# ------------------------------------------------------------
# 0) Helpers for generalized rolling forecasting & evaluation
# ------------------------------------------------------------

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from tqdm import tqdm
from ycml.config import load_yaml, curve_config
from m03_dns_config_and_initial_fit import weighted_rmse_scalar, extract_dns_factors, decode_dns_factors_to_yields, ns_yield

CFG = load_yaml("configs/base.yaml")
C = curve_config(CFG)

Y_COLS   = C["Y_COLS"]        # ['SVENY01', ..., 'SVENY30']
MAT_GRID = np.asarray(C["MAT_GRID"], dtype=float)  # [1.0, 2.0, ..., 30.0]
LAM      = C["LAM"]
WINDOW   = C["WINDOW"]
HORIZONS = C["HORIZONS"]
weights  = C["WEIGHTS"] 

df = pd.read_parquet("data/processed/spots.parquet")
Y_all = df[Y_COLS].dropna()


def _make_lagged_matrix(arr: np.ndarray, p: int, add_const: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lagged design matrix for VAR(p) (no exogenous).
    arr: (T x k) array, ordered by time.
    Returns: (Y, X) with shapes ((T-p) x k), ((T-p) x (k*p [+1]))
    """
    T, k = arr.shape
    if T <= p:
        raise ValueError("Not enough observations to build lagged matrix.")
    Y = arr[p:, :]
    X_list = []
    for i in range(1, p+1):
        X_list.append(arr[p-i:-i, :])
    X = np.concatenate(X_list, axis=1)  # (T-p) x (k*p)
    if add_const:
        X = np.concatenate([np.ones((T - p, 1)), X], axis=1)  # intercept first
    return Y, X

def _ridge_var_fit(Z: np.ndarray, p: int, alpha: float) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Closed-form ridge VAR(p) on standardized data Z (T x k).
    alpha: L2 penalty applied to all lag coefficients (not the intercept).
    Returns: (intercept (k,), [A1,...,Ap] each (k x k))
    """
    Y, X = _make_lagged_matrix(Z, p, add_const=True)        # Y:(T-p x k), X:(T-p x (1 + k*p))
    n_obs, k = Y.shape
    kxp = X.shape[1] - 1                                     # number of lagged regressors
    # Penalty matrix: do NOT penalize intercept (first column)
    Lambda = np.zeros((X.shape[1], X.shape[1]))
    Lambda[1:, 1:] = alpha * np.eye(kxp)
    # Solve (X'X + Lambda) B = X'Y
    XtX = X.T @ X
    B = np.linalg.solve(XtX + Lambda, X.T @ Y)               # B: ((1+k*p) x k)
    c = B[0, :]                                              # intercept (k,)
    coef = B[1:, :]                                          # (k*p) x k
    A_list = []
    for i in range(p):
        # Rows for lag i: (i*k : (i+1)*k)
        A_i = coef[i*k:(i+1)*k, :].T                         # shape (k x k)
        A_list.append(A_i)
    return c, A_list

def _ols_var_fit(Z: np.ndarray, p: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    OLS VAR(p) via statsmodels; returns intercept and A_1..A_p matrices.
    """
    res = VAR(Z).fit(p)
    c = res.intercept                                       # (k,)
    # res.coefs: (p, k, k) with order [lag, eqn, regressor-dim]
    A_list = [res.coefs[i].copy() for i in range(res.k_ar)] # list of (k x k)
    return c, A_list

def _iterative_forecast(last_hist: np.ndarray,
                        intercept: np.ndarray,
                        A_list: List[np.ndarray],
                        h: int,
                        damp: float = 1.0) -> np.ndarray:
    """
    Multi-step forecast given last p observations (p x k), intercept (k,), and A_lag matrices.
    Applies damping by scaling each lag matrix by 'damp'.
    Returns z_{t+h} forecast (k,)
    """
    p = len(A_list)
    k = last_hist.shape[1]
    hist = last_hist.copy()  # shape (p x k), rows: [z_{t-p+1}, ..., z_t]
    for _ in range(h):
        z_next = intercept.copy()
        for i in range(1, p+1):
            z_next += (damp * A_list[i-1]) @ hist[-i, :]
        # append and slide window
        hist = np.vstack([hist[1:, :], z_next[None, :]])
    return hist[-1, :]

# ------------------------------------------------------------
# 1) DNS — grid/search for (lambda, window, VAR order)
# ------------------------------------------------------------

def rolling_dns_var_forecast(Y_panel: pd.DataFrame,
                             lam: float,
                             h: int,
                             window: int,
                             order: int = 1,
                             standardize: bool = True,
                             model: str = "ols",
                             ridge_alpha: float = 0.0,
                             damp: float = 1.0,
                             difference: bool = False,
                             start_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Generalized rolling forecast using DNS factors (LEVEL, SLOPE, CURV) and VAR(p).
    Returns predicted yields aligned to target dates (index: target date; columns: Y_COLS).
    """
    factors = extract_dns_factors(Y_panel, lam=lam)
    dates = factors.index
    preds: List[Tuple[pd.Timestamp, pd.Series]] = []

    if start_idx is None:
        start_idx = max(window, 250)

    for i in tqdm(range(start_idx, len(dates) - h), desc=f"DNS VAR(p={order}) lam={lam} h={h}"):
        lo = max(0, i - window)
        train = factors.iloc[lo:i].dropna()
        if len(train) < max(120, order + 2):
            continue

        # (A) Choose series to fit (levels or differences)
        if difference:
            train_fit = train.diff().dropna()
            # last levels and last p deltas
            last_level = train.iloc[-1].copy()
            last_deltas = train.diff().dropna().iloc[-order:].copy()
        else:
            train_fit = train.copy()
            last_hist = train_fit.iloc[-order:].copy()

        # (B) Standardize (component-wise) if requested
        if standardize:
            mu = train_fit.mean()
            sd = train_fit.std().replace(0.0, 1.0)
            Z = (train_fit - mu) / sd
        else:
            mu = pd.Series(0.0, index=train_fit.columns)
            sd = pd.Series(1.0, index=train_fit.columns)
            Z = train_fit

        Z_arr = Z.values
        k = Z_arr.shape[1]

        # (C) Fit VAR
        if model == "ridge":
            c, A_list = _ridge_var_fit(Z_arr, p=order, alpha=float(ridge_alpha))
        else:  # "ols"
            c, A_list = _ols_var_fit(Z_arr, p=order)

        # (D) Prepare last history in Z-space
        if difference:
            # last deltas in Z space:
            last_deltas_Z = ((last_deltas - mu) / sd).values  # shape (order x k)
            z_h = _iterative_forecast(last_deltas_Z, c, A_list, h=h, damp=damp)
            # unscale cumulative deltas path
            # We'll forecast the entire path of deltas to accumulate:
            # quick approximation: replicate z_h for just the h-step endpoint (single-shot);
            # better: roll forward step-by-step to get all deltas; implement loop:
            hist = last_deltas_Z.copy()
            deltas_unscaled = []
            for _ in range(h):
                step = c.copy()
                for lag in range(1, order+1):
                    step += (damp * A_list[lag-1]) @ hist[-lag, :]
                hist = np.vstack([hist[1:, :], step[None, :]])
                deltas_unscaled.append(pd.Series(mu + step * sd, index=train_fit.columns))
            # sum deltas and add to last level
            f_h = last_level + pd.concat(deltas_unscaled, axis=1).sum(axis=1)
        else:
            last_hist_Z = ((last_hist - mu) / sd).values      # (order x k)
            z_h = _iterative_forecast(last_hist_Z, c, A_list, h=h, damp=damp)
            f_h = pd.Series(mu + z_h * sd, index=train_fit.columns)

        # (E) Decode to yields
        yhat = decode_dns_factors_to_yields(f_h, lam=lam)
        if np.isfinite(yhat.values).all():
            preds.append((dates[i + h], yhat))

    if not preds:
        return pd.DataFrame(columns=Y_COLS)

    pred_df = pd.DataFrame({d: y for d, y in preds}).T
    pred_df.index.name = "Date"
    pred_df = pred_df.loc[pred_df.index.intersection(Y_panel.index)]
    return pred_df

def grid_search_dns(Y_panel: pd.DataFrame,
                    horizons: List[int],
                    lam_grid: List[float],
                    window_grid: List[int],
                    order_grid: List[int]) -> pd.DataFrame:
    """
    Evaluate WRMSE for DNS-OLS over the specified grids.
    Returns a long DataFrame with columns: lam, window, order, h, wrmse_bps
    """
    rows = []
    for lam in lam_grid:
        for W in window_grid:
            for p in order_grid:
                for h in horizons:
                    dns_pred = rolling_dns_var_forecast(Y_panel, lam=lam, h=h, window=W, order=p,
                                                       standardize=True, model="ols", damp=1.0, difference=False)
                    if dns_pred.empty:
                        continue
                    y_true = Y_panel.loc[dns_pred.index, Y_COLS]
                    wrmse = weighted_rmse_scalar(y_true, dns_pred, weights) * 100.0
                    rows.append({"lam": lam, "window": W, "order": p, "h": h, "wrmse_bps": wrmse})
    return pd.DataFrame(rows).sort_values(["h", "wrmse_bps", "lam", "window", "order"])

# ------------------------------------------------------------
# 2) Regularisation: ridge VAR and damped forecasts (alpha<1)
# ------------------------------------------------------------

def dns_ridge_and_damped(Y_panel: pd.DataFrame,
                         lam: float,
                         h: int,
                         window: int,
                         order: int,
                         ridge_alphas: List[float],
                         damps: List[float]) -> pd.DataFrame:
    """
    Compare ridge VAR and damping parameters for a fixed (lam, window, order).
    Returns a summary DataFrame.
    """
    rows = []
    for alpha in ridge_alphas:
        for dmp in damps:
            pred = rolling_dns_var_forecast(Y_panel, lam=lam, h=h, window=window, order=order,
                                            standardize=True, model="ridge", ridge_alpha=alpha,
                                            damp=dmp, difference=False)
            if pred.empty:
                continue
            y_true = Y_panel.loc[pred.index, Y_COLS]
            wrmse = weighted_rmse_scalar(y_true, pred, weights) * 100.0
            rows.append({"lam": lam, "window": window, "order": order,
                         "ridge_alpha": alpha, "damp": dmp, "h": h, "wrmse_bps": wrmse})
    return pd.DataFrame(rows).sort_values(["h", "wrmse_bps"])

# ------------------------------------------------------------
# 3) Alternative factorisations
#    (a) Forecast NS parameters directly (log-transform tau1)
# ------------------------------------------------------------

def decode_ns_params_to_yields(params_ser: pd.Series) -> pd.Series:
    """
    Decode NS params (LEVEL=beta0, SLOPE=beta1, CURV=beta2, TAU1) into yields for MAT_GRID.
    params_ser keys: ["NS_BETA0","NS_BETA1","NS_BETA2","NS_TAU1"]
    """
    b0 = float(params_ser["NS_BETA0"])
    b1 = float(params_ser["NS_BETA1"])
    b2 = float(params_ser["NS_BETA2"])
    T1 = float(params_ser["NS_TAU1"])
    y = ns_yield(MAT_GRID, b0, b1, b2, T1)
    return pd.Series(y, index=Y_COLS)

def rolling_nsparam_var_forecast(df_with_ns: pd.DataFrame,
                                 h: int,
                                 window: int,
                                 order: int = 1,
                                 standardize: bool = True,
                                 model: str = "ols",
                                 ridge_alpha: float = 0.0,
                                 damp: float = 1.0,
                                 difference: bool = False,
                                 start_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Rolling VAR on NS parameters: [NS_BETA0, NS_BETA1, NS_BETA2, log(NS_TAU1)].
    """
    cols = ["NS_BETA0","NS_BETA1","NS_BETA2","NS_TAU1"]
    NS = df_with_ns[cols].dropna().copy()
    NS["NS_TAU1_LOG"] = np.log(NS["NS_TAU1"].clip(lower=1e-4))
    factors = NS[["NS_BETA0","NS_BETA1","NS_BETA2","NS_TAU1_LOG"]]
    dates = factors.index
    preds: List[Tuple[pd.Timestamp, pd.Series]] = []

    if start_idx is None:
        start_idx = max(window, 250)

    for i in tqdm(range(start_idx, len(dates) - h), desc=f"NSparam VAR(p={order}) h={h}"):
        lo = max(0, i - window)
        train = factors.iloc[lo:i].dropna()
        if len(train) < max(120, order + 2):
            continue

        if difference:
            train_fit = train.diff().dropna()
            last_level = train.iloc[-1].copy()
            last_deltas = train.diff().dropna().iloc[-order:].copy()
        else:
            train_fit = train.copy()
            last_hist = train_fit.iloc[-order:].copy()

        if standardize:
            mu = train_fit.mean()
            sd = train_fit.std().replace(0.0, 1.0)
            Z = (train_fit - mu) / sd
        else:
            mu = pd.Series(0.0, index=train_fit.columns)
            sd = pd.Series(1.0, index=train_fit.columns)
            Z = train_fit

        Z_arr = Z.values
        if model == "ridge":
            c, A_list = _ridge_var_fit(Z_arr, p=order, alpha=float(ridge_alpha))
        else:
            c, A_list = _ols_var_fit(Z_arr, p=order)

        if difference:
            last_deltas_Z = ((last_deltas - mu) / sd).values
            # step-by-step to cumulate deltas
            hist = last_deltas_Z.copy()
            deltas_unscaled = []
            for _ in range(h):
                step = c.copy()
                for lag in range(1, order+1):
                    step += (damp * A_list[lag-1]) @ hist[-lag, :]
                hist = np.vstack([hist[1:, :], step[None, :]])
                deltas_unscaled.append(pd.Series(mu + step * sd, index=train_fit.columns))
            f_h = last_level + pd.concat(deltas_unscaled, axis=1).sum(axis=1)
        else:
            last_hist_Z = ((last_hist - mu) / sd).values
            z_h = _iterative_forecast(last_hist_Z, c, A_list, h=h, damp=damp)
            f_h = pd.Series(mu + z_h * sd, index=train_fit.columns)

        # inverse transform of tau1
        f_decode = pd.Series(index=["NS_BETA0","NS_BETA1","NS_BETA2","NS_TAU1"], dtype=float)
        f_decode["NS_BETA0"] = f_h["NS_BETA0"]
        f_decode["NS_BETA1"] = f_h["NS_BETA1"]
        f_decode["NS_BETA2"] = f_h["NS_BETA2"]
        f_decode["NS_TAU1"]  = float(np.exp(f_h["NS_TAU1_LOG"]).clip(1e-4, 1e4))

        yhat = decode_ns_params_to_yields(f_decode)
        if np.isfinite(yhat.values).all():
            preds.append((dates[i + h], yhat))

    if not preds:
        return pd.DataFrame(columns=Y_COLS)
    pred_df = pd.DataFrame({d: y for d, y in preds}).T
    pred_df.index.name = "Date"
    pred_df = pred_df.loc[pred_df.index.intersection(Y_all.index)]
    return pred_df

# ------------------------------------------------------------
# 3) Alternative factorisations
#    (b) Rolling PCA factors (k components), VAR on scores
# ------------------------------------------------------------

def _rolling_pca_basis(X: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA basis from X (n x m), mean-centered; returns (mean, V_k, explained_var)
    V_k: (m x k) eigenvectors of covariance (columns sorted by eigenvalue desc)
    """
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # covariance across columns (m x m)
    S = np.cov(Xc, rowvar=False)
    vals, vecs = np.linalg.eigh(S)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]  # columns
    V_k = vecs[:, :k]    # (m x k)
    return mean.ravel(), V_k, vals[:k]

def rolling_pca_var_forecast(Y_panel: pd.DataFrame,
                             h: int,
                             window: int,
                             k_comp: int = 3,
                             order: int = 1,
                             standardize: bool = True,
                             model: str = "ols",
                             ridge_alpha: float = 0.0,
                             damp: float = 1.0,
                             difference: bool = False,
                             start_idx: Optional[int] = None) -> pd.DataFrame:
    """
    Rolling PCA: within each window, compute PCA basis on yields (Y), project to k scores,
    fit VAR(p) on scores (optionally standardized), forecast h steps, and reconstruct yields.
    """
    dates = Y_panel.index
    preds: List[Tuple[pd.Timestamp, pd.Series]] = []

    if start_idx is None:
        start_idx = max(window, 250)

    for i in tqdm(range(start_idx, len(dates) - h), desc=f"PCA(k={k_comp}) VAR(p={order}) h={h}"):
        lo = max(0, i - window)
        train_Y = Y_panel.iloc[lo:i].dropna()
        if len(train_Y) < max(120, order + 2):
            continue

        # Build PCA basis on training window
        mvec, V_k, _ = _rolling_pca_basis(train_Y.values, k=k_comp)  # mvec:(30,), V_k:(30 x k)
        # Scores for the entire window (n x k)
        Xc = train_Y.values - mvec[None, :]
        scores = Xc @ V_k                                     # (n x k)
        scores_df = pd.DataFrame(scores, index=train_Y.index, columns=[f"PC{i+1}" for i in range(k_comp)])

        if difference:
            train_fit = scores_df.diff().dropna()
            last_level_scores = scores_df.iloc[-1].copy()
            last_deltas = scores_df.diff().dropna().iloc[-order:].copy()
        else:
            train_fit = scores_df.copy()
            last_hist = train_fit.iloc[-order:].copy()

        if standardize:
            mu = train_fit.mean()
            sd = train_fit.std().replace(0.0, 1.0)
            Z = (train_fit - mu) / sd
        else:
            mu = pd.Series(0.0, index=train_fit.columns)
            sd = pd.Series(1.0, index=train_fit.columns)
            Z = train_fit

        Z_arr = Z.values
        if model == "ridge":
            c, A_list = _ridge_var_fit(Z_arr, p=order, alpha=float(ridge_alpha))
        else:
            c, A_list = _ols_var_fit(Z_arr, p=order)

        if difference:
            last_deltas_Z = ((last_deltas - mu) / sd).values
            # step-by-step to cumulate deltas in score space
            hist = last_deltas_Z.copy()
            deltas_unscaled = []
            for _ in range(h):
                step = c.copy()
                for lag in range(1, order+1):
                    step += (damp * A_list[lag-1]) @ hist[-lag, :]
                hist = np.vstack([hist[1:, :], step[None, :]])
                deltas_unscaled.append(pd.Series(mu + step * sd, index=train_fit.columns))
            scores_h = last_level_scores + pd.concat(deltas_unscaled, axis=1).sum(axis=1)
        else:
            last_hist_Z = ((last_hist - mu) / sd).values
            z_h = _iterative_forecast(last_hist_Z, c, A_list, h=h, damp=damp)
            scores_h = pd.Series(mu + z_h * sd, index=train_fit.columns)

        # Reconstruct yields: y = mean + V_k @ scores
        yhat_vec = mvec + (V_k @ scores_h.values)
        yhat = pd.Series(yhat_vec, index=Y_COLS)
        if np.isfinite(yhat.values).all():
            preds.append((dates[i + h], yhat))

    if not preds:
        return pd.DataFrame(columns=Y_COLS)
    pred_df = pd.DataFrame({d: y for d, y in preds}).T
    pred_df.index.name = "Date"
    pred_df = pred_df.loc[pred_df.index.intersection(Y_panel.index)]
    return pred_df

# ------------------------------------------------------------
# 4) Wrapper runners — produce forecasts & evaluations
# ------------------------------------------------------------

# (A) DNS grid search (lambda/window/order) — OLS VAR
LAM_GRID    = [0.1, 0.25, 0.5, 0.8, 1]
WINDOW_GRID = [252, 504, 756]
ORDER_GRID  = [1, 2]
dns_grid_results = grid_search_dns(Y_all, HORIZONS, LAM_GRID, WINDOW_GRID, ORDER_GRID)

# Pick a "best" DNS config per horizon (lowest WRMSE); build a dict for convenience
best_dns_configs = {}
for h in HORIZONS:
    sub = dns_grid_results[dns_grid_results["h"] == h]
    if len(sub):
        best_row = sub.nsmallest(1, "wrmse_bps").iloc[0]
        best_dns_configs[h] = {"lam": float(best_row["lam"]),
                               "window": int(best_row["window"]),
                               "order": int(best_row["order"])}

# (B) DNS ridge & damped around the best config for each horizon
ridge_alphas = [0.1, 1.0, 5.0]
damps = [1.0, 0.95, 0.9]
dns_ridge_damped_results = []
for h in HORIZONS:
    cfg = best_dns_configs.get(h, None)
    if not cfg:
        continue
    r = dns_ridge_and_damped(Y_all, lam=cfg["lam"], h=h, window=cfg["window"],
                              order=cfg["order"], ridge_alphas=ridge_alphas, damps=damps)
    if len(r):
        dns_ridge_damped_results.append(r)
dns_ridge_damped_results = pd.concat(dns_ridge_damped_results, axis=0) if dns_ridge_damped_results else pd.DataFrame()

# (C) DNS with differences (Δb_t), using the same best DNS config
dns_diff_results = []
dns_diff_preds = {}  # store predictions for later DM tests
rw_preds = {}        # store RW predictions for later DM tests
for h in HORIZONS:
    cfg = best_dns_configs.get(h, None)
    if not cfg:
        continue
    pred_dns_diff = rolling_dns_var_forecast(Y_all, lam=cfg["lam"], h=h, window=cfg["window"],
                                             order=cfg["order"], standardize=True, model="ols",
                                             damp=1.0, difference=True)
    if pred_dns_diff.empty:
        continue
    y_true = Y_all.loc[pred_dns_diff.index, Y_COLS]
    wrmse = weighted_rmse_scalar(y_true, pred_dns_diff, weights) * 100.0
    dns_diff_results.append({"h": h, "wrmse_bps": wrmse})
    dns_diff_preds[h] = pred_dns_diff.copy()
    # RW baseline for same dates
    rw_h = Y_all[Y_COLS].shift(h).loc[pred_dns_diff.index]
    rw_preds[h] = rw_h

dns_diff_results = pd.DataFrame(dns_diff_results)

# (D) NS-parameter VAR (with log tau1), OLS vs ridge, with differences option
nsparam_eval = []
nsparam_preds = {}
for h in HORIZONS:
    # choose a window/order (reuse the best DNS window/order to keep parity)
    cfg = best_dns_configs.get(h, {"window": WINDOW, "order": 1})
    for model_type in ["ols", "ridge"]:
        pred_ns = rolling_nsparam_var_forecast(df, h=h, window=cfg["window"], order=cfg["order"],
                                               standardize=True, model=model_type,
                                               ridge_alpha=1.0 if model_type == "ridge" else 0.0,
                                               damp=1.0, difference=False)
        if pred_ns.empty:
            continue
        y_true = Y_all.loc[pred_ns.index, Y_COLS]
        wrmse = weighted_rmse_scalar(y_true, pred_ns, weights) * 100.0
        nsparam_eval.append({"h": h, "model": model_type, "wrmse_bps": wrmse})
        nsparam_preds[(h, model_type)] = pred_ns.copy()
nsparam_eval = pd.DataFrame(nsparam_eval)

# (E) PCA factors (k=3), OLS vs ridge, with differences option
pca_eval = []
pca_preds = {}
for h in HORIZONS:
    cfg = best_dns_configs.get(h, {"window": WINDOW, "order": 1})
    for model_type in ["ols", "ridge"]:
        pred_pca = rolling_pca_var_forecast(Y_all, h=h, window=cfg["window"], k_comp=3, order=cfg["order"],
                                            standardize=True, model=model_type,
                                            ridge_alpha=1.0 if model_type == "ridge" else 0.0,
                                            damp=1.0, difference=False)
        if pred_pca.empty:
            continue
        y_true = Y_all.loc[pred_pca.index, Y_COLS]
        wrmse = weighted_rmse_scalar(y_true, pred_pca, weights) * 100.0
        pca_eval.append({"h": h, "model": model_type, "wrmse_bps": wrmse})
        pca_preds[(h, model_type)] = pred_pca.copy()
pca_eval = pd.DataFrame(pca_eval)



# ------------------------------------------------------------
# 6) Optional: Collate headline WRMSE summaries
# ------------------------------------------------------------

# DNS grid — best rows already in best_dns_configs; get their WRMSEs
dns_best_summary = []
for h, cfg in best_dns_configs.items():
    pred = rolling_dns_var_forecast(Y_all, lam=cfg["lam"], h=h, window=cfg["window"], order=cfg["order"],
                                    standardize=True, model="ols", damp=1.0, difference=False)
    if pred.empty:
        continue
    y_true = Y_all.loc[pred.index, Y_COLS]
    wrmse = weighted_rmse_scalar(y_true, pred, weights) * 100.0
    dns_best_summary.append({
        "h": h, "model": f"DNS_OLS(lam={cfg['lam']},W={cfg['window']},p={cfg['order']})",
        "wrmse_bps": wrmse
    })
dns_best_summary = pd.DataFrame(dns_best_summary)

# Ridge & damped summary (already computed)
dns_ridge_damped_summary = dns_ridge_damped_results.copy()

# NS-param summary
nsparam_summary = nsparam_eval.copy()

# PCA summary
pca_summary = pca_eval.copy()

# ------------------------------------------------------------
# 7) Quick printouts (optional; comment out if running in a notebook to avoid clutter)
# ------------------------------------------------------------
print("\n=== DNS OLS Grid Search (top 10 by h/wrmse) ===")
try:
    print(dns_grid_results.groupby("h").apply(lambda g: g.nsmallest(10, "wrmse_bps")))
except Exception:
    print(dns_grid_results.head(20))

print("\n=== DNS Best per Horizon (refit) ===")
print(dns_best_summary.sort_values("h"))

print("\n=== DNS Ridge + Damped (examples) ===")
print(dns_ridge_damped_summary.head(20))

print("\n=== DNS with Differences (WRMSE by horizon, bps) ===")
print(dns_diff_results.sort_values("h"))

print("\n=== NS-Param VAR Summary ===")
print(nsparam_summary.sort_values(["h","wrmse_bps"]))

print("\n=== PCA VAR Summary ===")
print(pca_summary.sort_values(["h","wrmse_bps"]))


for h, df in dns_diff_preds.items():
    # ensure Date is an index (it already is, but this is safe)
    df = df.sort_index()
    df.to_parquet(f"data/processed/predictions/dns_diff/dns_diff_predictions_h{h}.parquet", index=True) 


