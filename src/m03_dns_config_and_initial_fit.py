import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from tqdm import tqdm
from ycml.config import load_yaml, curve_config

# ----------------------------
# Config / constants
# ----------------------------
CFG = load_yaml("configs/base.yaml")
C = curve_config(CFG)

Y_COLS   = C["Y_COLS"]        # ['SVENY01', ..., 'SVENY30']
MAT_GRID = np.asarray(C["MAT_GRID"], dtype=float)  # [1.0, 2.0, ..., 30.0]
LAM      = C["LAM"]
WINDOW   = C["WINDOW"]
HORIZONS = C["HORIZONS"]
weights  = C["WEIGHTS"] 

df = pd.read_parquet("data/processed/spots.parquet")

# ----------------------------
# DNS loadings and factor extraction
# ----------------------------
def dl_loadings(tau, lam):
    """
    DNS loadings for yields (continuous compounding).
    tau: array-like maturities in years
    lam: positive scalar (1/years)
    Returns: array (len(tau) x 3) columns [f1, f2, f3]
    """
    tau = np.asarray(tau, dtype=float)
    lam = float(lam)
    if lam <= 0:
        raise ValueError("lambda must be positive")

    x  = lam * tau
    f1 = np.ones_like(tau, dtype=float)
    f2 = (1 - np.exp(-x)) / np.where(x == 0.0, 1.0, x)  # safe as x -> 0
    f3 = f2 - np.exp(-x)
    return np.column_stack([f1, f2, f3])

def extract_dns_factors(Y_df, lam=LAM):
    """
    Row-wise OLS: y_t ≈ F b_t, where F depends only on lam and MAT_GRID.
    Y_df: (n_days x 30) yields in PERCENT
    Returns: (n_days x 3) DataFrame: LEVEL, SLOPE, CURV (in percent)
    """
    F = dl_loadings(MAT_GRID, lam)           # (30 x 3)
    FtF_inv = np.linalg.inv(F.T @ F)
    P = FtF_inv @ F.T                         # (3 x 30) projection matrix
    B = (P @ Y_df.values.T).T                 # (n_days x 3)
    return pd.DataFrame(B, index=Y_df.index, columns=["LEVEL","SLOPE","CURV"])


# ----------------------------
# Decoding factors -> yields
# ----------------------------
def decode_dns_factors_to_yields(f_ser, lam=LAM):
    F = dl_loadings(MAT_GRID, lam)
    b = np.array([f_ser["LEVEL"], f_ser["SLOPE"], f_ser["CURV"]], dtype=float)
    y = F @ b
    return pd.Series(y, index=Y_COLS)


# ----------------------------
# Baselines
# ----------------------------
def rw_pred(df_yields, h):
    """
    Random-walk:  ŷ_{t+h} = y_t
    Returned index aligns to the target date (t+h) via shift(h).
    """
    return df_yields[Y_COLS].shift(h)

def rolling_var_dns_scaled(df_factors, h, window=WINDOW, start_idx=None):
    """
    DNS VAR(1) with per-window standardisation (z-scoring).
    Forecast in z-space, unscale, then decode to yields.
    Returns: DataFrame of predicted yields aligned to target dates.
    """
    dates = df_factors.index
    preds = []

    if start_idx is None:
        # ensure enough history before first forecast
        start_idx = max(window, 250)

    for i in tqdm(range(start_idx, len(dates)-h), desc=f"DNS VAR(scaled) h={h}"):
        lo = max(0, i - window)
        train = df_factors.iloc[lo:i].dropna().reset_index(drop=True)
        if len(train) < 120:  # need some history
            continue

        # per-window standardisation for numerical stability
        mu = train.mean()
        sd = train.std().replace(0.0, 1.0)
        Z  = (train - mu) / sd

        res  = VAR(Z).fit(1)
        last = Z.values[-res.k_ar:]
        z_h  = res.forecast(last, steps=h)[-1]
        f_h  = pd.Series(mu + z_h*sd, index=df_factors.columns)

        # decode to yields
        yhat = decode_dns_factors_to_yields(f_h)
        if np.isfinite(yhat.values).all():
            preds.append((dates[i+h], yhat))

    if not preds:
        return pd.DataFrame(columns=Y_COLS)

    pred_df = pd.DataFrame({d: y for d, y in preds}).T
    pred_df.index.name = "Date"
    return pred_df


# ----------------------------
# Metrics
# ----------------------------
def rmse_per_maturity(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    """
    Unweighted RMSE per maturity (column). Assumes y_true/y_pred aligned.
    Returns Series indexed by columns of Y_COLS.
    """
    err = (y_true - y_pred)
    return np.sqrt((err ** 2).mean(axis=0))

def weighted_rmse_scalar(y_true, y_pred, w):
    """
    Single weighted RMSE number (bps) across maturities & time.
    """
    err2 = (y_true - y_pred)**2
    w_aligned = w.reindex(err2.columns)
    return float(np.sqrt((err2 * w_aligned).mean().mean()))


if __name__ == "__main__":
    # ----------------------------
    # Prepare inputs
    # ----------------------------
    # Ensure yields exist and are clean (NS-only panel you produced)
    Y_all = df[Y_COLS].dropna()

    # Extract DNS factors once (on the same panel we forecast)
    factors_dns = extract_dns_factors(Y_all, lam=LAM)

    # ----------------------------
    # Run evaluations
    # ----------------------------
    start_idx = WINDOW + 1  # be conservative about first forecast date
    results_rmse = {}
    results_wrmse = {}

    for h in HORIZONS:
        # DNS VAR(scaled)
        dns_pred = rolling_var_dns_scaled(factors_dns, h=h, window=WINDOW, start_idx=start_idx)

        # Align truth & RW to DNS prediction dates
        y_true = Y_all.loc[dns_pred.index, Y_COLS]
        rw_h   = rw_pred(Y_all, h).loc[dns_pred.index]

        # RMSE per maturity (bps)
        rmse_dns = rmse_per_maturity(y_true, dns_pred) * 100.0
        rmse_rw  = rmse_per_maturity(y_true, rw_h)     * 100.0
        results_rmse[h] = pd.DataFrame({"RW": rmse_rw, "DNS_VAR(1)-scaled": rmse_dns})

        # Weighted RMSE (bps)
        wrmse_dns = weighted_rmse_scalar(y_true, dns_pred, weights) * 100.0
        wrmse_rw  = weighted_rmse_scalar(y_true, rw_h, weights)     * 100.0
        results_wrmse[h] = {"RW": wrmse_rw, "DNS_VAR(1)-scaled": wrmse_dns}


    print(results_rmse)