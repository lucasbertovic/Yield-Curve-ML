# ============================================================
# Autoencoder (AE) + VAR(p) on *latent differences* — PERIODIC REFRESH backtest
# Reproducible + AE loss equalized across maturities
# ============================================================
# - AE retrained periodically on a rolling window
# - AE trains to minimize *original-scale* MSE equally across maturities
#   (even though inputs are standardized for stability)
# - VAR is fit on *first differences* of latent codes (Δz_t), not levels
# - Within each origin, we:
#     * z-score Δz_t per window for numerics,
#     * fit VAR(p) on standardized Δz_t,
#     * forecast a path of future standardized Δz,
#     * unstandardize each step and cumulatively sum to get future z-levels,
#     * decode z-levels back to full yield curves,
# ============================================================

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from ycml.config import load_yaml, curve_config, base_seed
from m02_refit_ns_params_and_values import ns_yield
from m03_dns_config_and_initial_fit import rmse_per_maturity, weighted_rmse_scalar, extract_dns_factors, decode_dns_factors_to_yields
from m05_model_helpers import *

# ----------------------------
# Required globals
# ----------------------------   
CFG = load_yaml("configs/base.yaml")
C = curve_config(CFG)

Y_COLS   = C["Y_COLS"]        # ['SVENY01', ..., 'SVENY30']
MAT_GRID = np.asarray(C["MAT_GRID"], dtype=float)  # [1.0, 2.0, ..., 30.0]
WINDOW   = C["WINDOW"]
HORIZONS = C["HORIZONS"]
weights  = C["WEIGHTS"] 

df = pd.read_parquet("data/processed/spots.parquet")
Y_all = df[Y_COLS].dropna()

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set a single base seed for the entire run so results are identical each time
_BASE_SEED = base_seed(CFG)
set_global_seeds(_BASE_SEED, use_cuda=(device == "cuda"))

# ----------------------------
# AE model + train / encode / decode helpers
# ----------------------------
class AutoEncoder(nn.Module):
    def __init__(self, in_dim=30, latent_dim=3, hidden=(64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2), nn.ReLU(),
            nn.Linear(h2, h1), nn.ReLU(),
            nn.Linear(h1, in_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)

def train_autoencoder(Y_tr: pd.DataFrame,
                      latent_dim=3,
                      hidden=(64, 32),
                      epochs=25,
                      batch_size=256,
                      lr=1e-3,
                      device="cpu",
                      verbose=True,
                      seed: int = None):
    """
    Train AE on TRAIN window using *original-scale* MSE (equal per maturity).
    Implementation detail:
      - Inputs/outputs are standardized for stability.
      - We compute per-feature loss in standardized space and multiply by sd^2,
        making the objective identical to unstandardized MSE in yield units.
    Returns: (model, mu, sd)
    """
    # Ensure deterministic init and dataloader shuffling per call
    if seed is not None:
        torch.manual_seed(seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Train stats
    mu, sd = fit_standardizer(Y_tr)
    Z_tr = apply_standardizer(Y_tr, mu, sd).values.astype(np.float32)

    # loss re-weighting vector: sd^2 per maturity (undoes z-score weighting)
    sd2_vec = torch.tensor(sd.values.astype(np.float32) ** 2, device=device)  # shape [in_dim]

    ds = TensorDataset(torch.from_numpy(Z_tr))
    # Deterministic shuffling: seed a local generator
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    model = AutoEncoder(in_dim=Z_tr.shape[1], latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="none")

    model.train()
    for ep in range(epochs):
        total = 0.0
        for (xb,) in dl:
            xb = xb.to(device)  # standardized inputs
            opt.zero_grad()
            xhat = model(xb)    # standardized reconstruction
            # per-feature squared error in z-space ...
            loss_mat_std = mse(xhat, xb)  # shape [batch, in_dim]
            # ... reweight by sd^2 to get original-scale MSE per feature
            loss = (loss_mat_std * sd2_vec).mean()
            loss.backward()
            opt.step()
            total += float(loss.item()) * xb.size(0)
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"[AE] epoch {ep+1}/{epochs}  recon(MSE, original-scale)={total/len(ds):.6f}")
    return model, mu, sd

@torch.no_grad()
def encode_series(model: AutoEncoder, df: pd.DataFrame, mu: pd.Series, sd: pd.Series, device="cpu") -> pd.DataFrame:
    Z = apply_standardizer(df, mu, sd).values.astype(np.float32)
    z = model.encode(torch.from_numpy(Z).to(device)).cpu().numpy()
    return pd.DataFrame(z, index=df.index, columns=[f"z{i+1}" for i in range(z.shape[1])])

@torch.no_grad()
def decode_series(model: AutoEncoder, codes_df: pd.DataFrame, mu: pd.Series, sd: pd.Series, device="cpu") -> pd.DataFrame:
    yhat_z = model.decode(torch.from_numpy(codes_df.values.astype(np.float32)).to(device)).cpu().numpy()
    yhat_std = pd.DataFrame(yhat_z, index=codes_df.index, columns=Y_COLS)
    return invert_standardizer(yhat_std, mu, sd)

# ----------------------------
# Periodic refresh loop (deterministic)
# ----------------------------
def ae_var_periodic_refresh(Y_all: pd.DataFrame,
                            horizons,
                            var_window: int,
                            ae_train_window: int = 5000,
                            refresh_every: int = 22,
                            ae_latent: int = 3,
                            ae_hidden=(64, 32),
                            ae_epochs: int = 25,
                            ae_bs: int = 256,
                            ae_lr: float = 1e-3,
                            var_order: int = 1,
                            standardize_codes: bool = True,
                            base_seed: int = 1337):
    """
    Retrain AE every `refresh_every` obs on last `ae_train_window` obs,
    then within that block:
      - encode curves to latent codes z_t,
      - form *first differences* Δz_t = z_t - z_{t-1},
      - z-score Δz_t per window (if standardize_codes),
      - fit VAR(p) on standardized Δz_t,
      - forecast the path of future Δz,
      - unstandardize and cumulatively sum to recover future z-levels,
      - decode z-levels to yield curves for each requested horizon.

    Deterministic run: we seed once globally and derive per-refresh seeds
    from `base_seed + r_idx` to keep results identical across executions.
    """
    dates = Y_all.index
    T = len(dates)
    horizons = list(horizons)
    H_MAX = max(horizons)

    # Start once we have both AE training window and VAR window available
    start_origin = max(ae_train_window, int(var_window))
    if start_origin + H_MAX >= T:
        raise RuntimeError("[AE+VAR refresh] Not enough data to start backtest given ae_train_window/var_window/H_MAX.")

    preds_by_h = {h: [] for h in horizons}
    refresh_points = list(range(start_origin, T - H_MAX, refresh_every))
    print(f"[AE+VAR Δ (diff) refresh] total refreshes: {len(refresh_points)}, "
          f"train_win={ae_train_window}, step={refresh_every}, var_window={var_window}, H_MAX={H_MAX}")

    for r_idx, origin0 in enumerate(refresh_points, 1):
        # Derive a deterministic per-refresh seed so results are identical across runs
        refresh_seed = base_seed + r_idx

        # 1) Train AE on last `ae_train_window` rows ending at origin0
        tr_lo = origin0 - ae_train_window
        tr_hi = origin0
        train_slice = Y_all.iloc[tr_lo:tr_hi].dropna()
        if len(train_slice) < ae_train_window * 0.95:
            warnings.warn(f"[AE+VAR refresh] Training slice at origin {origin0} has NaNs; size={len(train_slice)}.")
        if r_idx % 100 == 0:
            print(f"[AE+VAR refresh] ({r_idx}/{len(refresh_points)}) AE training {train_slice.index[0].date()} -> {train_slice.index[-1].date()}")

        ae_model, ae_mu, ae_sd = train_autoencoder(
            train_slice,
            latent_dim=ae_latent, hidden=ae_hidden,
            epochs=ae_epochs, batch_size=ae_bs, lr=ae_lr,
            device=device, verbose=False, seed=refresh_seed
        )

        # 2) Define origin block [origin0, origin1)
        origin1 = min(origin0 + refresh_every, T - H_MAX)

        # For efficiency: pre-encode just the slice needed to build all VAR windows inside this block:
        # we will need codes from [encode_lo, origin1), where encode_lo = origin0 - var_window
        encode_lo = max(0, origin0 - int(var_window))
        encode_hi = origin1
        codes_block = encode_series(ae_model, Y_all.iloc[encode_lo:encode_hi], ae_mu, ae_sd, device=device)

        # 3) Roll through each forecast origin i in the block
        rng = range(origin0, origin1)
        for i_pos in tqdm(rng, desc=f"[AE+VAR Δ (diff) refresh] block {r_idx}/{len(refresh_points)}", leave=False):
            # relative index inside codes_block
            r = i_pos - encode_lo
            if r <= var_window:
                continue
            train_c = codes_block.iloc[r - int(var_window): r].dropna()
            if len(train_c) < max(120, var_order + 2):
                continue

            # ---- NEW: differences of latent codes within the window ----
            d_train = train_c.diff().dropna()               # Δz_t (levels to diffs)
            if len(d_train) < var_order:
                continue
            last_level = train_c.iloc[-1].copy()            # z_t (the last observed level before forecasting)
            last_deltas = d_train.iloc[-var_order:].copy()  # Δz_{t-var_order+1} ... Δz_t

            # Standardize *deltas* per window (stable numerics)
            if standardize_codes:
                mu_d = d_train.mean()
                sd_d = d_train.std().replace(0.0, 1.0)
                Zd = (d_train - mu_d) / sd_d
            else:
                mu_d = pd.Series(0.0, index=d_train.columns)
                sd_d = pd.Series(1.0, index=d_train.columns)
                Zd = d_train

            # Fit VAR on standardized deltas; reuse for all horizons via forecast path
            c0, A_list = ols_var_fit(Zd.values, p=var_order)

            # Prepare last history in standardized delta-space
            last_hist_Zd = ((last_deltas - mu_d) / sd_d).values  # shape (p, k)

            # Forecast standardized delta path for steps 1..H_MAX
            dz_std_path = iterative_forecast_path(last_hist_Zd, c0, A_list, h_max=H_MAX, damp=1.0)  # (H_MAX x k)

            # Unstandardize each step's delta and cumulatively sum to get future levels
            dz_path = dz_std_path * sd_d.values[None, :] + mu_d.values[None, :]   # (H_MAX x k) in original delta units
            z_level_path = last_level.values[None, :] + np.cumsum(dz_path, axis=0)  # z_{t+1}, z_{t+2}, ..., z_{t+H_MAX}

            # Decode for each requested horizon
            for h in horizons:
                z_h_level = z_level_path[h - 1]  # 1-step is index 0
                z_fore = pd.Series(z_h_level, index=train_c.columns)
                yhat = decode_series(ae_model, z_fore.to_frame().T, ae_mu, ae_sd, device=device).iloc[0]
                tgt_date = dates[i_pos + h]
                if np.isfinite(yhat.values).all():
                    preds_by_h[h].append((tgt_date, yhat))

        # free up GPU memory per refresh
        del ae_model

    # 4) Assemble DataFrames and evaluate WRMSE
    AE_RESULTS = {}
    AE_WRMSE = []
    for h in horizons:
        if preds_by_h[h]:
            df_h = pd.DataFrame({d: y for d, y in preds_by_h[h]}).T
            df_h.index.name = "Date"
            df_h = df_h.reindex(columns=Y_COLS)
        else:
            df_h = pd.DataFrame(columns=Y_COLS)
        AE_RESULTS[h] = df_h

        idx = Y_all.index.intersection(df_h.index)
        if len(idx):
            wrmse_bps = weighted_rmse_scalar(Y_all.loc[idx, Y_COLS], df_h.loc[idx, Y_COLS], weights) * 100.0
            AE_WRMSE.append({"model": "AE_dVAR_refresh", "h": h, "wrmse_bps": float(wrmse_bps)})

    AE_SUMMARY_DF = pd.DataFrame(AE_WRMSE).sort_values(["h", "wrmse_bps"]).reset_index(drop=True)
    return AE_RESULTS, AE_SUMMARY_DF

# ----------------------------
# Run periodic refresh backtest (deterministic)
# ----------------------------
AE_RESULTS, AE_SUMMARY_DF = ae_var_periodic_refresh(
    Y_all=Y_all,
    horizons=list(HORIZONS),
    var_window=int(WINDOW),     # your existing VAR lookback (e.g., 504)
    ae_train_window=5000,
    refresh_every=252,          # your current setting
    ae_latent=3,                # try 3–5
    ae_hidden=(64, 32),
    ae_epochs=100,
    ae_bs=256,
    ae_lr=1e-3,
    var_order=1,
    standardize_codes=True,
    base_seed=_BASE_SEED        # ensures identical results across runs
)

print("\n=== AE+VAR on Δ latent codes (Periodic Refresh) — WRMSE (bps) by horizon ===")
if len(AE_SUMMARY_DF):
    print(AE_SUMMARY_DF.to_string(index=False))
else:
    print("No AE+VAR(Δ) results produced.")

for h, df in AE_RESULTS.items():
    # ensure Date is an index (it already is, but this is safe)
    df = df.sort_index()
    df.to_parquet(f"data/processed/predictions/ae_var_diff/ae_var_diff_predictions_h{h}.parquet", index=True) 
