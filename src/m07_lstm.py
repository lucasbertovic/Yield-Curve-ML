# ============================================================
# LSTM — PERIODIC REFRESH backtest (multi-horizon, deterministic)
# ============================================================
# - Retrains every `refresh_every` obs on a rolling `train_window`
# - Inputs: last `seq_len` full yield curves (standardized w/ train stats)
# - Outputs: future yield curve for *all* requested horizons in one shot
# - Loss: original-scale MSE per maturity (equal weighting across maturities)
# ============================================================

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
# Dataset: seq -> multi-horizon targets
# ----------------------------
class SeqToMultiHorizonDataset(Dataset):
    """
    From standardized panel Z (T x k), create samples:
      X_t = Z[t-seq_len+1 : t+1], shape (seq_len, k)
      Y_t[h] = Z[t+h, :], for h in horizons
    Valid t: t ∈ [seq_len-1, T-1-H_MAX]
    """
    def __init__(self, Z: np.ndarray, horizons, seq_len: int, stride: int = 1):
        super().__init__()
        self.Z = Z.astype(np.float32)
        self.horizons = list(horizons)
        self.seq_len = int(seq_len)
        self.H_MAX = max(self.horizons)
        t_start = self.seq_len - 1
        t_end = Z.shape[0] - 1 - self.H_MAX
        if t_end < t_start:
            self.idxs = []
        else:
            self.idxs = list(range(t_start, t_end + 1, stride))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.Z[t - self.seq_len + 1 : t + 1, :]           # (seq_len, k)
        ys = [self.Z[t + h, :] for h in self.horizons]        # list of (k,)
        y = np.stack(ys, axis=0)                               # (H, k)
        return torch.from_numpy(x), torch.from_numpy(y)

# ----------------------------
# LSTM model
# ----------------------------
class LSTMForecaster(nn.Module):
    """
    Many-to-one LSTM that maps a sequence of shape (B, L, k) to all-horizon
    targets (B, H, k) in standardized space. We invert standardization later.
    """
    def __init__(self, in_dim=30, hidden_dim=128, num_layers=1, horizons=(1,5,10,22), dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizons = list(horizons)
        self.H = len(self.horizons)
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=(dropout if num_layers > 1 else 0.0),
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, self.H * in_dim)

    def forward(self, x):  # x: (B, L, k)
        out, _ = self.lstm(x)                # out: (B, L, hidden)
        last = out[:, -1, :]                 # (B, hidden)
        pred = self.head(last)               # (B, H*k)
        pred = pred.view(-1, self.H, self.in_dim)  # (B, H, k)
        return pred

# ----------------------------
# Train LSTM on a train slice (original-scale equal weighting)
# ----------------------------
def train_lstm(Y_tr: pd.DataFrame,
               horizons,
               seq_len: int = 252,
               stride: int = 1,
               hidden_dim: int = 128,
               num_layers: int = 1,
               dropout: float = 0.0,
               epochs: int = 20,
               batch_size: int = 256,
               lr: float = 1e-3,
               device: str = "cpu",
               seed: int = None,
               verbose: bool = False):
    """
    Returns: (model, mu, sd) and trains to minimize *original-scale* MSE per maturity.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Standardize with TRAIN stats only
    mu, sd = fit_standardizer(Y_tr)
    Z = apply_standardizer(Y_tr, mu, sd).values.astype(np.float32)
    sd2_vec = torch.tensor(sd.values.astype(np.float32) ** 2, device=device)  # (k,)

    ds = SeqToMultiHorizonDataset(Z, horizons=list(horizons), seq_len=seq_len, stride=stride)
    if len(ds) == 0:
        raise RuntimeError("[LSTM refresh] Training slice too short for seq_len/H_MAX.")
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    model = LSTMForecaster(in_dim=Z.shape[1], hidden_dim=hidden_dim,
                           num_layers=num_layers, horizons=list(horizons),
                           dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="none")

    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0; N = 0
        for xb, yb in dl:
            xb = xb.to(device)                 # (B, L, k) standardized
            yb = yb.to(device)                 # (B, H, k) standardized
            opt.zero_grad()
            pb = model(xb)                     # (B, H, k) standardized
            # original-scale MSE per maturity: (pb - yb)^2 * sd^2
            loss_mat = (pb - yb) ** 2 * sd2_vec  # broadcast to (B, H, k)
            loss = loss_mat.mean()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            N += xb.size(0)
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs):
            print(f"[LSTM] epoch {ep:3d}/{epochs}  recon(MSE, original-scale)={tot/max(1,N):.6f}")

    return model, mu, sd

@torch.no_grad()
def lstm_predict_block(model: LSTMForecaster,
                       Y_block: pd.DataFrame,
                       mu: pd.Series,
                       sd: pd.Series,
                       seq_len: int,
                       horizons,
                       device: str = "cpu") -> dict:
    """
    Produce predictions for each origin in Y_block.index *after* the first seq_len rows.
    Returns: dict h -> list[(date, pd.Series of yields)]
    """
    Z = apply_standardizer(Y_block, mu, sd).values.astype(np.float32)
    H = len(horizons)
    k = Z.shape[1]
    preds_by_h = {h: [] for h in horizons}
    dates = Y_block.index

    for end in range(seq_len, len(Z)):  # 'end' is exclusive index => last available index is end-1
        x = torch.from_numpy(Z[end - seq_len : end, :]).unsqueeze(0).to(device)  # (1, L, k)
        p = model(x).cpu().numpy()[0]  # (H, k) standardized
        # invert to yields
        p_df_std = pd.DataFrame(p, index=horizons, columns=Y_COLS)
        p_df = invert_standardizer(p_df_std, mu, sd)  # (H, k) original-scale
        origin_pos = end  # corresponds to i_pos in AE code
        for j, h in enumerate(horizons):
            tgt_idx = origin_pos + h
            if tgt_idx < len(dates):
                preds_by_h[h].append((dates[tgt_idx], p_df.iloc[j]))
    return preds_by_h

# ----------------------------
# Periodic refresh loop (deterministic)
# ----------------------------
def lstm_periodic_refresh(Y_all: pd.DataFrame,
                          horizons,
                          seq_len: int,
                          train_window: int = 5000,
                          refresh_every: int = 252,
                          hidden_dim: int = 128,
                          num_layers: int = 1,
                          dropout: float = 0.0,
                          epochs: int = 20,
                          batch_size: int = 256,
                          lr: float = 1e-3,
                          stride: int = 1,
                          base_seed: int = 1337):
    """
    Retrain LSTM every `refresh_every` obs on last `train_window` obs.
    For each origin i in a block, use last `seq_len` curves to predict all horizons.
    """
    dates = Y_all.index
    T = len(dates)
    horizons = list(horizons)
    H_MAX = max(horizons)

    start_origin = max(train_window, int(seq_len))  # need both a training window and L past inputs
    if start_origin + H_MAX >= T:
        raise RuntimeError("[LSTM refresh] Not enough data to start backtest.")

    preds_by_h_all = {h: [] for h in horizons}
    refresh_points = list(range(start_origin, T - H_MAX, refresh_every))
    print(f"[LSTM refresh] total refreshes: {len(refresh_points)}, "
          f"train_win={train_window}, step={refresh_every}, seq_len={seq_len}, H_MAX={H_MAX}")

    for r_idx, origin0 in enumerate(refresh_points, 1):
        refresh_seed = base_seed + r_idx
        # Train on trailing window
        tr_lo = origin0 - train_window
        tr_hi = origin0
        train_slice = Y_all.iloc[tr_lo:tr_hi].dropna()
        if len(train_slice) < train_window * 0.95:
            warnings.warn(f"[LSTM refresh] Training slice at origin {origin0} has NaNs; size={len(train_slice)}.")
        print(f"[LSTM refresh] ({r_idx}/{len(refresh_points)})")
        model, mu, sd = train_lstm(
            train_slice, horizons=horizons, seq_len=seq_len, stride=stride,
            hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout,
            epochs=epochs, batch_size=batch_size, lr=lr,
            device=device, seed=refresh_seed, verbose=False
        )

        # Define block [origin0, origin1)
        origin1 = min(origin0 + refresh_every, T - H_MAX)
        # Build a block of inputs that provides past seq_len and reaches to origin1 + H_MAX
        # We'll just use Y_all[origin0 - seq_len : origin1 + H_MAX] standardized with (mu, sd)
        block_lo = max(0, origin0 - seq_len)
        block_hi = origin1 + H_MAX
        Y_block = Y_all.iloc[block_lo:block_hi]

        preds_by_h = lstm_predict_block(model, Y_block, mu, sd,
                                        seq_len=seq_len, horizons=horizons, device=device)

        # Keep only origins inside [origin0, origin1); lstm_predict_block already aligned by date
        # so we only need to append (no duplicates across blocks)
        for h in horizons:
            preds_by_h_all[h].extend(preds_by_h[h])

        del model  # free GPU

    # Assemble & score
    LSTM_RESULTS = {}
    LSTM_WRMSE = []
    for h in horizons:
        if preds_by_h_all[h]:
            df_h = pd.DataFrame({d: y for d, y in preds_by_h_all[h]}).T
            df_h.index.name = "Date"
            df_h = df_h.reindex(columns=Y_COLS)
        else:
            df_h = pd.DataFrame(columns=Y_COLS)
        LSTM_RESULTS[h] = df_h

        idx = Y_all.index.intersection(df_h.index)
        if len(idx):
            wrmse_bps = weighted_rmse_scalar(Y_all.loc[idx, Y_COLS], df_h.loc[idx, Y_COLS], weights) * 100.0
            LSTM_WRMSE.append({"model": "LSTM_refresh", "h": h, "wrmse_bps": float(wrmse_bps)})

    LSTM_SUMMARY_DF = pd.DataFrame(LSTM_WRMSE).sort_values(["h", "wrmse_bps"]).reset_index(drop=True)
    return LSTM_RESULTS, LSTM_SUMMARY_DF

# ----------------------------
# Run periodic refresh backtest (deterministic)
# ----------------------------
# By default, use your existing WINDOW as the LSTM sequence length to keep it analogous.
LSTM_RESULTS, LSTM_SUMMARY_DF = lstm_periodic_refresh(
    Y_all=Y_all,
    horizons=list(HORIZONS),
    seq_len=int(WINDOW),          # analogous to VAR window, e.g., 504
    train_window=5000,            # trailing window LSTM trains on
    refresh_every=252,            # ~annual refresh
    hidden_dim=128,
    num_layers=1,
    dropout=0.0,
    epochs=20,                    # you can increase if training is stable/fast
    batch_size=256,
    lr=1e-3,
    stride=1,                     # >1 to subsample training steps for speed
    base_seed=_BASE_SEED
)

print("\n=== LSTM (Periodic Refresh) — WRMSE (bps) by horizon ===")
if len(LSTM_SUMMARY_DF):
    print(LSTM_SUMMARY_DF.to_string(index=False))
else:
    print("No LSTM results produced.")

for h, df in LSTM_RESULTS.items():
    # ensure Date is an index (it already is, but this is safe)
    df = df.sort_index()
    df.to_parquet(f"data/processed/predictions/lstm/lstm_predictions_h{h}.parquet", index=True) 
