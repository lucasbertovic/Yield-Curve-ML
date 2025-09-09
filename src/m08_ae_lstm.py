# ============================================================
# AE + LSTM — PERIODIC REFRESH backtest (deterministic)
# ============================================================
# - Train AE on trailing window (equal per-maturity reconstruction in original units)
# - Encode trailing window into latent codes
# - Train LSTM on sequences of latent codes → predict future latent codes (multi-horizon)
# - Decode predicted codes back to yields; evaluate WRMSE (bps)
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
from m06_ae_var_diff import AutoEncoder, train_autoencoder

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


@torch.no_grad()
def encode_series(model: AutoEncoder, df: pd.DataFrame, mu: pd.Series, sd: pd.Series, device="cpu") -> pd.DataFrame:
    Z = apply_standardizer(df, mu, sd).values.astype(np.float32)
    z = model.encode(torch.from_numpy(Z).to(device)).cpu().numpy()
    return pd.DataFrame(z, index=df.index, columns=[f"z{i+1}" for i in range(z.shape[1])])

def _freeze_model(m: nn.Module):
    for p in m.parameters(): p.requires_grad = False
    m.eval()
    return m

def _decode_latent_to_yield_torch(ae_model: AutoEncoder,
                                  z_bh: torch.Tensor,   # (B, H, latent_dim)
                                  mu_t: torch.Tensor,   # (k,)
                                  sd_t: torch.Tensor):  # (k,)
    """
    Decode latent codes to yields in original units (B, H, k), differentiable wrt z_bh.
    AE params are assumed frozen (no grad).
    """
    B, H, k_lat = z_bh.shape
    z2 = z_bh.reshape(B*H, k_lat)                 # (BH, latent)
    y_std = ae_model.decoder(z2)                   # (BH, k) standardized
    y = y_std * sd_t.unsqueeze(0) + mu_t.unsqueeze(0)  # (BH, k) original
    return y.reshape(B, H, -1)                     # (B, H, k)

# ----------------------------
# Dataset over latent codes: seq -> multi-horizon latent targets
# ----------------------------
class CodeSeqToMultiHorizonDataset(Dataset):
    """
    From latent code panel C (T x k_latent), create samples:
      X_t = C[t-seq_len+1 : t+1], shape (seq_len, k_latent)
      Y_t[h] = C[t+h, :], for h in horizons
    Valid t: t ∈ [seq_len-1, T-1-H_MAX]
    """
    def __init__(self, C: np.ndarray, horizons, seq_len: int, stride: int = 1):
        super().__init__()
        self.C = C.astype(np.float32)
        self.horizons = list(horizons)
        self.seq_len = int(seq_len)
        self.H_MAX = max(self.horizons)
        t_start = self.seq_len - 1
        t_end = C.shape[0] - 1 - self.H_MAX
        if t_end < t_start:
            self.idxs = []
        else:
            self.idxs = list(range(t_start, t_end + 1, stride))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        t = self.idxs[i]
        x = self.C[t - self.seq_len + 1 : t + 1, :]     # (seq_len, k_latent)
        ys = [self.C[t + h, :] for h in self.horizons]  # list of (k_latent,)
        y = np.stack(ys, axis=0)                        # (H, k_latent)
        return torch.from_numpy(x), torch.from_numpy(y)

# ----------------------------
# LSTM over latent codes
# ----------------------------
class LSTMLatentForecaster(nn.Module):
    """
    Many-to-one LSTM mapping a sequence of latent codes (B, L, k_latent) to
    multi-horizon latent targets (B, H, k_latent).
    """
    def __init__(self, in_dim=3, hidden_dim=128, num_layers=1, horizons=(1,5,10,22), dropout=0.0):
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

    def forward(self, x):  # x: (B, L, k_latent)
        out, _ = self.lstm(x)                # out: (B, L, hidden)
        last = out[:, -1, :]                 # (B, hidden)
        pred = self.head(last)               # (B, H*in_dim)
        pred = pred.view(-1, self.H, self.in_dim)  # (B, H, k_latent)
        return pred

# ----------------------------
# Train LSTM on latent codes (loss in yield space; AE decoder frozen)
# ----------------------------
def train_lstm_on_codes(C_tr: pd.DataFrame,
                        ae_model: AutoEncoder,
                        mu: pd.Series,
                        sd: pd.Series,
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
    Train LSTM on latent codes. To align with "equal per maturity" in yield space,
    we decode both predictions and targets back to yields (via AE decoder; frozen)
    and minimize unweighted MSE across maturities (original units).
    """
    if seed is not None:
        torch.manual_seed(seed)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    # Dataset over codes
    C_np = C_tr.values.astype(np.float32)
    ds = CodeSeqToMultiHorizonDataset(C_np, horizons=list(horizons), seq_len=seq_len, stride=stride)
    if len(ds) == 0:
        raise RuntimeError("[AE+LSTM refresh] Training slice too short for seq_len/H_MAX (codes).")
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

    # Model
    model = LSTMLatentForecaster(in_dim=C_np.shape[1], hidden_dim=hidden_dim,
                                 num_layers=num_layers, horizons=list(horizons),
                                 dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss(reduction="none")

    # Freeze AE (we still need gradients w.r.t. inputs; freezing params is fine)
    _freeze_model(ae_model)
    mu_t = torch.tensor(mu.values.astype(np.float32), device=device)
    sd_t = torch.tensor(sd.values.astype(np.float32), device=device)

    model.train()
    for ep in range(1, epochs + 1):
        tot = 0.0; N = 0
        for xb, yb in dl:
            xb = xb.to(device)          # (B, L, k_latent)
            yb = yb.to(device)          # (B, H, k_latent)
            opt.zero_grad()
            pb = model(xb)              # (B, H, k_latent)

            # Decode both predicted and target codes to yields (original units)
            yhat = _decode_latent_to_yield_torch(ae_model, pb, mu_t, sd_t)  # (B, H, k)
            ytar = _decode_latent_to_yield_torch(ae_model, yb, mu_t, sd_t)  # (B, H, k)

            # Equal per-maturity MSE in original units
            loss_mat = mse(yhat, ytar)             # (B, H, k)
            loss = loss_mat.mean()                 # scalar
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            N += xb.size(0)

        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs):
            print(f"[AE+LSTM] epoch {ep:3d}/{epochs}  loss(yield-space)={tot/max(1,N):.6f}")

    return model

@torch.no_grad()
def lstm_latent_predict_block(model: LSTMLatentForecaster,
                              ae_model: AutoEncoder,
                              Y_block: pd.DataFrame,
                              mu: pd.Series,
                              sd: pd.Series,
                              seq_len: int,
                              horizons,
                              device: str = "cpu") -> dict:
    """
    Produce predictions for each origin in Y_block.index *after* the first seq_len rows.
    Path: encode Y_block -> latent codes -> LSTM predict codes -> decode to yields.
    Returns: dict h -> list[(date, pd.Series of yields)]
    """
    # Encode block to codes with AE (no grad)
    C_block = encode_series(ae_model, Y_block, mu, sd, device=device).values.astype(np.float32)
    H = len(horizons)
    k_lat = C_block.shape[1]
    preds_by_h = {h: [] for h in horizons}
    dates = Y_block.index

    for end in range(seq_len, len(C_block)):
        x = torch.from_numpy(C_block[end - seq_len : end, :]).unsqueeze(0).to(device)  # (1, L, k_lat)
        p_codes = model(x).cpu().numpy()[0]  # (H, k_lat)

        # Decode each horizon prediction to yields (vectorized: use torch decoder for speed)
        with torch.no_grad():
            p_codes_t = torch.from_numpy(p_codes).to(device).unsqueeze(0)  # (1, H, k_lat)
            mu_t = torch.tensor(mu.values.astype(np.float32), device=device)
            sd_t = torch.tensor(sd.values.astype(np.float32), device=device)
            yhat_bhk = _decode_latent_to_yield_torch(_freeze_model(ae_model), p_codes_t, mu_t, sd_t)  # (1,H,k)
            yhat = yhat_bhk.squeeze(0).cpu().numpy()  # (H, k)

        origin_pos = end  # i_pos analog
        for j, h in enumerate(horizons):
            tgt_idx = origin_pos + h
            if tgt_idx < len(dates):
                preds_by_h[h].append((dates[tgt_idx],
                                      pd.Series(yhat[j, :], index=Y_COLS)))
    return preds_by_h

# ----------------------------
# Periodic refresh loop (deterministic)
# ----------------------------
def ae_lstm_periodic_refresh(Y_all: pd.DataFrame,
                             horizons,
                             code_seq_len: int,
                             ae_train_window: int = 5000,
                             refresh_every: int = 252,
                             ae_latent: int = 3,
                             ae_hidden=(64, 32),
                             ae_epochs: int = 25,
                             ae_bs: int = 256,
                             ae_lr: float = 1e-3,
                             lstm_hidden_dim: int = 128,
                             lstm_layers: int = 1,
                             lstm_dropout: float = 0.0,
                             lstm_epochs: int = 20,
                             lstm_bs: int = 256,
                             lstm_lr: float = 1e-3,
                             stride: int = 1,
                             base_seed: int = 1337):
    """
    Retrain AE every `refresh_every` obs on last `ae_train_window` yields.
    Encode to latent codes. Train LSTM on sequences of those codes (length = code_seq_len)
    to predict future codes for all horizons. Decode predictions back to yields.
    """
    dates = Y_all.index
    T = len(dates)
    horizons = list(horizons)
    H_MAX = max(horizons)

    # Need both an AE training window and code_seq_len history for LSTM
    start_origin = max(ae_train_window, int(code_seq_len))
    if start_origin + H_MAX >= T:
        raise RuntimeError("[AE+LSTM refresh] Not enough data to start backtest.")

    preds_by_h_all = {h: [] for h in horizons}
    refresh_points = list(range(start_origin, T - H_MAX, refresh_every))
    print(f"[AE+LSTM refresh] total refreshes: {len(refresh_points)}, "
          f"AE_train_win={ae_train_window}, step={refresh_every}, seq_len={code_seq_len}, H_MAX={H_MAX}")

    for r_idx, origin0 in enumerate(refresh_points, 1):
        refresh_seed = base_seed + r_idx

        # 1) Train AE on trailing window
        tr_lo = origin0 - ae_train_window
        tr_hi = origin0
        train_slice = Y_all.iloc[tr_lo:tr_hi].dropna()
        if len(train_slice) < ae_train_window * 0.95:
            warnings.warn(f"[AE+LSTM refresh] AE training slice at origin {origin0} has NaNs; size={len(train_slice)}.")
        print(f'Refresh {r_idx}')
        ae_model, ae_mu, ae_sd = train_autoencoder(
            train_slice,
            latent_dim=ae_latent, hidden=ae_hidden,
            epochs=ae_epochs, batch_size=ae_bs, lr=ae_lr,
            device=device, verbose=False, seed=refresh_seed
        )

        # 2) Encode the AE training slice to latent codes (for LSTM training)
        codes_train_df = encode_series(ae_model, train_slice, ae_mu, ae_sd, device=device)

        # 3) Train LSTM on latent codes (loss in yield space using frozen AE decoder)
        lstm_model = train_lstm_on_codes(
            codes_train_df, ae_model, ae_mu, ae_sd, horizons=horizons,
            seq_len=code_seq_len, stride=stride,
            hidden_dim=lstm_hidden_dim, num_layers=lstm_layers, dropout=lstm_dropout,
            epochs=lstm_epochs, batch_size=lstm_bs, lr=lstm_lr,
            device=device, seed=refresh_seed, verbose=False
        )

        # 4) Define forecasting block [origin0, origin1) and predict
        origin1 = min(origin0 + refresh_every, T - H_MAX)

        # Build a block providing past code_seq_len and reaching to origin1 + H_MAX
        block_lo = max(0, origin0 - code_seq_len)
        block_hi = origin1 + H_MAX
        Y_block = Y_all.iloc[block_lo:block_hi]

        preds_by_h = lstm_latent_predict_block(
            lstm_model, ae_model, Y_block, ae_mu, ae_sd,
            seq_len=code_seq_len, horizons=horizons, device=device
        )

        # Collect predictions
        for h in horizons:
            preds_by_h_all[h].extend(preds_by_h[h])

        # Free models per block
        del lstm_model, ae_model

    # 5) Assemble & score
    AE_LSTM_RESULTS = {}
    AE_LSTM_WRMSE = []
    for h in horizons:
        if preds_by_h_all[h]:
            df_h = pd.DataFrame({d: y for d, y in preds_by_h_all[h]}).T
            df_h.index.name = "Date"
            df_h = df_h.reindex(columns=Y_COLS)
        else:
            df_h = pd.DataFrame(columns=Y_COLS)
        AE_LSTM_RESULTS[h] = df_h

        idx = Y_all.index.intersection(df_h.index)
        if len(idx):
            wrmse_bps = weighted_rmse_scalar(Y_all.loc[idx, Y_COLS], df_h.loc[idx, Y_COLS], weights) * 100.0
            AE_LSTM_WRMSE.append({"model": "AE_LSTM_refresh", "h": h, "wrmse_bps": float(wrmse_bps)})

    AE_LSTM_SUMMARY_DF = pd.DataFrame(AE_LSTM_WRMSE).sort_values(["h", "wrmse_bps"]).reset_index(drop=True)
    return AE_LSTM_RESULTS, AE_LSTM_SUMMARY_DF

# ----------------------------
# Run periodic refresh backtest (deterministic)
# ----------------------------
AE_LSTM_RESULTS, AE_LSTM_SUMMARY_DF = ae_lstm_periodic_refresh(
    Y_all=Y_all,
    horizons=list(HORIZONS),
    code_seq_len=int(WINDOW),      # analogous to your VAR/LSTM window (e.g., 504)
    ae_train_window=5000,
    refresh_every=252,             # ~annual refresh (tune as you like)
    ae_latent=3,                   # 3–5 often reasonable
    ae_hidden=(64, 32),
    ae_epochs=50,                  # can increase; decoder is reused by LSTM training
    ae_bs=256,
    ae_lr=1e-3,
    lstm_hidden_dim=128,
    lstm_layers=1,
    lstm_dropout=0.0,
    lstm_epochs=20,
    lstm_bs=256,
    lstm_lr=1e-3,
    stride=1,
    base_seed=_BASE_SEED
)

print("\n=== AE+LSTM (Periodic Refresh) — WRMSE (bps) by horizon ===")
if len(AE_LSTM_SUMMARY_DF):
    print(AE_LSTM_SUMMARY_DF.to_string(index=False))
else:
    print("No AE+LSTM results produced.")

for h, df in AE_LSTM_RESULTS.items():
    # ensure Date is an index (it already is, but this is safe)
    df = df.sort_index()
    df.to_parquet(f"data/processed/predictions/ae_lstm/ae_lstm_predictions_h{h}.parquet", index=True) 
