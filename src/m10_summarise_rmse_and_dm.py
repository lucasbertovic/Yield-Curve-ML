from pathlib import Path
import numpy as np
import pandas as pd
from ycml.config import load_yaml, curve_config
from m09_plots import (
    load_model_predictions_for_horizons,
    build_random_walk_preds,
    common_index_for_horizon,
    print_index_diagnostics,
    normalize_dt_index,  # used for safety when aligning
)

# ----------------------------
# Paths
# ----------------------------
BASE_CFG_PATH = Path("configs/base.yaml")
PRED_ROOT = Path("data/processed/predictions")
TABLES_DIR = Path("outputs/tables")

# ----------------------------
# Metrics & tests (UNWEIGHTED)
# ----------------------------
def _per_date_mse_unweighted(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    """
    Per-date (row-wise) unweighted MSE across maturities (columns).
    Returns a Series indexed by dates.
    """
    err2 = (y_true.values - y_pred.values) ** 2
    mse = err2.mean(axis=1)  # equal weight across columns
    return pd.Series(mse, index=y_true.index)

def diebold_mariano(loss1: pd.Series, loss2: pd.Series, h: int) -> tuple[float, float, int]:
    """
    Diebold–Mariano test with Newey–West variance, lag = h-1.
    Returns (DM statistic, two-sided p-value, sample size).
    Negative DM => model1 better on average (lower loss).
    """
    d = (loss1 - loss2).dropna()
    T = len(d)
    if T < 5:
        return np.nan, np.nan, T
    d_mean = d.mean()
    q = max(h - 1, 0)

    # Long-run variance via NW
    gamma0 = ((d - d_mean) ** 2).sum() / T
    lr_var = gamma0
    for lag in range(1, q + 1):
        cov = ((d.iloc[lag:] - d_mean) * (d.shift(lag).dropna() - d_mean)).sum() / T
        w = 1.0 - lag / (q + 1)
        lr_var += 2.0 * w * cov

    dm_stat = d_mean / np.sqrt(lr_var / T) if lr_var > 0 else np.nan

    # normal approx p-value
    from math import erf, sqrt
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(dm_stat) / sqrt(2.0))))
    return float(dm_stat), float(p_val), int(T)

# ----------------------------
# Main
# ----------------------------
def main():
    # 1) Load config & data
    CFG = load_yaml(BASE_CFG_PATH)
    C = curve_config(CFG)
    Y_COLS   = C["Y_COLS"]
    HORIZONS = C["HORIZONS"]

    df = pd.read_parquet("data/processed/spots.parquet")
    # Ensure we have a Date index
    if "Date" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date").sort_index()
    df = normalize_dt_index(df)
    Y_all = df[Y_COLS].dropna(how="any")

    # 2) Load predictions (same machinery as plotting module)
    model_preds = load_model_predictions_for_horizons(Y_COLS, HORIZONS)
    rw_preds = build_random_walk_preds(Y_all, HORIZONS, Y_COLS)

    # We’ll build:
    #   rmse_table: rows=models, cols=horizons (RMSE bps on common intersection)
    #   dm_rows: one row per (h, pair)
    models_all = ["RW", "DNS-diff", "AE+VAR", "LSTM", "AE+LSTM"]
    rmse_table = pd.DataFrame(index=models_all, columns=sorted(HORIZONS), dtype=float)
    n_common = {h: np.nan for h in HORIZONS}

    dm_rows: list[dict] = []

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    for h in sorted(HORIZONS):
        # Which model predictions exist for this horizon?
        available_models = {name: preds[h] for name, preds in model_preds.items() if h in preds}
        # Always include RW
        available_models_with_rw = {"RW": rw_preds[h], **available_models}

        if not available_models:
            print(f"[summarize] No model predictions available for h={h}. Skipping.")
            continue

        # Diagnostics (optional but helpful)
        print_index_diagnostics(h, Y_all, available_models, rw_preds[h])

        # 3) Common intersection across *all* models at this horizon (for the RMSE table)
        idx_all = common_index_for_horizon(h, Y_all, available_models, rw_preds[h])
        if len(idx_all) < 5:
            print(f"[summarize] Too few overlapping dates for h={h} (N={len(idx_all)}). Skipping RMSE table fill.")
            continue
        n_common[h] = len(idx_all)

        y_true = Y_all.loc[idx_all, Y_COLS]

        # Fill RMSE table (bps) on the same aligned sample
        for name, df_pred in available_models_with_rw.items():
            dfp = df_pred.loc[idx_all, Y_COLS]
            per_date_mse = _per_date_mse_unweighted(y_true, dfp)
            rmse_bps = float(np.sqrt(per_date_mse.mean()) * 100.0)
            rmse_table.loc[name, h] = rmse_bps

        # 4) DM tests — small, meaningful set of pairs
        present = set(available_models_with_rw.keys())
        pairs = []
        # vs RW baseline
        for m in ["DNS-diff", "AE+VAR", "LSTM", "AE+LSTM"]:
            if {"RW", m}.issubset(present):
                pairs.append((m, "RW"))
        # structured intra-family comparisons
        if {"AE+VAR", "DNS-diff"}.issubset(present):
            pairs.append(("AE+VAR", "DNS-diff"))
        if {"LSTM", "AE+VAR"}.issubset(present):
            pairs.append(("LSTM", "AE+VAR"))
        if {"AE+LSTM", "LSTM"}.issubset(present):
            pairs.append(("AE+LSTM", "LSTM"))

        # Compute DM per pair using their *pairwise* intersection
        for m1, m2 in pairs:
            idx_pair = (
                normalize_dt_index(Y_all).index
                .intersection(normalize_dt_index(available_models_with_rw[m1]).index)
                .intersection(normalize_dt_index(available_models_with_rw[m2]).index)
            )
            if len(idx_pair) < 5:
                dm, p, N = np.nan, np.nan, len(idx_pair)
            else:
                yt = Y_all.loc[idx_pair, Y_COLS]
                y1 = available_models_with_rw[m1].loc[idx_pair, Y_COLS]
                y2 = available_models_with_rw[m2].loc[idx_pair, Y_COLS]
                l1 = _per_date_mse_unweighted(yt, y1)
                l2 = _per_date_mse_unweighted(yt, y2)
                dm, p, N = diebold_mariano(l1, l2, h=h)

            winner = None
            if np.isfinite(dm):
                winner = m1 if dm < 0 else m2  # negative => model1 better

            dm_rows.append({
                "h": int(h),
                "model1": m1,
                "model2": m2,
                "DM": dm,
                "p_value": p,
                "N": N,
                "winner": winner,
            })

    # Clean up RMSE table (drop all-NaN rows)
    rmse_table = rmse_table.dropna(how="all")
    # Add N_common row for reference
    rmse_table.loc["N_common"] = [n_common.get(h, np.nan) for h in sorted(HORIZONS)]

    # Save + print
    rmse_out = TABLES_DIR / "rmse_by_model_and_horizon.csv"
    rmse_table.to_csv(rmse_out)
    print("\n=== RMSE (bps, unweighted) by model × horizon (aligned across all models per h) ===")
    if not rmse_table.empty:
        print(rmse_table.to_string())
        print(f"[saved] {rmse_out.as_posix()}")
    else:
        print("No data to summarize.")

    dm_summary = pd.DataFrame(dm_rows)
    dm_out = TABLES_DIR / "dm_tests_summary.csv"
    if not dm_summary.empty:
        dm_summary = dm_summary.sort_values(["h", "model1", "model2"]).reset_index(drop=True)
        dm_summary.to_csv(dm_out, index=False)
        print("\n=== Diebold–Mariano tests (selected pairs; unweighted per-date MSE) ===")
        print(dm_summary.to_string(index=False))
        print(f"[saved] {dm_out.as_posix()}")
    else:
        print("\nNo DM results computed.")

if __name__ == "__main__":
    main()
