from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
# --- Load shared config + curve settings ---
from ycml.config import load_yaml, curve_config
from m03_dns_config_and_initial_fit import rmse_per_maturity

# ----------------------------
# Settings (paths & model map)
# ----------------------------
BASE_CFG_PATH = Path("configs/base.yaml")
PRED_ROOT = Path("data/processed/predictions")
PLOTS_DIR = Path("outputs/figures")  # change if you want a different folder

# Map: display name -> (subdir_name, file_prefix)
# Files must follow: {subdir}/{file_prefix}_predictions_h{h}.parquet
MODEL_SPECS = {
    "DNS-diff": ("dns_diff", "dns_diff"),
    "AE+VAR":   ("ae_var_diff", "ae_var_diff"),
    "LSTM":     ("lstm", "lstm"),
    "AE+LSTM":  ("ae_lstm", "ae_lstm"),
    # Random Walk handled separately (computed from Y_all)
}

# ----------------------------
# Utilities
# ----------------------------
def normalize_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DatetimeIndex (tz-naive) normalized to date (00:00)."""
    if df is None or df.empty:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to detect a Date column
        for cand in ["Date", "date", "DATE"]:
            if cand in df.columns:
                df = df.set_index(cand)
                break
    # Coerce to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return df  # best effort
    # Make tz-naive and normalized to date
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    # Sort and drop duplicates
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def safe_read_parquet_any(path: Path, y_cols) -> pd.DataFrame | None:
    """Read a parquet and normalize index; ensure columns limited to y_cols if present."""
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = normalize_dt_index(df)
    if df is None or df.empty:
        return None
    # Keep only known yield columns if present
    present = [c for c in y_cols if c in df.columns]
    if present:
        df = df.reindex(columns=y_cols).dropna(how="any")
    return df

def load_model_predictions_for_horizons(y_cols, horizons):
    """
    Returns: dict[model_name][h] -> DataFrame
    Accepts multiple filename prefixes per model, picks the first that exists.
    """
    out: dict[str, dict[int, pd.DataFrame]] = {}
    for display, (subdir, prefixes) in MODEL_SPECS.items():
        per_h: dict[int, pd.DataFrame] = {}
        for h in horizons:
            df_loaded = None
            # try exact filenames first
            for pref in prefixes:
                candidate = PRED_ROOT / subdir / f"{pref}_predictions_h{h}.parquet"
                if candidate.exists():
                    df_loaded = safe_read_parquet_any(candidate, y_cols)
                    if df_loaded is not None:
                        break
            # if none matched exactly, try glob
            if df_loaded is None:
                pattern = str(PRED_ROOT / subdir / f"*predictions_h{h}.parquet")
                for p in glob.glob(pattern):
                    df_loaded = safe_read_parquet_any(Path(p), y_cols)
                    if df_loaded is not None:
                        break
            if df_loaded is not None and not df_loaded.empty:
                per_h[h] = df_loaded
        if per_h:
            out[display] = per_h
    return out

def build_random_walk_preds(Y_all: pd.DataFrame, horizons, y_cols):
    rw = {}
    for h in horizons:
        df = Y_all[y_cols].shift(h)
        df = normalize_dt_index(df).dropna(how="any")
        rw[h] = df
    return rw

def common_index_for_horizon(h, Y_all, model_preds_h, rw_pred_h):
    idx = normalize_dt_index(Y_all).index
    idx = idx.intersection(normalize_dt_index(rw_pred_h).index)
    for df in model_preds_h.values():
        idx = idx.intersection(normalize_dt_index(df).index)
    return idx

def print_index_diagnostics(h, Y_all, model_preds_h, rw_pred_h):
    def _span(df):
        if df is None or df.empty: return ("∅", "∅", 0)
        return (str(df.index.min().date()), str(df.index.max().date()), len(df.index))
    rows = []
    rows.append(("Truth (Y_all)",) + _span(normalize_dt_index(Y_all)))
    rows.append((f"RW(h={h})",) + _span(normalize_dt_index(rw_pred_h)))
    for name, df in model_preds_h.items():
        rows.append((name,) + _span(normalize_dt_index(df)))
    w = max(len(r[0]) for r in rows)
    print(f"\n[h={h}] date coverage diagnostics:")
    for name, lo, hi, n in rows:
        print(f"  {name:<{w}} | {lo} → {hi}  (N={n})")

# ----------------------------
# Main
# ----------------------------
def main():
    # Load config & data
    CFG = load_yaml(BASE_CFG_PATH)
    C = curve_config(CFG)
    Y_COLS   = C["Y_COLS"]
    MAT_GRID = np.asarray(C["MAT_GRID"], float)
    HORIZONS = C["HORIZONS"]

    df = pd.read_parquet("data/processed/spots.parquet")
    Y_all = df[Y_COLS].dropna()
    print(df.columns.values)

    # Predictions
    model_preds = load_model_predictions_for_horizons(Y_COLS, HORIZONS)
    rw_preds = build_random_walk_preds(Y_all, HORIZONS, Y_COLS)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for h in sorted(HORIZONS):
        # models that have this horizon
        available_models = {name: preds[h] for name, preds in model_preds.items() if h in preds}
        if not available_models:
            print(f"[plot] No model predictions available for h={h}. Skipping.")
            continue

        # Diagnostics: show coverage
        print_index_diagnostics(h, Y_all, available_models, rw_preds[h])

        # Intersection of dates across truth, RW and all models
        idx = common_index_for_horizon(h, Y_all, available_models, rw_preds[h])

        if len(idx) < 5:
            print(f"[plot] Too few overlapping dates for h={h} (N={len(idx)}). Skipping.")
            continue

        y_true = Y_all.loc[idx, Y_COLS]

        # RMSE by maturity (bps)
        rmse_table = {}
        rmse_table["RW"] = rmse_per_maturity(y_true, rw_preds[h].loc[idx, Y_COLS]) * 100.0
        for name, pred_df in available_models.items():
            rmse_table[name] = rmse_per_maturity(y_true, pred_df.loc[idx, Y_COLS]) * 100.0

        # Order nicely if present
        desired = ["RW", "DNS-diff", "AE+VAR", "LSTM", "AE+LSTM"]
        cols = [c for c in desired if c in rmse_table]
        df_plot = pd.DataFrame({k: rmse_table[k] for k in cols}, index=Y_COLS)

        # Plot
        maturities = np.arange(1, len(Y_COLS) + 1)
        plt.figure(figsize=(9, 5.5))
        for col in df_plot.columns:
            plt.plot(maturities, df_plot[col].values, label=col)

        plt.title(f"RMSE by Maturity (bps) — horizon h={h}")
        plt.xlabel("Maturity (years)")
        plt.ylabel("RMSE (bps)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_png = PLOTS_DIR / f"rmse_by_maturity_h{h}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()
        print(f"[plot] Saved {out_png.as_posix()}")

if __name__ == "__main__":
    main()
