# src/ycml/config.py
from pathlib import Path
import yaml
import pandas as pd   # NEW: needed for weights

def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f) or {}
    
def base_seed(cfg: dict) -> int:
    """
    Extract the global random seed from the config dict.
    Defaults to 1337 if not provided.
    """
    return int(cfg.get("seed", 1337))

def curve_config(cfg: dict):
    c = cfg.get("curve", {})
    b = cfg.get("backtest", {})
    w = cfg.get("weights", None)

    m = int(c.get("maturities", 30))
    z = int(c.get("y_col_zero_pad", 2))
    prefix = c.get("y_col_prefix", "SVENY")
    y_cols = [f"{prefix}{i:0{z}d}" for i in range(1, m + 1)]

    # mat_grid: if given as [start, end], expand into full range
    mat_grid = c.get("mat_grid")
    if mat_grid and len(mat_grid) == 2:
        start, end = mat_grid
        mat_grid = list(range(int(start), int(end) + 1))
    elif not mat_grid:
        mat_grid = list(range(1, m + 1))

    lam = float(c.get("lambda", 1.37))
    window = int(b.get("window", 504))
    horizons = list(b.get("horizons", [1, 5, 10, 22]))

    # Weights: load into a pd.Series if provided
    weights = None
    if isinstance(w, dict) and w:
        weights = pd.Series(w, dtype=float)
        # normalize (mean = 1) just like your original code
        weights = weights / weights.mean()

    return {
        "Y_COLS": y_cols,
        "MAT_GRID": mat_grid,
        "LAM": lam,
        "WINDOW": window,
        "HORIZONS": horizons,
        "WEIGHTS": weights,   # NEW: include weights in returned dict
    }

def load_dns_best_configs(path: str | Path) -> dict[int, dict]:
    """
    Load best DNS hyperparameters from a YAML file (e.g., configs/model/dns_diff.yaml).
    
    YAML format:
    best_dns_configs:
      1: {lam: 0.25, window: 504, order: 1}
      5: {lam: 0.25, window: 756, order: 1}
      ...

    Returns a dictionary like:
      {
        1: {"lam": 0.25, "window": 504, "order": 1},
        5: {"lam": 0.25, "window": 756, "order": 1},
        ...
      }
    with integer keys for horizons.
    """
    cfg = load_yaml(path)
    dns_cfg = cfg.get("best_dns_configs", None)
    if not isinstance(dns_cfg, dict) or not dns_cfg:
        raise ValueError(f"[dns_best] No 'best_dns_configs' found in {path}")

    out = {}
    for k, v in dns_cfg.items():
        h = int(k)  # convert YAML string keys to int
        if not isinstance(v, dict):
            raise ValueError(f"[dns_best] Config for horizon {h} must be a dict")
        lam = float(v.get("lam", 1.37))
        window = int(v.get("window", 504))
        order = int(v.get("order", 1))
        if lam <= 0 or window <= 0 or order <= 0:
            raise ValueError(f"[dns_best] Invalid values for horizon {h}: {v}")
        out[h] = {"lam": lam, "window": window, "order": order}
    return out
