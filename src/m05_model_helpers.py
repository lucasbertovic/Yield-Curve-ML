import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from statsmodels.tsa.api import VAR

def set_global_seeds(seed: int = 1337, use_cuda: bool = torch.cuda.is_available()):
    """Set seeds for Python, NumPy, and Torch; force deterministic Torch ops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    # Make Torch deterministic as far as possible
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ols_var_fit(Z: np.ndarray, p: int):
    """
    OLS VAR(p) via statsmodels; returns intercept and A_1..A_p matrices.
    Z: (T x k) (typically standardized series in a window)
    """
    res = VAR(Z).fit(p)
    c = res.intercept
    A_list = [res.coefs[i].copy() for i in range(res.k_ar)]  # (p, k, k)
    return c, A_list

def iterative_forecast_path(last_hist: np.ndarray, c: np.ndarray, A_list, h_max: int, damp: float = 1.0) -> np.ndarray:
    """
    Produce the whole path [1..h_max]-step-ahead forecasts for a VAR(p).
    last_hist: (p x k) rows = [x_{t-p+1}, ..., x_t] in standardized space
    returns: (h_max x k) array; row j-1 is x_{t+j}
    """
    p = len(A_list)
    hist = last_hist.copy()
    outs = []
    for _ in range(h_max):
        x_next = c.copy()
        for i in range(1, p + 1):
            x_next += (damp * A_list[i - 1]) @ hist[-i, :]
        hist = np.vstack([hist[1:, :], x_next[None, :]])
        outs.append(hist[-1, :].copy())
    return np.asarray(outs)

# ----------------------------
# Standardization utilities (train-only stats)
# ----------------------------
def fit_standardizer(df_train: pd.DataFrame):
    mu = df_train.mean()
    sd = df_train.std().replace(0.0, 1.0)
    return mu, sd

def apply_standardizer(df: pd.DataFrame, mu: pd.Series, sd: pd.Series):
    # z-score: used for numerics, but loss will be re-weighted by sd^2 to be original-scale
    return (df - mu) / sd

def invert_standardizer(df_z: pd.DataFrame, mu: pd.Series, sd: pd.Series):
    return df_z * sd + mu
