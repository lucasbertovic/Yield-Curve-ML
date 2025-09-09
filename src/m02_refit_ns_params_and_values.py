import pandas as pd, numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

# ----------------------------
# Load & prepare
# ----------------------------
spot   = pd.read_parquet("data/processed/spot_us_gsw.parquet")   # Date, SVENY01..SVENY30
params = pd.read_parquet("data/processed/params_us_gsw.parquet") # Date, BETA*, TAU*
Y_COLS   = [f"SVENY{m:02d}" for m in range(1,31)]
MAT_GRID = np.arange(1,31, dtype=float)

df = (params.merge(spot, on="Date", how="inner")
             .sort_values("Date")
             .set_index("Date"))

# ----------------------------
# Nelson–Siegel (continuous)
# ----------------------------
def ns_yield(t, b0, b1, b2, T1):
    t = np.asarray(t, dtype=float)
    x = t / T1
    L = (1 - np.exp(-x)) / np.where(x == 0, 1, x)  # level loading
    S = L - np.exp(-x)                              # slope/curvature loading
    return b0 + b1*L + b2*S

def fit_ns_params_one_day(y_row, p0=None, min_pts=6):
    """
    Fit NS to one day's observed SVENY (may have NaNs).
    Returns (b0,b1,b2,T1) or None if fit fails.
    """
    mask = ~y_row.isna()
    if mask.sum() < min_pts:
        return None

    mats = np.array([int(c[-2:]) for c in y_row.index[mask]], dtype=float)
    yobs = y_row.values[mask].astype(float)

    # Robust initial guess:
    if p0 is None:
        b0_guess = float(np.nanmedian(yobs))               # approximate level
        b1_guess = float(yobs[0] - b0_guess) if len(yobs) > 0 else -1.0
        b2_guess = 0.0
        T1_guess = 1.5                                     # years (DNS literature ~1–3y)
        p0 = [b0_guess, b1_guess, b2_guess, T1_guess]

    # Bounds: wide for betas; positive, not tiny/huge T1
    bounds = ([-10.0, -20.0, -20.0, 0.05],
              [ 20.0,  20.0,  20.0, 30.00])

    try:
        popt, _ = curve_fit(ns_yield, mats, yobs, p0=p0, bounds=bounds, maxfev=20000)
        return popt  # (b0,b1,b2,T1)
    except Exception:
        return None


if __name__ == "__main__":
    # ----------------------------
    # Refit NS params for all days
    # ----------------------------
    ns_params_list = []
    prev_p = None

    # Use the yields present in df to fit NS each day (don’t reuse Svensson params)
    Y_panel = df[Y_COLS]

    for dt, y_row in tqdm(Y_panel.iterrows(), total=len(Y_panel), desc="Fitting NS per day"):
        p = fit_ns_params_one_day(y_row, p0=prev_p)
        if p is None:
            # mild fallback: try a neutral seed
            p = fit_ns_params_one_day(y_row, p0=[y_row.get("SVENY10", np.nan) if pd.notna(y_row.get("SVENY10", np.nan)) else 3.0,
                                                -1.0, 0.0, 1.5])
        ns_params_list.append(p)
        prev_p = p if p is not None else prev_p

    ns_params = pd.DataFrame(ns_params_list, index=df.index, columns=["NS_BETA0","NS_BETA1","NS_BETA2","NS_TAU1"])

    # ----------------------------
    # Build NS yields 1–30y from fitted params
    # ----------------------------
    ns_yields = pd.DataFrame(index=df.index, columns=Y_COLS, dtype=float)
    ok = ns_params.notna().all(axis=1)

    if ok.any():
        pars = ns_params.loc[ok].values
        # vectorized reconstruction
        yhat = np.vstack([ns_yield(MAT_GRID, *p) for p in pars])
        ns_yields.loc[ok, Y_COLS] = yhat

    # Attach NS outputs to df; replace yield panel with NS-only reconstruction
    df["NS_BETA0"] = ns_params["NS_BETA0"]
    df["NS_BETA1"] = ns_params["NS_BETA1"]
    df["NS_BETA2"] = ns_params["NS_BETA2"]
    df["NS_TAU1"]  = ns_params["NS_TAU1"]
    df[Y_COLS]     = ns_yields

    # Optional: drop days where fit failed entirely
    # df = df.dropna(subset=["NS_BETA0","NS_BETA1","NS_BETA2","NS_TAU1"])

    # Save
    df.index = pd.to_datetime(df.index)
    df.to_parquet("data/processed/spots.parquet", index=True)
    print("Saved NS-only reconstructed yield panel to parquet with fitted NS parameters.")
