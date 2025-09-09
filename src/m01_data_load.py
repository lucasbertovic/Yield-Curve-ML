import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
import pandas as pd, pathlib, io

path = pathlib.Path("data/raw/feds200628.csv")

# Find the header line
with path.open("r", encoding="utf-8") as f:
    lines = f.readlines()
header_idx = next(i for i, line in enumerate(lines) if line.startswith("Date,"))

# Parse from that header line onward
gsw = pd.read_csv(io.StringIO("".join(lines[header_idx:])))
gsw["Date"] = pd.to_datetime(gsw["Date"], dayfirst=True)

# Columns we'll use frequently
spot_cols = [c for c in gsw.columns if c.startswith("SVENY")]   # zero-coupon spots
par_cols  = [c for c in gsw.columns if c.startswith("SVENPY")]  # par yields
fwd_cols  = [c for c in gsw.columns if c.startswith("SVENF")]   # inst. forwards

# Drop holiday rows 
gsw_clean = gsw.dropna(subset=spot_cols, how="all").sort_values("Date").reset_index(drop=True)


keep = ["Date"] + [f"SVENY{m:02d}" for m in range(1,31)]
spot = gsw_clean[keep].dropna(how="all", subset=[c for c in keep if c!="Date"])
spot.to_parquet("data/processed/spot_us_gsw.parquet", index=False)

params = gsw_clean[["Date","BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]]
params.to_parquet("data/processed/params_us_gsw.parquet", index=False)