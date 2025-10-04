# prep_events_csv.py
import pandas as pd

IN = "events_20251001.csv"
OUT = "events_20251001_updated.csv"  # overwrite in place

df = pd.read_csv(IN)

# Basic checks
for need in ["symbol", "year", "quarter"]:
    if need not in df.columns:
        raise ValueError(f"Missing required column '{need}' in {IN}")

# Choose press-release anchor date: prefer 'date', else 'conference_date'
if "date" in df.columns:
    ann_base = pd.to_datetime(df["date"], errors="coerce")
elif "conference_date" in df.columns:
    ann_base = pd.to_datetime(df["conference_date"], errors="coerce")
else:
    raise ValueError("Need 'date' or 'conference_date' to anchor announcement timestamp")

# Call anchor from 'conference_date' if present
call_base = pd.to_datetime(df["conference_date"], errors="coerce") if "conference_date" in df.columns else pd.Series([pd.NaT]*len(df))

def to_et_with_time(ts_series, hour, minute):
    out = []
    for ts in ts_series:
        if pd.isna(ts):
            out.append(pd.NaT)
            continue
        ts = pd.Timestamp(ts)
        # If the timestamp already has a time, keep that time; otherwise set default time
        if ts.time() == pd.Timestamp("00:00:00").time():
            ts = ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
        # localize (if naive) or convert (if tz-aware) to America/New_York
        try:
            ts = ts.tz_convert("America/New_York")
        except TypeError:
            ts = ts.tz_localize("America/New_York")
        out.append(ts)
    return pd.Series(out, dtype="datetime64[ns, America/New_York]")

# Default assumptions:
# - If 'date' has no time, set press release at 16:10 ET
# - For calls, if no time is given, set 17:00 ET
ann_time_et = to_et_with_time(ann_base, 16, 10)
call_time_et = to_et_with_time(call_base, 17, 0)

def is_after_hours(ts):
    if pd.isna(ts): 
        return 1
    h, m = ts.hour, ts.minute
    # after close (>= 16:00) OR before open (< 9:30) = after-hours
    return 1 if (h > 16 or (h == 16 and m >= 0) or h < 9 or (h == 9 and m < 30)) else 0

after_hours = ann_time_et.apply(is_after_hours).astype(int)

out = pd.DataFrame({
    "ticker": df["symbol"].str.upper(),
    "year": df["year"].astype(int),
    "q": df["quarter"].astype(int),
    "ann_time_et": ann_time_et.dt.tz_convert("America/New_York").astype(str),
    "call_time_et": call_time_et.dt.tz_convert("America/New_York").astype(str),
    "after_hours": after_hours
}).dropna(subset=["ticker","year","q","ann_time_et"])

out.to_csv(OUT, index=False)
print(f"Wrote {len(out)} rows to {OUT}")
