import os, re, argparse, json
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

try:
    import pandas_market_calendars as pmc
except Exception:
    pmc = None

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util as st_util
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Tariff Sentiment Event Study — Clean (events_20251001_updated.csv only)")
    p.add_argument("--transcripts_dir", type=str, default="sp500_transcripts")
    p.add_argument("--summaries_dir",  type=str, default="sp500_summaries")
    p.add_argument("--tickers", type=str, default="", help="Comma-separated tickers subset (optional)")
    p.add_argument("--start", type=str, default="2023-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--market_proxy", type=str, default="SPY")
    p.add_argument("--save_dir", type=str, default="outputs_v2")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--top_quotes", type=int, default=10, help="How many top +/- quotes to save")
    return p.parse_args()

FN_RE = re.compile(r"^(?P<ticker>[A-Z]+)_(?P<year>\d{4})_Q(?P<q>[1-4])(?:_summary)?(?:\..+)?$")

def parse_file_id(name: str) -> Optional[Tuple[str,int,int]]:
    base = os.path.splitext(os.path.basename(name))[0]
    m = FN_RE.match(base)
    if not m: return None
    return (m.group("ticker"), int(m.group("year")), int(m.group("q")))

def list_firm_quarters(transcripts_dir: str, summaries_dir: str, subset: Optional[set]) -> pd.DataFrame:
    rows = []
    for d, kind in [(transcripts_dir, "transcript"), (summaries_dir, "summary")]:
        if not os.path.isdir(d): continue
        for f in os.listdir(d):
            m = parse_file_id(f)
            if not m: continue
            t,y,q = m
            if subset and t not in subset: continue
            rows.append({"ticker":t,"year":y,"q":q,"kind":kind,"path":os.path.join(d,f)})
    df = pd.DataFrame(rows).drop_duplicates()
    if df.empty:
        raise RuntimeError("No firm-quarters found. Check folder names and filename patterns.")
    return df.groupby(["ticker","year","q"])["kind"].apply(list).reset_index()

def fiscal_qtr_end(year:int, q:int) -> pd.Timestamp:
    m = q*3
    return pd.Timestamp(year=year, month=m, day=1) + pd.offsets.MonthEnd(0)

def make_calendar():
    if pmc is None: return None
    try: return pmc.get_calendar("XNYS")
    except Exception: return None

def _to_et_scalar(x):
    if x is None:
        return pd.NaT
    try:
        if pd.isna(x):
            return pd.NaT
    except Exception:
        pass
    ts = pd.to_datetime(x, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    try:
        return ts.tz_convert("America/New_York")
    except TypeError:
        return ts.tz_localize("America/New_York")

def _tz_to_et(obj):
    if isinstance(obj, pd.Series):
        ser = obj.apply(_to_et_scalar)
        try:
            ser = ser.astype("datetime64[ns, America/New_York]")
        except Exception:
            pass
        return ser
    else:
        return _to_et_scalar(obj)

def _infer_after_hours_from_ann(ts_et: pd.Timestamp) -> Optional[int]:
    if not isinstance(ts_et, pd.Timestamp) or pd.isna(ts_et):
        return None
    hhmm = ts_et.hour * 60 + ts_et.minute
    if hhmm >= 16*60:
        return 1
    if hhmm < 9*60 + 30:
        return 0
    return 0

def read_events_times_from_project_root() -> pd.DataFrame:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "events_20251001_updated.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError("Required events_20251001_updated.csv not found next to the .py file.")
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None
    c_tk   = pick("ticker", "symbol")
    c_y    = pick("year", "fyear", "fiscal_year")
    c_q    = pick("q", "fq", "quarter", "fiscal_quarter")
    c_ann  = pick("ann_time_et", "conference_date", "announcement_time_et",
                  "press_release_time_et", "press_release_ts", "ann_ts", "announcement_ts")
    c_call = pick("call_time_et", "call_start_time_et", "call_ts", "call_start_ts")
    c_ah   = pick("after_hours", "afterhours", "is_after_hours")
    need = [c_tk, c_y, c_q, c_ann]
    if any(x is None for x in need):
        raise ValueError(f"events file missing required columns (need ticker/year/q/ann_time_et). Found: {list(df.columns)}")
    out = pd.DataFrame({
        "ticker": df[c_tk].astype(str).str.upper(),
        "year":   pd.to_numeric(df[c_y], errors="coerce").astype("Int64"),
        "q":      pd.to_numeric(df[c_q], errors="coerce").astype("Int64"),
    })
    out["ann_time_et"]  = _tz_to_et(df[c_ann])
    out["call_time_et"] = _tz_to_et(df[c_call]) if c_call else pd.NaT
    if c_ah:
        ah_raw = pd.to_numeric(df[c_ah], errors="coerce")
        out["after_hours"] = ah_raw.where(ah_raw.isin([0,1]), pd.NA).astype("Int64")
    else:
        out["after_hours"] = pd.NA
    mask = out["after_hours"].isna() & out["ann_time_et"].notna()
    out.loc[mask, "after_hours"] = out.loc[mask, "ann_time_et"].apply(_infer_after_hours_from_ann).astype("Int64")
    out = out.dropna(subset=["ann_time_et"])
    return out

def to_day0(cal, ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        ts = ts.tz_localize("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    def _sessions_et_around(center_ts, days_before=7, days_after=10):
        if cal is None:
            return None
        sched = cal.schedule(
            start_date=(center_ts - pd.Timedelta(days=days_before)).date(),
            end_date=(center_ts + pd.Timedelta(days=days_after)).date()
        )
        return pd.DatetimeIndex(sched.index).tz_localize("UTC").tz_convert("America/New_York").normalize()
    def _next_trading_day_et(current_ts):
        if cal is not None:
            sessions_et = _sessions_et_around(current_ts)
            if sessions_et is not None:
                nxt = sessions_et[sessions_et > current_ts.normalize()]
                if len(nxt) > 0:
                    return nxt[0]
        bd = (current_ts + pd.tseries.offsets.BDay(1)).normalize()
        return bd.tz_convert("America/New_York") if bd.tzinfo else bd.tz_localize("America/New_York")
    hour, minute = ts.hour, ts.minute
    if (hour < 9) or (hour == 9 and minute < 30):
        d0 = ts.normalize()
        if cal is not None:
            sessions_et = _sessions_et_around(ts, 2, 2)
            if sessions_et is not None and d0 in sessions_et:
                pass
            else:
                d0 = _next_trading_day_et(ts)
        else:
            if d0.weekday() >= 5:
                d0 = _next_trading_day_et(ts)
    elif (hour > 16) or (hour == 16 and minute >= 0):
        d0 = _next_trading_day_et(ts)
    else:
        d0 = _next_trading_day_et(ts)
    if d0.weekday() >= 5:
        if cal is not None:
            sched = cal.schedule(start_date=d0.date(), end_date=(d0 + pd.Timedelta(days=7)).date())
            if not sched.empty:
                nxt = pd.DatetimeIndex(sched.index).tz_localize("UTC").tz_convert("America/New_York")
                nxt = nxt[nxt > d0]
                if len(nxt):
                    return nxt[0].normalize()
        return (d0 + pd.tseries.offsets.BDay(1)).tz_convert("America/New_York").normalize()
    return d0

def download_prices(tickers: List[str], start: str, end: Optional[str], mkt_proxy: str) -> Tuple[pd.DataFrame, pd.Series]:
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")
    symbols = sorted(set(tickers+[mkt_proxy]))
    data = yf.download(" ".join(symbols), start=start, end=end, auto_adjust=True, progress=False, group_by="ticker", threads=True)
    def close_of(sym):
        if isinstance(data.columns, pd.MultiIndex):
            if sym in data.columns and "Close" in data[sym].columns:
                return data[sym]["Close"].rename(sym)
        else:
            if "Close" in data.columns:
                return data["Close"].rename(sym)
        raise KeyError(f"Close not found for {sym}")
    closes = []
    for sym in symbols:
        try:
            closes.append(close_of(sym))
        except Exception:
            pass
    wide = pd.concat(closes, axis=1).dropna(how="all")
    ret = wide.pct_change().dropna(how="all")
    if mkt_proxy not in ret.columns:
        raise RuntimeError(f"Market proxy {mkt_proxy} not present in downloaded returns.")
    mkt = ret[mkt_proxy].rename("MKT")
    ret = ret.drop(columns=[mkt_proxy])
    ret_long = ret.stack().rename("RET").reset_index()
    ret_long.columns = ["date","ticker","RET"]
    ret_long["date"] = pd.to_datetime(ret_long["date"]).dt.tz_localize("America/New_York").dt.normalize()
    mkt.index = pd.to_datetime(mkt.index).tz_localize("America/New_York").normalize()
    return ret_long, mkt

def trading_idx(ret_long: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(sorted(ret_long["date"].unique()))

def window_dates(idx: pd.DatetimeIndex, day0: pd.Timestamp, start:int, end:int) -> List[pd.Timestamp]:
    if day0 not in idx:
        nxt = idx[idx >= day0]
        if len(nxt)==0: return []
        day0 = nxt[0]
    pos = idx.get_loc(day0)
    lo, hi = max(0, pos+start), min(len(idx)-1, pos+end)
    return idx[lo:hi+1].tolist()

def market_model_params(ret_s: pd.Series, mkt_s: pd.Series, est_dates: List[pd.Timestamp]) -> Optional[pd.Series]:
    df = pd.DataFrame({"RET":ret_s.reindex(est_dates).values,"MKT":mkt_s.reindex(est_dates).values}).dropna()
    if len(df)<30: return None
    X = sm.add_constant(df["MKT"])
    return sm.OLS(df["RET"], X).fit().params

def car_market_model(ret_s: pd.Series, mkt_s: pd.Series, idx: pd.DatetimeIndex, day0: pd.Timestamp, win=(0,1), est=(-250,-20)) -> Optional[float]:
    est_d = window_dates(idx, day0, est[0], est[1])
    win_d = window_dates(idx, day0, win[0], win[1])
    if len(est_d)<30 or len(win_d)==0: return None
    params = market_model_params(ret_s, mkt_s, est_d)
    if params is None: return None
    df = pd.DataFrame({"RET":ret_s.reindex(win_d).values,"MKT":mkt_s.reindex(win_d).values}).dropna()
    if df.empty: return None
    ar = df["RET"] - (params["const"] + params["MKT"]*df["MKT"])
    return float(ar.sum())

SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9])')

def split_sentences(text:str)->List[str]:
    t = re.sub(r"\s+"," ", (text or "").strip())
    if not t: return []
    s = SENT_SPLIT_RE.split(t)
    return [x.strip() for x in s if x and len(x.strip())>0]

QA_FLAG = re.compile(r"(question\s*&\s*answer|q\s*&\s*a|q\.?\s*&\s*a|operator[:\-].*questions?)", re.I)
def split_prepared_qa(text:str)->Tuple[str,str]:
    if not text: return "",""
    m = QA_FLAG.search(text)
    if not m: return text,""
    return text[:m.start()], text[m.start():]

TARIFF_KEYWORDS = [
    r"tariff(s)?", r"dut(y|ies)", r"lev(y|ies)", r"quota(s)?",
    r"section\s*301", r"countervailing", r"anti[- ]dump(ing)?",
    r"import tax(es)?", r"customs", r"retaliator(y|y measures)",
    r"exemption(s)?", r"exclusion(s)?"
]
KW_RE = re.compile(r"(" + r"|".join(TARIFF_KEYWORDS) + r")", re.I)

FX_RE = re.compile(r"\b(fx|foreign exchange|currency|currencies|dollar strength|usd strength)\b", re.I)

FWD_RE = re.compile(r"\b(expect(s|ed|ing)?|outlook|guide|guidance|plan(s|ned|ning)?|will|next (q(uarter)?|year))\b", re.I)

def dedup_keep_long(sents: List[str]) -> List[str]:
    out, seen = [], set()
    for s in sents:
        t = re.sub(r"\s+"," ", s).strip()
        if len(t)<20: continue
        k = t.lower()
        if k in seen: continue
        seen.add(k); out.append(t)
    return out

def load_models():
    print("Loading all-MiniLM-L6-v2 ...")
    embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Loading FinBERT ...")
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    clf = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").eval()
    return embed, tok, clf

SEED_QUERIES = [
    "impact of tariffs on costs", "duty increases on imports",
    "tariff headwinds or tailwinds", "tariff exclusions expired",
    "retaliatory tariffs from trading partners"
]

def tariff_hits(sentences: List[str], embed, seed_vec, sim_thresh=0.45) -> List[str]:
    hits = [s for s in sentences if KW_RE.search(s)]
    if len(hits) >= 3: return dedup_keep_long(hits)
    if not sentences: return []
    S = embed.encode(sentences, normalize_embeddings=True)
    sims = np.max(st_util.cos_sim(S, seed_vec).cpu().numpy(), axis=1)
    aug = [sentences[i] for i,sim in enumerate(sims) if sim>sim_thresh]
    return dedup_keep_long(list({*hits, *aug}))

def fx_hits(sentences: List[str]) -> List[str]:
    return dedup_keep_long([s for s in sentences if FX_RE.search(s)])

def finbert_scores(sents: List[str], tok, clf, batch=16) -> Tuple[float,float,int,float]:
    if not sents: return (np.nan, np.nan, 0, np.nan)
    pol, neg, n = [], 0, len(sents)
    fwd = [s for s in sents if FWD_RE.search(s)]
    with torch.no_grad():
        for i in range(0, len(sents), batch):
            chunk = sents[i:i+batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            probs = torch.softmax(clf(**enc).logits, dim=1).cpu().numpy()
            pol.extend(probs[:,2]-probs[:,0])
            neg += (probs[:,0] > probs[:,2]).sum()
    pol_mean = float(np.mean(pol))
    share_neg = float(neg/len(sents))
    pol_fwd = np.nan
    if fwd:
        with torch.no_grad():
            vals = []
            for i in range(0, len(fwd), batch):
                chunk = fwd[i:i+batch]
                enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
                probs = torch.softmax(clf(**enc).logits, dim=1).cpu().numpy()
                vals.extend(probs[:,2]-probs[:,0])
        pol_fwd = float(np.mean(vals))
    return pol_mean, share_neg, n, pol_fwd

def finbert_polarities(sents: List[str], tok, clf, batch=16) -> List[float]:
    if not sents: return []
    vals = []
    with torch.no_grad():
        for i in range(0, len(sents), batch):
            chunk = sents[i:i+batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            probs = torch.softmax(clf(**enc).logits, dim=1).cpu().numpy()
            vals.extend(list(probs[:,2]-probs[:,0]))
    return [float(x) for x in vals]

def get_eps_surprise_yf(ticker: str, anchor_date: pd.Timestamp, limit: int = 24):
    try:
        t = yf.Ticker(ticker)
        df = t.get_earnings_dates(limit=limit)
    except Exception as e:
        return None
    if df is None or len(df) == 0:
        return None
    cols_lower = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols_lower:
                return cols_lower[n]
        return None
    if "earnings date" in cols_lower:
        dt_col = cols_lower["earnings date"]
        idx = pd.to_datetime(df[dt_col], errors="coerce")
    else:
        idx = pd.to_datetime(df.index, errors="coerce")
    try:
        idx = idx.tz_localize("UTC", nonexistent="NaT", ambiguous="NaT").tz_convert("America/New_York")
    except Exception:
        try:
            idx = idx.tz_convert("America/New_York")
        except Exception:
            pass
    c_rep = pick("reported eps", "reported", "actual eps", "actual")
    c_est = pick("eps estimate", "estimate", "consensus eps", "consensus")
    c_spr = pick("surprise(%)", "surprise %", "surprise_pct", "surprise")
    if c_rep is None or c_est is None:
        return None
    try:
        diffs = (idx - anchor_date).days.to_numpy()
        i = int(np.nanargmin(np.abs(diffs)))
    except Exception:
        return None
    row = df.iloc[i]
    try:
        rep = pd.to_numeric(row.get(c_rep), errors="coerce")
        est = pd.to_numeric(row.get(c_est), errors="coerce")
    except Exception:
        return None
    if pd.isna(rep) or pd.isna(est):
        return None
    spr_pct = None
    if c_spr is not None:
        spr_pct = pd.to_numeric(row.get(c_spr), errors="coerce")
        if pd.isna(spr_pct):
            spr_pct = None
    return float(rep), float(est), (float(spr_pct) if spr_pct is not None else None)

def crude_eps_surprise_from_summary(text:str) -> Optional[float]:
    if not text: return None
    m = re.search(r"EPS:?\s*\$?([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    c = re.search(r"(consensus|estimate[s]?)[:\s]*\$?([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if m and c:
        try: return float(m.group(1)) - float(c.group(2))
        except Exception: return None
    b = re.search(r"(beat|miss)(?:ed)? (?:by )?\$?([0-9]+(?:\.[0-9]+)?)", text, flags=re.I)
    if b:
        val = float(b.group(2))
        return val if b.group(1).lower().startswith("beat") else -val
    return None

def size_sector_yf(ticker:str) -> Tuple[Optional[float], Optional[str]]:
    try:
        t = yf.Ticker(ticker)
        mc = None
        try:
            mc = getattr(t, "fast_info", {}).get("market_cap", None)
        except Exception:
            pass
        if not mc:
            info = t.info or {}
            mc = info.get("marketCap", None)
            sec = info.get("sector", None)
        else:
            sec = (t.info or {}).get("sector", None)
        sz = float(mc) if mc else None
        return sz, sec
    except Exception:
        return None, None

def momentum_12_2(ret_s: pd.Series, idx: pd.DatetimeIndex, day0: pd.Timestamp) -> Optional[float]:
    w = window_dates(idx, day0, -252, -42)
    vals = ret_s.reindex(w).dropna()
    if len(vals)<50: return None
    return float((1+vals).prod()-1)

def tariff_exposure_proxy(sector: Optional[str]) -> int:
    exposed = {"Technology","Consumer Cyclical","Industrials","Materials"}
    return 1 if (sector in exposed) else 0

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    subset = set([s.strip().upper() for s in args.tickers.split(",") if s.strip()]) if args.tickers else None
    events = list_firm_quarters(args.transcripts_dir, args.summaries_dir, subset)
    def build_map(root):
        out = {}
        if os.path.isdir(root):
            for f in os.listdir(root):
                m = parse_file_id(f)
                if not m: continue
                out[(m[0],m[1],m[2])] = os.path.join(root,f)
        return out
    tmap = build_map(args.transcripts_dir)
    smap = build_map(args.summaries_dir)
    ev_lookup = read_events_times_from_project_root()
    ev_keys = ev_lookup[["ticker","year","q"]].drop_duplicates()
    events = events.merge(ev_keys, on=["ticker","year","q"], how="inner")
    if events.empty:
        raise RuntimeError("No overlapping firm–quarters between folders and events_20251001_updated.csv.")
    embed, tok, clf = load_models()
    seed_vec = embed.encode(SEED_QUERIES, normalize_embeddings=True)
    cal = make_calendar()
    tickers = sorted(events["ticker"].unique().tolist())
    print(f"Downloading prices for {len(tickers)} tickers + market {args.market_proxy} ...")
    ret_long, mkt = download_prices(tickers, args.start, args.end, args.market_proxy)
    idx = trading_idx(ret_long)
    ret_panel = {t: ret_long.loc[ret_long["ticker"]==t, ["date","RET"]].set_index("date")["RET"] for t in tickers}
    quote_pool = []
    rows = []
    for _, ev in events.iterrows():
        ticker, year, q = ev["ticker"], int(ev["year"]), int(ev["q"])
        key = (ticker, year, q)
        print(f"\n=== {ticker} {year} Q{q} ===")
        transcript = ""
        if key in tmap and os.path.isfile(tmap[key]):
            transcript = open(tmap[key], "r", encoding="utf-8", errors="ignore").read()
        summary = ""
        if key in smap and os.path.isfile(smap[key]):
            summary = open(smap[key], "r", encoding="utf-8", errors="ignore").read()
        row_ev = ev_lookup[(ev_lookup["ticker"]==ticker) & (ev_lookup["year"]==year) & (ev_lookup["q"]==q)]
        if len(row_ev) == 0:
            raise KeyError(f"No event timing found in events_20251001_updated.csv for {ticker} {year} Q{q}")
        ann_ts = row_ev["ann_time_et"].iloc[0]
        call_ts = row_ev["call_time_et"].iloc[0] if "call_time_et" in row_ev.columns else pd.NaT
        ah = row_ev["after_hours"].iloc[0] if "after_hours" in row_ev.columns else pd.NA
        after_hours = int(ah) if not pd.isna(ah) else _infer_after_hours_from_ann(ann_ts)
        day0 = to_day0(cal, ann_ts)
        if day0.weekday() >= 5:
            print(f"[WARN] day0 fell on weekend for {ticker} {year} Q{q}: {day0} (ann_ts={ann_ts})")
        if ticker not in ret_panel:
            print("No returns for", ticker); continue
        rs = ret_panel[ticker]
        car_0p1  = car_market_model(rs, mkt, idx, day0, win=(0,1))
        car_m1p1 = car_market_model(rs, mkt, idx, day0, win=(-1,1))
        car_0p2  = car_market_model(rs, mkt, idx, day0, win=(0,2))
        if isinstance(call_ts, pd.Timestamp) and pd.notna(call_ts):
            call_day0 = to_day0(cal, call_ts)
            car_call_0p1 = car_market_model(rs, mkt, idx, call_day0, win=(0,1))
        else:
            car_call_0p1 = np.nan
        prep_txt, qa_txt = split_prepared_qa(transcript)
        prep_sents = split_sentences(prep_txt)
        qa_sents   = split_sentences(qa_txt)
        prep_tariff = tariff_hits(prep_sents, embed, seed_vec, sim_thresh=0.45)
        qa_tariff   = tariff_hits(qa_sents,   embed, seed_vec, sim_thresh=0.45)
        prep_pol, prep_share_neg, prep_mentions, prep_fwd = finbert_scores(prep_tariff, tok, clf, batch=args.batch_size)
        qa_pol,   qa_share_neg,   qa_mentions,   qa_fwd   = finbert_scores(qa_tariff,   tok, clf, batch=args.batch_size)
        prep_pols = finbert_polarities(prep_tariff, tok, clf, batch=args.batch_size)
        for s, p in zip(prep_tariff, prep_pols):
            quote_pool.append({"ticker":ticker,"year":year,"q":q,"sentence":s,"polarity":p})
        prep_fx = fx_hits(prep_sents)
        qa_fx   = fx_hits(qa_sents)
        prep_fx_pol, _, prep_fx_n, _ = finbert_scores(prep_fx, tok, clf, batch=args.batch_size)
        qa_fx_pol,   _, qa_fx_n,   _ = finbert_scores(qa_fx,   tok, clf, batch=args.batch_size)
        surp = get_eps_surprise_yf(ticker, day0)
        if surp:
            rep, est, spr_pct = surp
            eps_surp = rep - est
            eps_surp_pct = (rep - est)/abs(est) if est else np.nan
            eps_surp_pct_yf = spr_pct
        else:
            eps_surp = crude_eps_surprise_from_summary(summary)
            eps_surp_pct = np.nan
            eps_surp_pct_yf = np.nan
        size, sector = size_sector_yf(ticker)
        ln_size = float(np.log(size)) if size and size>0 else np.nan
        mom = momentum_12_2(rs, idx, day0)
        exposure = tariff_exposure_proxy(sector)
        rows.append({
            "ticker":ticker, "year":year, "q":q,
            "ann_ts":ann_ts, "call_ts":call_ts, "day0":day0, "after_hours":after_hours,
            "CAR_0p1":car_0p1, "CAR_m1p1":car_m1p1, "CAR_0p2":car_0p2,
            "CAR_call_0p1":car_call_0p1,
            "TariffSent_prep":prep_pol, "TariffShareNeg_prep":prep_share_neg, "TariffMentions_prep":prep_mentions, "TariffSent_fwd_prep":prep_fwd,
            "TariffSent_qa":qa_pol, "TariffMentions_qa":qa_mentions, "TariffSent_fwd_qa":qa_fwd,
            "FXSent_prep":prep_fx_pol, "FXMentions_prep":prep_fx_n,
            "FXSent_qa":qa_fx_pol, "FXMentions_qa":qa_fx_n,
            "EPS_surprise":eps_surp, "EPS_surprise_pct":eps_surp_pct, "EPS_surprise_pct_yf":eps_surp_pct_yf,
            "ln_size":ln_size, "momentum_12_2":mom, "sector":sector,
            "TariffExposure_proxy": exposure
        })
    panel = pd.DataFrame(rows)
    panel["cal_qtr"] = panel["year"].astype(str) + "Q" + panel["q"].astype(str)
    out_panel = os.path.join(args.save_dir, "panel_results.csv")
    panel.to_csv(out_panel, index=False)
    print(f"\nSaved panel to {out_panel}")
    try:
        ts = panel.groupby("cal_qtr")["TariffSent_prep"].mean().sort_index()
        ts.to_csv(os.path.join(args.save_dir, "sentiment_timeseries.csv"))
        plt.figure()
        ts.plot()
        plt.title("Average Tariff Sentiment (Prepared) by Quarter")
        plt.xlabel("Calendar Quarter"); plt.ylabel("Mean Polarity [-1,1]")
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "sentiment_timeseries.png"))
        plt.close()
        pivot = panel.pivot_table(values="TariffSent_prep", index="sector", columns="cal_qtr", aggfunc="mean")
        pivot.to_csv(os.path.join(args.save_dir, "sector_heatmap_table.csv"))
        if not pivot.empty:
            plt.figure(figsize=(max(6, 0.5*len(pivot.columns)), max(4, 0.3*len(pivot.index))))
            im = plt.imshow(pivot.values, aspect="auto")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(ticks=range(len(pivot.columns)), labels=list(pivot.columns), rotation=90)
            plt.yticks(ticks=range(len(pivot.index)), labels=list(pivot.index))
            plt.title("Tariff Sentiment (Prepared): Sector × Quarter")
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "sector_heatmap.png"))
            plt.close()
        pq = panel[["TariffSent_prep","TariffSent_qa"]].dropna()
        if not pq.empty:
            plt.figure()
            plt.scatter(pq["TariffSent_prep"], pq["TariffSent_qa"], s=10)
            plt.axhline(0, linewidth=0.8); plt.axvline(0, linewidth=0.8)
            plt.xlabel("Prepared sentiment"); plt.ylabel("Q&A sentiment")
            plt.title("Prepared vs Q&A Tariff Sentiment")
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "prepared_vs_qa_scatter.png"))
            plt.close()
    except Exception as e:
        print("Descriptive analytics failed:", e)
    def run_reg(dep: str, tone_col: str, suffix: str, include_surprise=True):
        df = panel.copy()
        surprise_col = None
        if include_surprise and df["EPS_surprise"].notna().mean() > 0.4:
            surprise_col = "EPS_surprise"
        elif include_surprise and df["EPS_surprise_pct_yf"].notna().mean() > 0.4:
            surprise_col = "EPS_surprise_pct_yf"
        base_cols = [tone_col, "after_hours", "ln_size", "momentum_12_2"]
        if surprise_col:
            base_cols.insert(1, surprise_col)
        keep_cols = [dep, "ticker", "sector", "cal_qtr"] + base_cols
        df = df[keep_cols].copy()
        df["sector"] = df["sector"].astype(str)
        for c in [dep] + base_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=[dep] + base_cols)
        if df.empty or df.shape[0] < 5:
            print(f"[WARN] Not enough rows for regression {suffix} (n={df.shape[0]}). Skipping.")
            return
        fe_blocks = []
        if df["sector"].nunique() >= 2 and df.shape[0] >= df["sector"].nunique() + 5:
            fe_blocks.append(pd.get_dummies(df["sector"], prefix="sector", drop_first=True))
        if df["cal_qtr"].nunique() >= 2 and df.shape[0] >= df["cal_qtr"].nunique() + 5:
            fe_blocks.append(pd.get_dummies(df["cal_qtr"], prefix="cal_qtr", drop_first=True))
        X = pd.concat([df[base_cols]] + fe_blocks, axis=1)
        X = X.apply(pd.to_numeric, errors="coerce").astype(float)
        X = sm.add_constant(X, has_constant="add")
        Y = pd.to_numeric(df[dep], errors="coerce").astype(float)
        mask = (~Y.isna()) & (~X.isna().any(axis=1))
        X, Y = X.loc[mask], Y.loc[mask]
        if X.empty or Y.empty or X.shape[0] < 5:
            print(f"[WARN] Not enough clean rows for regression {suffix} after NA scrub. Skipping.")
            return
        if X.shape[1] >= X.shape[0]:
            cols = [c for c in X.columns if not c.startswith("cal_qtr_")]
            X = X[cols]
        if X.shape[1] >= X.shape[0]:
            cols = [c for c in X.columns if not c.startswith("sector_")]
            X = X[cols]
        if X.shape[1] >= X.shape[0]:
            print(f"[WARN] Params ({X.shape[1]}) >= obs ({X.shape[0]}) for {suffix}. Skipping.")
            return
        n_groups = df.loc[X.index, "ticker"].nunique()
        if n_groups >= 2:
            model = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": df.loc[X.index, "ticker"]})
        else:
            model = sm.OLS(Y, X).fit(cov_type="HC1")
        summ = model.summary().as_text()
        with open(os.path.join(args.save_dir, f"regression_{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(summ)
        coefs = pd.DataFrame({
            "param": model.params.index,
            "coef": model.params.values,
            "t": model.tvalues,
            "p": model.pvalues
        })
        coefs.to_csv(os.path.join(args.save_dir, f"regression_coefs_{suffix}.csv"), index=False)
        print(f"\nSaved regression {suffix}")
        print(summ)
    def run_reg_with_interaction(dep: str, tone_col: str, suffix: str, include_surprise=True):
        df = panel.copy()
        df["TariffExposure_proxy"] = pd.to_numeric(df["TariffExposure_proxy"], errors="coerce")
        df[tone_col] = pd.to_numeric(df[tone_col], errors="coerce")
        df["TariffSent_x_Exposure"] = df[tone_col] * df["TariffExposure_proxy"]
        surprise_col = None
        if include_surprise and df["EPS_surprise"].notna().mean() > 0.4:
            surprise_col = "EPS_surprise"
        elif include_surprise and df["EPS_surprise_pct_yf"].notna().mean() > 0.4:
            surprise_col = "EPS_surprise_pct_yf"
        base_cols = [tone_col, "TariffExposure_proxy", "TariffSent_x_Exposure",
                     "after_hours", "ln_size", "momentum_12_2"]
        if surprise_col:
            base_cols.insert(1, surprise_col)
        keep_cols = [dep, "ticker", "sector", "cal_qtr"] + base_cols
        df = df[keep_cols].copy()
        df["sector"] = df["sector"].astype(str)
        for c in [dep] + base_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=[dep] + base_cols)
        if df.empty or df.shape[0] < 5:
            print(f"[WARN] Not enough rows for regression {suffix} (n={df.shape[0]}). Skipping.")
            return
        fe_blocks = []
        if df["sector"].nunique() >= 2 and df.shape[0] >= df["sector"].nunique() + 5:
            fe_blocks.append(pd.get_dummies(df["sector"], prefix="sector", drop_first=True))
        if df["cal_qtr"].nunique() >= 2 and df.shape[0] >= df["cal_qtr"].nunique() + 5:
            fe_blocks.append(pd.get_dummies(df["cal_qtr"], prefix="cal_qtr", drop_first=True))
        X = pd.concat([df[base_cols]] + fe_blocks, axis=1)
        X = X.apply(pd.to_numeric, errors="coerce").astype(float)
        X = sm.add_constant(X, has_constant="add")
        Y = pd.to_numeric(df[dep], errors="coerce").astype(float)
        mask = (~Y.isna()) & (~X.isna().any(axis=1))
        X, Y = X.loc[mask], Y.loc[mask]
        if X.empty or Y.empty or X.shape[0] < 5:
            print(f"[WARN] Not enough clean rows for regression {suffix} after NA scrub. Skipping.")
            return
        if X.shape[1] >= X.shape[0]:
            cols = [c for c in X.columns if not c.startswith("cal_qtr_")]
            X = X[cols]
        if X.shape[1] >= X.shape[0]:
            cols = [c for c in X.columns if not c.startswith("sector_")]
            X = X[cols]
        if X.shape[1] >= X.shape[0]:
            print(f"[WARN] Params ({X.shape[1]}) >= obs ({X.shape[0]}) for {suffix}. Skipping.")
            return
        n_groups = df.loc[X.index, "ticker"].nunique()
        if n_groups >= 2:
            model = sm.OLS(Y, X).fit(cov_type="cluster", cov_kwds={"groups": df.loc[X.index, "ticker"]})
        else:
            model = sm.OLS(Y, X).fit(cov_type="HC1")
        summ = model.summary().as_text()
        with open(os.path.join(args.save_dir, f"regression_{suffix}.txt"), "w", encoding="utf-8") as f:
            f.write(summ)
        coefs = pd.DataFrame({
            "param": model.params.index,
            "coef": model.params.values,
            "t": model.tvalues,
            "p": model.pvalues
        })
        coefs.to_csv(os.path.join(args.save_dir, f"regression_coefs_{suffix}.csv"), index=False)
        print(f"\nSaved regression {suffix}")
        print(summ)
    run_reg(dep="CAR_0p1", tone_col="TariffSent_prep", suffix="main_prep_CAR0p1", include_surprise=True)
    run_reg(dep="CAR_0p1", tone_col="TariffSent_qa", suffix="robust_qa_CAR0p1", include_surprise=True)
    run_reg(dep="CAR_0p2", tone_col="TariffSent_prep", suffix="robust_prep_CAR0p2", include_surprise=True)
    run_reg(dep="CAR_0p1", tone_col="FXSent_prep", suffix="placebo_FX_CAR0p1", include_surprise=True)
    run_reg(dep="CAR_0p1", tone_col="TariffSent_fwd_prep", suffix="forward_prep_CAR0p1", include_surprise=True)
    run_reg_with_interaction(dep="CAR_0p1", tone_col="TariffSent_prep", suffix="interaction_exposure_prep_CAR0p1", include_surprise=True)

if __name__ == "__main__":
    main()
