# engine/fundamentals_core.py
import math, time, random
from typing import Dict, List
import pandas as pd
import yfinance as yf

SAFE_KEYS = [
    ("longName", str),
    ("sector", str),
    ("industry", str),
    ("longBusinessSummary", str),
    ("marketCap", float),
    ("trailingPE", float),
    ("forwardPE", float),
    ("priceToBook", float),
    ("pegRatio", float),
    ("trailingPegRatio", float),
    ("returnOnEquity", float),
    ("profitMargins", float),
    ("grossMargins", float),
    ("operatingMargins", float),
    ("debtToEquity", float),
    ("totalDebt", float),
    ("totalCash", float),
    ("revenueGrowth", float),
    ("earningsGrowth", float),
    ("revenuePerShare", float),
    ("trailingEps", float),
    ("dividendYield", float),
]

def _safe(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
    except Exception:
        return None
    return v

def _coalesce(*vals):
    for v in vals:
        if _safe(v) is not None:
            return v
    return None

def _fetch_info_once(tk: yf.Ticker) -> Dict:
    """Try .get_info(), then .info; augment with fast_info.market_cap if missing."""
    info = {}
    try:
        if hasattr(tk, "get_info"):
            info = tk.get_info() or {}
    except Exception:
        info = {}
    try:
        if not info:
            info = tk.info or {}
    except Exception:
        pass
    try:
        if ("marketCap" not in info or not info.get("marketCap")) and getattr(tk, "fast_info", None):
            mc = getattr(tk.fast_info, "market_cap", None)
            if mc:
                info["marketCap"] = mc
    except Exception:
        pass
    return info or {}

def _row_from_info(sym: str, info: Dict) -> Dict:
    out = {"symbol": sym}
    for k, _t in SAFE_KEYS:
        out[k] = _safe(info.get(k))
    out["company"] = _coalesce(out.get("longName"), sym)
    out["peg"] = _coalesce(out.get("trailingPegRatio"), out.get("pegRatio"))
    return out

def _pct(x):
    return None if x is None else float(x) * 100.0

def _f(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def fetch_single_with_retry(sym: str, retries: int = 2, sleep_base: float = 0.6) -> Dict:
    tk = yf.Ticker(sym)
    last_err = None
    for _ in range(retries + 1):
        try:
            info = _fetch_info_once(tk)
            if info:
                row = _row_from_info(sym, info)
                row["error"] = None
                return row
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep_base * (1 + 0.5 * random.random()))
    return {"symbol": sym, "company": sym, "error": last_err or "info-unavailable"}

def compute_fundamentals(watchlist_csv: str, out_csv: str, throttle_sec: float = 0.3) -> int:
    """
    Fetch fundamentals for all symbols in watchlist and write a normalized CSV.
    Guarantees one row per symbol (with NA where data is missing).
    """
    wl = pd.read_csv(watchlist_csv)
    if wl.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return 0

    syms: List[str] = wl["symbol"].dropna().unique().tolist()
    rows: List[Dict] = []

    for s in syms:
        r = fetch_single_with_retry(s)
        rows.append(r)
        time.sleep(throttle_sec)

    df = pd.DataFrame(rows)

    out = pd.DataFrame({
        "symbol": df["symbol"],
        "company": df.get("company"),
        "sector": df.get("sector"),
        "industry": df.get("industry"),
        "mcap": df.get("marketCap").apply(_f),
        "pe_ttm": df.get("trailingPE").apply(_f),
        "pe_fwd": df.get("forwardPE").apply(_f),
        "pb": df.get("priceToBook").apply(_f),
        "peg": df.get("peg").apply(_f),
        "roe_pct": df.get("returnOnEquity").apply(_pct),
        "pm_pct": df.get("profitMargins").apply(_pct),
        "gm_pct": df.get("grossMargins").apply(_pct),
        "om_pct": df.get("operatingMargins").apply(_pct),
        "de_ratio": df.get("debtToEquity").apply(_f),
        "debt": df.get("totalDebt").apply(_f),
        "cash": df.get("totalCash").apply(_f),
        "rev_growth_pct": df.get("revenueGrowth").apply(_pct),
        "earn_growth_pct": df.get("earningsGrowth").apply(_pct),
        "rev_per_share": df.get("revenuePerShare").apply(_f),
        "eps_ttm": df.get("trailingEps").apply(_f),
        "div_yield_pct": df.get("dividendYield").apply(_pct),
        "summary": df.get("longBusinessSummary"),
        "error": df.get("error")
    }).drop_duplicates(subset=["symbol"], keep="last")

    out.to_csv(out_csv, index=False)
    return len(out)