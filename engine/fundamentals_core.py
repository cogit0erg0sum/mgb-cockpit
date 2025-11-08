# engine/fundamentals_core.py
import math, time, random
from typing import Dict, List
import pandas as pd
import yfinance as yf

def fetch_fundamentals_one(symbol: str) -> dict:
    """
    Returns a single fundamentals row dict for `symbol` (e.g., 'TATAMOTORS.NS').
    Non-throwing: returns {'symbol': symbol, 'error': '...'} if fetch fails.
    """
    row = {"symbol": symbol}

    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}

        row["company"]  = info.get("longName") or info.get("shortName")
        row["sector"]   = info.get("sector")
        row["industry"] = info.get("industry")
        row["mcap"]     = info.get("marketCap")
        row["pe_ttm"]   = info.get("trailingPE")
        row["pe_fwd"]   = info.get("forwardPE")
        row["pb"]       = info.get("priceToBook")
        row["peg"]      = info.get("pegRatio")
        dy = info.get("dividendYield")
        row["div_yield_pct"] = (dy * 100) if dy is not None else None
        row["rev_per_share"] = info.get("revenuePerShare")
        row["eps_ttm"]       = info.get("trailingEps")
        row["summary"]       = info.get("longBusinessSummary")

        # Operating margin, ROE best-effort from info
        om = info.get("operatingMargins")
        row["om_pct"]  = round(om * 100, 2) if om is not None else None
        roe = info.get("returnOnEquity")
        row["roe_pct"] = round(roe * 100, 2) if roe is not None else None
        row["de_ratio"] = info.get("debtToEquity")

        # Profit margin from financials if available
        try:
            fin = tk.financials
            if fin is not None and not fin.empty:
                if "Net Income" in fin.index and "Total Revenue" in fin.index:
                    ni  = float(fin.loc["Net Income"].iloc[0])
                    rev = float(fin.loc["Total Revenue"].iloc[0]) or None
                    row["pm_pct"] = round((ni / rev) * 100, 2) if rev else None
        except Exception:
            pass

        # Optional deltas + fund score if you defined these helpers in this file
        try:
            deltas = quarterly_deltas(tk)   # your helper from earlier step
            row.update(deltas or {})
        except Exception:
            pass

        try:
            row["fund_score"] = fundamental_score_row(row)
        except Exception:
            row["fund_score"] = None

        return row

    except Exception as e:
        row["error"] = str(e)
        return row


def upsert_fundamentals_csv(csv_path: str, row: dict) -> None:
    """Insert or update a single row in fundamentals.csv by symbol."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.DataFrame()

    if "symbol" not in df.columns:
        df = pd.DataFrame(columns=list(row.keys()))

    if (df["symbol"] == row["symbol"]).any():
        for k, v in row.items():
            df.loc[df["symbol"] == row["symbol"], k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(csv_path, index=False)

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