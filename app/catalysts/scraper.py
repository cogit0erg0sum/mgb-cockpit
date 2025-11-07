# app/catalysts/scraper.py
import pandas as pd
from datetime import datetime, timedelta

try:
    import yfinance as yf
except Exception:
    yf = None

def _tag(title: str) -> str:
    t = (title or "").lower()
    if any(k in t for k in ["order", "contract", "tender", "loa", "mou", "moa"]):
        return "Orders/Tenders"
    if any(k in t for k in ["result", "quarter", "q1", "q2", "q3", "q4", "earnings"]):
        return "Results"
    if any(k in t for k in ["capex", "plant", "capacity", "expansion", "commissioning"]):
        return "Capacity/Capex"
    if any(k in t for k in ["approval", "license", "clearance", "patent", "regulator"]):
        return "Regulatory/Approval"
    if any(k in t for k in ["promoter", "pledge", "buyback", "dividend", "bonus"]):
        return "Promoter/Capital"
    return "General"

def build_catalysts(symbols: list[str], days: int = 120) -> pd.DataFrame:
    """Return recent headlines for given Yahoo symbols (e.g., RELIANCE.NS) with tags."""
    if not yf:
        return pd.DataFrame(columns=["symbol","when","category","title","link"])
    rows = []
    cutoff = datetime.utcnow() - timedelta(days=days)
    for sym in symbols:
        try:
            tk = yf.Ticker(sym)
            news = tk.news or []
            for n in news:
                ts = n.get("providerPublishTime")
                when = datetime.utcfromtimestamp(ts) if isinstance(ts, (int, float)) else None
                if not when or when < cutoff:
                    continue
                title = n.get("title") or ""
                rows.append({
                    "symbol": sym,
                    "when": when,
                    "category": _tag(title),
                    "title": title,
                    "link": n.get("link") or "",
                })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["symbol","when","category","title","link"])
    df = df.sort_values("when", ascending=False).reset_index(drop=True)
    return df