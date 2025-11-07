# engine/screener_core.py
import os, math
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def sma(s, n):
    return s.rolling(n).mean()

def _compute_signals(df: pd.DataFrame, buy_low: float, buy_high: float, stop: float,
                     rsi_len=14, roll_max=55, vol_spike_thr=1.6) -> dict:
    c = df["Close"]; v = df["Volume"]
    last = float(c.iloc[-1])
    s20 = float(sma(c, 20).iloc[-1])
    s50 = float(sma(c, 50).iloc[-1])
    rsi_ = float(rsi(c, rsi_len).iloc[-1])
    max55 = float(c.rolling(roll_max).max().iloc[-1])
    min252 = float(c.rolling(252).min().iloc[-1])
    max252 = float(c.rolling(252).max().iloc[-1])
    avgv20 = float(v.rolling(20).mean().iloc[-1]) if v.notna().any() else np.nan
    vol_spike = float(v.iloc[-1] / avgv20) if avgv20 and avgv20 > 0 else np.nan

    in_buy_zone = (last >= buy_low) and (last <= buy_high)
    breakout = (last > max55) and (not math.isnan(vol_spike) and vol_spike >= vol_spike_thr)
    trend_ok = (last > s20 > s50)
    near_stop = (last <= stop * 1.03)

    signal = "HOLD"; reason = []
    if breakout and trend_ok:
        signal = "BUY"; reason.append("Breakout > 55D high with volume and MA alignment")
    elif in_buy_zone and trend_ok:
        signal = "ACCUMULATE"; reason.append("Within buy zone and trend ok")
    elif last < stop:
        signal = "SELL"; reason.append("Stop loss breached")
    elif near_stop:
        signal = "WATCH"; reason.append("Close to stop, reduce risk")

    return dict(
        last=last, sma20=s20, sma50=s50, rsi=rsi_,
        **{"52w_low": min252, "52w_high": max252},
        vol_spike=None if math.isnan(vol_spike) else vol_spike,
        breakout_55d=bool(breakout),
        in_buy_zone=bool(in_buy_zone),
        trend_ok=bool(trend_ok),
        near_stop=bool(near_stop),
        signal=signal,
        reason="; ".join(reason) if reason else "Neutral",
        dist_to_buy_low_pct=((buy_low - last) / last) * 100,
        dist_to_buy_high_pct=((buy_high - last) / last) * 100,
        dist_to_stop_pct=((last - stop) / last) * 100
    )

def compute_dashboard(watchlist_path: str, out_csv: str, days: int = 420) -> int:
    """Build dashboard.csv from a watchlist; returns number of rows written."""
    if not os.path.exists(watchlist_path):
        raise FileNotFoundError(f"Missing watchlist file: {watchlist_path}")
    wl = pd.read_csv(watchlist_path)
    if wl.empty:
        raise ValueError("watchlist.csv is empty")

    syms = wl["symbol"].dropna().unique().tolist()
    data = yf.download(syms, period=f"{days}d", auto_adjust=False, progress=False, group_by="ticker")

    rows = []
    for _, r in wl.iterrows():
        sym = r["symbol"]
        try:
            df = data[sym].dropna().copy()
        except KeyError:
            # Try fallback single fetch if grouped failed
            one = yf.download(sym, period=f"{days}d", auto_adjust=False, progress=False)
            if one is None or one.empty:
                print(f"No data for {sym}")
                continue
            df = one.dropna().copy()

        sig = _compute_signals(
            df,
            float(r["buy_low"]),
            float(r["buy_high"]),
            float(r["stop"])
        )
        rows.append({
            "symbol": sym, "name": r.get("name", sym), "segment": r.get("segment", ""),
            **sig,
            "buy_low": float(r["buy_low"]), "buy_high": float(r["buy_high"]), "stop": float(r["stop"]),
            "notes": r.get("notes", ""), "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return len(out)