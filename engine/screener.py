import os, math
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
from screener_core import compute_dashboard

# Project paths
ROOT = os.path.dirname(os.path.dirname(__file__))
WATCHLIST = os.path.join(ROOT, "app", "data", "watchlist.csv")
OUT_CSV   = os.path.join(ROOT, "app", "data", "dashboard.csv")

# Params
DAYS = 420         # ~18 months
RSI_LEN = 14
ROLL_MAX = 55      # breakout window
VOL_SPIKE = 1.6    # x avg volume vs 20d average

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def sma(s, n):
    return s.rolling(n).mean()

def compute_signals(df: pd.DataFrame, buy_low: float, buy_high: float, stop: float) -> dict:
    c = df["Close"]; v = df["Volume"]
    last = float(c.iloc[-1])
    s20 = float(sma(c, 20).iloc[-1])
    s50 = float(sma(c, 50).iloc[-1])
    rsi_ = float(rsi(c, RSI_LEN).iloc[-1])
    max55 = float(c.rolling(ROLL_MAX).max().iloc[-1])
    min252 = float(c.rolling(252).min().iloc[-1])
    max252 = float(c.rolling(252).max().iloc[-1])
    avgv20 = float(v.rolling(20).mean().iloc[-1]) if v.notna().any() else np.nan
    vol_spike = float(v.iloc[-1] / avgv20) if avgv20 and avgv20 > 0 else np.nan

    in_buy_zone = (last >= buy_low) and (last <= buy_high)
    breakout = (last > max55) and (not math.isnan(vol_spike) and vol_spike >= VOL_SPIKE)
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

def main():
    n = compute_dashboard(WATCHLIST, OUT_CSV, DAYS = 420)
    print(F"Wrote {OUT_CSV} with {n} rows")
    if not os.path.exists(WATCHLIST):
        raise SystemExit(f"Missing watchlist file: {WATCHLIST}")
    wl = pd.read_csv(WATCHLIST)
    if wl.empty:
        raise SystemExit("watchlist.csv is empty")

    syms = wl["symbol"].tolist()
    data = yf.download(syms, period=f"{DAYS}d", auto_adjust=False, progress=False, group_by="ticker")

    rows = []
    for _, r in wl.iterrows():
        sym = r["symbol"]
        try:
            df = data[sym].dropna().copy()
        except KeyError:
            print(f"No data for {sym}")
            continue

        sig = compute_signals(df, float(r["buy_low"]), float(r["buy_high"]), float(r["stop"]))
        rows.append({
            "symbol": sym, "name": r["name"], "segment": r["segment"],
            **sig,
            "buy_low": float(r["buy_low"]), "buy_high": float(r["buy_high"]), "stop": float(r["stop"]),
            "notes": r.get("notes", ""), "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV} with {len(out)} rows")

if __name__ == "__main__":
    main()