# app/services/data_loaders.py
from __future__ import annotations
import json, time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

from ..config import DATA, DASH, WL, FUND, META

try:
    from screener_core import compute_dashboard
except Exception:
    compute_dashboard = None

try:
    from fundamentals_core import (
        compute_fundamentals,
        fetch_fundamentals_one,
        upsert_fundamentals_csv,
    )
except Exception:
    compute_fundamentals = None
    fetch_fundamentals_one = None
    upsert_fundamentals_csv = None


def ensure_watchlist():
    DATA.mkdir(parents=True, exist_ok=True)
    if not WL.exists():
        pd.DataFrame(
            columns=["symbol","name","segment","buy_low","buy_high","stop","notes"]
        ).to_csv(WL, index=False)

def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def load_dash() -> pd.DataFrame:
    return load_csv(DASH)

def load_fund() -> pd.DataFrame:
    """Load fundamentals.csv from app/data. Always return a DataFrame."""
    path = DATA / "fundamentals.csv"
    if not path.exists():
        # file not found -> empty DF (so UI can show a helpful message)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # ensure at least the symbol column exists so tabs won’t break
        if "symbol" not in df.columns:
            df["symbol"] = ""
        return df
    except Exception:
        # on any read error, fail gracefully
        return pd.DataFrame()

def load_meta() -> dict:
    try:
        return json.loads(META.read_text()) if META.exists() else {}
    except Exception:
        return {}

def get_price(sym_full: str) -> Optional[float]:
    try:
        hist = yf.Ticker(sym_full).history(period="5d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None

def rebuild_signals():
    if compute_dashboard is None:
        st.warning("Screener core not available in this environment.")
        return
    with st.spinner("Building signals…"):
        n = compute_dashboard(str(WL), str(DASH), days=420)
    st.success(f"Updated signals for {n} stocks.")
    st.session_state["dash"] = load_dash()

def fetch_fund_one_and_upsert(sym_full: str):
    if not (fetch_fundamentals_one and upsert_fundamentals_csv):
        st.info("Fundamentals module not available here.")
        return
    row = fetch_fundamentals_one(sym_full)
    upsert_fundamentals_csv(str(FUND), row)
    st.session_state["fund"] = load_fund()

def fetch_fund_bulk(symbols: list[str], delay_sec: float = 0.35):
    if not (fetch_fundamentals_one and upsert_fundamentals_csv):
        st.info("Fundamentals module not available here.")
        return
    ok = 0
    with st.spinner("Fetching fundamentals for added symbols…"):
        for s in symbols:
            try:
                row = fetch_fundamentals_one(s)
                upsert_fundamentals_csv(str(FUND), row)
                ok += 1
                time.sleep(delay_sec)
            except Exception:
                pass
    st.session_state["fund"] = load_fund()
    st.toast(f"Fundamentals updated for {ok} symbols.", icon="✅")

def to_cr(x):
    try:
        return x / 1e7
    except Exception:
        return np.nan