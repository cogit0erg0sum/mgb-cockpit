# app/ui/components/banner.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import pytz
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[2] / "app" / "data"
DASH = DATA_DIR / "dashboard.csv"
FUND = DATA_DIR / "fundamentals.csv"
META = DATA_DIR / "metadata.json"

def _load_meta():
    if META.exists():
        try:
            return json.loads(META.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _fmt_ist(ts_utc_str: str | None) -> str:
    if not ts_utc_str:
        return "—"
    try:
        # expect ISO UTC string
        dt = datetime.fromisoformat(ts_utc_str.replace("Z", "+00:00"))
        ist = pytz.timezone("Asia/Kolkata")
        return dt.astimezone(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        return "—"

def render_banner():
    meta = _load_meta()
    last_updated_ist = _fmt_ist(meta.get("last_updated_utc"))

    # counts
    try:
        sig_cnt = len(pd.read_csv(DASH))
    except Exception:
        sig_cnt = 0
    try:
        fund_cnt = len(pd.read_csv(FUND))
    except Exception:
        fund_cnt = 0

    st.markdown(
        f"""
<div class="mb-card" style="display:flex;align-items:center;gap:16px;">
  <div style="font-weight:700;font-size:15px;">🕒 Last updated: {last_updated_ist}</div>
  <div class="badge">Signals: {sig_cnt}</div>
  <div class="badge">Fundamentals: {fund_cnt}</div>
</div>
""",
        unsafe_allow_html=True,
    )