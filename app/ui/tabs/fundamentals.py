# app/ui/tabs/fundamentals.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

METRICS = [
    # key, label, help, fmt, good_if ('high' or 'low'), good_thresh, bad_thresh
    ("pe_ttm",   "P/E (TTM)", "Price / Earnings (trailing 12M). Lower can be cheaper; compare within industry.", "0f", "low", 20, 60),
    ("pb",       "P/B",       "Price / Book. Asset-heavy firms often trade lower; compare peers.",                "0f", "low", 3, 8),
    ("roe_pct",  "ROE %",     "Return on Equity. Efficiency of equity capital.",                                   "0f", "high", 15, 8),
    ("om_pct",   "OPM %",     "Operating Margin %. Core operating profitability.",                                  "0f", "high", 15, 8),
    ("de_ratio", "D/E",       "Debt to Equity. Balance-sheet leverage.",                                           "2f", "low", 0.5, 1.5),
    ("rev_qoq",  "Rev QoQ %", "Quarter-over-Quarter revenue growth.",                                              "1f", "high", 5, 0),
    ("pat_qoq",  "PAT QoQ %", "Quarter-over-Quarter net profit growth.",                                           "1f", "high", 5, 0),
    ("rev_yoy",  "Rev YoY %", "Year-over-Year revenue growth.",                                                    "1f", "high", 15, 5),
    ("pat_yoy",  "PAT YoY %", "Year-over-Year net profit growth.",                                                 "1f", "high", 15, 5),
    ("mcap_cr",  "MCap (₹ Cr)","Market capitalization in Indian rupees (crores).",                                 "0f", "high", None, None),
]

NEEDED_COLS = [m[0] for m in METRICS] + ["symbol", "name", "sector", "industry", "summary", "mcap", "mcap_cr"]

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NEEDED_COLS:
        if col not in out.columns:
            out[col] = np.nan
    # derive market cap in crores if only raw mcap exists (assume INR if not tagged)
    if out["mcap_cr"].isna().all():
        if "mcap" in out.columns:
            out["mcap_cr"] = pd.to_numeric(out["mcap"], errors="coerce") / 1e7  # ₹ to ₹ Cr
    return out

def _fmt(series: pd.Series, fmt: str) -> pd.Series:
    ser = pd.to_numeric(series, errors="coerce")
    if fmt == "0f":
        return ser.map(lambda x: "" if pd.isna(x) else f"{x:.0f}")
    if fmt == "1f":
        return ser.map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    if fmt == "2f":
        return ser.map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    return ser.astype(str)

def _build_table(fdf: pd.DataFrame) -> pd.DataFrame:
    view_cols = ["symbol", "name"] + [m[0] for m in METRICS]
    df = fdf[view_cols].copy()

    # compute mcap_cr if still missing
    if df["mcap_cr"].isna().all() and "mcap" in fdf.columns:
        df["mcap_cr"] = pd.to_numeric(fdf["mcap"], errors="coerce") / 1e7

    # formatting
    for key, _, _, fmt, *_ in METRICS:
        if key in df.columns:
            df[key] = _fmt(df[key], fmt)

    return df

def _highlighter(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame of CSS strings with same shape as `data`,
    coloring cells green/red using thresholds defined above.
    Non-metric columns get "" (no style).
    """
    styles = pd.DataFrame("", index=data.index, columns=data.columns)
    for key, _, _, fmt, pref, good_thr, bad_thr in METRICS:
        if key not in data.columns:
            continue
        # parse numeric back from formatted strings
        colvals = pd.to_numeric(data[key].replace("", np.nan), errors="coerce")

        if pref == "high":
            # green if >= good_thr, red if <= bad_thr
            if good_thr is not None:
                styles.loc[colvals >= good_thr, key] = "background-color:#E8F5E9"  # light green
            if bad_thr is not None:
                styles.loc[colvals <= bad_thr, key] = "background-color:#FFEBEE"  # light red
        elif pref == "low":
            # green if <= good_thr, red if >= bad_thr
            if good_thr is not None:
                styles.loc[colvals <= good_thr, key] = "background-color:#E8F5E9"
            if bad_thr is not None:
                styles.loc[colvals >= bad_thr, key] = "background-color:#FFEBEE"
    return styles


def render():
    st.subheader("🧾 Fundamentals")

    # pull from session; if not present, explain
    if "fund" not in st.session_state:
        st.info("No fundamentals loaded yet. Use Add/Manage → Fetch Fundamentals.")
        return

    raw = st.session_state["fund"]
    if raw is None or raw.empty:
        st.info("Fundamentals file is empty.")
        return

    fdf = _ensure_columns(raw)

    # ----- Controls (optional filters; nothing enforced) -----
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        search = st.text_input("Search (symbol/company)")
    with c2:
        max_pe = st.number_input("Max P/E (TTM)", min_value=0.0, value=1000.0, step=1.0)
    with c3:
        max_pb = st.number_input("Max P/B", min_value=0.0, value=1000.0, step=0.5)
    with c4:
        min_roe = st.number_input("Min ROE (%)", min_value=0.0, value=0.0, step=1.0)

    view = fdf.copy()

    # apply optional filters only if user wants — keep broad by default
    pe_num = pd.to_numeric(view["pe_ttm"], errors="coerce")
    pb_num = pd.to_numeric(view["pb"], errors="coerce")
    roe_num = pd.to_numeric(view["roe_pct"], errors="coerce")

    view = view[(pe_num.isna()) | (pe_num <= max_pe)]
    view = view[(pb_num.isna()) | (pb_num <= max_pb)]
    view = view[(roe_num.isna()) | (roe_num >= min_roe)]

    if search:
        s = search.strip().lower()
        view = view[
            view["symbol"].str.lower().str.contains(s, na=False) |
            view["name"].str.lower().str.contains(s, na=False)
        ]

    # build nicely formatted table for display
    table = _build_table(view)

    # info tooltips (simple, above the grid)
    with st.expander("ℹ️ What the metrics mean", expanded=False):
        for key, label, help_txt, *_ in METRICS:
            st.markdown(f"**{label}** — {help_txt}")

    # show styled grid (green/red highlights)
    styled = table.style.apply(_highlighter, axis=None)
    st.dataframe(styled, use_container_width=True)

    # show record counts
    st.caption(f"Showing {len(table)} of {len(fdf)} stocks.")