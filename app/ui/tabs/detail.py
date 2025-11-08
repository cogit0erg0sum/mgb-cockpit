# app/ui/tabs/detail.py
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

def render(df: pd.DataFrame, fdf: pd.DataFrame, get_history_cached):
    st.subheader("📖 Detail / Decision Story")

    # Query params (new API)
    qp = st.query_params
    qp_symbol = qp.get("symbol")
    if isinstance(qp_symbol, list):
        qp_symbol = qp_symbol[0] if qp_symbol else None

    if df.empty:
        st.info("No signals yet. Rebuild from Add/Manage.")
        return

    symbols = df["symbol"].tolist()
    default_index = symbols.index(qp_symbol) if qp_symbol in symbols else 0
    sym = st.selectbox("Symbol", symbols, index=default_index)

    row = df[df["symbol"] == sym].iloc[0]

    # Intro / snapshot
    summary = None
    sector = industry = None
    if not fdf.empty and (fdf["symbol"] == sym).any():
        fr = fdf[fdf["symbol"] == sym].iloc[0]
        summary  = fr.get("summary")
        sector   = fr.get("sector")
        industry = fr.get("industry")

    st.markdown("### 🪪 Snapshot")
    c0, c1, c2 = st.columns([2, 1, 1])
    with c0:
        st.markdown(f"**{row.get('name', sym)}**  \n"
                    f"{'· ' + sector if isinstance(sector, str) and sector else ''}"
                    f"{' · ' + industry if isinstance(industry, str) and industry else ''}")
        if isinstance(summary, str) and summary.strip():
            brief = summary.strip()
            if len(brief) > 550:
                brief = brief[:550].rstrip() + "…"
            st.write(brief)
        else:
            st.write("No company description available yet.")
    with c1:
        st.metric("Signal", row.get("signal", "—"))
        rsi_v = row.get("rsi", np.nan)
        st.metric("RSI(14)", f"{rsi_v:.1f}" if pd.notna(rsi_v) else "—")
    with c2:
        vs = row.get("vol_spike", np.nan)
        st.metric("Vol spike (x)", f"{vs:.2f}" if pd.notna(vs) else "—")
        dstop = row.get("dist_to_stop_pct", np.nan)
        st.metric("Dist to Stop (%)", f"{dstop:.2f}" if pd.notna(dstop) else "—")

    st.divider()

    # Action levels
    st.markdown("### 🎯 Action levels")
    cA, cB, cC = st.columns(3)

    def fmt_money(x):
        try:
            return f"₹{float(x):,.2f}"
        except Exception:
            return "—"

    bz_low  = row.get("buy_low")
    bz_high = row.get("buy_high")
    stop    = row.get("stop")
    lo52    = row.get("52w_low")
    hi52    = row.get("52w_high")

    cA.metric("Buy zone", f"{fmt_money(bz_low)}–{fmt_money(bz_high)}" if pd.notna(bz_low) and pd.notna(bz_high) else "—")
    cB.metric("Stop", fmt_money(stop) if pd.notna(stop) else "—")
    cC.metric("52W range", f"{fmt_money(lo52)}–{fmt_money(hi52)}" if pd.notna(lo52) and pd.notna(hi52) else "—")

    # Fundamentals key
    if not fdf.empty and (fdf["symbol"] == sym).any():
        fr = fdf[fdf["symbol"] == sym].iloc[0]
        st.markdown("### 📊 Fundamentals (key)")

        def n(v):
            try: return float(v)
            except: return np.nan

        g1, g2, g3, g4, g5 = st.columns(5)
        g1.metric("P/E", f"{n(fr.get('pe_ttm')):.0f}" if pd.notna(n(fr.get('pe_ttm'))) else "—")
        g2.metric("P/B", f"{n(fr.get('pb')):.0f}" if pd.notna(n(fr.get('pb'))) else "—")
        g3.metric("ROE %", f"{n(fr.get('roe_pct')):.0f}" if pd.notna(n(fr.get('roe_pct'))) else "—")
        g4.metric("OPM %", f"{n(fr.get('om_pct')):.0f}" if pd.notna(n(fr.get('om_pct'))) else "—")
        g5.metric("D/E", f"{n(fr.get('de_ratio')):.2f}" if pd.notna(n(fr.get('de_ratio'))) else "—")

        r1, r2, r3, r4 = st.columns(4)
        rq = fr.get("rev_qoq"); pq = fr.get("pat_qoq"); ry = fr.get("rev_yoy"); py = fr.get("pat_yoy")
        r1.metric("Revenue QoQ", f"{n(rq):.1f}%" if pd.notna(n(rq)) else "—")
        r2.metric("PAT QoQ", f"{n(pq):.1f}%" if pd.notna(n(pq)) else "—")
        r3.metric("Revenue YoY", f"{n(ry):.1f}%" if pd.notna(n(ry)) else "—")
        r4.metric("PAT YoY", f"{n(py):.1f}%" if pd.notna(n(py)) else "—")
    else:
        st.info("No fundamentals found for this stock yet. Add it and refresh fundamentals.")

    # Conclusion
    st.markdown("### 🧠 Conclusion")
    notes = []
    sig = str(row.get("signal", "—"))
    if sig in ("BUY", "ACCUMULATE"):
        notes.append("Tape looks constructive; consider entries near the buy zone with the defined stop.")
    elif sig == "WATCH":
        notes.append("Setup forming; wait for a strong up-day with volume before committing.")
    elif sig == "SELL":
        notes.append("Weak tape; avoid fresh buys until strength returns.")
    else:
        notes.append("Insufficient signal; treat as neutral watchlist.")

    if not fdf.empty and (fdf["symbol"] == sym).any():
        fr = fdf[fdf["symbol"] == sym].iloc[0]
        def nz(x):
            try: return float(x)
            except: return np.nan
        if pd.notna(fr.get("roe_pct")) and nz(fr.get("roe_pct")) >= 15:
            notes.append("ROE looks healthy.")
        if pd.notna(fr.get("om_pct")) and nz(fr.get("om_pct")) >= 15:
            notes.append("Operating margins are solid.")
        if pd.notna(fr.get("de_ratio")) and nz(fr.get("de_ratio")) <= 0.5:
            notes.append("Low leverage profile.")
        if pd.notna(fr.get("peg")) and nz(fr.get("peg")) <= 1:
            notes.append("PEG at or below 1; growth justified.")
        if pd.notna(fr.get("rev_yoy")) and pd.notna(fr.get("pat_yoy")):
            if nz(fr.get("rev_yoy")) > 15 and nz(fr.get("pat_yoy")) > 15:
                notes.append("Both Revenue and PAT show strong YoY expansion.")

    st.write("• " + "\n• ".join(notes))

    st.divider()

    # Chart last
    if st.toggle("Show chart (SMA20/50)", value=False):
        hist = get_history_cached(sym, period_days=300, ttl_sec=6 * 3600)
        if not hist.empty:
            import matplotlib.pyplot as plt
            hist["SMA20"] = hist["Close"].rolling(20).mean()
            hist["SMA50"] = hist["Close"].rolling(50).mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hist.index, hist["Close"], label="Close")
            ax.plot(hist.index, hist["SMA20"], label="SMA20")
            ax.plot(hist.index, hist["SMA50"], label="SMA50")
            ax.legend(); ax.set_title(sym)
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("No chart data")