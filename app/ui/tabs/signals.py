# app/ui/tabs/signals.py
import pandas as pd
import streamlit as st

def render(df: pd.DataFrame):
    st.subheader("📋 Signals")
    if df.empty:
        st.info("No signals yet. Go to Add/Manage and press Rebuild Signals.")
        return

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        q = st.text_input("Search", "")
    with c2:
        sig = st.selectbox("Filter Signal", ["All", "BUY", "ACCUMULATE", "WATCH", "SELL"])
    with c3:
        sort_by = st.selectbox("Sort by", ["name","signal","rsi","vol_spike","dist_to_stop_pct","last"])

    view = df.copy()
    if q:
        view = view[
            view["name"].str.contains(q, case=False, na=False)
            | view["symbol"].str.contains(q, case=False, na=False)
        ]
    if sig != "All":
        view = view[view["signal"] == sig]
    view = view.sort_values(sort_by, ascending=True, na_position="last").reset_index(drop=True)
    if "symbol" in view.columns:
        view["detail"] = view["symbol"].apply(lambda s: f"[open](?symbol={s})")
    st.dataframe(
        view[["symbol","name","segment","last","signal","reason","rsi","vol_spike","buy_low","buy_high","stop","as_of","detail"]],
        use_container_width=True,
    )