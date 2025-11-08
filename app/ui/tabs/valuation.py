# app/ui/tabs/valuation.py
import pandas as pd
import streamlit as st

def render(ddf: pd.DataFrame, fdf: pd.DataFrame):
    st.subheader("💰 Valuation Sandbox")

    if ddf.empty:
        st.info("Build Signals first. Then fetch Fundamentals."); return
    if fdf.empty:
        st.info("Fetch Fundamentals first."); return

    common = sorted(set(ddf["symbol"].tolist()) & set(fdf["symbol"].tolist()))
    if not common:
        st.info("No overlap between signals and fundamentals yet."); return

    sym = st.selectbox("Symbol", common)
    last = float(ddf.loc[ddf["symbol"] == sym, "last"].iloc[0])
    eps  = fdf.loc[fdf["symbol"] == sym, "eps_ttm"].fillna(0).iloc[0]
    revp = fdf.loc[fdf["symbol"] == sym, "rev_per_share"].fillna(0).iloc[0]
    pm   = fdf.loc[fdf["symbol"] == sym, "pm_pct"].fillna(0).iloc[0]

    st.markdown("**Inputs**")
    c1, c2, c3, c4 = st.columns(4)
    with c1: target_pe = st.number_input("Target P/E", 1.0, value=25.0, step=1.0)
    with c2: target_ps = st.number_input("Target P/S", 0.1, value=3.0, step=0.1)
    with c3: margin_pct = st.number_input("Sustainable Net Margin (%)", 0.0, value=float(pm or 10.0), step=0.5)
    with c4: weight_pe  = st.slider("Weight to P/E model (%)", 0, 100, 70, 5)

    pe_price = target_pe * eps if eps and eps > 0 else None
    ps_price = target_ps * revp if revp and revp > 0 else None
    if ps_price is None:
        cur_pe = pd.to_numeric(fdf.loc[fdf["symbol"] == sym, "pe_ttm"], errors="coerce").fillna(0).iloc[0]
        eps_guess = last / cur_pe if cur_pe and cur_pe > 0 else None
        if eps_guess: ps_price = target_pe * eps_guess

    implied_list = [p for p in [pe_price, ps_price] if p is not None and p > 0]
    w_pe = weight_pe / 100.0

    if implied_list:
        implied = (pe_price or 0) * w_pe + (ps_price or 0) * (1 - w_pe) if (pe_price and ps_price) else implied_list[0]
        upside = (implied - last) / last * 100.0
        cA, cB, cC = st.columns(3)
        cA.metric("Last Price", f"₹{last:,.2f}")
        cB.metric("Implied Price", f"₹{implied:,.2f}")
        cC.metric("Upside", f"{upside:,.1f}%")
    else:
        st.info("Not enough data to compute implied price. Try adjusting inputs.")

    st.caption("Note: Simple multiples-based sandbox. Use as a scenario tool, not advice.")