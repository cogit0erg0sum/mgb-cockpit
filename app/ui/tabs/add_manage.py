# app/ui/tabs/add_manage.py
from __future__ import annotations
import pandas as pd
from datetime import datetime
import streamlit as st

from ...config import WL
from ...services.data_loaders import (
    ensure_watchlist, load_csv, get_price, rebuild_signals,
    fetch_fund_one_and_upsert, fetch_fund_bulk
)

def render():
    st.subheader("➕ Add / Manage Watchlist")
    ensure_watchlist()
    wl_df = load_csv(WL)

    # Add one
    st.markdown("**Add a single NSE symbol** (enter without .NS)")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        base = st.text_input("Symbol", placeholder="e.g., CUPID")
    with c2:
        seg = st.selectbox("Segment", ["User", "Microcap", "Smallcap", "Midcap"], index=0)
    with c3:
        add_one = st.button("Add", use_container_width=True)
    with c4:
        rebuild_now = st.button("Rebuild Signals", use_container_width=True)

    if add_one and base.strip():
        base_up = base.strip().upper()
        sym = f"{base_up}.NS"
        price = get_price(sym)
        if price is None:
            st.error("Symbol not found on Yahoo Finance. Check the code.")
        else:
            new_row = {
                "symbol": sym,
                "name": base_up,
                "segment": seg,
                "buy_low": round(price * 0.97, 2),
                "buy_high": round(price * 1.03, 2),
                "stop": round(price * 0.90, 2),
                "notes": f"Added {datetime.now():%Y-%m-%d}",
            }
            if not wl_df.empty and (wl_df["symbol"] == sym).any():
                for k, v in new_row.items():
                    wl_df.loc[wl_df["symbol"] == sym, k] = v
            else:
                wl_df = pd.concat([wl_df, pd.DataFrame([new_row])], ignore_index=True)
            wl_df.to_csv(WL, index=False)

            st.success(f"Added/updated {sym}. Rebuilding signals…")
            rebuild_signals()

            # Auto fetch fundamentals for this symbol
            try:
                fetch_fund_one_and_upsert(sym)
            except Exception as e:
                st.warning(f"Could not fetch fundamentals now: {e}")

            st.query_params.update({"symbol": sym})
            st.rerun()

    st.markdown("---")

    # Bulk add
    st.markdown("**Paste multiple NSE symbols** (comma or newline; without .NS)")
    pasted = st.text_area("Symbols", height=120, placeholder="CUPID, REDINGTON, TATAMOTORS")
    if st.button("Validate & Queue"):
        bases = [s.strip().upper() for s in pasted.replace(",", "\n").splitlines() if s.strip()]
        queued = []
        for b in bases:
            sym_i = f"{b}.NS"
            price_i = get_price(sym_i)
            queued.append({
                "symbol": sym_i, "name": b,
                "status": "✅" if price_i is not None else "❌ not found",
                "buy_low": round(price_i * 0.97, 2) if price_i else None,
                "buy_high": round(price_i * 1.03, 2) if price_i else None,
                "stop": round(price_i * 0.90, 2) if price_i else None,
            })
        qdf = pd.DataFrame(queued)
        st.dataframe(qdf, use_container_width=True)
        st.session_state["queued_rows"] = [r for r in queued if r["status"] == "✅"]

    if st.button("Add queued to Watchlist"):
        rows = st.session_state.get("queued_rows", [])
        if not rows:
            st.warning("Nothing queued.")
        else:
            add_df = pd.DataFrame(rows)
            add_df["segment"] = "User"
            add_df["notes"] = f"Bulk add {datetime.now():%Y-%m-%d}"
            base_df = wl_df if not wl_df.empty else pd.DataFrame(columns=add_df.columns)
            merged = pd.concat([base_df, add_df], ignore_index=True).drop_duplicates(subset=["symbol"], keep="last")
            merged.to_csv(WL, index=False)
            st.success(f"Added {len(add_df)} symbols. Rebuilding signals…")
            rebuild_signals()

            try:
                fetch_fund_bulk([r["symbol"] for r in rows], delay_sec=0.35)
            except Exception as e:
                st.warning(f"Fundamentals bulk fetch had issues: {e}")

            st.rerun()

    if rebuild_now:
        rebuild_signals()

    st.markdown("---")
    st.caption("Current watchlist")
    if wl_df.empty:
        st.info("Watchlist is empty. Add some symbols.")
    else:
        st.dataframe(wl_df, use_container_width=True)