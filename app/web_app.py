# app/web_app.py
import os, sys, pathlib
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime

# -------- Paths & imports --------
ROOT = pathlib.Path(__file__).resolve().parents[1]   # .../mgb-cockpit
APP_DIR = ROOT / "app"
ENGINE_DIR = ROOT / "engine"
for p in (APP_DIR, ENGINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from indicators import rsi
from screener_core import compute_dashboard
from fundamentals_core import compute_fundamentals

DATA = APP_DIR / "data"
DASH = DATA / "dashboard.csv"
WL   = DATA / "watchlist.csv"
FUND = DATA / "fundamentals.csv"

# -------- App setup --------
st.set_page_config(page_title="Multibagger Cockpit (Simple)", layout="wide")
st.title("🟢 Multibagger Cockpit — Simple")

# -------- Helpers --------
def _get_price(sym_full: str) -> float | None:
    try:
        hist = yf.Ticker(sym_full).history(period="5d")
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception:
        return None

def _load_dash() -> pd.DataFrame:
    if not DASH.exists():
        return pd.DataFrame()
    return pd.read_csv(DASH)

def _load_fund() -> pd.DataFrame:
    if not FUND.exists():
        return pd.DataFrame()
    return pd.read_csv(FUND)

def _ensure_watchlist():
    if not WL.exists():
        pd.DataFrame(
            columns=["symbol","name","segment","buy_low","buy_high","stop","notes"]
        ).to_csv(WL, index=False)

def _rebuild():
    try:
        with st.spinner("Building signals…"):
            n = compute_dashboard(str(WL), str(DASH), days=420)
        st.success(f"Updated signals for {n} stocks.")
    except Exception as e:
        st.error(f"Rebuild failed: {e}")
    finally:
        st.session_state["dash"] = _load_dash()

def _to_cr(x):
    try:
        return x / 1e7  # ₹ to Cr
    except Exception:
        return None

# Keep dashboard cached in session
if "dash" not in st.session_state:
    st.session_state["dash"] = _load_dash()

# -------- Tabs --------
tabs = st.tabs(["➕ Add/Manage", "📋 Signals", "📈 Detail", "🧾 Fundamentals", "💰 Valuation", "ℹ️ Help"])

# ========== Tab 1: Add / Manage ==========
with tabs[0]:
    st.subheader("➕ Add / Manage Watchlist")
    _ensure_watchlist()
    wl_df = pd.read_csv(WL)

    # A) Add ONE symbol quickly
    st.markdown("**Add a single NSE symbol** (enter without .NS)")
    c1, c2, c3, c4 = st.columns([2,1,1,1])
    with c1:
        base = st.text_input("Symbol", placeholder="e.g., CUPID")
    with c2:
        seg = st.selectbox("Segment", ["User","Microcap","Smallcap","Midcap"], index=0)
    with c3:
        add_one = st.button("Add")
    with c4:
        rebuild_now = st.button("Rebuild Signals")

    if add_one and base.strip():
        base_up = base.strip().upper()
        sym = f"{base_up}.NS"
        price = _get_price(sym)
        if price is None:
            st.error("Symbol not found on Yahoo Finance. Check the code.")
        else:
            row = {
                "symbol": sym,
                "name": base_up,
                "segment": seg,
                "buy_low": round(price*0.97, 2),
                "buy_high": round(price*1.03, 2),
                "stop": round(price*0.90, 2),
                "notes": f"Added {datetime.now():%Y-%m-%d}"
            }
            if (wl_df["symbol"] == sym).any():
                wl_df.loc[wl_df["symbol"] == sym, row.keys()] = list(row.values())
            else:
                wl_df = pd.concat([wl_df, pd.DataFrame([row])], ignore_index=True)
            wl_df.to_csv(WL, index=False)
            st.success(f"Added/updated {sym}. Rebuilding signals…")
            _rebuild()
            st.rerun()

    # B) Paste MANY
    st.markdown("---")
    st.markdown("**Paste multiple NSE symbols** (comma or newline; without .NS)")
    pasted = st.text_area("Symbols", height=100, placeholder="CUPID, RELIANCE, HDFCBANK")
    if st.button("Validate & Queue"):
        bases = [s.strip().upper() for s in pasted.replace(",", "\n").splitlines() if s.strip()]
        queued = []
        with st.spinner("Validating…"):
            for b in bases:
                sym = f"{b}.NS"
                price = _get_price(sym)
                if price is None:
                    queued.append({"symbol": sym, "name": b, "status": "❌ not found"})
                else:
                    queued.append({
                        "symbol": sym, "name": b, "status": "✅",
                        "buy_low": round(price*0.97,2),
                        "buy_high": round(price*1.03,2),
                        "stop": round(price*0.90,2),
                    })
        qdf = pd.DataFrame(queued)
        st.dataframe(qdf, use_container_width=True)
        st.session_state["queued_rows"] = [r for r in queued if r.get("status") == "✅"]

    if st.button("Add queued to Watchlist"):
        rows = st.session_state.get("queued_rows", [])
        if not rows:
            st.warning("Nothing queued.")
        else:
            add_df = pd.DataFrame(rows)
            add_df["segment"] = "User"
            add_df["notes"] = f"Bulk add {datetime.now():%Y-%m-%d}"
            base = wl_df if not wl_df.empty else pd.DataFrame(columns=add_df.columns)
            merged = pd.concat([base, add_df], ignore_index=True)
            merged = merged.drop_duplicates(subset=["symbol"], keep="last")
            merged.to_csv(WL, index=False)
            st.success(f"Added {len(add_df)} symbols. Rebuilding signals…")
            _rebuild()
            st.rerun()

    # C) CSV upload (optional)
    st.markdown("---")
    st.markdown("**Upload CSV** with columns: symbol,name,segment,buy_low,buy_high,stop,notes")
    up = st.file_uploader("Upload watchlist rows", type=["csv"])
    if up is not None:
        try:
            up_df = pd.read_csv(up)
            up_df["symbol"] = up_df["symbol"].astype(str).str.strip().str.upper().apply(
                lambda s: s if s.endswith(".NS") else f"{s}.NS"
            )
            base = wl_df if not wl_df.empty else pd.DataFrame(columns=up_df.columns)
            merged = pd.concat([base, up_df], ignore_index=True).drop_duplicates(subset=["symbol"], keep="last")
            merged.to_csv(WL, index=False)
            st.success(f"Merged {len(up_df)} rows into watchlist. Rebuilding signals…")
            _rebuild()
            st.rerun()
        except Exception as e:
            st.error(f"CSV parse failed: {e}")

    # Manual rebuild button
    if rebuild_now:
        _rebuild()

    st.markdown("---")
    st.caption("Current watchlist")
    st.dataframe(wl_df, use_container_width=True)

# ========== Tab 2: Signals ==========
with tabs[1]:
    st.subheader("📋 Signals")
    df = st.session_state["dash"]
    if df.empty:
        st.info("No signals yet. Go to Add/Manage and press Rebuild Signals.")
    else:
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            q = st.text_input("Search", "")
        with c2:
            sig = st.selectbox("Filter Signal", ["All","BUY","ACCUMULATE","WATCH","SELL"])
        with c3:
            sort_by = st.selectbox("Sort by", ["name","signal","rsi","vol_spike","dist_to_stop_pct"])

        view = df.copy()
        if q:
            view = view[view["name"].str.contains(q, case=False) | view["symbol"].str.contains(q, case=False)]
        if sig != "All":
            view = view[view["signal"] == sig]
        view = view.sort_values(sort_by, ascending=True, na_position="last").reset_index(drop=True)

        st.dataframe(
            view[["symbol","name","segment","last","signal","reason","rsi","vol_spike","buy_low","buy_high","stop","as_of"]],
            use_container_width=True
        )

# ========== Tab 3: Detail (Decision Card, chart optional) ==========
with tabs[2]:
    st.subheader("📈 Detail / Decision Card")
    df = st.session_state["dash"]
    if df.empty:
        st.info("No signals yet. Rebuild from Add/Manage.")
    else:
        sym = st.selectbox("Symbol", df["symbol"].tolist())
        row = df[df["symbol"] == sym].iloc[0]

        # quick insights
        try:
            hist = yf.download(sym, period="300d", auto_adjust=False, progress=False)
        except Exception:
            hist = pd.DataFrame()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Signal", row["signal"])
        c2.metric("RSI(14)", f'{row["rsi"]:.1f}')
        c3.metric("Vol spike (x)", f'{row["vol_spike"]:.2f}' if pd.notna(row["vol_spike"]) else "—")
        c4.metric("Dist to Stop (%)", f'{row["dist_to_stop_pct"]:.2f}')

        st.markdown("### 🎯 Action levels")
        cA, cB, cC = st.columns(3)
        cA.metric("Buy zone", f"₹{row['buy_low']:.2f}–₹{row['buy_high']:.2f}")
        cB.metric("Stop", f"₹{row['stop']:.2f}")
        cC.metric("52W range", f"₹{row['52w_low']:.2f}–₹{row['52w_high']:.2f}")

        # simple conclusion
        verdict = "HOLD"
        reasons = []
        if row["signal"] in ("BUY","ACCUMULATE"): verdict = "BUY"; reasons.append("Model signal favorable")
        if row.get("rsi", np.nan) > 70: reasons.append("RSI hot; expect pullbacks")
        if row.get("rsi", np.nan) < 40: reasons.append("RSI weak; wait for strength")
        try:
            dstop = float(row["dist_to_stop_pct"])
            if dstop < 4:  reasons.append("Stop very tight; size carefully")
            if dstop > 12: reasons.append("Stop wide; reduce size")
        except Exception:
            pass
        st.success(f"**Conclusion:** {verdict}. " + (" · ".join(reasons) if reasons else "Neutral."))

        # Optional chart (off by default)
        if st.toggle("Show chart", value=False):
            if not hist.empty:
                import matplotlib.pyplot as plt
                hist["SMA20"] = hist["Close"].rolling(20).mean()
                hist["SMA50"] = hist["Close"].rolling(50).mean()
                fig, ax = plt.subplots(figsize=(10,4))
                ax.plot(hist.index, hist["Close"], label="Close")
                ax.plot(hist.index, hist["SMA20"], label="SMA20")
                ax.plot(hist.index, hist["SMA50"], label="SMA50")
                ax.legend(); ax.set_title(sym)
                st.pyplot(fig, use_container_width=True)
            else:
                st.info("No chart data")

# ========== Tab 4: Fundamentals (Show all by default; optional filters; tooltips; formatting; highlights) ==========
with tabs[3]:
    st.subheader("🧾 Fundamentals")

    # Top actions
    left, right = st.columns([1,3])
    with left:
        if st.button("Fetch / Refresh Fundamentals"):
            try:
                with st.spinner("Fetching fundamentals (free Yahoo)…"):
                    n = compute_fundamentals(str(WL), str(FUND), throttle_sec=0.3)
                st.success(f"Saved fundamentals for {n} symbols.")
            except Exception as e:
                st.error(f"Failed: {e}")
            st.session_state["fund"] = _load_fund()
    with right:
        enable_filters = st.toggle("Enable filters", value=False)

    # Load
    if "fund" not in st.session_state:
        st.session_state["fund"] = _load_fund()

    fdf = st.session_state["fund"].copy()
    if fdf.empty:
        st.info("No fundamentals yet. Click **Fetch / Refresh Fundamentals**.")
        st.stop()

    # Ensure required columns exist
    needed_cols = [
        "symbol","company","sector","industry","mcap",
        "pe_ttm","pe_fwd","pb","peg","roe_pct","pm_pct","om_pct",
        "de_ratio","div_yield_pct","summary","error","rev_per_share","eps_ttm"
    ]
    for c in needed_cols:
        if c not in fdf.columns:
            fdf[c] = np.nan

    # Coerce numeric cols once
    num_cols = ["mcap","pe_ttm","pe_fwd","pb","peg","roe_pct","pm_pct","om_pct",
                "de_ratio","div_yield_pct","rev_per_share","eps_ttm"]
    for c in num_cols:
        fdf[c] = pd.to_numeric(fdf[c], errors="coerce")

    # By default show ALL rows (no filters)
    view = fdf.copy()
    total_rows = len(view)

    # Optional filters
    if enable_filters:
        colf1, colf2, colf3, colf4, colf5, colf6 = st.columns([2,1,1,1,1,1])
        with colf1:
            q = st.text_input("Search (symbol/company)", "")
        with colf2:
            pe_max = st.number_input("Max P/E (TTM)", min_value=0.0, value=60.0, step=1.0,
                                     help="Price / Earnings (TTM). Lower can be cheaper, but cyclicals can mislead.")
        with colf3:
            pb_max = st.number_input("Max P/B", min_value=0.0, value=10.0, step=0.5,
                                     help="Price / Book Value. For lenders, compare with ROE.")
        with colf4:
            roe_min = st.number_input("Min ROE (%)", min_value=0.0, value=0.0, step=1.0,
                                      help="Return on Equity. Higher suggests efficient use of capital.")
        with colf5:
            show_only_valid = st.checkbox("Require any of: P/E, P/B, ROE, Mcap", value=False)
        with colf6:
            if st.button("Reset"):
                st.rerun()

        if q:
            mask = view["symbol"].str.contains(q, case=False, na=False) | \
                   view["company"].fillna("").str.contains(q, case=False)
            view = view[mask]

        if show_only_valid:
            key_ok = view["pe_ttm"].notna() | view["pb"].notna() | view["roe_pct"].notna() | view["mcap"].notna()
            view = view[key_ok]

        view = view[view["pe_ttm"].fillna(1e12) <= pe_max]
        view = view[view["pb"].fillna(1e12) <= pb_max]
        view = view[view["roe_pct"].fillna(-1e12) >= roe_min]

    # Row count
    st.caption(f"Showing {len(view)} of {total_rows} symbols")

    # ---- Table render (safe: always defines `display`) ----
    if view.empty:
        st.info("No rows to show. If filters are on, try Reset.")
    else:
        # Helper: ₹ → Cr
        def _to_cr_local(x):
            return x/1e7 if pd.notna(x) else np.nan

        # Base columns
        base_cols = [
            "symbol","company","sector","industry",
            "mcap","pe_ttm","pe_fwd","pb","peg",
            "roe_pct","pm_pct","om_pct","de_ratio","div_yield_pct"
        ]
        for c in base_cols:
            if c not in view.columns:
                view[c] = np.nan

        display = view[base_cols].copy()

        # Formatted columns (0 decimals) + Mcap Cr
        display["Mcap (₹ Cr)"] = display["mcap"].apply(_to_cr_local).round(0)
        display["P/E"]         = display["pe_ttm"].round(0)
        display["Fwd P/E"]     = display["pe_fwd"].round(0)
        display["P/B"]         = display["pb"].round(0)
        display["PEG"]         = display["peg"].round(0)
        display["ROE %"]       = display["roe_pct"].round(0)
        display["Profit %"]    = display["pm_pct"].round(0)
        display["OPM %"]       = display["om_pct"].round(0)
        display["D/E"]         = display["de_ratio"].round(0)
        display["Div %"]       = display["div_yield_pct"].round(0)

        # Drop raw numeric cols now
        display = display.drop(columns=[
            "mcap","pe_ttm","pe_fwd","pb","peg","roe_pct","pm_pct","om_pct","de_ratio","div_yield_pct"
        ])

        colorize = st.toggle("Highlight good/bad cells", value=True, help="Turn on conditional coloring (light green/red).")

        if colorize:
            import pandas as pd

            def color_cell(val, col):
                if pd.isna(val): return ""
                good = bad = False
                if col == "P/E":        good, bad = (val <= 25, val >= 60)
                elif col == "Fwd P/E":  good, bad = (val <= 22, val >= 55)
                elif col == "P/B":      good, bad = (val <= 3,  val >= 8)
                elif col == "PEG":      good, bad = (val <= 1,  val >= 2)
                elif col == "ROE %":    good, bad = (val >= 15, val <= 8)
                elif col == "Profit %": good, bad = (val >= 10, val <= 2)
                elif col == "OPM %":    good, bad = (val >= 15, val <= 8)
                elif col == "D/E":      good, bad = (val <= 0.5,val >= 2)
                elif col == "Div %":    good, bad = (val >= 2,  False)
                if good: return "background-color:#e7f6e7;"
                if bad:  return "background-color:#fde8e8;"
                return ""

            metric_cols = ["Mcap (₹ Cr)","P/E","Fwd P/E","P/B","PEG","ROE %","Profit %","OPM %","D/E","Div %"]

            def style_fn(df_):
                styles = pd.DataFrame("", index=df_.index, columns=df_.columns)
                for c in metric_cols:
                    if c in df_.columns:
                        styles[c] = df_[c].apply(lambda v: color_cell(v, c))
                return styles

            styled = display.style.apply(style_fn, axis=None)
            st.dataframe(styled, use_container_width=True)

            with st.expander("Legend & thresholds"):
                st.markdown("""
- Good: **P/E ≤ 25**, **Fwd P/E ≤ 22**, **P/B ≤ 3**, **PEG ≤ 1**, **ROE ≥ 15%**, **OPM ≥ 15%**, **Profit ≥ 10%**, **D/E ≤ 0.5**, **Div ≥ 2%**  
- Red flags: **P/E ≥ 60**, **P/B ≥ 8**, **ROE ≤ 8%**, **OPM ≤ 8%**, **Profit ≤ 2%**, **D/E ≥ 2**  
_Tweak per sector later if you like (banks vs FMCG behave differently)._
""")
        else:
            st.dataframe(
                display,
                use_container_width=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol", help="NSE code (with .NS suffix)."),
                    "company": st.column_config.TextColumn("Company", help="Registered name as per exchange / Yahoo."),
                    "sector": st.column_config.TextColumn("Sector", help="High-level industry group."),
                    "industry": st.column_config.TextColumn("Industry", help="Specific line of business."),
                    "Mcap (₹ Cr)": st.column_config.NumberColumn("Mcap (₹ Cr)", format="%.0f",
                        help="Market Capitalization in Crores."),
                    "P/E": st.column_config.NumberColumn("P/E", format="%.0f",
                        help="Price / Earnings (TTM). Lower can be cheaper, but cyclicals can mislead."),
                    "Fwd P/E": st.column_config.NumberColumn("Fwd P/E", format="%.0f",
                        help="Price / forward 12m expected Earnings."),
                    "P/B": st.column_config.NumberColumn("P/B", format="%.0f",
                        help="Price / Book Value. For lenders, compare with ROE."),
                    "PEG": st.column_config.NumberColumn("PEG", format="%.0f",
                        help="P/E divided by earnings growth. ~1 can be fair for growth stocks."),
                    "ROE %": st.column_config.NumberColumn("ROE %", format="%.0f",
                        help="Return on Equity: Net Income / Equity. Quality proxy!"),
                    "Profit %": st.column_config.NumberColumn("Profit %", format="%.0f",
                        help="Net Profit Margin. Pricing power & efficiency."),
                    "OPM %": st.column_config.NumberColumn("OPM %", format="%.0f",
                        help="Operating Profit Margin (before interest & tax)."),
                    "D/E": st.column_config.NumberColumn("D/E", format="%.0f",
                        help="Debt / Equity. >2 can be risky unless stable cash flows."),
                    "Div %": st.column_config.NumberColumn("Div %", format="%.0f",
                        help="Dividend yield (annual)."),
                }
            )

    # Company summary
    st.markdown("### 📜 Company Summary")
    if not view.empty:
        sym_pick = st.selectbox("Pick a symbol", view["symbol"].tolist(), key="fund_pick")
        picked = view[view["symbol"] == sym_pick].iloc[0]
        summary = picked.get("summary")
        if isinstance(summary, str) and summary.strip():
            st.write(summary)
        else:
            st.info("Summary unavailable.")
    else:
        st.info("No rows match the filters above.")

    # Diagnostics if available
    if "error" in fdf.columns:
        errs = fdf[fdf["error"].notna()]
        if not errs.empty:
            with st.expander("🔎 Diagnostics: symbols with missing/partial data"):
                st.dataframe(errs[["symbol","error"]], use_container_width=True)
                st.caption("Yahoo’s free endpoint can be intermittent. Refresh later if needed.")

# ========== Tab 5: Valuation ==========
with tabs[4]:
    st.subheader("💰 Valuation Sandbox")
    ddf = st.session_state.get("dash", _load_dash())
    fdf = st.session_state.get("fund", _load_fund())

    if ddf.empty:
        st.info("Build Signals first. Then fetch Fundamentals.")
    elif fdf.empty:
        st.info("Fetch Fundamentals first.")
    else:
        common = sorted(set(ddf["symbol"].tolist()) & set(fdf["symbol"].tolist()))
        if not common:
            st.info("No overlap between signals and fundamentals yet.")
        else:
            sym = st.selectbox("Symbol", common)
            last = float(ddf.loc[ddf["symbol"] == sym, "last"].iloc[0])
            eps  = fdf.loc[fdf["symbol"] == sym, "eps_ttm"].fillna(0).iloc[0] if not fdf.empty else 0.0
            revps = fdf.loc[fdf["symbol"] == sym, "rev_per_share"].fillna(0).iloc[0] if not fdf.empty else 0.0
            pm   = fdf.loc[fdf["symbol"] == sym, "pm_pct"].fillna(0).iloc[0] if not fdf.empty else 0.0

            st.markdown("**Inputs**")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                target_pe = st.number_input("Target P/E", min_value=1.0, value=25.0, step=1.0)
            with c2:
                target_ps = st.number_input("Target P/S", min_value=0.1, value=3.0, step=0.1)
            with c3:
                margin_pct = st.number_input("Sustainable Net Margin (%)", min_value=0.0, value=float(pm or 10.0), step=0.5)
            with c4:
                weight_pe = st.slider("Weight to P/E model (%)", 0, 100, 70, 5)

            pe_price = None
            if eps and eps > 0:
                pe_price = target_pe * eps

            ps_price = None
            if revps and revps > 0:
                ps_price = target_ps * revps
            else:
                cur_pe = fdf.loc[fdf["symbol"] == sym, "pe_ttm"].fillna(0).iloc[0]
                eps_guess = last / cur_pe if cur_pe and cur_pe > 0 else None
                if eps_guess:
                    ps_price = target_pe * eps_guess

            w_pe = weight_pe / 100.0
            implied_list = [p for p in [pe_price, ps_price] if p is not None and p > 0]
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

# ========== Tab 6: Help ==========
with tabs[5]:
    st.subheader("ℹ️ How to use")
    st.markdown("""
**Add/Manage:**  
- Type a symbol (without .NS) and click **Add**.  
- Or paste many symbols and **Validate & Queue**, then **Add queued**.  
- Signals rebuild automatically after add.

**Signals:**  
- Filter and sort. Then open **Detail**.

**Detail:**  
- Key levels and a quick conclusion. Toggle the chart as needed.

**Fundamentals:**  
- Click **Fetch / Refresh Fundamentals**.  
- By default all rows show. Toggle **Enable filters** to refine.  
- Use **Highlight good/bad cells** for quick scanning, or turn it off to see header info buttons.

**Valuation:**  
- Set target P/E or P/S, adjust margin and weight, see **Implied Price** and **Upside**.
""")