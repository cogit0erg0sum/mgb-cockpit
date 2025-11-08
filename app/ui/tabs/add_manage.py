# app/ui/tabs/add_manage.py
from __future__ import annotations
import json, time, subprocess, importlib
from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parents[3] / "app" / "data"
WL_CSV   = DATA_DIR / "watchlist.csv"

def _load_watchlist() -> pd.DataFrame:
    if WL_CSV.exists():
        try:
            return pd.read_csv(WL_CSV)
        except Exception:
            pass
    return pd.DataFrame(columns=["symbol","name","segment","buy_low","buy_high","stop","notes"])

def render():
    st.subheader("➕ Add / Manage")
    DATA_DIR = Path(__file__).resolve().parents[3] / "app" / "data"
    META    = DATA_DIR / "metadata.json"

    # --- DEBUG breadcrumb so we know this tab rendered
    st.caption("DEBUG: add_manage tab loaded")

    # =============== Fetch Fundamentals (the button you need) ===============
    st.markdown("#### 🔄 Fundamentals")
    col_ff1, col_ff2 = st.columns([1, 3])
    with col_ff1:
        do_fetch = st.button("Fetch fundamentals", key="btn_fetch_fundamentals")
    with col_ff2:
        st.caption("Fetch/refresh fundamentals for all symbols in your watchlist.")

    if do_fetch:
        try:
            # Prefer in-process call to engine.fundamentals_core
            fc = importlib.import_module("engine.fundamentals_core")
            if hasattr(fc, "main"):
                fc.main()
            elif hasattr(fc, "compute_fundamentals"):
                fc.compute_fundamentals()
            else:
                # Fallback to script run (works on Streamlit Cloud too)
                subprocess.run(["python", "engine/fundamentals_core.py"], check=True)

            # Touch metadata
            meta_path = DATA_DIR / "metadata.json"
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            meta["fundamentals_last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

            # Reload into session if loader available
            try:
                from app.services.data_loaders import load_fund
                st.session_state["fund"] = load_fund()
            except Exception:
                pass

            st.success("Fundamentals fetched. Reloading …")
            st.rerun()
        except Exception as e:
            st.error(f"Fetch failed: {e}")

    st.divider()

    # =============== Manage Watchlist (simple view) ===============
    st.markdown("#### 📄 Current Watchlist")
    wl = _load_watchlist()
    st.dataframe(wl, use_container_width=True)