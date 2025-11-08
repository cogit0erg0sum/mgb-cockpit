

# ===== app/web_app.py =====
# app/web_app.py
import os, sys, pathlib
from app.services.data_loaders import load_dash, load_fund
from app.ui.tabs import add_manage, signals, detail, fundamentals, valuation, help_

# -------- Paths & imports (make packages importable) --------
ROOT = pathlib.Path(__file__).resolve().parents[1]   # .../mgb-cockpit
APP_DIR = ROOT / "app"
ENGINE_DIR = ROOT / "engine"

# 1) add project root so `import app...` works
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# 2) also add module folders for direct imports (optional)
for p in (APP_DIR, ENGINE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime
from app.ui.tabs import add_manage, signals, detail, fundamentals, valuation, help_, diag
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

# ---- Last updated banner (IST) ----
from datetime import datetime, timezone
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:
    ZoneInfo = None

def _fmt_ist(ts: float | None):
    if not ts:
        return "—"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    if ZoneInfo:
        dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
    return dt.strftime("%Y-%m-%d %H:%M:%S IST")

dash_mtime = DATA.joinpath("dashboard.csv").stat().st_mtime if DATA.joinpath("dashboard.csv").exists() else None
fund_mtime = DATA.joinpath("fundamentals.csv").stat().st_mtime if DATA.joinpath("fundamentals.csv").exists() else None

st.caption(
    f"🕒 Last updated — Dashboard: **{_fmt_ist(dash_mtime)}** · Fundamentals: **{_fmt_ist(fund_mtime)}**"
)

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
if "fund" not in st.session_state:
    st.session_state["fund"] = _load_fund()
st.caption(f"🧪 Debug — Fundamentals rows: {len(st.session_state['fund'])}")  

# -------- Tabs --------
tabs = st.tabs(["➕ Add/Manage", "📋 Signals", "📈 Detail", "🧾 Fundamentals", "💰 Valuation", "ℹ️ Help"])

# ========== Tab 1: Add / Manage ==========
with tabs[0]:
    add_manage.render()
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

# replace the whole Tab 4 block with:
from app.ui.tabs import fundamentals as fundamentals_tab
with tabs[3]:
    fundamentals.render()

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

# ===== app/services/data_loaders.py =====
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

# ===== app/ui/components/banner.py =====
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

# ===== app/ui/components/layout.py =====
# app/ui/components/layout.py
from __future__ import annotations
import contextlib
import streamlit as st

# --- Page meta (put this near the top of web_app.py) ---
def set_page_meta(title: str = "Multibagger Cockpit", layout: str = "wide"):
    # must be the FIRST Streamlit call on the page
    st.set_page_config(page_title=title, layout=layout)

# --- Global CSS & light theming ---
def inject_global_css():
    st.markdown(
        """
        <style>
          /* Hide Streamlit chrome */
          #MainMenu, header, footer {visibility:hidden;}
          /* Typography + spacing */
          html, body, [class*="block-container"] {
            font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
            padding-top: 0.8rem;
          }
          h1,h2,h3 {letter-spacing:-0.01em}
          [data-testid="stMetricValue"] {font-weight:700}
          /* Card */
          .mb-card {border: 1px solid #eee; border-radius: 16px; padding: 18px; background: #fff;}
          .mb-muted {color:#637083;}
          /* Badges */
          .badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px;
                  background:#F1F5F9; color:#0F172A;}
          .badge.green {background:#E8F5E9; color:#1B5E20;}
          .badge.red   {background:#FFEBEE; color:#B71C1C;}
          .badge.blue  {background:#E3F2FD; color:#0D47A1;}
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Small UI helpers ---
@contextlib.contextmanager
def card():
    st.markdown('<div class="mb-card">', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)

def badge(text: str, kind: str = "neutral"):
    cls = "badge"
    if kind == "green": cls += " green"
    elif kind == "red": cls += " red"
    elif kind == "blue": cls += " blue"
    st.markdown(f'<span class="{cls}">{text}</span>', unsafe_allow_html=True)

def section(title: str, subtitle: str | None = None):
    """Section header with optional subtle subtitle."""
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="mb-muted">{subtitle}</div>', unsafe_allow_html=True)

def metric_help(label: str, value: str, help_text: str | None = None):
    """Metric with an optional small help icon tooltip."""
    if help_text:
        st.metric(label, value, help=help_text)
    else:
        st.metric(label, value)

# ===== app/ui/tabs/add_manage.py =====
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

# ===== app/ui/tabs/signals.py =====
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

# ===== app/ui/tabs/detail.py =====
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

# ===== app/ui/tabs/fundamentals.py =====
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

# ===== app/ui/tabs/valuation.py =====
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

# ===== engine/screener_core.py =====
# engine/screener_core.py
import os, math
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(0)

def sma(s, n):
    return s.rolling(n).mean()

def _compute_signals(df: pd.DataFrame, buy_low: float, buy_high: float, stop: float,
                     rsi_len=14, roll_max=55, vol_spike_thr=1.6) -> dict:
    c = df["Close"]; v = df["Volume"]
    last = float(c.iloc[-1])
    s20 = float(sma(c, 20).iloc[-1])
    s50 = float(sma(c, 50).iloc[-1])
    rsi_ = float(rsi(c, rsi_len).iloc[-1])
    max55 = float(c.rolling(roll_max).max().iloc[-1])
    min252 = float(c.rolling(252).min().iloc[-1])
    max252 = float(c.rolling(252).max().iloc[-1])
    avgv20 = float(v.rolling(20).mean().iloc[-1]) if v.notna().any() else np.nan
    vol_spike = float(v.iloc[-1] / avgv20) if avgv20 and avgv20 > 0 else np.nan

    in_buy_zone = (last >= buy_low) and (last <= buy_high)
    breakout = (last > max55) and (not math.isnan(vol_spike) and vol_spike >= vol_spike_thr)
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

def compute_dashboard(watchlist_path: str, out_csv: str, days: int = 420) -> int:
    """Build dashboard.csv from a watchlist; returns number of rows written."""
    if not os.path.exists(watchlist_path):
        raise FileNotFoundError(f"Missing watchlist file: {watchlist_path}")
    wl = pd.read_csv(watchlist_path)
    if wl.empty:
        raise ValueError("watchlist.csv is empty")

    syms = wl["symbol"].dropna().unique().tolist()
    data = yf.download(syms, period=f"{days}d", auto_adjust=False, progress=False, group_by="ticker")

    rows = []
    for _, r in wl.iterrows():
        sym = r["symbol"]
        try:
            df = data[sym].dropna().copy()
        except KeyError:
            # Try fallback single fetch if grouped failed
            one = yf.download(sym, period=f"{days}d", auto_adjust=False, progress=False)
            if one is None or one.empty:
                print(f"No data for {sym}")
                continue
            df = one.dropna().copy()

        sig = _compute_signals(
            df,
            float(r["buy_low"]),
            float(r["buy_high"]),
            float(r["stop"])
        )
        rows.append({
            "symbol": sym, "name": r.get("name", sym), "segment": r.get("segment", ""),
            **sig,
            "buy_low": float(r["buy_low"]), "buy_high": float(r["buy_high"]), "stop": float(r["stop"]),
            "notes": r.get("notes", ""), "as_of": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    # append at the very end, just before return
    from datetime import datetime, timezone
    import json, pathlib
    DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "app" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (meta := {"last_updated_utc": datetime.now(timezone.utc).isoformat()}).clear() or None
    (DATA_DIR / "metadata.json").write_text(json.dumps({"last_updated_utc": datetime.now(timezone.utc).isoformat()}))
    return len(out)

# ===== engine/fundamentals_core.py =====
# engine/fundamentals_core.py
import math, time, random
from typing import Dict, List
import pandas as pd
import yfinance as yf

def fetch_fundamentals_one(symbol: str) -> dict:
    """
    Returns a single fundamentals row dict for `symbol` (e.g., 'TATAMOTORS.NS').
    Non-throwing: returns {'symbol': symbol, 'error': '...'} if fetch fails.
    """
    row = {"symbol": symbol}

    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}

        row["company"]  = info.get("longName") or info.get("shortName")
        row["sector"]   = info.get("sector")
        row["industry"] = info.get("industry")
        row["mcap"]     = info.get("marketCap")
        row["pe_ttm"]   = info.get("trailingPE")
        row["pe_fwd"]   = info.get("forwardPE")
        row["pb"]       = info.get("priceToBook")
        row["peg"]      = info.get("pegRatio")
        dy = info.get("dividendYield")
        row["div_yield_pct"] = (dy * 100) if dy is not None else None
        row["rev_per_share"] = info.get("revenuePerShare")
        row["eps_ttm"]       = info.get("trailingEps")
        row["summary"]       = info.get("longBusinessSummary")

        # Operating margin, ROE best-effort from info
        om = info.get("operatingMargins")
        row["om_pct"]  = round(om * 100, 2) if om is not None else None
        roe = info.get("returnOnEquity")
        row["roe_pct"] = round(roe * 100, 2) if roe is not None else None
        row["de_ratio"] = info.get("debtToEquity")

        # Profit margin from financials if available
        try:
            fin = tk.financials
            if fin is not None and not fin.empty:
                if "Net Income" in fin.index and "Total Revenue" in fin.index:
                    ni  = float(fin.loc["Net Income"].iloc[0])
                    rev = float(fin.loc["Total Revenue"].iloc[0]) or None
                    row["pm_pct"] = round((ni / rev) * 100, 2) if rev else None
        except Exception:
            pass

        # Optional deltas + fund score if you defined these helpers in this file
        try:
            deltas = quarterly_deltas(tk)   # your helper from earlier step
            row.update(deltas or {})
        except Exception:
            pass

        try:
            row["fund_score"] = fundamental_score_row(row)
        except Exception:
            row["fund_score"] = None

        return row

    except Exception as e:
        row["error"] = str(e)
        return row


def upsert_fundamentals_csv(csv_path: str, row: dict) -> None:
    """Insert or update a single row in fundamentals.csv by symbol."""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.DataFrame()

    if "symbol" not in df.columns:
        df = pd.DataFrame(columns=list(row.keys()))

    if (df["symbol"] == row["symbol"]).any():
        for k, v in row.items():
            df.loc[df["symbol"] == row["symbol"], k] = v
    else:
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df.to_csv(csv_path, index=False)

SAFE_KEYS = [
    ("longName", str),
    ("sector", str),
    ("industry", str),
    ("longBusinessSummary", str),
    ("marketCap", float),
    ("trailingPE", float),
    ("forwardPE", float),
    ("priceToBook", float),
    ("pegRatio", float),
    ("trailingPegRatio", float),
    ("returnOnEquity", float),
    ("profitMargins", float),
    ("grossMargins", float),
    ("operatingMargins", float),
    ("debtToEquity", float),
    ("totalDebt", float),
    ("totalCash", float),
    ("revenueGrowth", float),
    ("earningsGrowth", float),
    ("revenuePerShare", float),
    ("trailingEps", float),
    ("dividendYield", float),
]

def _safe(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
    except Exception:
        return None
    return v

def _coalesce(*vals):
    for v in vals:
        if _safe(v) is not None:
            return v
    return None

def _fetch_info_once(tk: yf.Ticker) -> Dict:
    """Try .get_info(), then .info; augment with fast_info.market_cap if missing."""
    info = {}
    try:
        if hasattr(tk, "get_info"):
            info = tk.get_info() or {}
    except Exception:
        info = {}
    try:
        if not info:
            info = tk.info or {}
    except Exception:
        pass
    try:
        if ("marketCap" not in info or not info.get("marketCap")) and getattr(tk, "fast_info", None):
            mc = getattr(tk.fast_info, "market_cap", None)
            if mc:
                info["marketCap"] = mc
    except Exception:
        pass
    return info or {}

def _row_from_info(sym: str, info: Dict) -> Dict:
    out = {"symbol": sym}
    for k, _t in SAFE_KEYS:
        out[k] = _safe(info.get(k))
    out["company"] = _coalesce(out.get("longName"), sym)
    out["peg"] = _coalesce(out.get("trailingPegRatio"), out.get("pegRatio"))
    return out

def _pct(x):
    return None if x is None else float(x) * 100.0

def _f(x):
    try:
        return None if x is None else float(x)
    except Exception:
        return None

def fetch_single_with_retry(sym: str, retries: int = 2, sleep_base: float = 0.6) -> Dict:
    tk = yf.Ticker(sym)
    last_err = None
    for _ in range(retries + 1):
        try:
            info = _fetch_info_once(tk)
            if info:
                row = _row_from_info(sym, info)
                row["error"] = None
                return row
        except Exception as e:
            last_err = str(e)
        time.sleep(sleep_base * (1 + 0.5 * random.random()))
    return {"symbol": sym, "company": sym, "error": last_err or "info-unavailable"}

def compute_fundamentals(watchlist_csv: str, out_csv: str, throttle_sec: float = 0.3) -> int:
    """
    Fetch fundamentals for all symbols in watchlist and write a normalized CSV.
    Guarantees one row per symbol (with NA where data is missing).
    """
    wl = pd.read_csv(watchlist_csv)
    if wl.empty:
        pd.DataFrame().to_csv(out_csv, index=False)
        return 0

    syms: List[str] = wl["symbol"].dropna().unique().tolist()
    rows: List[Dict] = []

    for s in syms:
        r = fetch_single_with_retry(s)
        rows.append(r)
        time.sleep(throttle_sec)

    df = pd.DataFrame(rows)

    out = pd.DataFrame({
        "symbol": df["symbol"],
        "company": df.get("company"),
        "sector": df.get("sector"),
        "industry": df.get("industry"),
        "mcap": df.get("marketCap").apply(_f),
        "pe_ttm": df.get("trailingPE").apply(_f),
        "pe_fwd": df.get("forwardPE").apply(_f),
        "pb": df.get("priceToBook").apply(_f),
        "peg": df.get("peg").apply(_f),
        "roe_pct": df.get("returnOnEquity").apply(_pct),
        "pm_pct": df.get("profitMargins").apply(_pct),
        "gm_pct": df.get("grossMargins").apply(_pct),
        "om_pct": df.get("operatingMargins").apply(_pct),
        "de_ratio": df.get("debtToEquity").apply(_f),
        "debt": df.get("totalDebt").apply(_f),
        "cash": df.get("totalCash").apply(_f),
        "rev_growth_pct": df.get("revenueGrowth").apply(_pct),
        "earn_growth_pct": df.get("earningsGrowth").apply(_pct),
        "rev_per_share": df.get("revenuePerShare").apply(_f),
        "eps_ttm": df.get("trailingEps").apply(_f),
        "div_yield_pct": df.get("dividendYield").apply(_pct),
        "summary": df.get("longBusinessSummary"),
        "error": df.get("error")
    }).drop_duplicates(subset=["symbol"], keep="last")

    out.to_csv(out_csv, index=False)
    # append at the very end, just before return
    from datetime import datetime, timezone
    import json, pathlib
    DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "app" / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (meta := {"last_updated_utc": datetime.now(timezone.utc).isoformat()}).clear() or None
    (DATA_DIR / "metadata.json").write_text(json.dumps({"last_updated_utc": datetime.now(timezone.utc).isoformat()}))
    return len(out)