# app/ui/tabs/diag.py
from __future__ import annotations
import hashlib, pathlib
import streamlit as st

FILES_TO_CHECK = [
    "app/web_app.py",
    "app/ui/tabs/detail.py",
    "app/ui/tabs/fundamentals.py",
    "app/ui/tabs/signals.py",
    "app/services/data_loaders.py",
    "engine/screener_core.py",
    "engine/fundamentals_core.py",
    "app/data/watchlist.csv",
    "app/data/dashboard.csv",
    "app/data/fundamentals.csv",
]

def _md5(path: pathlib.Path) -> str:
    try:
        m = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                m.update(chunk)
        return m.hexdigest()[:10]
    except Exception:
        return "—"

def render():
    st.subheader("🧪 Diagnostics / Sync Check")

    root = pathlib.Path(__file__).resolve().parents[2]  # repo root
    rows = []
    for rel in FILES_TO_CHECK:
        p = root / rel
        rows.append((rel, p.exists(), _md5(p), str(p) if p.exists() else ""))

    st.caption("MD5 shown here should match your local file if we’re on the same code.")
    st.table({"file": [r[0] for r in rows],
              "exists": [r[1] for r in rows],
              "md5_10": [r[2] for r in rows],
              "abs_path": [r[3] for r in rows]})