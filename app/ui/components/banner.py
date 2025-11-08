# app/ui/components/banner.py
import streamlit as st
from ...services.data_loaders import load_meta

def render_banner():
    meta = load_meta()
    if meta.get("last_updated_ist"):
        st.caption(
            f"🕒 Last updated: {meta['last_updated_ist']} · "
            f"Signals: {meta.get('rows_dashboard','?')} · "
            f"Fundamentals: {meta.get('rows_fundamentals','?')}"
        )
    else:
        st.caption("🕒 Last updated: unknown (run refresh once)")