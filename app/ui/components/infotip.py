# app/ui/components/infotip.py
import streamlit as st

# small ℹ️ helper you can drop next to any label
def info(label: str, help_text: str):
    col1, col2 = st.columns([1, 0.06])
    with col1:
        st.markdown(label)
    with col2:
        st.markdown(
            f'<span title="{help_text}" style="cursor:help;">ℹ️</span>',
            unsafe_allow_html=True,
        )