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