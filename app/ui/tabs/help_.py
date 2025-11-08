# app/ui/tabs/help_.py
import streamlit as st

def render():
    st.subheader("ℹ️ How to use")
    st.markdown("""
**Add/Manage**  
- Add one or many symbols (without .NS). Signals rebuild automatically.  
- Fundamentals for new symbols are fetched automatically (rate-limited).

**Signals**  
- Filter and sort; click **open** to deep-link a stock into Detail.

**Detail**  
- Story-first view: snapshot → action levels → fundamentals → conclusion → chart at the end.

**Fundamentals**  
- Fetch/refresh all fundamentals, filter, and scan with green/red highlights.
""")