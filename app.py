"""
Adaptive Personalized Energy Advisor - main entry point.
Run with:  python -m streamlit run app.py
Each page lives in the views/ folder.
"""
import streamlit as st

st.set_page_config(page_title="Adaptive Personalized Energy Advisor",
                   page_icon="⚡", layout="wide")

from utils import inject_css, load_or_train_model, get_settings  # noqa: E402

inject_css()
get_settings()                 # makes sure settings exist
load_or_train_model()          # loads (or retrains) the ANN once, cached

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon="🏠", default=True),
    st.Page("views/analytics.py", title="Energy Analytics", icon="📊"),
    st.Page("views/weekly_report.py", title="Weekly Report", icon="🗓️"),
    st.Page("views/advice.py", title="Personalized Advice", icon="💡"),
    st.Page("views/calculator.py", title="Energy Calculator", icon="🧮"),
    st.Page("views/prediction.py", title="Energy Prediction", icon="🔮"),
    st.Page("views/adaptive.py", title="Adaptive Learning", icon="🧠"),
    st.Page("views/settings.py", title="Settings", icon="⚙️"),
]

with st.sidebar:
    st.markdown("## ⚡ Energy Advisor")
    st.caption("Soft Computing Project · ANN approach")
    st.divider()

st.navigation(pages).run()
