import streamlit as st
from utils import get_settings, page_header

page_header("Settings", "Tariff, currency and usage-level thresholds")
s = get_settings()

st.subheader("Billing")
c1, c2 = st.columns(2)
st.session_state["currency"] = c1.text_input("Currency symbol", value=s["currency"], max_chars=4)
st.session_state["tariff"] = c2.number_input("Electricity price per kWh", min_value=0.0, value=float(s["tariff"]), step=0.5)

st.subheader("Low / Medium / High thresholds")
st.caption("A prediction is Low if it is below the first percentile of historical hourly power, "
           "High if above the second, otherwise Medium.")
lo, hi = st.slider("Percentile limits", 5, 95, (int(s["low_pct"]), int(s["high_pct"])))
st.session_state["low_pct"], st.session_state["high_pct"] = lo, hi

st.subheader("About this application")
st.markdown("""
- **Model:** MLPRegressor (64, 32), ReLU, Adam, 50 iterations, StandardScaler
- **Data:** `prepared_hourly_data.csv` (hourly household power data, 2006–2010)
- **Target:** next-hour electricity consumption (`target_next_hour`)
- **Files kept untouched:** `train_model.py`, all CSVs, `artifacts/`, `sc_project.ipynb`
""")
st.caption("Settings last for your current browser session.")
if st.button("Reset settings"):
    for k in ["currency", "tariff", "low_pct", "high_pct"]:
        st.session_state.pop(k, None)
    st.rerun()
