import streamlit as st
from utils import get_settings, page_header

page_header("Settings", "Set your price per unit and your Low / Medium / High limits")
s = get_settings()

st.subheader("Electricity price")
c1, c2 = st.columns(2)
st.session_state["currency"] = c1.text_input("Money symbol", value=s["currency"], max_chars=4)
st.session_state["tariff"] = c2.number_input("Price for 1 unit (1 kWh) of electricity", min_value=0.0,
                                             value=float(s["tariff"]), step=0.5,
                                             help="Look at your electricity bill to find this price.")

st.subheader("Low, Medium and High limits")
st.caption("We compare a prediction with your past hourly readings. If it is in the lowest part, it is Low. "
           "If it is in the highest part, it is High. Everything in between is Medium.")
lo, hi = st.slider("Move the two ends to set the limits", 5, 95, (int(s["low_pct"]), int(s["high_pct"])),
                   help="For example 33 and 66 means: the lowest third is Low and the highest third is High.")
st.session_state["low_pct"], st.session_state["high_pct"] = lo, hi

st.subheader("About this website")
st.markdown("""
- **Model:** an ANN (a computer model that learns from past data) with two hidden layers (64 and 32 neurons)
- **Data:** `prepared_hourly_data.csv`, hourly household power readings from 2006 to 2010
- **What it predicts:** the power your home will use in the next hour
- **Files left unchanged:** `train_model.py`, all CSV files, the `artifacts/` folder and `sc_project.ipynb`
""")
st.caption("Your settings stay until you close or refresh the browser tab.")
if st.button("Reset settings"):
    for k in ["currency", "tariff", "low_pct", "high_pct"]:
        st.session_state.pop(k, None)
    st.rerun()
