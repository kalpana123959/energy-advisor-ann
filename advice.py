import streamlit as st
from utils import (load_hourly_data, get_settings, page_header, make_recommendations, tip_html,
                   consumption_level, level_badge, SUBMETER_NAMES)

page_header("Personalized Advice", "Energy-saving tips built from the ANN prediction and your usage pattern")
df = load_hourly_data()
s = get_settings()
res = st.session_state.get("last_result")

if not res:
    st.info("No prediction yet. Open **Energy Prediction**, enter or load a row, and click "
            "**Predict next hour**. The advice here is generated from that real ANN result.")
    st.stop()

inputs, pred = res["inputs"], res["predicted"]
level, lo, hi = consumption_level(pred, df, s)
c1, c2, c3 = st.columns(3)
c1.metric("Predicted next hour (kW)", f"{pred:.3f}")
c2.markdown(f"**Predicted level**<br>{level_badge(level)}", unsafe_allow_html=True)
sm = {k: inputs[k] for k in SUBMETER_NAMES}
top = max(sm, key=sm.get)
c3.metric("Largest sub-meter", SUBMETER_NAMES[top], help=f"Reading: {sm[top]:.2f}")

tips, reasons = make_recommendations(inputs, pred)
kind = {"High": "alert", "Medium": "warn", "Low": ""}[level]
st.subheader("Recommended actions")
for t in tips:
    st.markdown(tip_html(t, kind), unsafe_allow_html=True)

with st.expander("Why these tips were shown"):
    for r in reasons:
        st.write(f"- {r}")

st.subheader("Estimated saving if you follow the advice (what-if)")
pct = st.slider("Assume you reduce the next hour's load by (%)", 0, 40, 10)
saved = pred * pct / 100.0
a, b = st.columns(2)
a.metric("Energy saved this hour (kWh)", f"{saved:.3f}")
b.metric("Cost saved this hour", f"{s['currency']}{saved * s['tariff']:.2f}")
st.caption("This is an illustrative what-if from the prediction, not a measured saving. "
           "Tips come from rules applied to the ANN output and your inputs.")
