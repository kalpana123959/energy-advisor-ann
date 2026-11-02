import streamlit as st
from utils import (go, load_hourly_data, get_settings, page_header, make_recommendations, tip_html,
                   consumption_level, level_badge, kpi_panel, SUBMETER_NAMES)

page_header("Personalized Advice", "Simple tips to save power, based on your prediction")
df = load_hourly_data()
s = get_settings()
res = st.session_state.get("last_result")

if not res:
    st.info("You have not made a prediction yet. Go to **Energy Prediction**, click **Predict next hour**, "
            "and your tips will appear here.")
    st.button("Go to Energy Prediction →", on_click=go, args=("prediction",), key="to_pred")
    st.stop()

inputs, pred = res["inputs"], res["predicted"]
level, lo, hi = consumption_level(pred, df, s)
sm = {k: inputs[k] for k in SUBMETER_NAMES}
top = max(sm, key=sm.get)
kpi_panel([
    {"label": "Guess for next hour (kW)", "value": f"{pred:.3f}"},
    {"label": "Your level", "value": level_badge(level)},
    {"label": "Meter using the most", "value": SUBMETER_NAMES[top], "help": f"Reading: {sm[top]:.2f}"},
])

tips, reasons = make_recommendations(inputs, pred)
kind = {"High": "alert", "Medium": "warn", "Low": ""}[level]
st.subheader("What you can do")
for t in tips:
    st.markdown(tip_html(t, kind), unsafe_allow_html=True)

with st.expander("Why am I seeing these tips?"):
    for r in reasons:
        st.write(f"- {r}")

st.subheader("How much could you save? (just an estimate)")
pct = st.slider("Imagine you use this much less power in the next hour (%)", 0, 40, 10)
saved = pred * pct / 100.0
kpi_panel([
    {"label": "Power saved in this hour (kWh)", "value": f"{saved:.3f}"},
    {"label": "Money saved in this hour", "value": f"{s['currency']}{saved * s['tariff']:.2f}"},
])
st.caption("This is only an estimate, not a measured saving. The tips come from simple rules "
           "applied to your prediction and readings.")
