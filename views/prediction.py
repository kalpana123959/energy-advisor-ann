import pandas as pd
import streamlit as st
from utils import (load_or_train_model, load_hourly_data, load_test_sample_table, get_settings,
                   page_header, predict_next_hour, predict_many, consumption_level, level_badge,
                   DAY_NAMES, SUBMETER_HELP)
from train_model import FEATURES, TARGET

page_header("Energy Prediction", "Next-hour electricity consumption predicted by your trained ANN")
model, scaler, metrics = load_or_train_model()
samples = load_test_sample_table()
df = load_hourly_data()
s = get_settings()

st.write("The 9 inputs below are the same features the ANN was trained on.")
c_a, c_b = st.columns(2)
use_sample = c_a.checkbox("Load a real row from X_test.csv", value=True)
idx = c_b.number_input("Test-set row number", 0, max(0, len(samples) - 1), 0, 1, disabled=not use_sample)

d = samples.iloc[int(idx)].to_dict() if use_sample else {
    "Global_active_power": 1.0, "Global_reactive_power": 0.1, "Voltage": 240.0,
    "Global_intensity": 4.0, "Sub_metering_1": 0.0, "Sub_metering_2": 0.0,
    "Sub_metering_3": 1.0, "hour": 18, "day_of_week": 5, TARGET: None}

c1, c2, c3 = st.columns(3)
with c1:
    gap = st.number_input("Global active power (kW)", value=float(d["Global_active_power"]), min_value=0.0, format="%.4f")
    grp = st.number_input("Global reactive power (kVAR)", value=float(d["Global_reactive_power"]), min_value=0.0, format="%.4f")
    voltage = st.number_input("Voltage (V)", value=float(d["Voltage"]), min_value=0.0, format="%.4f")
with c2:
    inten = st.number_input("Global intensity (A)", value=float(d["Global_intensity"]), min_value=0.0, format="%.4f")
    sm1 = st.number_input("Sub-metering 1 (kitchen)", value=float(d["Sub_metering_1"]), min_value=0.0, format="%.4f", help=SUBMETER_HELP["Sub_metering_1"])
    sm2 = st.number_input("Sub-metering 2 (laundry)", value=float(d["Sub_metering_2"]), min_value=0.0, format="%.4f", help=SUBMETER_HELP["Sub_metering_2"])
with c3:
    sm3 = st.number_input("Sub-metering 3 (water heater / AC)", value=float(d["Sub_metering_3"]), min_value=0.0, format="%.4f", help=SUBMETER_HELP["Sub_metering_3"])
    hour = st.slider("Hour of day", 0, 23, int(d["hour"]))
    dow = st.selectbox("Day of week", list(range(7)), index=int(d["day_of_week"]) % 7,
                       format_func=lambda i: f"{i} — {DAY_NAMES[i]}")

inputs = {"Global_active_power": gap, "Global_reactive_power": grp, "Voltage": voltage,
          "Global_intensity": inten, "Sub_metering_1": sm1, "Sub_metering_2": sm2,
          "Sub_metering_3": sm3, "hour": hour, "day_of_week": dow}

if st.button("Predict next hour", type="primary"):
    st.session_state["last_result"] = {
        "inputs": inputs, "predicted": predict_next_hour(model, scaler, inputs),
        "actual": d.get(TARGET) if use_sample else None,
        "use_sample": use_sample, "idx": int(idx)}

res = st.session_state.get("last_result")
if res:
    pred, now, actual = res["predicted"], res["inputs"]["Global_active_power"], res["actual"]
    level, lo, hi = consumption_level(pred, df, s)
    st.subheader("Result")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current hour (kW)", f"{now:.3f}")
    m2.metric("ANN predicted next hour (kW)", f"{pred:.3f}", delta=f"{pred - now:.3f} vs now")
    if actual is not None and not pd.isna(actual):
        m3.metric("Actual next hour (test CSV)", f"{float(actual):.3f}")
    else:
        m3.metric("Estimated cost of that hour", f"{s['currency']}{pred * s['tariff']:.2f}")
    m4.markdown(f"**Level**<br>{level_badge(level)}", unsafe_allow_html=True)
    st.caption(f"Level: Low below {lo:.2f} kW, High above {hi:.2f} kW "
               f"(percentile limits from the dataset; change them in Settings).")

    if res["use_sample"]:
        st.markdown("**Nearby hours in the test set**")
        a, b = max(0, res["idx"] - 12), min(len(samples), res["idx"] + 13)
        w = samples.iloc[a:b].copy()
        w["ANN prediction"] = predict_many(model, scaler, w)
        w = w.rename(columns={"Global_active_power": "Current-hour power", TARGET: "Actual next hour"})
        st.line_chart(w[["Current-hour power", "Actual next hour", "ANN prediction"]].reset_index(drop=True))
    st.info("Go to **Personalized Advice** to see tips generated from this prediction.")
