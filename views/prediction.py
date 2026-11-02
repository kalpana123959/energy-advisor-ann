import pandas as pd
import streamlit as st
from utils import (go, load_or_train_model, load_hourly_data, load_test_sample_table, get_settings,
                   page_header, predict_next_hour, predict_many, consumption_level, level_badge,
                   line_chart, kpi_panel, DAY_NAMES)
from train_model import FEATURES, TARGET

page_header("Energy Prediction", "Find out how much power your home may use in the next hour")
model, scaler, metrics = load_or_train_model()
samples = load_test_sample_table()
df = load_hourly_data()
s = get_settings()

st.write("Fill in what your home is using **this hour**. You can also load a real hour from the saved "
         "test data. Then click **Predict next hour**.")
c_a, c_b = st.columns(2)
use_sample = c_a.checkbox("Use a real hour from the test data", value=True)
idx = c_b.number_input("Row number in the test data", 0, max(0, len(samples) - 1), 0, 1, disabled=not use_sample)

d = samples.iloc[int(idx)].to_dict() if use_sample else {
    "Global_active_power": 1.0, "Global_reactive_power": 0.1, "Voltage": 240.0,
    "Global_intensity": 4.0, "Sub_metering_1": 0.0, "Sub_metering_2": 0.0,
    "Sub_metering_3": 1.0, "hour": 18, "day_of_week": 5, TARGET: None}

c1, c2, c3 = st.columns(3)
with c1:
    gap = st.number_input("Power used this hour (kW)", value=float(d["Global_active_power"]), min_value=0.0,
                          format="%.4f", help="How much electricity the whole house is using now.")
    grp = st.number_input("Reactive power (kVAR)", value=float(d["Global_reactive_power"]), min_value=0.0,
                          format="%.4f", help="A technical meter reading. You can keep the default value.")
    voltage = st.number_input("Voltage (V)", value=float(d["Voltage"]), min_value=0.0, format="%.4f",
                              help="The strength of the electricity supply.")
with c2:
    inten = st.number_input("Current (A)", value=float(d["Global_intensity"]), min_value=0.0, format="%.4f",
                            help="How much electricity is flowing into the house.")
    sm1 = st.number_input("Kitchen meter", value=float(d["Sub_metering_1"]), min_value=0.0, format="%.4f",
                          help="Dishwasher, oven and microwave.")
    sm2 = st.number_input("Laundry meter", value=float(d["Sub_metering_2"]), min_value=0.0, format="%.4f",
                          help="Washing machine, dryer and refrigerator.")
with c3:
    sm3 = st.number_input("Water heater / AC meter", value=float(d["Sub_metering_3"]), min_value=0.0,
                          format="%.4f", help="Water heater and air conditioner.")
    hour = st.slider("Hour of the day (0 = midnight)", 0, 23, int(d["hour"]))
    dow = st.selectbox("Day of the week", list(range(7)), index=int(d["day_of_week"]) % 7,
                       format_func=lambda i: DAY_NAMES[i])

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
    st.subheader("Your result")
    if actual is not None and not pd.isna(actual):
        third = {"label": "Real next hour (test data)", "value": f"{float(actual):.3f}"}
    else:
        third = {"label": "Cost of that hour", "value": f"{s['currency']}{pred * s['tariff']:.2f}"}
    kpi_panel([
        {"label": "Power now (kW)", "value": f"{now:.3f}"},
        {"label": "Guess for next hour (kW)", "value": f"{pred:.3f}", "delta": f"{pred - now:+.3f} vs now"},
        third,
        {"label": "Level", "value": level_badge(level)},
    ])
    st.caption(f"Low means under {lo:.2f} kW and High means over {hi:.2f} kW. These limits come from "
               "your past data. You can change them in Settings.")

    if res["use_sample"]:
        st.markdown("**The hours around this one**")
        a, b = max(0, res["idx"] - 12), min(len(samples), res["idx"] + 13)
        w = samples.iloc[a:b].copy()
        w["ANN prediction"] = predict_many(model, scaler, w)
        w = w.rename(columns={"Global_active_power": "Power now", TARGET: "Real next hour",
                              "ANN prediction": "Model's guess"})
        w = w[["Power now", "Real next hour", "Model's guess"]].reset_index(drop=True)
        w.index = w.index + 1
        line_chart(w, ylabel="Power (kW)", xlabel="Hours in a row (1 = first hour shown)")
    st.button("Get my saving tips →", on_click=go, args=("advice",), key="to_advice")
