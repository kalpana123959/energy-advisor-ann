"""
Member 3 prototype: Personalized Advice and Application.

Original files (CSVs and sc_project.ipynb) are only read, never changed.
"""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from train_model import (
    ARTIFACT_DIR,
    FEATURES,
    TARGET,
    chronological_split,
    load_and_prepare,
    train_and_save,
)

BASE_DIR = Path(__file__).resolve().parent

SUBMETER_HELP = {
    "Sub_metering_1": "Kitchen (dishwasher, oven, microwave)",
    "Sub_metering_2": "Laundry (washing machine, tumble-dryer, refrigerator)",
    "Sub_metering_3": "Water heater and air-conditioner",
}

DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


@st.cache_resource(show_spinner="Loading preprocessing and ANN (same settings as the notebook)...")
def load_or_train_model():
    model_path = ARTIFACT_DIR / "ann_model.joblib"
    scaler_path = ARTIFACT_DIR / "scaler.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"

    if model_path.exists() and scaler_path.exists() and metrics_path.exists():
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        return model, scaler, metrics

    metrics = train_and_save()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, metrics


@st.cache_data
def load_test_sample_table():
    x_test_path = BASE_DIR / "X_test.csv"
    y_test_path = BASE_DIR / "y_test.csv"
    if x_test_path.exists() and y_test_path.exists():
        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path)
        sample = x_test.copy()
        sample[TARGET] = y_test[TARGET]
        return sample.reset_index(drop=True)

    data = load_and_prepare()
    _, x_test, _, y_test = chronological_split(data)
    sample = x_test.copy()
    sample[TARGET] = y_test.values
    return sample.reset_index(drop=True)


def make_recommendations(inputs, predicted):
    """ANN-based advice: prediction plus household usage features."""
    tips = []
    reasons = []
    current = float(inputs["Global_active_power"])
    hour = int(inputs["hour"])
    kitchen = float(inputs["Sub_metering_1"])
    laundry = float(inputs["Sub_metering_2"])
    climate = float(inputs["Sub_metering_3"])
    voltage = float(inputs["Voltage"])
    intensity = float(inputs["Global_intensity"])

    change_pct = 0.0 if current == 0 else 100.0 * (predicted - current) / current

    if predicted > current * 1.15:
        tips.append(
            "Next-hour use is predicted to rise. Finish high-power tasks soon, "
            "or delay extra appliances."
        )
        reasons.append(f"ANN prediction is {change_pct:.1f}% above current-hour power.")
    elif predicted < current * 0.85:
        tips.append(
            "Next-hour use is predicted to fall. If a chore cannot wait until "
            "late night, this lower window is the better choice."
        )
        reasons.append(f"ANN prediction is {abs(change_pct):.1f}% below current-hour power.")
    else:
        tips.append(
            "Next-hour use is predicted to stay close to now. Keep only "
            "necessary appliances running."
        )
        reasons.append("ANN prediction is close to current-hour power.")

    if 17 <= hour <= 22:
        tips.append(
            "This is a typical evening peak (5–10 pm). Avoid overlapping "
            "the oven, washing machine, and water heater."
        )
        reasons.append(f"Hour = {hour}, which is an evening peak period in this dataset.")
    elif 0 <= hour <= 6:
        tips.append(
            "This is an off-peak night/early-morning hour. Shift flexible "
            "loads here when you can."
        )
        reasons.append(f"Hour = {hour}, which is an off-peak period in this dataset.")

    highest = max(
        ("kitchen", kitchen),
        ("laundry", laundry),
        ("water heater / AC", climate),
        key=lambda item: item[1],
    )
    if highest[1] >= 10:
        if highest[0] == "kitchen":
            tips.append(
                "Kitchen use is high. Batch cooking, skip unused oven preheat, "
                "and run the dishwasher only when full."
            )
        elif highest[0] == "laundry":
            tips.append(
                "Laundry use is high. Use a cold wash when possible and "
                "air-dry instead of the dryer."
            )
        else:
            tips.append(
                "Water-heater / AC use is high. Raise the AC setpoint a little, "
                "or heat water only as needed."
            )
        reasons.append(
            f"Highest sub-meter is {highest[0]} at {highest[1]:.2f}."
        )
    elif kitchen + laundry + climate < 2 and current > 1.5:
        tips.append(
            "The three sub-meters are low, but whole-house power is still high. "
            "Check lights, electronics, and other unmetered plug loads."
        )
        reasons.append(
            "Sub-meters are low while Global_active_power is still above 1.5 kW."
        )

    if voltage < 230:
        tips.append(
            "Voltage is on the low side of this dataset. Reduce extra loads "
            "until voltage returns closer to the usual household range."
        )
        reasons.append(f"Voltage = {voltage:.2f} V.")
    if intensity > 15:
        tips.append(
            "Current (global intensity) is high. Turn off unused high-draw devices."
        )
        reasons.append(f"Global intensity = {intensity:.2f} A.")

    if not tips:
        tips.append("Usage looks moderate. Continue switching off idle devices.")
        reasons.append("No high-load rule was triggered.")
    return tips, reasons


def plot_current_vs_predicted(current, predicted, actual=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = ["Current hour", "ANN next hour"]
    values = [current, predicted]
    colors = ["#5b8def", "#2bb673"]
    if actual is not None and not pd.isna(actual):
        labels.append("Actual next hour (test CSV)")
        values.append(float(actual))
        colors.append("#f4a261")
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Electricity consumption (kW)")
    ax.set_title("Electricity consumption: current vs ANN prediction")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_submeters(kitchen, laundry, climate):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(
        ["Kitchen", "Laundry", "Water heater / AC"],
        [kitchen, laundry, climate],
        color=["#e76f51", "#2a9d8f", "#264653"],
    )
    ax.set_ylabel("Sub-meter reading")
    ax.set_title("Household usage pattern this hour")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_nearby_hours(samples, center_index, model, scaler):
    start = max(0, center_index - 12)
    end = min(len(samples), center_index + 13)
    window = samples.iloc[start:end].copy()
    scaled = scaler.transform(window[FEATURES])
    window["ann_next_hour"] = np.maximum(model.predict(scaled), 0.0)
    x = np.arange(len(window))

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(x, window["Global_active_power"], marker="o", label="Current-hour power")
    ax.plot(x, window[TARGET], marker="s", label="Actual next-hour target")
    ax.plot(x, window["ann_next_hour"], marker="^", label="ANN next-hour prediction")
    mark = center_index - start
    ax.axvline(mark, color="gray", linestyle="--", alpha=0.7, label="Selected hour")
    ax.set_xlabel("Nearby hours in the test set")
    ax.set_ylabel("Electricity consumption (kW)")
    ax.set_title("Consumption around the selected test-set hour")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Adaptive Personalized Energy Advisor",
        page_icon="⚡",
        layout="wide",
    )
    st.title("Adaptive Personalized Energy Advisor")
    st.caption(
        "Member 3 prototype: next-hour ANN prediction, consumption graphs, "
        "and personalized energy-saving advice."
    )

    try:
        model, scaler, metrics = load_or_train_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    samples = load_test_sample_table()

    with st.sidebar:
        st.header("How to use")
        st.write(
            "1. Load a test-set row or type this hour's readings.\n"
            "2. Click **Predict next hour**.\n"
            "3. Read the prediction, graphs, and personal tips."
        )
        st.divider()
        st.subheader("Integrated ANN")
        st.write("Preprocessing: numeric cleanup, then StandardScaler")
        st.write("Split: first 80% train, last 20% test (time order)")
        st.write("ANN: MLPRegressor, hidden layers 64 and 32")
        st.write("Activation: ReLU · Solver: Adam · Max iterations: 50")
        st.write("Target: `target_next_hour`")

        nb = metrics.get("notebook_output_from_sc_project", {})
        if nb:
            st.subheader("Notebook test results")
            st.caption(nb.get("note", "From sc_project.ipynb output"))
            st.write(
                f"ANN MAE **{nb['ann']['MAE']:.4f}** · "
                f"RMSE **{nb['ann']['RMSE']:.4f}** · "
                f"R² **{nb['ann']['R2']:.4f}**"
            )
            st.write(
                f"Baseline MAE **{nb['baseline']['MAE']:.4f}** · "
                f"RMSE **{nb['baseline']['RMSE']:.4f}** · "
                f"R² **{nb['baseline']['R2']:.4f}**"
            )

        if "ann" in metrics and metrics.get("source"):
            st.subheader("This computer's retrain")
            st.caption(metrics.get("source", ""))
            st.write(
                f"ANN MAE **{metrics['ann']['MAE']:.4f}** · "
                f"RMSE **{metrics['ann']['RMSE']:.4f}** · "
                f"R² **{metrics['ann']['R2']:.4f}**"
            )

    tab_predict, tab_docs = st.tabs(["Advisor app", "Prototype notes"])

    with tab_predict:
        st.subheader("1. Current-hour household inputs")
        st.write("These 9 features match `sc_project.ipynb`.")

        col_a, col_b = st.columns(2)
        with col_a:
            use_sample = st.checkbox("Load a real row from X_test.csv", value=True)
        with col_b:
            sample_index = st.number_input(
                "Test-set row number",
                min_value=0,
                max_value=max(0, len(samples) - 1),
                value=0,
                step=1,
                disabled=not use_sample,
            )

        defaults = samples.iloc[int(sample_index)].to_dict() if use_sample else {
            "Global_active_power": 1.0,
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 0.0,
            "Sub_metering_3": 1.0,
            "hour": 18,
            "day_of_week": 5,
            TARGET: None,
        }

        c1, c2, c3 = st.columns(3)
        with c1:
            gap = st.number_input(
                "Global active power (kW)",
                value=float(defaults["Global_active_power"]),
                min_value=0.0,
                format="%.4f",
                help="Household active power this hour.",
            )
            grp = st.number_input(
                "Global reactive power (kVAR)",
                value=float(defaults["Global_reactive_power"]),
                min_value=0.0,
                format="%.4f",
            )
            voltage = st.number_input(
                "Voltage (V)",
                value=float(defaults["Voltage"]),
                min_value=0.0,
                format="%.4f",
            )
        with c2:
            intensity = st.number_input(
                "Global intensity (A)",
                value=float(defaults["Global_intensity"]),
                min_value=0.0,
                format="%.4f",
                help="Current drawn by the household.",
            )
            sm1 = st.number_input(
                "Sub-metering 1 (kitchen)",
                value=float(defaults["Sub_metering_1"]),
                min_value=0.0,
                format="%.4f",
                help=SUBMETER_HELP["Sub_metering_1"],
            )
            sm2 = st.number_input(
                "Sub-metering 2 (laundry)",
                value=float(defaults["Sub_metering_2"]),
                min_value=0.0,
                format="%.4f",
                help=SUBMETER_HELP["Sub_metering_2"],
            )
        with c3:
            sm3 = st.number_input(
                "Sub-metering 3 (water heater / AC)",
                value=float(defaults["Sub_metering_3"]),
                min_value=0.0,
                format="%.4f",
                help=SUBMETER_HELP["Sub_metering_3"],
            )
            hour = st.slider("Hour of day", 0, 23, int(defaults["hour"]))
            day_of_week = st.selectbox(
                "Day of week",
                options=list(range(7)),
                index=int(defaults["day_of_week"]) % 7,
                format_func=lambda d: f"{d} — {DAY_NAMES[d]}",
            )

        inputs = {
            "Global_active_power": gap,
            "Global_reactive_power": grp,
            "Voltage": voltage,
            "Global_intensity": intensity,
            "Sub_metering_1": sm1,
            "Sub_metering_2": sm2,
            "Sub_metering_3": sm3,
            "hour": hour,
            "day_of_week": day_of_week,
        }

        predict = st.button("Predict next hour", type="primary")
        if predict:
            row = pd.DataFrame([inputs], columns=FEATURES)
            scaled = scaler.transform(row)
            predicted = float(np.maximum(model.predict(scaled), 0.0)[0])
            actual = defaults.get(TARGET)
            st.session_state["last_result"] = {
                "inputs": inputs,
                "predicted": predicted,
                "actual": actual if use_sample else None,
                "use_sample": use_sample,
                "sample_index": int(sample_index),
            }

        result = st.session_state.get("last_result")
        if result:
            predicted = result["predicted"]
            gap_now = result["inputs"]["Global_active_power"]
            actual = result["actual"]

            st.subheader("2. Predicted electricity consumption")
            m1, m2, m3 = st.columns(3)
            m1.metric("Current hour (kW)", f"{gap_now:.3f}")
            m2.metric(
                "ANN predicted next hour (kW)",
                f"{predicted:.3f}",
                delta=f"{predicted - gap_now:.3f} vs now",
            )
            if actual is not None and not pd.isna(actual):
                m3.metric(
                    "Actual next hour in test CSV (kW)",
                    f"{float(actual):.3f}",
                    help="Saved target for this X_test row, not a claimed accuracy score.",
                )
            else:
                m3.metric(
                    "Day / hour",
                    f"{DAY_NAMES[int(result['inputs']['day_of_week'])]}, "
                    f"{int(result['inputs']['hour']):02d}:00",
                )

            st.subheader("3. Electricity-consumption graphs")
            g1, g2 = st.columns(2)
            with g1:
                fig1 = plot_current_vs_predicted(gap_now, predicted, actual)
                st.pyplot(fig1)
                plt.close(fig1)
            with g2:
                fig2 = plot_submeters(
                    result["inputs"]["Sub_metering_1"],
                    result["inputs"]["Sub_metering_2"],
                    result["inputs"]["Sub_metering_3"],
                )
                st.pyplot(fig2)
                plt.close(fig2)

            if result["use_sample"]:
                fig3 = plot_nearby_hours(
                    samples, result["sample_index"], model, scaler
                )
                st.pyplot(fig3)
                plt.close(fig3)

            st.subheader("4. Personalized energy-saving suggestions")
            st.caption(
                "Workflow: ANN prediction + hour + sub-meter usage → rule-based tips. "
                "The tips are project advice, not a measured saving result."
            )
            tips, reasons = make_recommendations(result["inputs"], predicted)
            for tip in tips:
                st.write(f"- {tip}")
            with st.expander("Why these tips were shown"):
                for reason in reasons:
                    st.write(f"- {reason}")

    with tab_docs:
        st.markdown(
            """
**What this prototype does**

1. Reads your prepared hourly data (original files are not changed).
2. Applies the notebook preprocessing: keep 9 features, convert to numbers, drop invalid rows, scale with `StandardScaler`.
3. Uses the same ANN as Member 2: `MLPRegressor` with hidden layers (64, 32), ReLU, Adam, 50 iterations.
4. Predicts `target_next_hour` (next-hour electricity consumption).
5. Shows graphs and personalized tips from the prediction and household features.

**Files that are new (safe to submit with the prototype)**

- `app.py` — Streamlit user interface
- `train_model.py` — trains/saves the ANN without editing original CSVs
- `requirements.txt` — Python packages
- `artifacts/` — created after the first run (`ann_model.joblib`, `scaler.joblib`, `metrics.json`)

**How to run**

```text
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

The first app launch can also train the model if `artifacts/` is missing. Training may take a few minutes.

**Notebook test results (from `sc_project.ipynb` output, not invented)**

- ANN: MAE 0.3405, RMSE 0.4823, R² 0.5795
- Baseline (use current-hour power as the next-hour guess): MAE 0.3968, RMSE 0.6027, R² 0.3435

If this computer retrains the model, the sidebar also shows those newly computed numbers. They can differ slightly from Colab even with the same settings.
            """
        )


if __name__ == "__main__":
    main()
