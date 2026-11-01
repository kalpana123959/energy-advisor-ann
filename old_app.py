"""
Adaptive Personalized Energy Advisor: Streamlit application.

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


@st.cache_resource(show_spinner="Loading preprocessing and ANN...")
def load_or_train_model():
    """Load the saved ANN. If the saved files are missing or were created with a
    different scikit-learn version (a common reason the app fails to open),
    retrain automatically and save fresh files."""
    model_path = ARTIFACT_DIR / "ann_model.joblib"
    scaler_path = ARTIFACT_DIR / "scaler.joblib"
    metrics_path = ARTIFACT_DIR / "metrics.json"

    try:
        if model_path.exists() and scaler_path.exists() and metrics_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            import sklearn
            if metrics.get("sklearn_version") != sklearn.__version__:
                raise RuntimeError("Saved model was built with a different scikit-learn version")
            # quick sanity check that the loaded objects really work
            model.predict(scaler.transform(pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)))
            return model, scaler, metrics
    except Exception:
        pass  # fall through and retrain

    metrics = train_and_save()
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler, metrics


@st.cache_data
def load_hourly_data():
    """Full hourly dataset with timestamps, used for the weekly report."""
    path = BASE_DIR / "prepared_hourly_data.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df


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


def build_weekly_report(df, week_start, model, scaler):
    """Summarise one Monday-to-Sunday week and compare it with the previous week."""
    week_start = pd.Timestamp(week_start)
    week_end = week_start + pd.Timedelta(days=7)
    cur = df[(df["datetime"] >= week_start) & (df["datetime"] < week_end)].copy()
    prev = df[
        (df["datetime"] >= week_start - pd.Timedelta(days=7))
        & (df["datetime"] < week_start)
    ].copy()
    if cur.empty:
        return None

    # Global_active_power is the hourly average in kW, so each row is about 1 kWh per kW.
    cur["kwh"] = cur["Global_active_power"]
    cur["date"] = cur["datetime"].dt.date
    daily = cur.groupby("date")["kwh"].sum()
    hourly_profile = cur.groupby("hour")["kwh"].mean()

    total = float(cur["kwh"].sum())
    prev_total = float(prev["Global_active_power"].sum()) if len(prev) else None
    change_pct = (
        100.0 * (total - prev_total) / prev_total if prev_total and prev_total > 0 else None
    )

    sm_cols = ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]
    sm_totals = cur[sm_cols].sum()
    sm_sum = float(sm_totals.sum())
    sm_share = (sm_totals / sm_sum * 100.0) if sm_sum > 0 else sm_totals * 0.0

    evening = cur[(cur["hour"] >= 17) & (cur["hour"] <= 22)]["kwh"].sum()
    night = cur[cur["hour"] <= 6]["kwh"].sum()

    scaled = scaler.transform(cur[FEATURES])
    cur["ann_pred"] = np.maximum(model.predict(scaled), 0.0)
    mae = float(np.mean(np.abs(cur["ann_pred"] - cur[TARGET])))

    weekday_mask = cur["day_of_week"] < 5
    weekday_avg = float(cur.loc[weekday_mask, "kwh"].mean()) if weekday_mask.any() else 0.0
    weekend_avg = float(cur.loc[~weekday_mask, "kwh"].mean()) if (~weekday_mask).any() else 0.0

    return {
        "data": cur,
        "daily": daily,
        "hourly_profile": hourly_profile,
        "total": total,
        "prev_total": prev_total,
        "change_pct": change_pct,
        "avg_hourly": float(cur["kwh"].mean()),
        "peak_hour": int(hourly_profile.idxmax()),
        "peak_value": float(hourly_profile.max()),
        "highest_day": daily.idxmax(),
        "highest_day_kwh": float(daily.max()),
        "lowest_day": daily.idxmin(),
        "lowest_day_kwh": float(daily.min()),
        "sm_share": sm_share,
        "evening_share": float(100.0 * evening / total) if total > 0 else 0.0,
        "night_share": float(100.0 * night / total) if total > 0 else 0.0,
        "ann_mae": mae,
        "weekday_avg": weekday_avg,
        "weekend_avg": weekend_avg,
        "hours_covered": int(len(cur)),
    }


def weekly_recommendations(rep):
    tips = []
    if rep["change_pct"] is not None:
        if rep["change_pct"] > 10:
            tips.append(
                f"Consumption rose {rep['change_pct']:.1f}% compared with last week. "
                "Review which appliances were used more often."
            )
        elif rep["change_pct"] < -10:
            tips.append(
                f"Well done: consumption fell {abs(rep['change_pct']):.1f}% compared with last week. "
                "Keep up the same habits."
            )
        else:
            tips.append("Weekly consumption is stable compared with last week.")

    if rep["evening_share"] > 40:
        tips.append(
            f"{rep['evening_share']:.0f}% of this week's energy was used between 5 pm and 10 pm. "
            "Spread heavy appliances (washing machine, oven, water heater) across the day."
        )
    if rep["night_share"] > 25:
        tips.append(
            f"{rep['night_share']:.0f}% of energy was used between midnight and 6 am. "
            "Check for devices left running overnight (standby loads)."
        )

    names = {
        "Sub_metering_1": "kitchen",
        "Sub_metering_2": "laundry",
        "Sub_metering_3": "water heater / AC",
    }
    top = rep["sm_share"].idxmax()
    if rep["sm_share"][top] > 0:
        tips.append(
            f"The {names[top]} circuit had the largest share of sub-metered energy "
            f"({rep['sm_share'][top]:.0f}%). Start your savings there."
        )

    if rep["weekend_avg"] > rep["weekday_avg"] * 1.15:
        tips.append("Weekends use noticeably more power than weekdays. Plan chores to avoid overlapping loads.")
    elif rep["weekday_avg"] > rep["weekend_avg"] * 1.15:
        tips.append("Weekday usage is higher than weekends. Check what runs while the house is empty.")

    if not tips:
        tips.append("Usage looks balanced this week. Continue switching off idle devices.")
    return tips


def plot_daily_totals(daily):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = [pd.Timestamp(d).strftime("%a %d %b") for d in daily.index]
    ax.bar(labels, daily.values, color="#5b8def")
    ax.set_ylabel("Energy used (kWh)")
    ax.set_title("Daily consumption this week")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_hourly_profile(profile):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(profile.index, profile.values, marker="o", color="#2bb673")
    ax.axvspan(17, 22, color="#f4a261", alpha=0.2, label="Evening peak (5-10 pm)")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average power (kW)")
    ax.set_title("Typical day this week")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_week_actual_vs_pred(data):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(data["datetime"], data[TARGET], label="Actual next hour", linewidth=1.2)
    ax.plot(data["datetime"], data["ann_pred"], label="ANN prediction", linewidth=1.2, alpha=0.85)
    ax.set_ylabel("Electricity consumption (kW)")
    ax.set_title("Actual vs ANN prediction for the week")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_submeter_share(share):
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(
        ["Kitchen", "Laundry", "Water heater / AC"],
        share.values,
        color=["#e76f51", "#2a9d8f", "#264653"],
    )
    ax.set_ylabel("Share of sub-metered energy (%)")
    ax.set_title("Where the sub-metered energy went")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def render_weekly_report(model, scaler):
    st.subheader("Weekly electricity report")
    st.caption(
        "Pick a week from the held-out test period (data the ANN did not train on). "
        "Weeks run Monday to Sunday."
    )
    df = load_hourly_data()
    split_time = df["datetime"].iloc[int(len(df) * 0.80)]
    test_df = df[df["datetime"] >= split_time]
    mondays = sorted(
        {
            (t - pd.Timedelta(days=int(t.dayofweek))).normalize()
            for t in test_df["datetime"]
        }
    )
    # keep only weeks with enough data (at least 5 full days of hours)
    mondays = [
        m for m in mondays
        if len(df[(df["datetime"] >= m) & (df["datetime"] < m + pd.Timedelta(days=7))]) >= 120
    ]
    if not mondays:
        st.warning("Not enough data to build a weekly report.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        week_start = st.selectbox(
            "Select week",
            options=mondays,
            index=len(mondays) - 1,
            format_func=lambda m: f"{m:%d %b %Y} to {(m + pd.Timedelta(days=6)):%d %b %Y}",
        )
    with c2:
        tariff = st.number_input(
            "Electricity price (per kWh)",
            min_value=0.0,
            value=7.0,
            step=0.5,
            help="Used only for the cost estimate. Enter your own tariff.",
        )

    rep = build_weekly_report(df, week_start, model, scaler)
    if rep is None:
        st.warning("No data for this week.")
        return

    k1, k2, k3, k4 = st.columns(4)
    delta = None if rep["change_pct"] is None else f"{rep['change_pct']:+.1f}% vs last week"
    k1.metric("Total energy (kWh)", f"{rep['total']:.1f}", delta=delta, delta_color="inverse")
    k2.metric("Estimated cost", f"{rep['total'] * tariff:,.0f}")
    k3.metric("Average per hour (kW)", f"{rep['avg_hourly']:.2f}")
    k4.metric("Peak hour", f"{rep['peak_hour']:02d}:00", help=f"Average {rep['peak_value']:.2f} kW")

    d1, d2 = st.columns(2)
    d1.write(
        f"**Highest day:** {pd.Timestamp(rep['highest_day']):%A %d %b} "
        f"({rep['highest_day_kwh']:.1f} kWh)"
    )
    d2.write(
        f"**Lowest day:** {pd.Timestamp(rep['lowest_day']):%A %d %b} "
        f"({rep['lowest_day_kwh']:.1f} kWh)"
    )
    if rep["hours_covered"] < 168:
        st.caption(f"Note: this week has {rep['hours_covered']} of 168 hourly readings, so totals are partial.")

    g1, g2 = st.columns(2)
    with g1:
        fig = plot_daily_totals(rep["daily"])
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        fig = plot_hourly_profile(rep["hourly_profile"])
        st.pyplot(fig)
        plt.close(fig)

    g3, g4 = st.columns(2)
    with g3:
        fig = plot_submeter_share(rep["sm_share"])
        st.pyplot(fig)
        plt.close(fig)
    with g4:
        st.write("**Usage split**")
        st.write(f"- Evening peak (5-10 pm): **{rep['evening_share']:.0f}%** of the week's energy")
        st.write(f"- Night (12-6 am): **{rep['night_share']:.0f}%**")
        st.write(f"- Weekday average: **{rep['weekday_avg']:.2f} kW**")
        st.write(f"- Weekend average: **{rep['weekend_avg']:.2f} kW**")
        st.write(f"- ANN error this week (MAE): **{rep['ann_mae']:.3f} kW**")

    fig = plot_week_actual_vs_pred(rep["data"])
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Weekly energy-saving suggestions")
    for tip in weekly_recommendations(rep):
        st.write(f"- {tip}")

    st.subheader("Estimated savings (illustrative)")
    pct = st.slider("If you cut evening-peak usage by (%)", 0, 30, 10)
    saved_kwh = rep["total"] * rep["evening_share"] / 100.0 * pct / 100.0
    s1, s2 = st.columns(2)
    s1.metric("Energy saved per week (kWh)", f"{saved_kwh:.1f}")
    s2.metric("Cost saved per week", f"{saved_kwh * tariff:,.0f}")
    st.caption("This is a what-if estimate from your own usage pattern, not a measured result.")

    report_df = pd.DataFrame(
        {
            "Metric": [
                "Week start", "Total kWh", "Previous week kWh", "Change vs last week (%)",
                "Average kW per hour", "Peak hour", "Highest day", "Lowest day",
                "Evening share (%)", "Night share (%)", "ANN MAE (kW)",
            ],
            "Value": [
                f"{pd.Timestamp(week_start):%Y-%m-%d}", f"{rep['total']:.2f}",
                "n/a" if rep["prev_total"] is None else f"{rep['prev_total']:.2f}",
                "n/a" if rep["change_pct"] is None else f"{rep['change_pct']:.1f}",
                f"{rep['avg_hourly']:.3f}", f"{rep['peak_hour']:02d}:00",
                f"{pd.Timestamp(rep['highest_day']):%Y-%m-%d}",
                f"{pd.Timestamp(rep['lowest_day']):%Y-%m-%d}",
                f"{rep['evening_share']:.1f}", f"{rep['night_share']:.1f}", f"{rep['ann_mae']:.3f}",
            ],
        }
    )
    st.download_button(
        "Download weekly report (CSV)",
        report_df.to_csv(index=False).encode("utf-8"),
        file_name=f"weekly_report_{pd.Timestamp(week_start):%Y-%m-%d}.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(
        page_title="Adaptive Personalized Energy Advisor",
        page_icon="⚡",
        layout="wide",
    )
    st.title("Adaptive Personalized Energy Advisor")
    st.caption(
        "ANN-based next-hour prediction, consumption graphs, weekly reports "
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

    tab_predict, tab_weekly, tab_docs = st.tabs(["Advisor app", "Weekly report", "About"])

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

    with tab_weekly:
        render_weekly_report(model, scaler)

    with tab_docs:
        st.markdown(
            """
**What this prototype does**

1. Reads your prepared hourly data (original files are not changed).
2. Applies the notebook preprocessing: keep 9 features, convert to numbers, drop invalid rows, scale with `StandardScaler`.
3. Uses an ANN (`MLPRegressor`) with hidden layers (64, 32), ReLU, Adam, 50 iterations.
4. Predicts `target_next_hour` (next-hour electricity consumption).
5. Shows graphs and personalized tips from the prediction and household features.

**Project files**

- `app.py` — Streamlit user interface
- `train_model.py` — trains/saves the ANN without editing original CSVs
- `requirements.txt` — Python packages
- `artifacts/` — saved model files, recreated automatically if missing or incompatible (`ann_model.joblib`, `scaler.joblib`, `metrics.json`)

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
