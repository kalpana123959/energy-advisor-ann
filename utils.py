"""
Shared logic for every page of the Adaptive Personalized Energy Advisor.

Nothing here changes your ANN, scaler, CSV files or train_model.py.
It only LOADS them and reuses the same functions that used to live in app.py.
"""

from pathlib import Path
import json

import joblib
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

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SUBMETER_NAMES = {
    "Sub_metering_1": "Kitchen",
    "Sub_metering_2": "Laundry",
    "Sub_metering_3": "Water heater / AC",
}
SUBMETER_HELP = {
    "Sub_metering_1": "Kitchen (dishwasher, oven, microwave)",
    "Sub_metering_2": "Laundry (washing machine, tumble-dryer, refrigerator)",
    "Sub_metering_3": "Water heater and air-conditioner",
}

# ---------------------------------------------------------------- styling ----
CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
[data-testid="stSidebar"] {background: #0f1d3a;}
[data-testid="stSidebar"] * {color: #e8eefc !important;}
h1, h2, h3 {color: #0f172a; letter-spacing: -0.3px;}
.hero {background: linear-gradient(120deg, #0f1d3a 0%, #2563eb 100%);
       color: #fff; padding: 22px 28px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0; font-size: 1.8rem;}
.hero p {color: #cfe0ff; margin: 4px 0 0 0;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e3e8f2;
       border-left: 5px solid #2563eb; padding: 14px 16px; border-radius: 12px;
       box-shadow: 0 1px 3px rgba(15,23,42,.06);}
.card {background:#fff; border:1px solid #e3e8f2; border-radius:14px;
       padding:16px 18px; margin-bottom:12px; box-shadow:0 1px 3px rgba(15,23,42,.05);}
.tip {background:#fff; border-left:5px solid #16a34a; border-radius:10px;
      padding:12px 16px; margin-bottom:10px; box-shadow:0 1px 3px rgba(15,23,42,.05);}
.tip.warn {border-left-color:#f59e0b;}
.tip.alert {border-left-color:#dc2626;}
.badge {display:inline-block; padding:3px 12px; border-radius:999px;
        font-weight:600; font-size:.85rem;}
.badge.low {background:#dcfce7; color:#166534;}
.badge.medium {background:#fef3c7; color:#92400e;}
.badge.high {background:#fee2e2; color:#991b1b;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


# ------------------------------------------------------------ settings ------
def get_settings():
    """User settings kept in session_state (edited on the Settings page)."""
    defaults = {
        "currency": "₹",
        "tariff": 7.0,
        "low_pct": 33,       # percentile limits used for Low / Medium / High
        "high_pct": 66,
        "peak_start": 17,
        "peak_end": 22,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
    return {k: st.session_state[k] for k in defaults}


# ------------------------------------------------------- model + data -------
@st.cache_resource(show_spinner="Loading preprocessing and ANN...")
def load_or_train_model():
    """Same behaviour as before: load saved ANN, retrain if missing/incompatible."""
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
                raise RuntimeError("Saved model built with a different scikit-learn version")
            model.predict(scaler.transform(pd.DataFrame([[1.0] * len(FEATURES)], columns=FEATURES)))
            return model, scaler, metrics
    except Exception:
        pass
    metrics = train_and_save()
    return joblib.load(model_path), joblib.load(scaler_path), metrics


@st.cache_data
def load_hourly_data():
    df = pd.read_csv(BASE_DIR / "prepared_hourly_data.csv", parse_dates=["datetime"])
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


@st.cache_data
def load_test_sample_table():
    x_test_path, y_test_path = BASE_DIR / "X_test.csv", BASE_DIR / "y_test.csv"
    if x_test_path.exists() and y_test_path.exists():
        sample = pd.read_csv(x_test_path)
        sample[TARGET] = pd.read_csv(y_test_path)[TARGET]
        return sample.reset_index(drop=True)
    data = load_and_prepare()
    _, x_test, _, y_test = chronological_split(data)
    sample = x_test.copy()
    sample[TARGET] = y_test.values
    return sample.reset_index(drop=True)


def predict_next_hour(model, scaler, inputs):
    """Real ANN prediction for one set of inputs (dict with the 9 FEATURES)."""
    row = pd.DataFrame([inputs], columns=FEATURES)
    return float(np.maximum(model.predict(scaler.transform(row)), 0.0)[0])


def predict_many(model, scaler, frame):
    return np.maximum(model.predict(scaler.transform(frame[FEATURES])), 0.0)


def split_index(df):
    return int(len(df) * 0.80)


def consumption_level(value, df, settings):
    """Low / Medium / High from percentile limits of the real hourly data."""
    lo = float(np.percentile(df["Global_active_power"], settings["low_pct"]))
    hi = float(np.percentile(df["Global_active_power"], settings["high_pct"]))
    if value < lo:
        return "Low", lo, hi
    if value > hi:
        return "High", lo, hi
    return "Medium", lo, hi


def level_badge(level):
    return f'<span class="badge {level.lower()}">{level}</span>'


# ------------------------------------------------ recommendations (reused) --
def make_recommendations(inputs, predicted):
    """Unchanged logic from your original app.py."""
    tips, reasons = [], []
    current = float(inputs["Global_active_power"])
    hour = int(inputs["hour"])
    kitchen = float(inputs["Sub_metering_1"])
    laundry = float(inputs["Sub_metering_2"])
    climate = float(inputs["Sub_metering_3"])
    voltage = float(inputs["Voltage"])
    intensity = float(inputs["Global_intensity"])
    change_pct = 0.0 if current == 0 else 100.0 * (predicted - current) / current

    if predicted > current * 1.15:
        tips.append("Next-hour use is predicted to rise. Finish high-power tasks soon, or delay extra appliances.")
        reasons.append(f"ANN prediction is {change_pct:.1f}% above current-hour power.")
    elif predicted < current * 0.85:
        tips.append("Next-hour use is predicted to fall. If a chore cannot wait until late night, this lower window is the better choice.")
        reasons.append(f"ANN prediction is {abs(change_pct):.1f}% below current-hour power.")
    else:
        tips.append("Next-hour use is predicted to stay close to now. Keep only necessary appliances running.")
        reasons.append("ANN prediction is close to current-hour power.")

    if 17 <= hour <= 22:
        tips.append("This is a typical evening peak (5–10 pm). Avoid overlapping the oven, washing machine, and water heater.")
        reasons.append(f"Hour = {hour}, which is an evening peak period in this dataset.")
    elif 0 <= hour <= 6:
        tips.append("This is an off-peak night/early-morning hour. Shift flexible loads here when you can.")
        reasons.append(f"Hour = {hour}, which is an off-peak period in this dataset.")

    highest = max(("kitchen", kitchen), ("laundry", laundry), ("water heater / AC", climate), key=lambda i: i[1])
    if highest[1] >= 10:
        if highest[0] == "kitchen":
            tips.append("Kitchen use is high. Batch cooking, skip unused oven preheat, and run the dishwasher only when full.")
        elif highest[0] == "laundry":
            tips.append("Laundry use is high. Use a cold wash when possible and air-dry instead of the dryer.")
        else:
            tips.append("Water-heater / AC use is high. Raise the AC setpoint a little, or heat water only as needed.")
        reasons.append(f"Highest sub-meter is {highest[0]} at {highest[1]:.2f}.")
    elif kitchen + laundry + climate < 2 and current > 1.5:
        tips.append("The three sub-meters are low, but whole-house power is still high. Check lights, electronics, and other unmetered plug loads.")
        reasons.append("Sub-meters are low while Global_active_power is still above 1.5 kW.")

    if voltage < 230:
        tips.append("Voltage is on the low side of this dataset. Reduce extra loads until voltage returns closer to the usual household range.")
        reasons.append(f"Voltage = {voltage:.2f} V.")
    if intensity > 15:
        tips.append("Current (global intensity) is high. Turn off unused high-draw devices.")
        reasons.append(f"Global intensity = {intensity:.2f} A.")
    if not tips:
        tips.append("Usage looks moderate. Continue switching off idle devices.")
        reasons.append("No high-load rule was triggered.")
    return tips, reasons


def tip_html(text, kind=""):
    return f'<div class="tip {kind}">{text}</div>'


# --------------------------------------------------- weekly report (reused) -
def available_mondays(df, test_only=True):
    pool = df[df["datetime"] >= df["datetime"].iloc[split_index(df)]] if test_only else df
    mondays = sorted({(t - pd.Timedelta(days=int(t.dayofweek))).normalize() for t in pool["datetime"]})
    return [m for m in mondays
            if len(df[(df["datetime"] >= m) & (df["datetime"] < m + pd.Timedelta(days=7))]) >= 120]


def build_weekly_report(df, week_start, model, scaler):
    """Unchanged logic from your original app.py."""
    week_start = pd.Timestamp(week_start)
    week_end = week_start + pd.Timedelta(days=7)
    cur = df[(df["datetime"] >= week_start) & (df["datetime"] < week_end)].copy()
    prev = df[(df["datetime"] >= week_start - pd.Timedelta(days=7)) & (df["datetime"] < week_start)].copy()
    if cur.empty:
        return None

    cur["kwh"] = cur["Global_active_power"]  # hourly mean kW ~ kWh for that hour
    cur["date"] = cur["datetime"].dt.date
    daily = cur.groupby("date")["kwh"].sum()
    hourly_profile = cur.groupby("hour")["kwh"].mean()

    total = float(cur["kwh"].sum())
    prev_total = float(prev["Global_active_power"].sum()) if len(prev) else None
    change_pct = 100.0 * (total - prev_total) / prev_total if prev_total and prev_total > 0 else None

    sm_cols = list(SUBMETER_NAMES)
    sm_totals = cur[sm_cols].sum()
    sm_sum = float(sm_totals.sum())
    sm_share = (sm_totals / sm_sum * 100.0) if sm_sum > 0 else sm_totals * 0.0

    evening = cur[(cur["hour"] >= 17) & (cur["hour"] <= 22)]["kwh"].sum()
    night = cur[cur["hour"] <= 6]["kwh"].sum()

    cur["ann_pred"] = predict_many(model, scaler, cur)
    mae = float(np.mean(np.abs(cur["ann_pred"] - cur[TARGET])))

    wk = cur["day_of_week"] < 5
    weekday_avg = float(cur.loc[wk, "kwh"].mean()) if wk.any() else 0.0
    weekend_avg = float(cur.loc[~wk, "kwh"].mean()) if (~wk).any() else 0.0

    return {
        "data": cur, "daily": daily, "hourly_profile": hourly_profile,
        "total": total, "prev_total": prev_total, "change_pct": change_pct,
        "avg_hourly": float(cur["kwh"].mean()),
        "peak_hour": int(hourly_profile.idxmax()), "peak_value": float(hourly_profile.max()),
        "highest_day": daily.idxmax(), "highest_day_kwh": float(daily.max()),
        "lowest_day": daily.idxmin(), "lowest_day_kwh": float(daily.min()),
        "sm_share": sm_share,
        "evening_share": float(100.0 * evening / total) if total > 0 else 0.0,
        "night_share": float(100.0 * night / total) if total > 0 else 0.0,
        "ann_mae": mae, "weekday_avg": weekday_avg, "weekend_avg": weekend_avg,
        "hours_covered": int(len(cur)),
    }


def weekly_recommendations(rep):
    """Unchanged logic from your original app.py."""
    tips = []
    if rep["change_pct"] is not None:
        if rep["change_pct"] > 10:
            tips.append(f"Consumption rose {rep['change_pct']:.1f}% compared with last week. Review which appliances were used more often.")
        elif rep["change_pct"] < -10:
            tips.append(f"Well done: consumption fell {abs(rep['change_pct']):.1f}% compared with last week. Keep up the same habits.")
        else:
            tips.append("Weekly consumption is stable compared with last week.")
    if rep["evening_share"] > 40:
        tips.append(f"{rep['evening_share']:.0f}% of this week's energy was used between 5 pm and 10 pm. Spread heavy appliances (washing machine, oven, water heater) across the day.")
    if rep["night_share"] > 25:
        tips.append(f"{rep['night_share']:.0f}% of energy was used between midnight and 6 am. Check for devices left running overnight (standby loads).")
    top = rep["sm_share"].idxmax()
    if rep["sm_share"][top] > 0:
        tips.append(f"The {SUBMETER_NAMES[top].lower()} circuit had the largest share of sub-metered energy ({rep['sm_share'][top]:.0f}%). Start your savings there.")
    if rep["weekend_avg"] > rep["weekday_avg"] * 1.15:
        tips.append("Weekends use noticeably more power than weekdays. Plan chores to avoid overlapping loads.")
    elif rep["weekday_avg"] > rep["weekend_avg"] * 1.15:
        tips.append("Weekday usage is higher than weekends. Check what runs while the house is empty.")
    if not tips:
        tips.append("Usage looks balanced this week. Continue switching off idle devices.")
    return tips


# ------------------------------------------------------ adaptive learning ---
ADAPTED_MODEL_PATH = ARTIFACT_DIR / "adapted_model.joblib"


def evaluate(model, scaler, frame):
    """Real MAE / RMSE / R2 of a model on a frame that has FEATURES + TARGET."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    pred = predict_many(model, scaler, frame)
    y = frame[TARGET].to_numpy()
    return {"MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred))}


def adapt_model(model, scaler, new_data, epochs=5):
    """Fine-tune a COPY of the ANN on new data with partial_fit (Adam keeps going
    from the existing weights). The original model object is not modified."""
    import copy
    adapted = copy.deepcopy(model)
    X = scaler.transform(new_data[FEATURES])
    y = new_data[TARGET].to_numpy()
    for _ in range(int(epochs)):
        adapted.partial_fit(X, y)
    return adapted
