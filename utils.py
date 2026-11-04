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
.stApp {background: #fbf8f1;}
.block-container {padding-top: 1.4rem; max-width: 1200px;}
h1, h2, h3, h4 {color: #1f2937; letter-spacing: -0.2px;}
p, li, label, span {color: #1f2937;}
.hero {background: #e8f3ea; border: 1px solid #d3e6d8; color: #1f2937;
       padding: 18px 24px; border-radius: 16px; margin-bottom: 16px;}
.hero h1 {color: #1f2937; margin: 0; font-size: 1.7rem;}
.hero p {color: #4b5b52; margin: 4px 0 0 0; font-size: 1rem;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e4e9df;
       border-left: 5px solid #7fbf9e; padding: 12px 14px; border-radius: 12px;
       box-shadow: 0 1px 3px rgba(60,80,60,.06);}
.tip {background: #ffffff; border: 1px solid #e4e9df; border-left: 5px solid #7fbf9e;
      border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; color: #1f2937;
      font-size: 1.02rem; line-height: 1.5;}
.tip.warn {border-left-color: #e6c36a;}
.tip.alert {border-left-color: #e29a8e;}
.badge {display: inline-block; padding: 3px 14px; border-radius: 999px;
        font-weight: 600; font-size: .9rem;}
.badge.low {background: #dff1e4; color: #25613b;}
.badge.medium {background: #fbf0cf; color: #7a5b13;}
.badge.high {background: #f9dcd7; color: #8a3a2f;}
.note {color: #5b6b62; font-size: .92rem;}
.kpi-panel {display: flex; flex-wrap: wrap; background: #ffffff; border: 1px solid #e4e9df;
            border-radius: 14px; padding: 6px 0; margin: 4px 0 16px 0; box-shadow: 0 1px 3px rgba(60,80,60,.05);}
.kpi {flex: 1 1 150px; padding: 10px 18px; border-right: 1px solid #edf0e8; min-width: 140px;}
.kpi:last-child {border-right: none;}
.kpi-l {font-size: .82rem; color: #5b6b62; margin-bottom: 2px;}
.kpi-v {font-size: 1.55rem; font-weight: 700; color: #1f2937; line-height: 1.25;}
.kpi-d {font-size: .8rem; margin-top: 2px; color: #6b7a70;}
.kpi-d.good {color: #25834f;}
.kpi-d.bad {color: #b5483a;}
[data-testid="stSidebar"] {background: #eef6f0;}
[data-testid="stSidebar"] h3 {margin-top: 0;}
@media (max-width: 640px) {
  .block-container {padding-left: .9rem; padding-right: .9rem;}
  .hero {padding: 14px 16px;}
  .hero h1 {font-size: 1.35rem;}
  .kpi {border-right: none; flex-basis: 45%;}
}
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
    """Same rules and limits as before. Only the wording is simpler."""
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
        tips.append("You will probably use more power in the next hour. Finish heavy work soon, "
                    "and do not switch on extra appliances at the same time.")
        reasons.append(f"The model expects {change_pct:.1f}% more power than this hour.")
    elif predicted < current * 0.85:
        tips.append("You will probably use less power in the next hour. If a chore can wait "
                    "(like washing clothes), this is a better time than a busy hour.")
        reasons.append(f"The model expects {abs(change_pct):.1f}% less power than this hour.")
    else:
        tips.append("Your power use will stay about the same. Keep only the appliances you need "
                    "switched on, and switch off lights when you leave a room.")
        reasons.append("The model expects about the same power as this hour.")

    if 17 <= hour <= 22:
        tips.append("It is evening (5 pm to 10 pm), when most homes use the most power. "
                    "Do not run the oven, washing machine and water heater all at the same time.")
        reasons.append(f"The hour is {hour}:00, which is an evening busy time in this data.")
    elif 0 <= hour <= 6:
        tips.append("It is night or early morning, when power use is usually low. "
                    "This is a good time for chores that can wait, like running the washing machine.")
        reasons.append(f"The hour is {hour}:00, which is a quiet time in this data.")

    highest = max(("kitchen", kitchen), ("laundry", laundry), ("water heater / AC", climate),
                  key=lambda i: i[1])
    if highest[1] >= 10:
        if highest[0] == "kitchen":
            tips.append("Kitchen use is high. Cook several things together, do not heat the oven "
                        "longer than needed, and run the dishwasher only with a full load.")
        elif highest[0] == "laundry":
            tips.append("Laundry use is high. Use the washing machine only with a full load, "
                        "wash in cold water when you can, and dry clothes in the air instead of a dryer.")
        else:
            tips.append("Water heater or AC use is high. Set the AC a little warmer (around 24 to 26 °C) "
                        "and heat water only when you need it.")
        reasons.append(f"The {highest[0]} meter has the highest reading ({highest[1]:.2f}).")
    elif kitchen + laundry + climate < 2 and current > 1.5:
        tips.append("The kitchen, laundry and heater/AC meters are low, but the house still uses a lot "
                    "of power. Check lights, TV, computers and chargers, and switch off what you are not using.")
        reasons.append("The three meters are low, but total power is above 1.5 kW.")

    if voltage < 230:
        tips.append("The voltage is a little low. Switch off a few appliances you do not need "
                    "until it comes back to normal.")
        reasons.append(f"Voltage is {voltage:.2f} V.")
    if intensity > 15:
        tips.append("The house is drawing a lot of current. Switch off appliances that nobody is using.")
        reasons.append(f"Current is {intensity:.2f} A.")
    if not tips:
        tips.append("Your use looks normal. Keep switching off things you are not using.")
        reasons.append("No high-use rule was triggered.")
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
    """Same rules as before. Only the wording is simpler."""
    tips = []
    if rep["change_pct"] is not None:
        if rep["change_pct"] > 10:
            tips.append(f"You used {rep['change_pct']:.1f}% more power than last week. "
                        "Think about which appliances you used more often.")
        elif rep["change_pct"] < -10:
            tips.append(f"Well done! You used {abs(rep['change_pct']):.1f}% less power than last week. "
                        "Keep up the good habits.")
        else:
            tips.append("Your power use is about the same as last week.")
    if rep["evening_share"] > 40:
        tips.append(f"{rep['evening_share']:.0f}% of this week's power was used between 5 pm and 10 pm. "
                    "Try not to run the washing machine, oven and water heater at the same time.")
    if rep["night_share"] > 25:
        tips.append(f"{rep['night_share']:.0f}% of the power was used between midnight and 6 am. "
                    "Check if something is left on at night, like a TV, charger or heater.")
    top = rep["sm_share"].idxmax()
    if rep["sm_share"][top] > 0:
        tips.append(f"The {SUBMETER_NAMES[top].lower()} meter used the most power ({rep['sm_share'][top]:.0f}% "
                    "of the three meters). Start saving there.")
    if rep["weekend_avg"] > rep["weekday_avg"] * 1.15:
        tips.append("You use more power on weekends than on weekdays. Spread out chores like washing and cooking.")
    elif rep["weekday_avg"] > rep["weekend_avg"] * 1.15:
        tips.append("You use more power on weekdays. Check what stays switched on when nobody is home.")
    if not tips:
        tips.append("Your use looks balanced this week. Keep switching off things you are not using.")
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


# ------------------------------------------------------------ navigation ----
# route key -> (icon, title, short description, file inside views/)
SECTIONS = {
    "dashboard":  ("🏠", "Dashboard", "A quick summary of your power use, cost and how well the model works."),
    "analytics":  ("📊", "Energy Analytics", "See when and where your power is used."),
    "weekly":     ("🗓️", "Weekly Report", "Check one week at a time and get saving tips."),
    "prediction": ("🔮", "Energy Prediction", "Find out how much power you may use in the next hour."),
    "advice":     ("💡", "Personalized Advice", "Simple tips to save power, based on your prediction."),
    "calculator": ("🧮", "Energy Calculator", "Work out the power and cost of your appliances."),
    "adaptive":   ("🧠", "Adaptive Learning", "Teach the model with newer data and see if it improves."),
    "settings":   ("⚙️", "Settings", "Set your price per unit and your Low / Medium / High limits."),
}
ROUTE_FILES = {
    "dashboard": "dashboard.py", "analytics": "analytics.py", "weekly": "weekly_report.py",
    "prediction": "prediction.py", "advice": "advice.py", "calculator": "calculator.py",
    "adaptive": "adaptive.py", "settings": "settings.py",
}


def go(route):
    """Used as a button callback: switch the visible page."""
    st.session_state["route"] = route


# ------------------------------------------------ charts (Streamlit's own style) ---
CHART_H = 260
PIE_COLORS = ["#5aa886", "#6fa8dc", "#e0b66a", "#c98f8f", "#9b8fc9", "#7fc4c4", "#c9b38f", "#aab4a0"]


def _native(fn, data, **kw):
    """Call a Streamlit chart with optional extras; drop the extras on older versions."""
    for attempt in (kw, {k: v for k, v in kw.items() if k in ("height",)}, {}):
        try:
            return fn(data, **attempt)
        except TypeError:
            continue
    return fn(data)


def bar_chart(data, title="", ylabel="", xlabel="", horizontal=None, height=None):
    """Streamlit bar chart. Text labels use horizontal bars, so names are never sideways."""
    if title:
        st.markdown(f"**{title}**")
    df_ = data.to_frame() if isinstance(data, pd.Series) else data
    numeric_index = pd.api.types.is_numeric_dtype(df_.index)
    horiz = (not numeric_index) if horizontal is None else horizontal
    if horiz:
        df_ = df_.copy()
        df_.index = [str(i).replace("\n", " ") for i in df_.index]
        _native(st.bar_chart, df_, horizontal=True, height=height or CHART_H)
    else:
        _native(st.bar_chart, df_, height=height or CHART_H, x_label=xlabel or None, y_label=ylabel or None)


def line_chart(data, title="", ylabel="", xlabel="", height=None, markers=False, fill=False):
    """Streamlit line chart (or filled area chart)."""
    if title:
        st.markdown(f"**{title}**")
    fn = st.area_chart if fill else st.line_chart
    _native(fn, data, height=height or CHART_H, x_label=xlabel or None, y_label=ylabel or None)


def pie_chart(data, title=""):
    """Donut chart drawn with Streamlit's built-in chart engine (Vega-Lite)."""
    s = data.iloc[:, 0] if isinstance(data, pd.DataFrame) else data
    s = s[s > 0]
    total = float(s.sum())
    if total <= 0:
        st.info("There is nothing to show in this chart yet.")
        return
    if len(s) > 6:
        small = s[s / total < 0.03]
        if len(small) > 1:
            s = pd.concat([s[s / total >= 0.03], pd.Series({"Other": small.sum()})])
    if title:
        st.markdown(f"**{title}**")
    chart_df = pd.DataFrame({"Part": [f"{k} ({100 * v / total:.0f}%)" for k, v in s.items()],
                             "Value": [float(v) for v in s.values]})
    spec = {
        "mark": {"type": "arc", "innerRadius": 55, "outerRadius": 100},
        "height": 230,
        "encoding": {
            "theta": {"field": "Value", "type": "quantitative", "stack": True},
            "color": {"field": "Part", "type": "nominal", "sort": None,
                      "scale": {"range": PIE_COLORS[: len(chart_df)]},
                      "legend": {"orient": "right", "title": None}},
            "tooltip": [{"field": "Part", "type": "nominal"},
                        {"field": "Value", "type": "quantitative", "format": ".1f"}],
        },
    }
    try:
        st.vega_lite_chart(chart_df, spec, width="stretch")
    except TypeError:
        st.vega_lite_chart(chart_df, spec, use_container_width=True)


# --------------------------------------- stretch-to-width helpers (any version) ---
def btn(label, **kw):
    """st.button that fills its column, working on old and new Streamlit versions."""
    try:
        return st.button(label, width="stretch", **kw)
    except TypeError:
        return st.button(label, use_container_width=True, **kw)


def table(df_or_styler):
    try:
        st.dataframe(df_or_styler, width="stretch")
    except TypeError:
        st.dataframe(df_or_styler, use_container_width=True)


def editor(df, **kw):
    try:
        return st.data_editor(df, width="stretch", **kw)
    except TypeError:
        return st.data_editor(df, use_container_width=True, **kw)


# ------------------------------------------------ one-panel number summary ---
def kpi_panel(items):
    """Show several numbers inside ONE panel (instead of many separate boxes).
    items: list of dicts: label, value, delta (text, optional), delta_val (number, optional),
           better ("lower" or "higher"), help (optional). `value` may contain simple HTML."""
    cells = []
    for it in items:
        cls, d = "", it.get("delta")
        dv, better = it.get("delta_val"), it.get("better", "lower")
        if d and dv is not None and dv != 0:
            good = (dv < 0) if better == "lower" else (dv > 0)
            cls = "good" if good else "bad"
        title = f' title="{it["help"]}"' if it.get("help") else ""
        delta_html = f'<div class="kpi-d {cls}">{d}</div>' if d else ""
        cells.append(f'<div class="kpi"{title}><div class="kpi-l">{it["label"]}</div>'
                     f'<div class="kpi-v">{it["value"]}</div>{delta_html}</div>')
    st.markdown('<div class="kpi-panel">' + "".join(cells) + "</div>", unsafe_allow_html=True)
