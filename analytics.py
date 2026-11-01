import pandas as pd
import streamlit as st
from utils import load_hourly_data, page_header, DAY_NAMES, SUBMETER_NAMES, get_settings

page_header("Energy Analytics", "Explore patterns in your real consumption data")
df = load_hourly_data()
s = get_settings()

dmin, dmax = df["datetime"].min().date(), df["datetime"].max().date()
_r = st.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(_r, (tuple, list)) or len(_r) < 2:
    st.info("Pick both a start and an end date.")
    st.stop()
start, end = _r[0], _r[1]
sel = df[(df["datetime"].dt.date >= start) & (df["datetime"].dt.date <= end)]
if sel.empty:
    st.warning("No data in this range.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total energy (kWh)", f"{sel['Global_active_power'].sum():,.0f}")
k2.metric("Estimated cost", f"{s['currency']}{sel['Global_active_power'].sum() * s['tariff']:,.0f}")
k3.metric("Average power (kW)", f"{sel['Global_active_power'].mean():.2f}")
k4.metric("Highest hourly reading (kW)", f"{sel['Global_active_power'].max():.2f}")

t1, t2, t3, t4 = st.tabs(["Trend", "Hour of day", "Day of week", "Sub-meters"])
with t1:
    freq = st.radio("Group by", ["Day", "Week", "Month"], horizontal=True)
    rule = {"Day": "D", "Week": "W", "Month": "MS"}[freq]
    trend = sel.set_index("datetime")["Global_active_power"].resample(rule).sum()
    st.line_chart(trend.rename("kWh"))
with t2:
    st.bar_chart(sel.groupby("hour")["Global_active_power"].mean().rename("Average kW"))
    st.caption("Shows when the household typically uses the most power.")
with t3:
    dow = sel.groupby("day_of_week")["Global_active_power"].mean()
    dow.index = [DAY_NAMES[i] for i in dow.index]
    st.bar_chart(dow.rename("Average kW"))
with t4:
    tot = sel[list(SUBMETER_NAMES)].sum()
    tot.index = [SUBMETER_NAMES[i] for i in tot.index]
    st.bar_chart(tot.rename("Sub-meter total"))
    st.caption("Sub-meter values come from the dataset's three metered circuits.")

with st.expander("Correlation of features with next-hour consumption"):
    cols = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
            "Sub_metering_1", "Sub_metering_2", "Sub_metering_3", "hour", "day_of_week"]
    corr = sel[cols + ["target_next_hour"]].corr()["target_next_hour"].drop("target_next_hour")
    st.bar_chart(corr.rename("Correlation"))
