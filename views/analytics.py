import pandas as pd
import streamlit as st
from utils import (load_hourly_data, page_header, bar_chart, line_chart, pie_chart, kpi_panel, DAY_NAMES,
                   SUBMETER_NAMES, get_settings)

page_header("Energy Analytics", "See when and where your power is used")
df = load_hourly_data()
s = get_settings()

dmin, dmax = df["datetime"].min().date(), df["datetime"].max().date()
_r = st.date_input("Choose the dates to look at", value=(dmin, dmax), min_value=dmin, max_value=dmax)
if not isinstance(_r, (tuple, list)) or len(_r) < 2:
    st.info("Please pick both a start date and an end date.")
    st.stop()
start, end = _r[0], _r[1]
sel = df[(df["datetime"].dt.date >= start) & (df["datetime"].dt.date <= end)]
if sel.empty:
    st.warning("There is no data for these dates.")
    st.stop()

kpi_panel([
    {"label": "Power used (kWh)", "value": f"{sel['Global_active_power'].sum():,.0f}"},
    {"label": "Estimated cost", "value": f"{s['currency']}{sel['Global_active_power'].sum() * s['tariff']:,.0f}"},
    {"label": "Average power (kW)", "value": f"{sel['Global_active_power'].mean():.2f}"},
    {"label": "Highest hour (kW)", "value": f"{sel['Global_active_power'].max():.2f}"},
])

t1, t2, t3, t4 = st.tabs(["Over time", "By hour", "By weekday", "By meter"])
with t1:
    span_days = (sel["datetime"].max() - sel["datetime"].min()).days
    auto = 2 if span_days > 400 else (1 if span_days > 60 else 0)     # Month / Week / Day
    options = ["Day", "Week", "Month"]
    freq = st.radio("Show one point for each", options, index=auto, horizontal=True,
                    help="For a long time period, Week or Month is easier to read than Day.")
    rule = {"Day": "D", "Week": "W", "Month": "MS"}[freq]
    per_day = sel.set_index("datetime")["Global_active_power"].resample("D").sum()
    per_day = per_day[per_day > 0]                      # ignore days with no readings
    trend = per_day if freq == "Day" else per_day.resample(rule).mean().dropna()
    label = "kWh per day" if freq == "Day" else "Average kWh per day"
    line_chart(trend.rename("kWh"), ylabel=label, xlabel="Date", markers=(freq == "Month"))
    st.caption("Each point shows how much power your home used in a day"
               + ("." if freq == "Day" else f" on average during that {freq.lower()}.")
               + " Higher points mean more power used.")
with t2:
    line_chart(sel.groupby("hour")["Global_active_power"].mean().rename("Average kW"),
               ylabel="Average power (kW)", xlabel="Hour of the day (0 = midnight)", fill=True)
    st.caption("Tall bars are the hours when your home uses the most power.")
with t3:
    dow = sel.groupby("day_of_week")["Global_active_power"].mean()
    dow.index = [DAY_NAMES[i] for i in dow.index]
    bar_chart(dow.rename("Average kW"), ylabel="Average power (kW)")
with t4:
    tot = sel[list(SUBMETER_NAMES)].sum()
    tot.index = [SUBMETER_NAMES[i] for i in tot.index]
    pie_chart(tot.rename("Meter total"), title="Share of the three meters")
    st.caption("These are the three parts of the house that have their own meter.")

with st.expander("Which readings are linked to next-hour power use?"):
    names = {"Global_active_power": "Power now", "Global_reactive_power": "Reactive power",
             "Voltage": "Voltage", "Global_intensity": "Current",
             "Sub_metering_1": "Kitchen meter", "Sub_metering_2": "Laundry meter",
             "Sub_metering_3": "Water heater / AC meter", "hour": "Hour of day", "day_of_week": "Day of week"}
    corr = sel[list(names) + ["target_next_hour"]].corr()["target_next_hour"].drop("target_next_hour")
    corr.index = [names[i] for i in corr.index]
    bar_chart(corr.rename("Link"), ylabel="Link strength (-1 to 1)")
    st.caption("A bar further from zero means that reading moves more closely with next hour's power use.")
