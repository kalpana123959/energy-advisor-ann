import pandas as pd
import streamlit as st
from utils import (load_or_train_model, load_hourly_data, get_settings, page_header,
                   available_mondays, build_weekly_report, weekly_recommendations,
                   tip_html, bar_chart, line_chart, pie_chart, kpi_panel, SUBMETER_NAMES)

page_header("Weekly Report", "Look at one week (Monday to Sunday) and get simple saving tips")
model, scaler, metrics = load_or_train_model()
df = load_hourly_data()
s = get_settings()

scope = st.radio("Which weeks do you want to see?",
                 ["Newer weeks (the model has not seen these)", "All weeks"], horizontal=True)
mondays = available_mondays(df, test_only=scope.startswith("Newer"))
if not mondays:
    st.warning("There is not enough data to make a weekly report.")
    st.stop()

week_start = st.selectbox("Choose a week", options=mondays, index=len(mondays) - 1,
                          format_func=lambda m: f"{m:%d %b %Y} to {(m + pd.Timedelta(days=6)):%d %b %Y}")
rep = build_weekly_report(df, week_start, model, scaler)
if rep is None:
    st.warning("There is no data for this week.")
    st.stop()

tariff, cur = s["tariff"], s["currency"]
kpi_panel([
    {"label": "Power used (kWh)", "value": f"{rep['total']:.1f}",
     "delta": None if rep["change_pct"] is None else f"{rep['change_pct']:+.1f}% vs last week",
     "delta_val": rep["change_pct"], "better": "lower"},
    {"label": "Estimated cost", "value": f"{cur}{rep['total'] * tariff:,.0f}"},
    {"label": "Average power (kW)", "value": f"{rep['avg_hourly']:.2f}"},
    {"label": "Busiest hour", "value": f"{rep['peak_hour']:02d}:00", "help": f"Average {rep['peak_value']:.2f} kW"},
])

d1, d2 = st.columns(2)
d1.write(f"**Highest day:** {pd.Timestamp(rep['highest_day']):%A %d %b} ({rep['highest_day_kwh']:.1f} kWh)")
d2.write(f"**Lowest day:** {pd.Timestamp(rep['lowest_day']):%A %d %b} ({rep['lowest_day_kwh']:.1f} kWh)")
if rep["hours_covered"] < 168:
    st.caption(f"Note: this week has {rep['hours_covered']} of 168 hourly readings, so the totals are not complete.")

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Power used each day**")
    daily = rep["daily"].copy()
    daily.index = [pd.Timestamp(d).strftime("%a\n%d %b") for d in daily.index]
    bar_chart(daily.rename("kWh"), ylabel="Power used (kWh)")
with g2:
    st.markdown("**A typical day this week**")
    line_chart(rep["hourly_profile"].rename("kW"), ylabel="Average power (kW)",
               xlabel="Hour of the day (0 = midnight)", fill=True)

g3, g4 = st.columns(2)
with g3:
    st.markdown("**Share of each meter**")
    share = rep["sm_share"].copy()
    share.index = [SUBMETER_NAMES[i] for i in share.index]
    pie_chart(share.rename("%"))
with g4:
    st.markdown("**When was the power used?**")
    st.write(f"- Evening (5 pm to 10 pm): **{rep['evening_share']:.0f}%**")
    st.write(f"- Night (midnight to 6 am): **{rep['night_share']:.0f}%**")
    st.write(f"- Average on weekdays: **{rep['weekday_avg']:.2f} kW**")
    st.write(f"- Average on weekends: **{rep['weekend_avg']:.2f} kW**")
    st.write(f"- Model's average error this week: **{rep['ann_mae']:.3f} kW**")

st.markdown("**Real power use and the model's guess (for the next hour)**")
chart = rep["data"].set_index("datetime")[["target_next_hour", "ann_pred"]]
chart.columns = ["Real next hour", "Model's guess"]
line_chart(chart, ylabel="Power (kW)", xlabel="Date")

st.subheader("Tips for this week")
for tip in weekly_recommendations(rep):
    st.markdown(tip_html(tip), unsafe_allow_html=True)

st.subheader("How much could you save? (just an estimate)")
pct = st.slider("If you use this much less in the evening (5 pm to 10 pm) (%)", 0, 30, 10)
saved = rep["total"] * rep["evening_share"] / 100.0 * pct / 100.0
kpi_panel([
    {"label": "Power saved in a week (kWh)", "value": f"{saved:.1f}"},
    {"label": "Money saved in a week", "value": f"{cur}{saved * tariff:,.0f}"},
])
st.caption("This is only an estimate from this week's use. It is not a measured result.")

report_df = pd.DataFrame({
    "Metric": ["Week start", "Total kWh", "Previous week kWh", "Change vs last week (%)",
               "Average kW per hour", "Peak hour", "Evening share (%)", "Night share (%)", "ANN MAE (kW)"],
    "Value": [f"{pd.Timestamp(week_start):%Y-%m-%d}", f"{rep['total']:.2f}",
              "n/a" if rep["prev_total"] is None else f"{rep['prev_total']:.2f}",
              "n/a" if rep["change_pct"] is None else f"{rep['change_pct']:.1f}",
              f"{rep['avg_hourly']:.3f}", f"{rep['peak_hour']:02d}:00",
              f"{rep['evening_share']:.1f}", f"{rep['night_share']:.1f}", f"{rep['ann_mae']:.3f}"]})
st.download_button("Download this report (CSV file)", report_df.to_csv(index=False).encode("utf-8"),
                   file_name=f"weekly_report_{pd.Timestamp(week_start):%Y-%m-%d}.csv", mime="text/csv")
