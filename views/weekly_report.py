import pandas as pd
import streamlit as st
from utils import (load_or_train_model, load_hourly_data, get_settings, page_header,
                   available_mondays, build_weekly_report, weekly_recommendations,
                   tip_html, SUBMETER_NAMES)

page_header("Weekly Report", "Monday-to-Sunday summary with ANN predictions and suggestions")
model, scaler, metrics = load_or_train_model()
df = load_hourly_data()
s = get_settings()

scope = st.radio("Weeks to choose from", ["Held-out test period (ANN has not seen it)", "All data"], horizontal=True)
mondays = available_mondays(df, test_only=scope.startswith("Held"))
if not mondays:
    st.warning("Not enough data to build a weekly report.")
    st.stop()

week_start = st.selectbox("Select week", options=mondays, index=len(mondays) - 1,
                          format_func=lambda m: f"{m:%d %b %Y} to {(m + pd.Timedelta(days=6)):%d %b %Y}")
rep = build_weekly_report(df, week_start, model, scaler)
if rep is None:
    st.warning("No data for this week.")
    st.stop()

tariff, cur = s["tariff"], s["currency"]
delta = None if rep["change_pct"] is None else f"{rep['change_pct']:+.1f}% vs last week"
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total energy (kWh)", f"{rep['total']:.1f}", delta=delta, delta_color="inverse")
k2.metric("Estimated cost", f"{cur}{rep['total'] * tariff:,.0f}")
k3.metric("Average per hour (kW)", f"{rep['avg_hourly']:.2f}")
k4.metric("Peak hour", f"{rep['peak_hour']:02d}:00", help=f"Average {rep['peak_value']:.2f} kW")

d1, d2 = st.columns(2)
d1.write(f"**Highest day:** {pd.Timestamp(rep['highest_day']):%A %d %b} ({rep['highest_day_kwh']:.1f} kWh)")
d2.write(f"**Lowest day:** {pd.Timestamp(rep['lowest_day']):%A %d %b} ({rep['lowest_day_kwh']:.1f} kWh)")
if rep["hours_covered"] < 168:
    st.caption(f"Note: this week has {rep['hours_covered']} of 168 hourly readings, so totals are partial.")

g1, g2 = st.columns(2)
with g1:
    st.markdown("**Daily consumption**")
    daily = rep["daily"].copy()
    daily.index = [pd.Timestamp(d).strftime("%a %d %b") for d in daily.index]
    st.bar_chart(daily.rename("kWh"))
with g2:
    st.markdown("**Typical day this week (avg kW by hour)**")
    st.line_chart(rep["hourly_profile"].rename("kW"))

g3, g4 = st.columns(2)
with g3:
    st.markdown("**Sub-meter share (%)**")
    share = rep["sm_share"].copy()
    share.index = [SUBMETER_NAMES[i] for i in share.index]
    st.bar_chart(share.rename("%"))
with g4:
    st.markdown("**Usage split**")
    st.write(f"- Evening peak (5–10 pm): **{rep['evening_share']:.0f}%**")
    st.write(f"- Night (12–6 am): **{rep['night_share']:.0f}%**")
    st.write(f"- Weekday average: **{rep['weekday_avg']:.2f} kW**")
    st.write(f"- Weekend average: **{rep['weekend_avg']:.2f} kW**")
    st.write(f"- ANN error this week (MAE): **{rep['ann_mae']:.3f} kW**")

st.markdown("**Actual vs ANN prediction (next hour)**")
chart = rep["data"].set_index("datetime")[["target_next_hour", "ann_pred"]]
chart.columns = ["Actual next hour", "ANN prediction"]
st.line_chart(chart)

st.subheader("Weekly energy-saving suggestions")
for tip in weekly_recommendations(rep):
    st.markdown(tip_html(tip), unsafe_allow_html=True)

st.subheader("What-if savings (illustrative)")
pct = st.slider("If you cut evening-peak usage by (%)", 0, 30, 10)
saved = rep["total"] * rep["evening_share"] / 100.0 * pct / 100.0
a, b = st.columns(2)
a.metric("Energy saved per week (kWh)", f"{saved:.1f}")
b.metric("Cost saved per week", f"{cur}{saved * tariff:,.0f}")
st.caption("A what-if estimate from this week's own pattern, not a measured result.")

report_df = pd.DataFrame({
    "Metric": ["Week start", "Total kWh", "Previous week kWh", "Change vs last week (%)",
               "Average kW per hour", "Peak hour", "Evening share (%)", "Night share (%)", "ANN MAE (kW)"],
    "Value": [f"{pd.Timestamp(week_start):%Y-%m-%d}", f"{rep['total']:.2f}",
              "n/a" if rep["prev_total"] is None else f"{rep['prev_total']:.2f}",
              "n/a" if rep["change_pct"] is None else f"{rep['change_pct']:.1f}",
              f"{rep['avg_hourly']:.3f}", f"{rep['peak_hour']:02d}:00",
              f"{rep['evening_share']:.1f}", f"{rep['night_share']:.1f}", f"{rep['ann_mae']:.3f}"]})
st.download_button("Download weekly report (CSV)", report_df.to_csv(index=False).encode("utf-8"),
                   file_name=f"weekly_report_{pd.Timestamp(week_start):%Y-%m-%d}.csv", mime="text/csv")
