import pandas as pd
import streamlit as st
from utils import (load_or_train_model, load_hourly_data, get_settings, page_header,
                   split_index, SUBMETER_NAMES)

page_header("Dashboard", "Overview of your household consumption and the ANN model")
model, scaler, metrics = load_or_train_model()
df = load_hourly_data()
s = get_settings()

latest_day = df["datetime"].dt.normalize().iloc[-1]
last30 = df[df["datetime"] >= latest_day - pd.Timedelta(days=30)]
prev30 = df[(df["datetime"] >= latest_day - pd.Timedelta(days=60)) &
            (df["datetime"] < latest_day - pd.Timedelta(days=30))]
tot30 = last30["Global_active_power"].sum()
tot_prev = prev30["Global_active_power"].sum()
delta = None if tot_prev == 0 else f"{100 * (tot30 - tot_prev) / tot_prev:+.1f}% vs previous 30 days"

st.caption(f"Data covers {df['datetime'].min():%d %b %Y} to {df['datetime'].max():%d %b %Y} "
           f"({len(df):,} hourly readings). 'Last 30 days' means the final 30 days in the dataset.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Energy, last 30 days (kWh)", f"{tot30:,.0f}", delta=delta, delta_color="inverse")
c2.metric("Estimated cost", f"{s['currency']}{tot30 * s['tariff']:,.0f}")
c3.metric("Average power (kW)", f"{last30['Global_active_power'].mean():.2f}")
hourly_avg = df.groupby("hour")["Global_active_power"].mean()
c4.metric("Usual peak hour", f"{int(hourly_avg.idxmax()):02d}:00")

st.subheader("ANN model performance (held-out test data)")
a, b = metrics["ann"], metrics["baseline"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("ANN MAE (kW)", f"{a['MAE']:.4f}", delta=f"{a['MAE'] - b['MAE']:.4f} vs baseline", delta_color="inverse")
m2.metric("ANN RMSE (kW)", f"{a['RMSE']:.4f}", delta=f"{a['RMSE'] - b['RMSE']:.4f} vs baseline", delta_color="inverse")
m3.metric("ANN R²", f"{a['R2']:.4f}", delta=f"{a['R2'] - b['R2']:+.4f} vs baseline")
m4.metric("Test rows", f"{metrics['n_test']:,}")
st.caption("Baseline = use this hour's power as the next-hour guess. These numbers come from "
           "artifacts/metrics.json, computed by train_model.py on your real split.")

left, right = st.columns([3, 2])
with left:
    st.subheader("Daily consumption – last 60 days of data")
    daily = (df[df["datetime"] >= latest_day - pd.Timedelta(days=60)]
             .groupby(df["datetime"].dt.date)["Global_active_power"].sum())
    st.line_chart(daily.rename("kWh per day"))
with right:
    st.subheader("Where sub-metered energy goes")
    share = last30[list(SUBMETER_NAMES)].sum()
    share.index = [SUBMETER_NAMES[i] for i in share.index]
    st.bar_chart(share.rename("Sub-meter total (last 30 days)"))
    st.caption("The rest of the household load is not covered by the three sub-meters.")

st.subheader("Average day profile")
st.line_chart(hourly_avg.rename("Average kW by hour of day"))
