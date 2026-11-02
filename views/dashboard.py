import pandas as pd
import streamlit as st
from utils import (load_or_train_model, load_hourly_data, get_settings, page_header,
                   pie_chart, line_chart, kpi_panel, SUBMETER_NAMES)

page_header("Dashboard", "A quick summary of your power use and how well the model works")
model, scaler, metrics = load_or_train_model()
df = load_hourly_data()
s = get_settings()

latest_day = df["datetime"].dt.normalize().iloc[-1]
last30 = df[df["datetime"] >= latest_day - pd.Timedelta(days=30)]
prev30 = df[(df["datetime"] >= latest_day - pd.Timedelta(days=60)) &
            (df["datetime"] < latest_day - pd.Timedelta(days=30))]
tot30 = last30["Global_active_power"].sum()
tot_prev = prev30["Global_active_power"].sum()
chg = None if tot_prev == 0 else 100 * (tot30 - tot_prev) / tot_prev
hourly_avg = df.groupby("hour")["Global_active_power"].mean()

st.caption(f"The data runs from {df['datetime'].min():%d %b %Y} to {df['datetime'].max():%d %b %Y} "
           f"({len(df):,} hourly readings). \"Last 30 days\" means the final 30 days of this data.")

kpi_panel([
    {"label": "Power used, last 30 days (kWh)", "value": f"{tot30:,.0f}",
     "delta": None if chg is None else f"{chg:+.1f}% vs the 30 days before", "delta_val": chg, "better": "lower"},
    {"label": "Estimated cost", "value": f"{s['currency']}{tot30 * s['tariff']:,.0f}"},
    {"label": "Average power (kW)", "value": f"{last30['Global_active_power'].mean():.2f}"},
    {"label": "Busiest hour of the day", "value": f"{int(hourly_avg.idxmax()):02d}:00"},
])

st.subheader("How well does the model guess?")
a, b = metrics["ann"], metrics["baseline"]
kpi_panel([
    {"label": "Average error (kW)", "value": f"{a['MAE']:.4f}", "delta": f"{a['MAE'] - b['MAE']:+.4f} vs simple guess",
     "delta_val": a['MAE'] - b['MAE'], "better": "lower"},
    {"label": "Typical error, RMSE (kW)", "value": f"{a['RMSE']:.4f}", "delta": f"{a['RMSE'] - b['RMSE']:+.4f} vs simple guess",
     "delta_val": a['RMSE'] - b['RMSE'], "better": "lower"},
    {"label": "Fit score, R²", "value": f"{a['R2']:.4f}", "delta": f"{a['R2'] - b['R2']:+.4f} vs simple guess",
     "delta_val": a['R2'] - b['R2'], "better": "higher"},
    {"label": "Hours used for testing", "value": f"{metrics['n_test']:,}"},
])
with st.expander("What do these numbers mean?"):
    st.markdown("""
- **Average error:** on average, the model's guess is off by this many kW. **Smaller is better.**
- **Typical error (RMSE):** like the average error, but big mistakes count more. **Smaller is better.**
- **Fit score (R²):** how closely the guess follows the real ups and downs. **Closer to 1 is better.**
- **Simple guess:** just saying "next hour will be the same as this hour". The model should beat it.
- These numbers were measured on hours the model had never seen before.
""")

left, right = st.columns([3, 2])
with left:
    st.subheader("Power used each day (last 60 days of data)")
    daily = (df[df["datetime"] >= latest_day - pd.Timedelta(days=60)]
             .groupby(df["datetime"].dt.date)["Global_active_power"].sum())
    line_chart(daily.rename("kWh per day"), ylabel="Power used (kWh)", xlabel="Date")
with right:
    st.subheader("Where the power goes")
    share = last30[list(SUBMETER_NAMES)].sum()
    share.index = [SUBMETER_NAMES[i] for i in share.index]
    pie_chart(share.rename("Meter total"), title="Share of the three meters (last 30 days)")
    st.caption("Only three parts of the house have their own meter. The rest is not shown here.")

st.subheader("A typical day")
line_chart(hourly_avg.rename("Average kW"), ylabel="Average power (kW)", xlabel="Hour of the day (0 = midnight)", fill=True)
