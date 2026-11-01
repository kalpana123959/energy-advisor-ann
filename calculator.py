import pandas as pd
import streamlit as st
from utils import get_settings, page_header

page_header("Energy Calculator", "Estimate the energy and cost of your appliances")
s = get_settings()
cur, tariff = s["currency"], s["tariff"]

st.caption("Wattages below are typical ASSUMPTIONS for you to edit with your own appliance ratings "
           "(check the label). This calculator is plain arithmetic: kWh = watts × hours ÷ 1000. "
           "It does not use the ANN.")

default = pd.DataFrame({
    "Appliance": ["Air conditioner", "Ceiling fan", "Television", "Refrigerator",
                  "Washing machine", "Water heater", "LED lights"],
    "Watts": [1500, 75, 100, 150, 500, 2000, 40],
    "Quantity": [1, 4, 1, 1, 1, 1, 6],
    "Hours per day": [6.0, 8.0, 4.0, 24.0, 1.0, 0.5, 5.0],
})
edited = st.data_editor(default, num_rows="dynamic", use_container_width=True, key="appl_table",
                        column_config={"Watts": st.column_config.NumberColumn(min_value=0),
                                       "Quantity": st.column_config.NumberColumn(min_value=0, step=1),
                                       "Hours per day": st.column_config.NumberColumn(min_value=0.0, max_value=24.0)})
calc = edited.dropna(subset=["Appliance"]).copy()
for c in ["Watts", "Quantity", "Hours per day"]:
    calc[c] = pd.to_numeric(calc[c], errors="coerce").fillna(0)
calc["kWh / day"] = calc["Watts"] * calc["Quantity"] * calc["Hours per day"] / 1000.0
calc["kWh / month"] = calc["kWh / day"] * 30
calc[f"Cost / month ({cur})"] = calc["kWh / month"] * tariff

day, month = calc["kWh / day"].sum(), calc["kWh / month"].sum()
a, b, c = st.columns(3)
a.metric("Energy per day (kWh)", f"{day:.2f}")
b.metric("Energy per month (kWh)", f"{month:.1f}")
c.metric(f"Cost per month ({cur})", f"{month * tariff:,.0f}", help=f"Tariff {cur}{tariff}/kWh (change in Settings)")

if len(calc) and month > 0:
    st.bar_chart(calc.set_index("Appliance")["kWh / month"])
    top = calc.loc[calc["kWh / month"].idxmax()]
    st.markdown(f"**Biggest consumer:** {top['Appliance']} — {top['kWh / month'] / month * 100:.0f}% of the monthly estimate.")

st.subheader("What-if: use an appliance less")
if len(calc):
    name = st.selectbox("Appliance", calc["Appliance"].tolist())
    row = calc[calc["Appliance"] == name].iloc[0]
    less = st.slider("Reduce daily hours by", 0.0, float(max(row["Hours per day"], 0.5)), min(1.0, float(row["Hours per day"])), 0.5)
    saved = row["Watts"] * row["Quantity"] * less / 1000.0 * 30
    x, y = st.columns(2)
    x.metric("Energy saved per month (kWh)", f"{saved:.1f}")
    y.metric(f"Cost saved per month ({cur})", f"{saved * tariff:,.0f}")
