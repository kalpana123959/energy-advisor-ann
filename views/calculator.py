import pandas as pd
import streamlit as st
from utils import get_settings, page_header, pie_chart, kpi_panel, table, editor

page_header("Energy Calculator", "Work out how much power your appliances use and what it costs")
s = get_settings()
cur, tariff = s["currency"], s["tariff"]

st.write("Change the table to match your home, then click **Calculate**. "
         "Energy in kWh = watts × number of appliances × hours ÷ 1000.")
st.caption("The watts shown are common examples. Check the label on your own appliance and change them if needed. "
           "You can also add or delete rows.")

default = pd.DataFrame({
    "Appliance": ["Air conditioner", "Ceiling fan", "Television", "Refrigerator",
                  "Washing machine", "Water heater", "LED lights"],
    "Watts": [1500, 75, 100, 150, 500, 2000, 40],
    "Quantity": [1, 4, 1, 1, 1, 1, 6],
    "Hours per day": [6.0, 8.0, 4.0, 24.0, 1.0, 0.5, 5.0],
})

with st.form("calc_form"):
    edited = editor(default, num_rows="dynamic", key="appl_table",
                    column_config={"Watts": st.column_config.NumberColumn(min_value=0),
                                   "Quantity": st.column_config.NumberColumn(min_value=0, step=1),
                                   "Hours per day": st.column_config.NumberColumn(min_value=0.0, max_value=24.0)})
    submitted = st.form_submit_button("Calculate", type="primary")

if submitted:
    calc = edited.dropna(subset=["Appliance"]).copy()
    for c in ["Watts", "Quantity", "Hours per day"]:
        calc[c] = pd.to_numeric(calc[c], errors="coerce").fillna(0)
    calc["kWh / day"] = calc["Watts"] * calc["Quantity"] * calc["Hours per day"] / 1000.0
    calc["kWh / month"] = calc["kWh / day"] * 30
    calc["Cost / month"] = calc["kWh / month"] * tariff
    st.session_state["calc_result"] = {"calc": calc, "cur": cur, "tariff": tariff}
    st.session_state.pop("calc_saving", None)

res = st.session_state.get("calc_result")
if res is None:
    st.info("Your results will appear here after you click **Calculate**.")
    st.stop()

calc, rcur, rtariff = res["calc"], res["cur"], res["tariff"]
day, month = calc["kWh / day"].sum(), calc["kWh / month"].sum()
st.subheader("Your results")
kpi_panel([
    {"label": "Power per day (kWh)", "value": f"{day:.2f}"},
    {"label": "Power per month (kWh)", "value": f"{month:.1f}"},
    {"label": f"Cost per month ({rcur})", "value": f"{month * rtariff:,.0f}",
     "help": f"Price used: {rcur}{rtariff} per kWh (change it in Settings)"},
])

if len(calc) and month > 0:
    left, right = st.columns([3, 2])
    with left:
        pie_chart(calc.set_index("Appliance")["kWh / month"], title="Share of power used by each appliance")
    with right:
        show = calc[["Appliance", "kWh / month", "Cost / month"]].copy()
        show.columns = ["Appliance", "kWh per month", f"Cost per month ({rcur})"]
        table(show.round(1).set_index("Appliance"))
    top = calc.loc[calc["kWh / month"].idxmax()]
    st.markdown(f"**Uses the most power:** {top['Appliance']} ({top['kWh / month'] / month * 100:.0f}% of the total).")

st.subheader("What if you use one appliance less?")
if len(calc):
    with st.form("saving_form"):
        name = st.selectbox("Choose an appliance", calc["Appliance"].tolist())
        less = st.slider("Use it this many hours less each day", 0.0, 12.0, 1.0, 0.5)
        go_save = st.form_submit_button("Calculate saving")
    if go_save:
        row = calc[calc["Appliance"] == name].iloc[0]
        hours_cut = min(less, float(row["Hours per day"]))
        saved = row["Watts"] * row["Quantity"] * hours_cut / 1000.0 * 30
        st.session_state["calc_saving"] = {"name": name, "hours": hours_cut, "saved": saved}
    sv = st.session_state.get("calc_saving")
    if sv:
        kpi_panel([
            {"label": "Power saved per month (kWh)", "value": f"{sv['saved']:.1f}"},
            {"label": f"Money saved per month ({rcur})", "value": f"{sv['saved'] * rtariff:,.0f}"},
        ])
        st.caption(f"This is for {sv['name']}, using it {sv['hours']:.1f} hours less per day. "
                   "You cannot save more hours than it is used.")
