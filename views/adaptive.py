import joblib
import pandas as pd
import streamlit as st
from utils import (load_or_train_model, load_hourly_data, page_header, split_index,
                   evaluate, adapt_model, bar_chart, table, kpi_panel, ADAPTED_MODEL_PATH)

page_header("Adaptive Learning", "Teach the model with newer data and see if it gets better")
model, scaler, metrics = load_or_train_model()
df = load_hourly_data()

test = df.iloc[split_index(df):].reset_index(drop=True)
st.write(
    "The model first learned from the older 80% of the data. The newest 20% is treated like "
    "**new data that arrives later**. Part of it is used to teach a copy of the model again, and the "
    "rest is used to check if the copy got better. The scores below are worked out live from your data.")

c1, c2 = st.columns(2)
frac = c1.slider("How much of the new data should be used to teach the model? (%)", 10, 80, 50, 5)
epochs = c2.slider("Number of learning rounds", 1, 30, 2,
                   help="One round means the model looks at the new data once.")
cut = int(len(test) * frac / 100)
new_part, eval_part = test.iloc[:cut], test.iloc[cut:]
st.caption(f"Hours used for teaching: {len(new_part):,}. Hours used for checking (kept apart): {len(eval_part):,}.")

if st.button("Teach the model with new data", type="primary"):
    with st.spinner("Teaching the model..."):
        before = evaluate(model, scaler, eval_part)
        adapted = adapt_model(model, scaler, new_part, epochs)
        after = evaluate(adapted, scaler, eval_part)
    st.session_state["adapt_result"] = {"before": before, "after": after, "model": adapted,
                                        "frac": frac, "epochs": epochs}

r = st.session_state.get("adapt_result")
if r:
    b, a = r["before"], r["after"]
    st.subheader("Before and after")
    kpi_panel([
        {"label": "Average error (kW)", "value": f"{a['MAE']:.4f}", "delta": f"{a['MAE'] - b['MAE']:+.4f}",
         "delta_val": a["MAE"] - b["MAE"], "better": "lower"},
        {"label": "Typical error, RMSE (kW)", "value": f"{a['RMSE']:.4f}", "delta": f"{a['RMSE'] - b['RMSE']:+.4f}",
         "delta_val": a["RMSE"] - b["RMSE"], "better": "lower"},
        {"label": "Fit score, R²", "value": f"{a['R2']:.4f}", "delta": f"{a['R2'] - b['R2']:+.4f}",
         "delta_val": a["R2"] - b["R2"], "better": "higher"},
    ])
    st.caption("For the two error scores, a smaller number is better. For R², a number closer to 1 is better.")
    tbl = pd.DataFrame({"Original model": b, "Updated model": a}).T
    tbl.columns = ["Average error", "Typical error (RMSE)", "Fit score (R²)"]
    table(tbl.style.format("{:.4f}"))
    bar_chart(pd.DataFrame({"Original model": [b["MAE"], b["RMSE"]],
                            "Updated model": [a["MAE"], a["RMSE"]]},
                           index=["Average error", "Typical error (RMSE)"]),
              title="Error before and after (smaller is better)", ylabel="Error (kW)")
    if a["MAE"] < b["MAE"]:
        st.success("The updated model made smaller mistakes on the hours it had not seen.")
    else:
        st.warning("The updated model did not get better this time. Try fewer learning rounds or more data. "
                   "It is fine to report this honestly.")
    if st.button("Save the updated model (as a separate file)"):
        joblib.dump(r["model"], ADAPTED_MODEL_PATH)
        st.success(f"Saved as {ADAPTED_MODEL_PATH.name}. Your original model file is not changed.")

st.info("Note: this is one test on one set of data. Try a few different settings before drawing a conclusion.")
