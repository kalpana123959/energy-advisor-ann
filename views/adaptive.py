import joblib
import pandas as pd
import streamlit as st
from utils import (load_or_train_model, load_hourly_data, page_header, split_index,
                   evaluate, adapt_model, ADAPTED_MODEL_PATH)

page_header("Adaptive Learning", "Update the ANN with newer household data and measure the real effect")
model, scaler, metrics = load_or_train_model()
df = load_hourly_data()

test = df.iloc[split_index(df):].reset_index(drop=True)
st.write(
    "The ANN was trained on the first 80% of the data. The held-out test period is treated as "
    "**new household data arriving later**. The first part of it is used to update (fine-tune) a "
    "copy of the model, and the remaining part is used to check whether the update helped. "
    "Both scores are computed live from your data; nothing is pre-filled.")

c1, c2 = st.columns(2)
frac = c1.slider("Share of new data used for updating (%)", 10, 80, 50, 5)
epochs = c2.slider("Update passes (epochs)", 1, 30, 5)
cut = int(len(test) * frac / 100)
new_part, eval_part = test.iloc[:cut], test.iloc[cut:]
st.caption(f"Update data: {len(new_part):,} hours · Evaluation data (never used for updating): {len(eval_part):,} hours")

if st.button("Update ANN with new data", type="primary"):
    with st.spinner("Updating the ANN..."):
        before = evaluate(model, scaler, eval_part)
        adapted = adapt_model(model, scaler, new_part, epochs)
        after = evaluate(adapted, scaler, eval_part)
    st.session_state["adapt_result"] = {"before": before, "after": after, "model": adapted,
                                        "frac": frac, "epochs": epochs}

r = st.session_state.get("adapt_result")
if r:
    b, a = r["before"], r["after"]
    st.subheader("Before vs after (on the evaluation hours)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (kW)", f"{a['MAE']:.4f}", delta=f"{a['MAE'] - b['MAE']:+.4f}", delta_color="inverse")
    m2.metric("RMSE (kW)", f"{a['RMSE']:.4f}", delta=f"{a['RMSE'] - b['RMSE']:+.4f}", delta_color="inverse")
    m3.metric("R²", f"{a['R2']:.4f}", delta=f"{a['R2'] - b['R2']:+.4f}")
    table = pd.DataFrame({"Original ANN": b, "Adapted ANN": a}).T
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
    st.bar_chart(pd.DataFrame({"Original ANN": [b["MAE"], b["RMSE"]],
                               "Adapted ANN": [a["MAE"], a["RMSE"]]}, index=["MAE", "RMSE"]))
    if a["MAE"] < b["MAE"]:
        st.success("The update reduced the error on unseen hours.")
    else:
        st.warning("The update did not reduce the error this time. Try fewer passes or more data. "
                   "Reporting this honestly is part of a good evaluation.")
    if st.button("Save adapted model (separate file)"):
        joblib.dump(r["model"], ADAPTED_MODEL_PATH)
        st.success(f"Saved to {ADAPTED_MODEL_PATH.name}. Your original ann_model.joblib is unchanged.")

st.info("Limitation: one update run on one dataset. For the report, repeat with several settings "
        "and describe the average behaviour.")
