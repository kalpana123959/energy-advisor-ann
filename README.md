# Adaptive Personalized Energy Advisor (multi-page)

Run (Windows): double-click `run_app.bat`, or in a terminal inside this folder:

    pip install -r requirements.txt
    python -m streamlit run app.py

Pages (folder `views/`): Dashboard, Energy Analytics, Weekly Report, Personalized Advice,
Energy Calculator, Energy Prediction, Adaptive Learning, Settings.

Files: `app.py` (navigation) · `utils.py` (shared logic, reused from the old app) ·
`train_model.py`, CSVs, `artifacts/` (unchanged) · `old_app.py` (your previous single-page app, kept as backup).
Use the Energy Prediction page first, then open Personalized Advice.
