# Adaptive Personalized Energy Advisor

Run locally (Windows): double-click `run_app.bat`
or in a terminal inside this folder:

    pip install -r requirements.txt
    python -m streamlit run app.py

If the app does not open:
- Use `python -m streamlit run app.py` (plain `streamlit` may not be on PATH).
- If the saved model came from a different scikit-learn version, the app retrains automatically (about a minute) and saves fresh files in `artifacts/`.
- Keep all CSV files in the same folder as `app.py`.

Tabs: Advisor app (next-hour prediction, graphs, tips), Weekly report, About.
