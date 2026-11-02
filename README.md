# Adaptive Personalized Energy Advisor

How to run (Windows): double-click `run_app.bat`, or open a terminal in this folder and type:

    pip install -r requirements.txt
    python -m streamlit run app.py

How it works: Welcome page -> "Get Started" -> cards (2 per row) -> a section. Inside a section, the "All sections" side panel switches pages and the Home button returns to the start.

Files
- app.py        : welcome page, card menu and page switching (no default sidebar menu)
- utils.py      : shared code (model loading, predictions, tips, charts, theme)
- views/*.py    : the 8 sections (keep the file names as they are)
- .streamlit/config.toml : light colour theme
- train_model.py, CSV files, artifacts/, sc_project.ipynb : unchanged
- old_app.py    : your earlier single-page app (backup)
