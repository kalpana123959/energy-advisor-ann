"""
Train the same ANN as sc_project.ipynb and save new files only.

Does not modify or delete the original CSV files or notebook.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "prepared_hourly_data.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"

FEATURES = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
    "hour",
    "day_of_week",
]
TARGET = "target_next_hour"


def load_and_prepare():
    data = pd.read_csv(DATA_PATH)
    data = data[FEATURES + [TARGET]].copy()
    for col in FEATURES + [TARGET]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)
    return data


def chronological_split(data):
    split_index = int(len(data) * 0.80)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    X_test = test_data[FEATURES]
    y_test = test_data[TARGET]
    return X_train, X_test, y_train, y_test


def train_and_save():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.name}")

    data = load_and_prepare()
    X_train, X_test, y_train, y_test = chronological_split(data)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Same ANN settings as the notebook
    ann = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=50,
        early_stopping=False,
        random_state=42,
        verbose=True,
    )
    ann.fit(X_train_scaled, y_train)

    y_pred = ann.predict(X_test_scaled)
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    baseline = X_test["Global_active_power"].to_numpy()
    base_mae = float(mean_absolute_error(y_test, baseline))
    base_rmse = float(np.sqrt(mean_squared_error(y_test, baseline)))
    base_r2 = float(r2_score(y_test, baseline))

    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(ann, ARTIFACT_DIR / "ann_model.joblib")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")

    metrics = {
        "source": "Computed by train_model.py using the same ANN and split as sc_project.ipynb",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "ann": {"MAE": mae, "RMSE": rmse, "R2": r2},
        "baseline": {"MAE": base_mae, "RMSE": base_rmse, "R2": base_r2},
        "notebook_output_from_sc_project": {
            "note": "Copied from saved Colab output in sc_project.ipynb (not invented)",
            "ann": {"MAE": 0.3405, "RMSE": 0.4823, "R2": 0.5795},
            "baseline": {"MAE": 0.3968, "RMSE": 0.6027, "R2": 0.3435},
        },
    }
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to artifacts/ann_model.joblib")
    print("ANN MAE/RMSE/R2:", mae, rmse, r2)
    return metrics


if __name__ == "__main__":
    train_and_save()
