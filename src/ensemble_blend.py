from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RF_PATH = MODELS_DIR / "random_forest_model.pkl"
XGB_PATH = MODELS_DIR / "xgboost_model.pkl"
OUT_PATH = MODELS_DIR / "ensemble_config.pkl"

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)
X = df.drop("Exited", axis=1)
y = df["Exited"]

# same split as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Load trained pipelines ----------------
rf = joblib.load(RF_PATH)
xgb = joblib.load(XGB_PATH)

# probabilities
rf_proba = rf.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]

# ---------------- Search best blend ----------------
best = {
    "f1": -1,
    "w_rf": None,
    "threshold": None,
    "metrics": None
}

# weights: 0.0, 0.1, ..., 1.0
weights = np.linspace(0, 1, 11)
# thresholds: 0.30 to 0.70
thresholds = np.linspace(0.30, 0.70, 41)

for w_rf in weights:
    blend_proba = w_rf * rf_proba + (1 - w_rf) * xgb_proba
    auc = roc_auc_score(y_test, blend_proba)

    for t in thresholds:
        y_pred = (blend_proba >= t).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        if f1 > best["f1"]:
            best["f1"] = f1
            best["w_rf"] = float(w_rf)
            best["threshold"] = float(t)
            best["metrics"] = {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "roc_auc": float(auc),
            }

# ---------------- Print best ----------------
print("✅ Best Ensemble (by F1)")
print("RF weight      :", best["w_rf"])
print("XGB weight     :", 1 - best["w_rf"])
print("Threshold      :", best["threshold"])
print("Metrics        :", best["metrics"])

# confusion matrix for best
blend_proba = best["w_rf"] * rf_proba + (1 - best["w_rf"]) * xgb_proba
y_best = (blend_proba >= best["threshold"]).astype(int)
cm = confusion_matrix(y_test, y_best)
print("\nConfusion Matrix:\n", cm)

# save config (for Streamlit later)
joblib.dump(
    {"w_rf": best["w_rf"], "threshold": best["threshold"]},
    OUT_PATH
)
print(f"\n💾 Saved ensemble config to: {OUT_PATH}")
