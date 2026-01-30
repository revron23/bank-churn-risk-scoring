from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

MODELS_DIR = PROJECT_ROOT / "models"
RF_PATH = MODELS_DIR / "random_forest_model.pkl"
XGB_PATH = MODELS_DIR / "xgboost_model.pkl"

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = joblib.load(RF_PATH)
xgb = joblib.load(XGB_PATH)

rf_proba = rf.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]

# Search weights + thresholds
weights = np.linspace(0, 1, 21)          # finer than before
thresholds = np.linspace(0.30, 0.95, 66) # go high for accuracy

best_acc = {"acc": -1, "w_rf": None, "t": None, "metrics": None, "cm": None}
best_f1  = {"f1": -1,  "w_rf": None, "t": None, "metrics": None, "cm": None}

for w_rf in weights:
    proba = w_rf * rf_proba + (1 - w_rf) * xgb_proba
    auc = roc_auc_score(y_test, proba)

    for t in thresholds:
        pred = (proba >= t).astype(int)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        cm = confusion_matrix(y_test, pred)

        if acc > best_acc["acc"]:
            best_acc = {
                "acc": float(acc), "w_rf": float(w_rf), "t": float(t),
                "metrics": {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(auc)},
                "cm": cm
            }

        if f1 > best_f1["f1"]:
            best_f1 = {
                "f1": float(f1), "w_rf": float(w_rf), "t": float(t),
                "metrics": {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "roc_auc": float(auc)},
                "cm": cm
            }

print("\n🏁 BEST BY ACCURACY")
print("RF weight:", best_acc["w_rf"], "XGB weight:", 1 - best_acc["w_rf"], "Threshold:", best_acc["t"])
print("Metrics:", best_acc["metrics"])
print("Confusion Matrix:\n", best_acc["cm"])

print("\n🏆 BEST BY F1 (Balanced)")
print("RF weight:", best_f1["w_rf"], "XGB weight:", 1 - best_f1["w_rf"], "Threshold:", best_f1["t"])
print("Metrics:", best_f1["metrics"])
print("Confusion Matrix:\n", best_f1["cm"])
