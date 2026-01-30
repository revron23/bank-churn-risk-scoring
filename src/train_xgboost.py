from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from xgboost import XGBClassifier

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.pkl"

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)
assert "Exited" in df.columns, "Target column 'Exited' not found!"

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- Handle class imbalance ----------------
# scale_pos_weight = (negative / positive)
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

# ---------------- Load preprocessor ----------------
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ---------------- XGBoost Model ----------------
xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    min_child_weight=2,
    gamma=0.0,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb)
])

# ---------------- Train ----------------
clf.fit(X_train, y_train)

# ---------------- Predict ----------------
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# ---------------- Evaluate ----------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("✅ XGBoost Trained!")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- Save ----------------
joblib.dump(clf, MODEL_PATH)
print(f"\n💾 Saved trained model to: {MODEL_PATH}")
