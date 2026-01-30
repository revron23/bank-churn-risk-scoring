from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "random_forest_model.pkl"

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

# ---------------- Load preprocessor ----------------
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ---------------- Model ----------------
rf = RandomForestClassifier(
    n_estimators=400,      # stronger than default
    max_depth=None,        # let trees grow, RF controls overfit via averaging
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", rf)
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

print("✅ Random Forest Trained!")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------- Save model ----------------
joblib.dump(clf, MODEL_PATH)
print(f"\n💾 Saved trained model to: {MODEL_PATH}")
