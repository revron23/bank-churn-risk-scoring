from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Your current dataset path (as per your setup)
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "logistic_model.pkl"

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)

# If target isn't present, crash early (safety)
assert "Exited" in df.columns, "Target column 'Exited' not found!"

X = df.drop("Exited", axis=1)
y = df["Exited"]

# Stratified split (same as preprocessing stage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------- Load preprocessor ----------------
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ---------------- Build model pipeline ----------------
# Using class_weight='balanced' because churn is imbalanced (~20%)
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    solver="lbfgs"
)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
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

print("✅ Logistic Regression Trained!")
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
