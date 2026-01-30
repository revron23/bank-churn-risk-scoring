from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "models"
PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.pkl"
MODEL_PATH = ARTIFACTS_DIR / "rf_tuned_model.pkl"

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = joblib.load(PREPROCESSOR_PATH)

# ---------------- Base RF ----------------
rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("model", rf)
])

# ---------------- Hyperparameter space ----------------
param_grid = {
    "model__n_estimators": [300, 500, 800],
    "model__max_depth": [None, 6, 10, 14],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.5]
}

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_grid,
    n_iter=20,
    scoring="f1",   # optimize for balance
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

# ---------------- Train ----------------
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("\n🏆 Best RF Params:", search.best_params_)

# ---------------- Evaluate ----------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)

print("\n✅ Tuned Random Forest Results")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nConfusion Matrix:\n", cm)

# ---------------- Save ----------------
joblib.dump(best_model, MODEL_PATH)
print(f"\n💾 Saved tuned RF to: {MODEL_PATH}")
