from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ---------------- Load ----------------
df = pd.read_csv(DATA_PATH)

# Drop non-informative columns (as per spec)
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, errors="ignore")

# Split X/y
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Stratified train-test split (important for churn imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Identify column types
categorical_cols = ["Geography", "Gender"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# Preprocessing for each type
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine into one preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Fit only on training data (IMPORTANT: no data leakage)
preprocessor.fit(X_train)

# Transform train and test
X_train_ready = preprocessor.transform(X_train)
X_test_ready = preprocessor.transform(X_test)

# Save artifacts for later use in training + Streamlit
joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
joblib.dump({"numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, ARTIFACTS_DIR / "columns.pkl")

print("✅ Preprocessing pipeline created & saved!")
print("Train shape:", X_train_ready.shape, "Test shape:", X_test_ready.shape)
print("Train churn %:", y_train.mean(), "Test churn %:", y_test.mean())
