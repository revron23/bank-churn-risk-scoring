from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import shap

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

MODELS_DIR = PROJECT_ROOT / "models"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Choose which model to explain:
# Option A (recommended): tuned RF (more explainable)
RF_TUNED_PATH = MODELS_DIR / "rf_tuned_model.pkl"

# Option B: base RF (if tuned not available)
RF_BASE_PATH = MODELS_DIR / "random_forest_model.pkl"

# ---------------- Load data ----------------
df = pd.read_csv(DATA_PATH)
X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Load model pipeline ----------------
# tuned RF is a sklearn Pipeline (preprocessor + model)
if RF_TUNED_PATH.exists():
    pipeline = joblib.load(RF_TUNED_PATH)
    print("✅ Loaded tuned RF:", RF_TUNED_PATH)
elif RF_BASE_PATH.exists():
    pipeline = joblib.load(RF_BASE_PATH)
    print("✅ Loaded base RF:", RF_BASE_PATH)
else:
    raise FileNotFoundError("No RF model file found in /models folder.")

preprocessor = pipeline.named_steps["preprocessor"]
rf_model = pipeline.named_steps["model"]

# ---------------- Transform test data ----------------
X_test_transformed = preprocessor.transform(X_test)

# Get feature names after one-hot encoding
# Works for sklearn >= 1.0
feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrame for nicer SHAP plots
X_test_transformed_df = pd.DataFrame(
    X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
    columns=feature_names
)

# ---------------- SHAP Explainer ----------------
# TreeExplainer is perfect for RandomForest
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_transformed_df)

# For binary classification, shap_values is usually [class0, class1]
# We want churn class (1)
shap_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

# ---------------- 1) SHAP Summary Plot (global) ----------------
plt.figure()
shap.summary_plot(shap_class1, X_test_transformed_df, show=False)
plt.title("SHAP Summary Plot (Churn Class)")
plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- 2) SHAP Bar Plot (global importance) ----------------
plt.figure()
shap.summary_plot(shap_class1, X_test_transformed_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Churn Class)")
plt.savefig(PLOTS_DIR / "shap_importance_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- SHAP Explainer (modern) ----------------
explainer = shap.TreeExplainer(rf_model)

# This returns a shap.Explanation object (best format)
sv = explainer(X_test_transformed_df)

# For binary classifier, sv has shape: (n_samples, n_features, 2)
# We want class 1 (churn)
if len(sv.shape) == 3:
    sv_class1 = sv[:, :, 1]
else:
    # sometimes it's already (n_samples, n_features)
    sv_class1 = sv

# ---------------- 1) SHAP Summary Plot ----------------
plt.figure()
shap.summary_plot(sv_class1.values, X_test_transformed_df, show=False)
plt.title("SHAP Summary Plot (Churn Class)")
plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- 2) SHAP Bar Plot ----------------
plt.figure()
shap.summary_plot(sv_class1.values, X_test_transformed_df, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Churn Class)")
plt.savefig(PLOTS_DIR / "shap_importance_bar.png", dpi=300, bbox_inches="tight")
plt.close()

# ---------------- 3) Waterfall for ONE customer ----------------
idx = 0
plt.figure()
shap.plots.waterfall(sv_class1[idx], show=False)
plt.title("SHAP Waterfall (One Customer Explanation)")
plt.savefig(PLOTS_DIR / "shap_waterfall_one_customer.png", dpi=300, bbox_inches="tight")
plt.close()

# --- Sanity check 1: baseline vs mean predicted probability ---
proba_mean = pipeline.predict_proba(X_test)[:, 1].mean()
print("Mean predicted churn probability on test:", proba_mean)

# expected_value from SHAP object
try:
    print("SHAP base value (first 5):", sv_class1.base_values[:5])
    print("SHAP base value mean:", np.mean(sv_class1.base_values))
except Exception as e:
    print("Could not print base_values:", e)

# --- Sanity check 2: additivity check for ONE sample ---
idx = 0
pred_p = pipeline.predict_proba(X_test.iloc[[idx]])[0, 1]
shap_sum = sv_class1.base_values[idx] + sv_class1.values[idx].sum()
print("Model predicted prob (idx=0):", pred_p)
print("SHAP reconstructed output (idx=0):", shap_sum)

print("\n✅ SHAP plots saved in:", PLOTS_DIR)

print(" - shap_summary.png")
print(" - shap_importance_bar.png")
print(" - shap_waterfall_one_customer.png")
