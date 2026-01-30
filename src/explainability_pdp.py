import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rf_tuned_model.pkl"
PLOTS_DIR = PROJECT_ROOT / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Load model
rf_model = joblib.load(MODEL_PATH)
print("✅ Loaded RF model")

# Features to analyze
features = [
    "Age",
    "IsActiveMember",
    "Balance",
    "NumOfProducts"
]

# Generate PDP plots
fig, ax = plt.subplots(2, 2, figsize=(12, 10))

PartialDependenceDisplay.from_estimator(
    rf_model,
    X,
    features,
    grid_resolution=20,
    ax=ax
)

plt.suptitle("Partial Dependence Plots (Churn Risk)", fontsize=16)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "pdp_churn.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ PDP plot saved to:", PLOTS_DIR / "pdp_churn.png")
