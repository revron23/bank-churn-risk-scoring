from pathlib import Path
import numpy as np
import pandas as pd

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "dataset" / "Churn_Modelling.csv"
OUT_DIR = PROJECT_ROOT / "dataset" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "churn_fe.csv"

# ---------------- Load ----------------
df = pd.read_csv(RAW_PATH)

# Drop non-informative columns (as per spec)
df = df.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# ---------------- Feature Engineering ----------------
# 1) Balance-to-Salary ratio (safe division)
df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1e-6)

# 2) Products per tenure-year (tenure 0 safe)
df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

# 3) Engagement-product interaction
df["engagement_product"] = df["IsActiveMember"] * df["NumOfProducts"]

# 4) Age-tenure relationship
df["tenure_age_ratio"] = df["Tenure"] / (df["Age"] + 1e-6)

# (Optional but useful + very explainable)
# 5) Is balance zero?
df["is_zero_balance"] = (df["Balance"] == 0).astype(int)

# 6) Is senior customer? (simple business bucket)
df["is_senior"] = (df["Age"] >= 50).astype(int)

# 7) Has more than 1 product?
df["multi_product"] = (df["NumOfProducts"] > 1).astype(int)

# ---------------- Save ----------------
df.to_csv(OUT_PATH, index=False)

print("✅ Feature engineering done!")
print("Saved to:", OUT_PATH)
print("Shape:", df.shape)
print("New columns added:",
      ["balance_salary_ratio", "products_per_tenure", "engagement_product",
       "tenure_age_ratio", "is_zero_balance", "is_senior", "multi_product"])
