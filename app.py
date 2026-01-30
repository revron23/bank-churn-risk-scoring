import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ---------------- Paths ----------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "final_ensemble.joblib"

# ---------------- Load model (cached) ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()
@st.cache_data
def load_reference_data():
    df = pd.read_csv(PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv")
    return df

ref_df = load_reference_data()

# ---------------- Feature Engineering (same as your script) ----------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["balance_salary_ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1e-6)
    df["products_per_tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["engagement_product"] = df["IsActiveMember"] * df["NumOfProducts"]
    df["tenure_age_ratio"] = df["Tenure"] / (df["Age"] + 1e-6)

    df["is_zero_balance"] = (df["Balance"] == 0).astype(int)
    df["is_senior"] = (df["Age"] >= 50).astype(int)
    df["multi_product"] = (df["NumOfProducts"] > 1).astype(int)

    return df

# ---------------- UI ----------------
st.set_page_config(page_title="Churn Risk Scoring", page_icon="📉", layout="centered")
st.title("📉 Customer Churn Risk Scoring")
st.write("Enter customer details to predict churn probability.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        CreditScore = st.number_input("CreditScore", min_value=300, max_value=900, value=650)
        Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        Gender = st.selectbox("Gender", ["Male", "Female"])
        Age = st.number_input("Age", min_value=18, max_value=100, value=35)
        Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)

    with col2:
        Balance = st.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
        NumOfProducts = st.number_input("NumOfProducts", min_value=1, max_value=4, value=2)
        HasCrCard = st.selectbox("HasCrCard", [0, 1], index=1)
        IsActiveMember = st.selectbox("IsActiveMember", [0, 1], index=1)
        EstimatedSalary = st.number_input("EstimatedSalary", min_value=0.0, max_value=300000.0, value=100000.0, step=1000.0)


    submitted = st.form_submit_button("Predict Churn Risk")
if st.button("Load a real churned customer example"):
    churned = ref_df[ref_df["Exited"] == 1].sample(1, random_state=42)
    st.write("Sample churned row (from dataset):")
    st.dataframe(churned)

if submitted:
    # Build input dataframe (raw columns)
    input_df = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }])

    # Add engineered features
    final_df = add_features(input_df)
    st.write("### 🧪 Sanity Check (Input vs Dataset Range)")

    cols_to_check = ["CreditScore","Age","Tenure","Balance","NumOfProducts","EstimatedSalary"]
    stats = ref_df[cols_to_check].describe(percentiles=[0.01,0.05,0.5,0.95,0.99]).T

    user_vals = final_df[cols_to_check].iloc[0]
    out_flags = {}
    for c in cols_to_check:
     p01, p99 = stats.loc[c, "1%"], stats.loc[c, "99%"]
     out_flags[c] = not (p01 <= user_vals[c] <= p99)

    check_table = pd.DataFrame({
     "Your Value": user_vals,
     "P01 (1%)": stats["1%"],
     "P99 (99%)": stats["99%"],
     "Out of Range?": pd.Series(out_flags)
     })

    st.dataframe(check_table)

    # Predict
    proba = float(model.predict_proba(final_df)[0, 1])

    st.subheader("📌 Prediction Result")
    st.metric("Churn Probability", f"{proba*100:.2f}%")

    # Risk badge
    if proba < 0.30:
        st.success("🟢 LOW RISK — customer likely to stay")
    elif proba < 0.60:
        st.warning("🟠 MEDIUM RISK — monitor & engage")
    else:
        st.error("🔴 HIGH RISK — retention action recommended")

    st.write("### Probability Visual")
    st.progress(int(proba * 100))

    # Show model threshold (your ensemble uses this)
    st.write(f"**Model Decision Threshold:** {model.threshold:.2f}")
    st.caption("If probability ≥ threshold → predicted as churn (1).")
