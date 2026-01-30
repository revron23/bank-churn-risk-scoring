import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# ✅ make sure root imports work (for final_ensemble_model during unpickle)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "models" / "final_ensemble.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

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

st.title("🧪 What-If Scenario Simulator")
st.write("Change values and see how churn probability changes.")

st.subheader("Base Customer")

col1, col2 = st.columns(2)
with col1:
    CreditScore = st.number_input("CreditScore", 300, 900, 650)
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.slider("Age", 18, 100, 35)
    Tenure = st.slider("Tenure", 0, 10, 5)

with col2:
    Balance = st.slider("Balance", 0, 300000, 50000, step=1000)
    NumOfProducts = st.slider("NumOfProducts", 1, 4, 2)
    HasCrCard = st.selectbox("HasCrCard", [0, 1], index=1)
    IsActiveMember = st.selectbox("IsActiveMember", [0, 1], index=1)
    EstimatedSalary = st.slider("EstimatedSalary", 0, 300000, 100000, step=1000)

base_df = pd.DataFrame([{
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

base_final = add_features(base_df)
base_proba = float(model.predict_proba(base_final)[0, 1])

st.divider()
st.subheader("Try Changes (What-If)")

Age2 = st.slider("What-if Age", 18, 100, Age)
Balance2 = st.slider("What-if Balance", 0, 300000, Balance, step=1000)
NumOfProducts2 = st.slider("What-if NumOfProducts", 1, 4, NumOfProducts)
IsActiveMember2 = st.selectbox("What-if IsActiveMember", [0, 1], index=IsActiveMember)

what_df = base_df.copy()
what_df.loc[0, "Age"] = Age2
what_df.loc[0, "Balance"] = Balance2
what_df.loc[0, "NumOfProducts"] = NumOfProducts2
what_df.loc[0, "IsActiveMember"] = IsActiveMember2

what_final = add_features(what_df)
what_proba = float(model.predict_proba(what_final)[0, 1])
delta = (what_proba - base_proba) * 100

st.subheader("📌 Comparison")
a, b, c = st.columns(3)
a.metric("Base Probability", f"{base_proba*100:.2f}%")
b.metric("What-If Probability", f"{what_proba*100:.2f}%")
c.metric("Change (Δ)", f"{delta:+.2f}%")

st.write("### Visual")
st.progress(int(what_proba * 100))
st.caption(f"Decision threshold: {model.threshold:.2f} (prob ≥ threshold → churn)")
