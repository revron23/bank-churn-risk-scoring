import streamlit as st
import json
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Model Overview", page_icon="📊", layout="wide")
st.title("📊 Model Overview")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.json"
DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"

# -------- Dataset summary --------
st.subheader("Dataset Summary")
df = pd.read_csv(DATA_PATH)
st.write(f"Rows: **{df.shape[0]}** | Columns: **{df.shape[1]}**")
st.write("Target: **Exited** (1 = churn, 0 = stay)")
st.write("Class distribution:")
st.bar_chart(df["Exited"].value_counts())

st.divider()

# -------- Feature Engineering --------
st.subheader("Feature Engineering (Added Features)")
added = [
    "balance_salary_ratio",
    "products_per_tenure",
    "engagement_product",
    "tenure_age_ratio",
    "is_zero_balance",
    "is_senior",
    "multi_product"
]
st.write(added)

st.divider()

# -------- Model details --------
st.subheader("Final Model + Metrics")

if METRICS_PATH.exists():
    data = json.loads(METRICS_PATH.read_text())

    st.write(f"**Model:** {data.get('model_name', 'N/A')}")
    st.write(f"**Threshold:** {data.get('threshold', 'N/A')}")

    # weights
    w_rf = data.get("best_rf_weight", None)
    if w_rf is not None:
        st.write(f"**Blend Weights:** RF = {w_rf}, XGB = {1 - float(w_rf):.2f}")

    # metrics table
    st.subheader("Performance Metrics")

    m = data.get("metrics", {})

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{m['accuracy']:.2f}")
    c2.metric("Precision", f"{m['precision']:.2f}")
    c3.metric("Recall", f"{m['recall']:.2f}")
    c4.metric("F1-score", f"{m['f1']:.2f}")
    c5.metric("ROC-AUC", f"{m['roc_auc']:.2f}")


    # confusion matrix
    cm = data.get("confusion_matrix", None)
    if cm is not None:
        st.write("**Confusion Matrix** ( [[TN, FP],[FN, TP]] )")
        st.subheader("Confusion Matrix")

        cm_df = pd.DataFrame(
         cm,
         index=["Actual: Stay (0)", "Actual: Churn (1)"],
         columns=["Predicted: Stay (0)", "Predicted: Churn (1)"]
        )

        st.dataframe(cm_df, use_container_width=True)


else:
    st.error("❌ models/model_metrics.json not found. Create it using your ensemble_blend results.")
