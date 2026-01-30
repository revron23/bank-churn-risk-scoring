import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Explainability", page_icon="🧠", layout="wide")
st.title("🧠 Model Explainability (SHAP + PDP)")
st.write("These plots explain *why* the model predicts churn risk.")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOTS_DIR = PROJECT_ROOT / "plots"

# filenames you showed
files = {
    "SHAP Feature Importance (Global)": PLOTS_DIR / "shap_importance_bar.png",
    "SHAP Summary (Global Impact)": PLOTS_DIR / "shap_summary.png",
    "SHAP Waterfall (Single Customer Explanation)": PLOTS_DIR / "shap_waterfall_one_customer.png",
    "Partial Dependence Plots (PDP)": PLOTS_DIR / "pdp_churn.png",
}

# helper to show nicely
def show_image(title, path):
    st.subheader(title)
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.error(f"❌ Missing file: {path.name} (expected in plots/)")

# Layout
col1, col2 = st.columns(2)
with col1:
    show_image("SHAP Feature Importance (Global)", files["SHAP Feature Importance (Global)"])
with col2:
    show_image("SHAP Summary (Global Impact)", files["SHAP Summary (Global Impact)"])

st.divider()
show_image("SHAP Waterfall (Single Customer Explanation)", files["SHAP Waterfall (Single Customer Explanation)"])

st.divider()
show_image("Partial Dependence Plots (PDP)", files["Partial Dependence Plots (PDP)"])

st.caption("Tip: SHAP shows feature contribution direction (+ increases churn risk, − decreases). PDP shows average effect trends.")
