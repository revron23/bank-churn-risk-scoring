# Customer Churn Risk Scoring System 🏦📉

An end-to-end machine learning system to predict customer churn risk for a banking dataset using ensemble learning and explainable AI.

## 🔥 Highlights
- Full ML pipeline: preprocessing → feature engineering → training → evaluation → deployment  
- Final model: **Weighted Ensemble (Random Forest + XGBoost)**  
- Explainability: **SHAP (global + local)** + **Partial Dependence Plots (PDP)**  
- Deployment: **Streamlit Web Dashboard**  
  - Risk calculator  
  - Probability visualization  
  - What-if simulator  
  - Explainability page  

## 📊 Performance (Test Set)
- **Accuracy:** 0.86  
- **Precision:** 0.74  
- **Recall:** 0.58  
- **F1-score:** 0.65  
- **ROC-AUC:** 0.88  

## 🧠 Explainability Outputs
Plots are available in the `plots/` folder:
- SHAP summary plot  
- SHAP feature importance  
- SHAP waterfall (single customer)  
- PDP plots  

## 🖥️ How to Run
[live demo link]{https://bank-churn-risk-scoring-bk5g98hnwabutwr9dqfcgf.streamlit.app/}

