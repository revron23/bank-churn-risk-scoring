import pandas as pd

df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Documents\\PythonProjects\\bank-churn-risk-scoring\\dataset\\Churn_Modelling.csv")
print(df.shape)
print(df.columns.tolist())
print(df["Exited"].value_counts(normalize=True))
