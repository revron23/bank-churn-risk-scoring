import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
os.makedirs("plots", exist_ok=True)

sns.set(style="whitegrid")

df = pd.read_csv("C:\\Users\\Admin\\OneDrive\\Documents\\PythonProjects\\bank-churn-risk-scoring\\dataset\\Churn_Modelling.csv")

print("Dataset Shape:", df.shape)
df.head()

df.info()

df.isnull().sum()
df.describe()
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
# Encode Gender using Label Encoding
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# One-hot encode Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

df.head()
plt.figure(figsize=(7,5))
sns.countplot(x='Exited', data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Exited")
plt.ylabel("Count")
plt.savefig("plots/churn_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(7,5))
sns.boxplot(x='Exited', y='Age', data=df)
plt.title("Age vs Customer Churn")
plt.savefig("plots/age_vs_churn.png", dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(7,5))
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title("Balance vs Customer Churn")
plt.savefig("plots/balance_vs_churn.png", dpi=300, bbox_inches='tight')
plt.show()
X = df.drop('Exited', axis=1)
y = df['Exited']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)
cm
print(classification_report(y_test, y_pred))
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("plots/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

feature_importance.head(10)
plt.figure(figsize=(8,6))
feature_importance.head(10).plot(kind='bar')
plt.title("Top 10 Feature Importances")
plt.ylabel("Importance Score")
plt.savefig("plots/feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()