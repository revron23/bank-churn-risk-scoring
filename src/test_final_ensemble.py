from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "dataset" / "processed" / "churn_fe.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "final_ensemble.joblib"

def main():
    # load one sample row
    df = pd.read_csv(DATA_PATH)
    X = df.drop("Exited", axis=1)

    sample = X.iloc[[0]]  # keep as DataFrame (important)
    model = joblib.load(MODEL_PATH)

    proba = model.predict_proba(sample)[0, 1]
    pred = model.predict(sample)[0]

    print("✅ Sample prediction works!")
    print("Churn probability:", round(float(proba), 4))
    print("Predicted class  :", int(pred))

if __name__ == "__main__":
    main()
