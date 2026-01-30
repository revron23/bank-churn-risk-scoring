from pathlib import Path
import joblib

from final_ensemble_model import FinalEnsembleModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

RF_PATH = MODELS_DIR / "random_forest_model.pkl"
XGB_PATH = MODELS_DIR / "xgboost_model.pkl"
CFG_PATH = MODELS_DIR / "ensemble_config.pkl"

OUT_PATH = MODELS_DIR / "final_ensemble.joblib"

def main():
    rf = joblib.load(RF_PATH)
    xgb = joblib.load(XGB_PATH)
    cfg = joblib.load(CFG_PATH)

    model = FinalEnsembleModel(
        rf_pipeline=rf,
        xgb_pipeline=xgb,
        w_rf=cfg["w_rf"],
        threshold=cfg["threshold"]
    )

    joblib.dump(model, OUT_PATH)
    print(f"✅ Saved final ensemble model to: {OUT_PATH}")

if __name__ == "__main__":
    main()
