import numpy as np

class FinalEnsembleModel:
    """
    Wraps two trained sklearn pipelines (rf, xgb) + blend config (w_rf, threshold)
    into a single object with predict_proba and predict methods.
    """

    def __init__(self, rf_pipeline, xgb_pipeline, w_rf: float, threshold: float):
        self.rf = rf_pipeline
        self.xgb = xgb_pipeline
        self.w_rf = float(w_rf)
        self.threshold = float(threshold)

    def predict_proba(self, X):
        # X can be a DataFrame with raw columns (same as training)
        rf_proba = self.rf.predict_proba(X)[:, 1]
        xgb_proba = self.xgb.predict_proba(X)[:, 1]
        blend = self.w_rf * rf_proba + (1 - self.w_rf) * xgb_proba

        # return in sklearn style: [P(class0), P(class1)]
        return np.vstack([1 - blend, blend]).T

    def predict(self, X):
        proba_1 = self.predict_proba(X)[:, 1]
        return (proba_1 >= self.threshold).astype(int)
