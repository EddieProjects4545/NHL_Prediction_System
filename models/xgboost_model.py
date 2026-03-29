"""
XGBoost classifier wrapper for moneyline prediction.
Primary model in the ensemble (50% weight).
Captures non-linear interactions between features that logistic regression misses.
"""
import joblib
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import XGBOOST_ML_PARAMS, SAVED_MODELS_DIR


class XGBoostModel:
    def __init__(self, params: dict = None):
        self.params = params or XGBOOST_ML_PARAMS
        self.model = XGBClassifier(**self.params)
        self.calibrated = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series,
            X_val: pd.DataFrame = None, y_val: pd.Series = None,
            calibrate: bool = True,
            sample_weight: np.ndarray = None,
            scale_pos_weight: float = None) -> "XGBoostModel":
        self.feature_names = list(X.columns)
        X_arr = X.values
        y_arr = y.values

        if scale_pos_weight is not None:
            self.model.set_params(scale_pos_weight=scale_pos_weight)

        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        if X_val is not None and y_val is not None:
            self.model.fit(
                X_arr, y_arr,
                eval_set=[(X_val[self.feature_names].values, y_val.values)],
                verbose=False,
                **fit_kwargs
            )
        else:
            self.model.fit(X_arr, y_arr, verbose=False, **fit_kwargs)

        if calibrate:
            base = XGBClassifier(**{**self.params, "n_estimators": self.model.best_iteration + 1
                                    if hasattr(self.model, "best_iteration") else self.params["n_estimators"]})
            self.calibrated = CalibratedClassifierCV(base, cv=5, method="isotonic")
            self.calibrated.fit(X_arr, y_arr,
                                **{"sample_weight": sample_weight} if sample_weight is not None else {})

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self.feature_names].fillna(0).values
        if self.calibrated is not None:
            return self.calibrated.predict_proba(X_arr)[:, 1]
        return self.model.predict_proba(X_arr)[:, 1]

    def feature_importance(self, top_n: int = 20) -> dict:
        """Return top-N features by XGBoost gain importance."""
        scores = self.model.get_booster().get_score(importance_type="gain")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_scores[:top_n])

    def save(self, name: str = "xgboost_ml") -> str:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, name: str = "xgboost_ml") -> "XGBoostModel":
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        return joblib.load(path)
