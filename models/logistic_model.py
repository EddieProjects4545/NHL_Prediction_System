"""
Logistic Regression model wrapper.
Provides well-calibrated probability estimates and interpretable coefficients.
Used as 25% of the final ensemble.
"""
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import LOGISTIC_PARAMS, SAVED_MODELS_DIR


class ScaledLogisticEstimator(BaseEstimator, ClassifierMixin):
    """scikit-learn compatible estimator with explicit weighted scaling."""
    _estimator_type = "classifier"

    def __init__(self, params: dict = None):
        self.params = params or LOGISTIC_PARAMS
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(**self.params)
        self.keep_mask_ = None

    def fit(self, X, y, sample_weight=None):
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        variances = np.nanvar(X_arr, axis=0)
        self.keep_mask_ = np.isfinite(variances) & (variances > 1e-12)
        if not np.any(self.keep_mask_):
            self.keep_mask_ = np.ones(X_arr.shape[1], dtype=bool)
        X_arr = X_arr[:, self.keep_mask_]
        sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.scaler.fit(X_arr, **sw_kw)
        X_scaled = self.scaler.transform(X_arr)
        self.clf.fit(X_scaled, y_arr, **sw_kw)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X)
        if self.keep_mask_ is not None:
            X_arr = X_arr[:, self.keep_mask_]
        X_scaled = self.scaler.transform(X_arr)
        return self.clf.predict_proba(X_scaled)

    def decision_function(self, X):
        X_arr = np.asarray(X)
        if self.keep_mask_ is not None:
            X_arr = X_arr[:, self.keep_mask_]
        X_scaled = self.scaler.transform(X_arr)
        return self.clf.decision_function(X_scaled)


class LogisticModel:
    def __init__(self):
        self.model = ScaledLogisticEstimator()
        self.calibrated = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray = None,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None,
            calibrate: bool = True) -> "LogisticModel":
        self.feature_names = list(X.columns)
        X_arr = X[self.feature_names].fillna(0).values
        y_arr = y.values
        sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
        self.model = ScaledLogisticEstimator()
        self.model.fit(X_arr, y_arr, **sw_kw)
        self.calibrated = None
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self.feature_names].fillna(0).values
        if self.calibrated is not None:
            return self.calibrated.predict_proba(X_arr)[:, 1]
        return self.model.predict_proba(X_arr)[:, 1]

    def get_coefficients(self) -> dict:
        """Return top feature importances (logistic coefficients)."""
        try:
            coef = self.model.clf.coef_[0]
            return dict(sorted(
                zip(self.feature_names, coef),
                key=lambda x: abs(x[1]), reverse=True
            )[:20])
        except Exception:
            return {}

    def save(self, name: str = "logistic_ml") -> str:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, name: str = "logistic_ml") -> "LogisticModel":
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        return joblib.load(path)
