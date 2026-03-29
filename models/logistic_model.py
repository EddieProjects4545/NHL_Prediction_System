"""
Logistic Regression model wrapper.
Provides well-calibrated probability estimates and interpretable coefficients.
Used as 25% of the final ensemble.
"""
import joblib
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from config import LOGISTIC_PARAMS, SAVED_MODELS_DIR


class LogisticModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**LOGISTIC_PARAMS)),
        ])
        self.calibrated = None
        self.feature_names = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray = None,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None,
            calibrate: bool = True) -> "LogisticModel":
        self.feature_names = list(X.columns)
        if calibrate:
            if X_val is not None and y_val is not None:
                # Fit on training data, calibrate on held-out val set — no leakage
                self.pipeline.fit(X.values, y.values)
                self.calibrated = CalibratedClassifierCV(
                    self.pipeline, cv="prefit", method="isotonic"
                )
                self.calibrated.fit(
                    X_val[self.feature_names].fillna(0).values,
                    y_val.values,
                )
            else:
                base = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(**LOGISTIC_PARAMS)),
                ])
                self.calibrated = CalibratedClassifierCV(base, cv=3, method="isotonic")
                sw_kw = {"sample_weight": sample_weight} if sample_weight is not None else {}
                self.calibrated.fit(X.values, y.values, **sw_kw)
        else:
            self.pipeline.fit(X.values, y.values)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self.feature_names].fillna(0).values
        if self.calibrated is not None:
            return self.calibrated.predict_proba(X_arr)[:, 1]
        return self.pipeline.predict_proba(X_arr)[:, 1]

    def get_coefficients(self) -> dict:
        """Return top feature importances (logistic coefficients)."""
        if self.calibrated is not None:
            # Extract coefficients from calibrated estimator
            try:
                clf = self.calibrated.calibrated_classifiers_[0].estimator
                coef = clf.named_steps["clf"].coef_[0]
                return dict(sorted(
                    zip(self.feature_names, coef),
                    key=lambda x: abs(x[1]), reverse=True
                )[:20])
            except Exception:
                return {}
        coef = self.pipeline.named_steps["clf"].coef_[0]
        return dict(sorted(
            zip(self.feature_names, coef),
            key=lambda x: abs(x[1]), reverse=True
        )[:20])

    def save(self, name: str = "logistic_ml") -> str:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, name: str = "logistic_ml") -> "LogisticModel":
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        return joblib.load(path)
