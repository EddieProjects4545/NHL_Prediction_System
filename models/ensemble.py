"""
Weighted ensemble combiner for moneyline prediction.

Combines:
  - Logistic Regression  → 25%
  - XGBoost              → 50%
  - Elo                  → 25%

Design rationale:
  XGBoost gets the highest weight because it captures non-linear feature
  interactions. Elo and Logistic Regression serve as calibration anchors
  and reduce variance from XGBoost overfitting on small late-season samples.

The ensemble also computes model agreement (std dev across models) which
feeds into the confidence scorer.
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple

from models.logistic_model import LogisticModel
from models.xgboost_model  import XGBoostModel
from models.elo_model       import EloModel
from config import ENSEMBLE_WEIGHTS


class EnsembleModel:
    def __init__(self,
                 logistic: LogisticModel = None,
                 xgboost:  XGBoostModel  = None,
                 elo:      EloModel      = None,
                 weights:  Dict[str, float] = None):
        self.logistic = logistic or LogisticModel()
        self.xgboost  = xgboost  or XGBoostModel()
        self.elo      = elo      or EloModel()
        self.weights  = weights  or ENSEMBLE_WEIGHTS
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray = None,
            X_val: pd.DataFrame = None,
            y_val: pd.Series = None) -> "EnsembleModel":
        print("  Training Logistic Regression...")
        self.logistic.fit(X, y)
        print("  Training XGBoost...")
        self.xgboost.fit(X, y, X_val=X_val, y_val=y_val,
                         sample_weight=sample_weight)
        print("  Fitting Elo model...")
        self.elo.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """
        Return (ensemble_prob_home_win, component_probs dict).
        component_probs: {"logistic", "xgboost", "elo", "std"}
        """
        p_lr  = self.logistic.predict_proba(X)
        p_xgb = self.xgboost.predict_proba(X)
        p_elo = self.elo.predict_proba(X)

        w = self.weights
        p_ensemble = (
            w["logistic"] * p_lr +
            w["xgboost"]  * p_xgb +
            w["elo"]       * p_elo
        )

        # Model agreement: std across the three models
        stack = np.vstack([p_lr, p_xgb, p_elo])
        model_std = np.std(stack, axis=0)

        components = {
            "logistic" : p_lr,
            "xgboost"  : p_xgb,
            "elo"       : p_elo,
            "std"       : model_std,
        }

        return p_ensemble, components

    def predict_single(self, X_row: pd.Series) -> Tuple[float, Dict]:
        X_df = pd.DataFrame([X_row])
        probs, components = self.predict_proba(X_df)
        return float(probs[0]), {k: float(v[0]) for k, v in components.items()}
