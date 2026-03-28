"""
Elo-based probability model.
Wraps the Elo rating system into a model interface.
Used as 25% of the final ensemble.
Provides real-time team strength updates that season-aggregate stats miss.
"""
import joblib
import os
import numpy as np
import pandas as pd
from typing import Dict, List

from features.elo import build_elo_ratings, elo_probability
from config import SAVED_MODELS_DIR, ELO_HOME_BONUS


class EloModel:
    """
    Elo model: reads pre-computed Elo features from the feature matrix.
    The actual Elo computation happens in features/elo.py.
    This class provides the model interface (predict_proba) using
    'elo_prob_home' column already in X.
    """
    def __init__(self):
        self.feature_names = None
        self.is_fitted = True  # Elo is always ready — no training needed

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight=None) -> "EloModel":
        self.feature_names = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return Elo win probability for the home team."""
        if "elo_prob_home" in X.columns:
            return X["elo_prob_home"].fillna(0.5).values
        # Fallback: derive from elo_diff if direct prob not available
        if "elo_diff" in X.columns:
            diffs = X["elo_diff"].fillna(0).values
            probs = 1.0 / (1.0 + 10 ** ((-diffs - ELO_HOME_BONUS) / 400))
            return probs
        return np.full(len(X), 0.5)

    def save(self, name: str = "elo_model") -> str:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, name: str = "elo_model") -> "EloModel":
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        return joblib.load(path)
