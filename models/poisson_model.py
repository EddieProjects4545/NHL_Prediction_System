"""
Poisson regression model for Over/Under totals prediction.

Goals in hockey arrive as approximately independent events → Poisson distribution.
We fit two separate Poisson GLMs:
  1. Home team goals (offensive rating vs away defensive rating + goalie)
  2. Away team goals (offensive rating vs home defensive rating + goalie)

P(over line) is derived by integrating the Poisson joint distribution.
P(under line) = 1 - P(over line) - P(exactly line, if integer)

Key features used:
  - combined_gf_pg, combined_ga_pg
  - delta_gf_pg, delta_ga_pg
  - combined_save_pct_l10
  - combined_shooting_pct_l10
  - h2h_avg_total
  - days_rest (fatigued teams score less)
  - is_playoff (tighter, lower-scoring games)
"""
import joblib
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import factorial
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple

from config import SAVED_MODELS_DIR

# Features most relevant for totals prediction
TOTALS_FEATURES = [
    "combined_gf_pg", "combined_ga_pg",
    "delta_gf_pg", "delta_ga_pg",
    "h_shots_for_pg", "a_shots_for_pg",
    "combined_shots_pg",
    "combined_goal_diff_l10",
    "combined_shot_diff_l10",
    "combined_save_pct_l10",
    "combined_shooting_pct_l10",
    "delta_goal_diff_l10",
    "delta_shot_diff_l10",
    "delta_save_pct_l10",
    "delta_shooting_pct_l10",
    "h2h_avg_total",
    "h2h_over_rate",
    "h_days_rest", "a_days_rest",
    "either_b2b",
    "is_playoff",
    "ou_line",
    "h_save_pct_l10", "a_save_pct_l10",
    "h_shooting_pct_l10", "a_shooting_pct_l10",
    "h_ot_rate_l10", "a_ot_rate_l10",
    "elo_diff",
]


class PoissonModel:
    def __init__(self):
        self.model_home = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", PoissonRegressor(alpha=1.0, max_iter=300)),
        ])
        self.model_away = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", PoissonRegressor(alpha=1.0, max_iter=300)),
        ])
        self.feature_names = None
        self.is_fitted = False
        self.calib_home = 1.0   # ratio: mean(actual_home) / mean(pred_home)
        self.calib_away = 1.0   # ratio: mean(actual_away) / mean(pred_away)

    def _get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in TOTALS_FEATURES if f in X.columns]
        return X[available].fillna(0)

    def fit(self, X: pd.DataFrame,
            y_goals_home: pd.Series,
            y_goals_away: pd.Series,
            sample_weight: np.ndarray = None) -> "PoissonModel":
        Xf = self._get_features(X)
        self.feature_names = list(Xf.columns)

        kw = {"reg__sample_weight": sample_weight} if sample_weight is not None else {}

        self.model_home.fit(Xf.values, y_goals_home.values, **kw)
        self.model_away.fit(Xf.values, y_goals_away.values, **kw)

        # Calibration: scale predictions to match training-set mean actuals
        pred_h = np.clip(self.model_home.predict(Xf.values), 1.0, 6.0)
        pred_a = np.clip(self.model_away.predict(Xf.values), 1.0, 6.0)
        self.calib_home = float(y_goals_home.mean() / pred_h.mean()) if pred_h.mean() > 0 else 1.0
        self.calib_away = float(y_goals_away.mean() / pred_a.mean()) if pred_a.mean() > 0 else 1.0

        self.is_fitted = True
        return self

    def predict_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Return (expected_home_goals, expected_away_goals) arrays."""
        Xf = X[self.feature_names].fillna(0)
        mu_home = self.model_home.predict(Xf.values)
        mu_away = self.model_away.predict(Xf.values)
        # Apply calibration factor learned during fit, then clip to NHL range
        mu_home = np.clip(mu_home * self.calib_home, 1.0, 6.0)
        mu_away = np.clip(mu_away * self.calib_away, 1.0, 6.0)
        return mu_home, mu_away

    def predict_over_prob(self, X: pd.DataFrame,
                          lines: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute P(total > line) and P(total < line) for each game.

        Uses Poisson convolution: P(sum = k) = sum_j P(home=j)*P(away=k-j)
        Computed up to k=15 goals total (>99.9% coverage).
        """
        mu_h, mu_a = self.predict_goals(X)
        n = len(mu_h)
        p_over  = np.zeros(n)
        p_under = np.zeros(n)

        MAX_GOALS = 15

        for i in range(n):
            lam_h = mu_h[i]
            lam_a = mu_a[i]
            line  = lines[i] if hasattr(lines, "__len__") else lines

            # Build joint distribution of total goals
            p_total = np.zeros(MAX_GOALS * 2 + 1)
            for h in range(MAX_GOALS + 1):
                ph = stats.poisson.pmf(h, lam_h)
                for a in range(MAX_GOALS + 1):
                    pa = stats.poisson.pmf(a, lam_a)
                    p_total[h + a] += ph * pa

            # Integrate
            over_prob  = 0.0
            under_prob = 0.0
            for k, p in enumerate(p_total):
                if k > line:
                    over_prob  += p
                elif k < line:
                    under_prob += p
                # k == line → push (split evenly conceptually, but neither wins)

            p_over[i]  = over_prob
            p_under[i] = under_prob

        return p_over, p_under

    def predict_total_prob_single(self, mu_home: float, mu_away: float,
                                  line: float) -> Tuple[float, float]:
        """Convenience method for a single game."""
        MAX_GOALS = 15
        p_total = np.zeros(MAX_GOALS * 2 + 1)
        for h in range(MAX_GOALS + 1):
            ph = stats.poisson.pmf(h, mu_home)
            for a in range(MAX_GOALS + 1):
                pa = stats.poisson.pmf(a, mu_away)
                p_total[h + a] += ph * pa
        over_prob  = sum(p for k, p in enumerate(p_total) if k > line)
        under_prob = sum(p for k, p in enumerate(p_total) if k < line)
        return round(over_prob, 4), round(under_prob, 4)

    def save(self, name: str = "poisson_totals") -> str:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, name: str = "poisson_totals") -> "PoissonModel":
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        return joblib.load(path)
