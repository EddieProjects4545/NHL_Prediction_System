"""
Puck Line model — predicts probability of covering -1.5 / +1.5.

Key insight: Puck line coverage is DECOUPLED from moneyline win probability.
Teams that win by blowouts (high Corsi, elite PP, deep scoring) cover -1.5
far more often than their win% suggests. Conversely, goalie-dependent teams
win close games but rarely by 2+.

Critical features:
  - cover_rate_minus1_5: team's historical rate of winning by 2+
  - avg_margin_when_win: average margin in wins
  - blowout_rate: % of games won by 3+
  - en_goals_for/against: empty net goal tendencies (3rd period, leading by 1)
  - corsi_pct_5v5: possession dominance → score inflation
  - net_pp_advantage: PP-PK differential drives blowout potential
  - pl_matchup_score: combined cover/fade matchup index
  - one_goal_game_rate: teams that play tight → fade on puck line
  - ot_game_rate: high OT rate teams rarely cover -1.5

Two separate models:
  - home_model: P(home covers -1.5)
  - away_model: P(away covers +1.5)
"""
import joblib
import os
import re
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from config import XGBOOST_PL_PARAMS, SAVED_MODELS_DIR

# Features most relevant for puck line coverage
PUCKLINE_FEATURES = [
    # Cover rate history
    "h_cover_rate_minus1_5", "a_cover_rate_minus1_5",
    "h_cover_rate_plus1_5",  "a_cover_rate_plus1_5",
    "h_avg_margin_when_win", "a_avg_margin_when_win",
    "h_blowout_rate",        "a_blowout_rate",
    "h_one_goal_game_rate",  "a_one_goal_game_rate",
    "h_ot_game_rate",        "a_ot_game_rate",
    # Empty net
    "h_en_cover_proxy",
    # Possession / shot quality
    # Scoring rates
    "delta_gf_pg",
    "delta_goal_diff_pg",
    "delta_goal_diff_l10",
    "delta_shot_diff_pg",
    "delta_shot_diff_l10",
    "h_gf_pg",               "a_ga_pg",
    "h_gf_ga_ratio",
    "h_goal_diff_l10",       "a_goal_diff_l10",
    "h_shot_diff_l10",       "a_shot_diff_l10",
    "delta_save_pct_l10",
    "delta_shooting_pct_l10",
    "h_save_pct_l10",        "a_save_pct_l10",
    "h_shooting_pct_l10",    "a_shooting_pct_l10",
    # H2H
    "h2h_cover_rate_minus1_5",
    "h2h_goal_diff_avg",
    # Elo strength differential
    "elo_diff",
    "elo_prob_home",
    # Matchup composite
    "pl_matchup_score",
    # Form
    "delta_l10_win_pct",
    "delta_streak_value",
    "delta_regulation_win_pct_l10",
    "h_one_goal_rate_l10",   "a_one_goal_rate_l10",
    "h_ot_rate_l10",         "a_ot_rate_l10",
    # Context
    "either_b2b",
    "h_is_back_to_back",     "a_is_back_to_back",
    "is_playoff",
    "home_advantage",
]


class PuckLineModel:
    """Predicts P(home covers -1.5) and P(away covers +1.5) independently."""

    def __init__(self):
        self.home_model   = XGBClassifier(**XGBOOST_PL_PARAMS)
        self.away_model   = XGBClassifier(**XGBOOST_PL_PARAMS)
        self.home_calib   = None
        self.away_calib   = None
        self.feature_names = None
        self.is_fitted    = False

    def _get_features(self, X: pd.DataFrame) -> pd.DataFrame:
        available = [f for f in PUCKLINE_FEATURES if f in X.columns]
        return X[available].fillna(0)

    def fit(self, X: pd.DataFrame,
            y_pl_home: pd.Series,
            y_pl_away: pd.Series,
            sample_weight: np.ndarray = None) -> "PuckLineModel":

        Xf = self._get_features(X)
        self.feature_names = list(Xf.columns)
        Xv = Xf.values

        kw = {"sample_weight": sample_weight} if sample_weight is not None else {}

        # Home covers -1.5
        self.home_model.fit(Xv, y_pl_home.values, verbose=False, **kw)
        self.home_calib = CalibratedClassifierCV(
            XGBClassifier(**XGBOOST_PL_PARAMS), cv=5, method="isotonic"
        )
        self.home_calib.fit(Xv, y_pl_home.values, **kw)

        # Away covers +1.5
        self.away_model.fit(Xv, y_pl_away.values, verbose=False, **kw)
        self.away_calib = CalibratedClassifierCV(
            XGBClassifier(**XGBOOST_PL_PARAMS), cv=5, method="isotonic"
        )
        self.away_calib.fit(Xv, y_pl_away.values, **kw)

        self.is_fitted = True
        return self

    def predict_proba_home_minus1_5(self, X: pd.DataFrame) -> np.ndarray:
        """P(home team wins by 2+ goals, covering -1.5)."""
        Xf = X[self.feature_names].fillna(0)
        if self.home_calib:
            return self.home_calib.predict_proba(Xf.values)[:, 1]
        return self.home_model.predict_proba(Xf.values)[:, 1]

    def predict_proba_away_plus1_5(self, X: pd.DataFrame) -> np.ndarray:
        """P(away team covers +1.5 — loses by ≤1 or wins)."""
        Xf = X[self.feature_names].fillna(0)
        if self.away_calib:
            return self.away_calib.predict_proba(Xf.values)[:, 1]
        return self.away_model.predict_proba(Xf.values)[:, 1]

    def feature_importance_home(self, top_n: int = 15) -> dict:
        scores = self.home_model.get_booster().get_score(importance_type="gain")
        return self._map_feature_scores(scores, top_n)

    def feature_importance_away(self, top_n: int = 15) -> dict:
        scores = self.away_model.get_booster().get_score(importance_type="gain")
        return self._map_feature_scores(scores, top_n)

    def _map_feature_scores(self, scores: dict, top_n: int) -> dict:
        mapped_scores = {}
        for key, value in scores.items():
            mapped = key
            match = re.fullmatch(r"f(\d+)", key)
            if match and self.feature_names:
                idx = int(match.group(1))
                if 0 <= idx < len(self.feature_names):
                    mapped = self.feature_names[idx]
            mapped_scores[mapped] = value
        return dict(sorted(mapped_scores.items(), key=lambda x: x[1], reverse=True)[:top_n])

    def save(self, name: str = "puckline") -> str:
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, name: str = "puckline") -> "PuckLineModel":
        path = os.path.join(SAVED_MODELS_DIR, f"{name}.joblib")
        return joblib.load(path)
