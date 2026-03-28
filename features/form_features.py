"""
Rolling form features computed from raw game-result lists.

For each team, we build:
  - Rolling GF/GA averages over L5, L10, L20
  - Rolling win rate over L5, L10, L20
  - Scoring trend (slope of GF over L10)
  - Days-of-rest (fatigue / freshness)
  - Back-to-back flag
  - Home/road recent form split

These are computed per-game (pre-game snapshot) for use in the training
matrix, and also as current-state values for prediction.
"""
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional
import numpy as np

from config import ROLLING_WINDOWS, MIN_GAMES_FOR_FEAT


# ─── Team game log builder ────────────────────────────────────────────────────

def build_team_game_logs(game_results: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Return per-team ordered game log.
    Each entry: {date, gf, ga, home, win, ot}
    """
    logs: Dict[str, List[Dict]] = defaultdict(list)
    for g in sorted(game_results, key=lambda x: x.get("date", "")):
        ht = g.get("home_team")
        at = g.get("away_team")
        hg = g.get("home_goals", 0)
        ag = g.get("away_goals", 0)
        ot = g.get("ot_flag", False)
        if ht:
            logs[ht].append({
                "date": g["date"], "gf": hg, "ga": ag,
                "home": True, "win": hg > ag, "ot": ot,
                "game_id": g.get("game_id"),
            })
        if at:
            logs[at].append({
                "date": g["date"], "gf": ag, "ga": hg,
                "home": False, "win": ag > hg, "ot": ot,
                "game_id": g.get("game_id"),
            })
    return dict(logs)


def _rolling_stats(log: List[Dict], window: int) -> Dict:
    """Compute rolling stats over last `window` games."""
    recent = log[-window:]
    n = len(recent)
    if n == 0:
        return {
            f"gf_last{window}": 0, f"ga_last{window}": 0,
            f"win_pct_last{window}": 0.5,
            f"goal_diff_last{window}": 0,
        }
    gf_vals = [g["gf"] for g in recent]
    ga_vals = [g["ga"] for g in recent]
    wins    = sum(1 for g in recent if g["win"])
    return {
        f"gf_last{window}"       : round(sum(gf_vals) / n, 4),
        f"ga_last{window}"       : round(sum(ga_vals) / n, 4),
        f"win_pct_last{window}"  : round(wins / n, 4),
        f"goal_diff_last{window}": round((sum(gf_vals) - sum(ga_vals)) / n, 4),
    }


def _scoring_trend(log: List[Dict], window: int = 10) -> float:
    """
    Linear slope of GF over last `window` games.
    Positive = improving offence; negative = declining.
    """
    recent = log[-window:]
    if len(recent) < 3:
        return 0.0
    y = np.array([g["gf"] for g in recent], dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return round(float(slope), 4)


def _days_rest(log: List[Dict], game_date: str) -> int:
    """
    Days since last game. Returns 7 (capped) if no prior games.
    """
    if not log:
        return 7
    last_date = date.fromisoformat(log[-1]["date"])
    current   = date.fromisoformat(game_date)
    diff      = (current - last_date).days
    return min(diff, 7)


def get_current_form(team_logs: Dict[str, List[Dict]],
                     game_date: str = None) -> Dict[str, Dict]:
    """
    Return current form features for all teams (for prediction).
    """
    if game_date is None:
        game_date = date.today().isoformat()
    result = {}
    for team, log in team_logs.items():
        feats: Dict = {}
        for w in ROLLING_WINDOWS:
            feats.update(_rolling_stats(log, w))
        feats["scoring_trend_l10"]   = _scoring_trend(log, 10)
        feats["days_rest"]           = _days_rest(log, game_date)
        feats["is_back_to_back"]     = int(feats["days_rest"] <= 1)
        feats["games_played_season"] = len(log)

        # Home / road split from recent 10
        recent10 = log[-10:]
        home_games = [g for g in recent10 if g["home"]]
        road_games = [g for g in recent10 if not g["home"]]
        feats["home_form_l10"] = (
            round(sum(g["win"] for g in home_games) / len(home_games), 4)
            if home_games else 0.5
        )
        feats["road_form_l10"] = (
            round(sum(g["win"] for g in road_games) / len(road_games), 4)
            if road_games else 0.5
        )
        result[team] = feats
    return result


def get_pregame_form(team: str, log_before_game: List[Dict],
                     game_date: str) -> Dict:
    """
    Form snapshot for a single team *before* a specific game.
    Used when building the training feature matrix.
    """
    feats: Dict = {}
    for w in ROLLING_WINDOWS:
        feats.update(_rolling_stats(log_before_game, w))
    feats["scoring_trend_l10"]   = _scoring_trend(log_before_game, 10)
    feats["days_rest"]           = _days_rest(log_before_game, game_date)
    feats["is_back_to_back"]     = int(feats["days_rest"] <= 1)
    feats["games_played_season"] = len(log_before_game)

    recent10 = log_before_game[-10:]
    home_g = [g for g in recent10 if g["home"]]
    road_g = [g for g in recent10 if not g["home"]]
    feats["home_form_l10"] = (
        round(sum(g["win"] for g in home_g) / len(home_g), 4) if home_g else 0.5
    )
    feats["road_form_l10"] = (
        round(sum(g["win"] for g in road_g) / len(road_g), 4) if road_g else 0.5
    )

    return feats
