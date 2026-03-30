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
                "sf": g.get("home_sog", 0), "sa": g.get("away_sog", 0),
                "home": True, "win": hg > ag, "ot": ot,
                "game_id": g.get("game_id"),
            })
        if at:
            logs[at].append({
                "date": g["date"], "gf": ag, "ga": hg,
                "sf": g.get("away_sog", 0), "sa": g.get("home_sog", 0),
                "home": False, "win": ag > hg, "ot": ot,
                "game_id": g.get("game_id"),
            })
    return dict(logs)


def build_pregame_team_snapshot(log_before_game: List[Dict],
                                game_date: str) -> Dict:
    """
    Team snapshot using only games played before `game_date`.
    This is the training-safe replacement for season aggregates.
    """
    n = len(log_before_game)
    if n == 0:
        return {
            "gp": 0,
            "gf_pg": 0.0,
            "ga_pg": 0.0,
            "gf_ga_ratio": 1.0,
            "shots_for_pg": 0.0,
            "shots_against_pg": 0.0,
            "goal_diff_pg": 0.0,
            "shot_diff_pg": 0.0,
            "win_pct": 0.5,
            "regulation_win_pct": 0.5,
            "home_win_pct": 0.5,
            "road_win_pct": 0.5,
            "l10_win_pct": 0.5,
            "home_form_l10": 0.5,
            "road_form_l10": 0.5,
            "streak_value": 0,
            "scoring_trend_l10": 0.0,
            "goal_diff_l5": 0.0,
            "goal_diff_l10": 0.0,
            "shot_diff_l5": 0.0,
            "shot_diff_l10": 0.0,
            "save_pct_l5": 0.900,
            "save_pct_l10": 0.900,
            "shooting_pct_l5": 0.100,
            "shooting_pct_l10": 0.100,
            "one_goal_rate_l10": 0.35,
            "ot_rate_l10": 0.25,
            "regulation_win_pct_l10": 0.5,
            "days_rest": 7,
            "is_back_to_back": 0,
            "games_played_season": 0,
        }

    gf = [g["gf"] for g in log_before_game]
    ga = [g["ga"] for g in log_before_game]
    sf = [g.get("sf", 0) for g in log_before_game]
    sa = [g.get("sa", 0) for g in log_before_game]
    wins = [int(g["win"]) for g in log_before_game]
    regulation_wins = [int(g["win"] and not g["ot"]) for g in log_before_game]

    home_games = [g for g in log_before_game if g["home"]]
    road_games = [g for g in log_before_game if not g["home"]]
    recent10 = log_before_game[-10:]
    recent5 = log_before_game[-5:]
    l10_win_pct = round(sum(1 for g in recent10 if g["win"]) / len(recent10), 4) if recent10 else 0.5
    home_recent10 = [g for g in recent10 if g["home"]]
    road_recent10 = [g for g in recent10 if not g["home"]]

    streak = 0
    if log_before_game:
        last_win = log_before_game[-1]["win"]
        for g in reversed(log_before_game):
            if g["win"] == last_win:
                streak += 1
            else:
                break
        streak = streak if last_win else -streak

    def _avg_margin(games: List[Dict]) -> float:
        if not games:
            return 0.0
        return round(sum(g["gf"] - g["ga"] for g in games) / len(games), 4)

    def _avg_shot_margin(games: List[Dict]) -> float:
        if not games:
            return 0.0
        return round(sum(g.get("sf", 0) - g.get("sa", 0) for g in games) / len(games), 4)

    def _save_pct(games: List[Dict]) -> float:
        shots_against = sum(g.get("sa", 0) for g in games)
        goals_against = sum(g["ga"] for g in games)
        if shots_against <= 0:
            return 0.900
        return round(max(0.0, 1.0 - (goals_against / shots_against)), 4)

    def _shooting_pct(games: List[Dict]) -> float:
        shots_for = sum(g.get("sf", 0) for g in games)
        goals_for = sum(g["gf"] for g in games)
        if shots_for <= 0:
            return 0.100
        return round(goals_for / shots_for, 4)

    def _one_goal_rate(games: List[Dict]) -> float:
        if not games:
            return 0.35
        return round(sum(1 for g in games if abs(g["gf"] - g["ga"]) == 1) / len(games), 4)

    def _ot_rate(games: List[Dict]) -> float:
        if not games:
            return 0.25
        return round(sum(1 for g in games if g.get("ot")) / len(games), 4)

    def _regulation_win_pct(games: List[Dict]) -> float:
        if not games:
            return 0.5
        return round(sum(1 for g in games if g["win"] and not g.get("ot")) / len(games), 4)

    return {
        "gp": n,
        "gf_pg": round(sum(gf) / n, 4),
        "ga_pg": round(sum(ga) / n, 4),
        "gf_ga_ratio": round((sum(gf) / max(sum(ga), 1)), 4),
        "shots_for_pg": round(sum(sf) / n, 4),
        "shots_against_pg": round(sum(sa) / n, 4),
        "goal_diff_pg": round((sum(gf) - sum(ga)) / n, 4),
        "shot_diff_pg": round((sum(sf) - sum(sa)) / n, 4),
        "win_pct": round(sum(wins) / n, 4),
        "regulation_win_pct": round(sum(regulation_wins) / n, 4),
        "home_win_pct": round(sum(1 for g in home_games if g["win"]) / len(home_games), 4) if home_games else 0.5,
        "road_win_pct": round(sum(1 for g in road_games if g["win"]) / len(road_games), 4) if road_games else 0.5,
        "l10_win_pct": l10_win_pct,
        "home_form_l10": round(sum(1 for g in home_recent10 if g["win"]) / len(home_recent10), 4) if home_recent10 else 0.5,
        "road_form_l10": round(sum(1 for g in road_recent10 if g["win"]) / len(road_recent10), 4) if road_recent10 else 0.5,
        "streak_value": streak,
        "scoring_trend_l10": _scoring_trend(log_before_game, 10),
        "goal_diff_l5": _avg_margin(recent5),
        "goal_diff_l10": _avg_margin(recent10),
        "shot_diff_l5": _avg_shot_margin(recent5),
        "shot_diff_l10": _avg_shot_margin(recent10),
        "save_pct_l5": _save_pct(recent5),
        "save_pct_l10": _save_pct(recent10),
        "shooting_pct_l5": _shooting_pct(recent5),
        "shooting_pct_l10": _shooting_pct(recent10),
        "one_goal_rate_l10": _one_goal_rate(recent10),
        "ot_rate_l10": _ot_rate(recent10),
        "regulation_win_pct_l10": _regulation_win_pct(recent10),
        "days_rest": _days_rest(log_before_game, game_date),
        "is_back_to_back": int(_days_rest(log_before_game, game_date) <= 1),
        "games_played_season": n,
    }


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
