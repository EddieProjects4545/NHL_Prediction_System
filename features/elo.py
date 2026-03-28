"""
Elo rating system for NHL teams.

Ratings are initialised at 1500, updated game-by-game through the full
season history, and regressed toward 1500 at each season boundary.

Key design choices
──────────────────
• K = 4 (regular season), K = 6 (playoffs) — higher stakes
• Home team receives +35 Elo points before probability calculation
• 35% regression toward 1500 at season start (captures roster turnover)
• OT/SO wins counted as 0.75 win (lower certainty than regulation wins)
• Elo probability is converted via the standard logistic function (400-base)
"""
import math
from copy import deepcopy
from typing import Dict, List, Tuple

from config import (
    ELO_INITIAL, ELO_K_REGULAR, ELO_K_PLAYOFF,
    ELO_HOME_BONUS, ELO_RESET_FACTOR, TEAM_ABBREVS,
)


# ─── Core Elo Math ────────────────────────────────────────────────────────────

def expected_score(rating_a: float, rating_b: float,
                   home_bonus: float = ELO_HOME_BONUS) -> float:
    """P(A beats B) given A has home-ice advantage."""
    return 1.0 / (1.0 + math.pow(10, (rating_b - (rating_a + home_bonus)) / 400))


def update_elo(winner_r: float, loser_r: float,
               k: float, margin: float = 1.0) -> Tuple[float, float]:
    """
    Update Elo ratings after a game.
    margin: goal-differential multiplier (>1 for blowouts).
    Returns (new_winner_rating, new_loser_rating).
    """
    expected_win = 1.0 / (1.0 + math.pow(10, (loser_r - winner_r) / 400))
    # Margin of victory multiplier (capped at 2x)
    mov_mult = min(math.log(abs(margin) + 1) + 0.5, 2.0)
    change = k * mov_mult * (1.0 - expected_win)
    return winner_r + change, loser_r - change


def regress_to_mean(rating: float, factor: float = ELO_RESET_FACTOR) -> float:
    """Pull rating toward 1500 by factor (e.g. 0.35 → 35% regression)."""
    return rating + factor * (ELO_INITIAL - rating)


# ─── Season Simulation ────────────────────────────────────────────────────────

def build_elo_ratings(game_results: List[Dict],
                      prev_results: List[Dict] = None) -> Dict[str, float]:
    """
    Simulate all games in chronological order and return current Elo ratings.

    Parameters
    ----------
    game_results : current-season game results (list of dicts from nhl_api)
    prev_results : prior-season game results (optional, applied first with
                   season-end regression before current season)

    Returns
    -------
    Dict mapping team_abbrev → current Elo rating
    """
    ratings: Dict[str, float] = {t: ELO_INITIAL for t in TEAM_ABBREVS}

    def _process(results: List[Dict], k_override: float = None):
        for g in sorted(results, key=lambda x: x.get("date", "")):
            home = g.get("home_team")
            away = g.get("away_team")
            hg   = g.get("home_goals", 0)
            ag   = g.get("away_goals", 0)
            game_type = g.get("game_type", 2)

            if not home or not away or home not in ratings or away not in ratings:
                continue

            k = k_override or (ELO_K_PLAYOFF if game_type == 3 else ELO_K_REGULAR)

            # OT/SO: winning team gets 0.75 credit
            ot = g.get("ot_flag", False)
            k_eff = k * (0.75 if ot else 1.0)

            margin = abs(hg - ag)

            if hg > ag:
                ratings[home], ratings[away] = update_elo(
                    ratings[home], ratings[away], k_eff, margin)
            elif ag > hg:
                ratings[away], ratings[home] = update_elo(
                    ratings[away], ratings[home], k_eff, margin)
            # Tie should not occur in NHL (OT/SO resolves it) — skip

    # Process prior season first
    if prev_results:
        _process(prev_results)
        # Regress all ratings at season boundary
        for team in ratings:
            ratings[team] = regress_to_mean(ratings[team])

    # Process current season
    _process(game_results)

    return ratings


def elo_probability(home_rating: float, away_rating: float,
                    home_bonus: float = ELO_HOME_BONUS) -> float:
    """Return P(home wins) from Elo ratings."""
    return expected_score(home_rating, away_rating, home_bonus)


# ─── Per-game Elo trajectory (for training features) ─────────────────────────

def build_elo_trajectory(game_results: List[Dict],
                         prev_results: List[Dict] = None) -> Dict[str, List[Dict]]:
    """
    Return per-game Elo snapshots: after each game the pre-game and
    post-game ratings are recorded.

    Returns dict: {game_id: {home_elo_pre, away_elo_pre, elo_prob_home}}
    """
    ratings: Dict[str, float] = {t: ELO_INITIAL for t in TEAM_ABBREVS}

    if prev_results:
        for g in sorted(prev_results, key=lambda x: x.get("date", "")):
            home = g.get("home_team")
            away = g.get("away_team")
            hg   = g.get("home_goals", 0)
            ag   = g.get("away_goals", 0)
            if not home or not away or home not in ratings or away not in ratings:
                continue
            k   = ELO_K_REGULAR
            ot  = g.get("ot_flag", False)
            k_e = k * (0.75 if ot else 1.0)
            m   = abs(hg - ag)
            if hg > ag:
                ratings[home], ratings[away] = update_elo(ratings[home], ratings[away], k_e, m)
            elif ag > hg:
                ratings[away], ratings[home] = update_elo(ratings[away], ratings[home], k_e, m)
        for t in ratings:
            ratings[t] = regress_to_mean(ratings[t])

    snapshots: Dict[int, Dict] = {}
    for g in sorted(game_results, key=lambda x: x.get("date", "")):
        home = g.get("home_team")
        away = g.get("away_team")
        hg   = g.get("home_goals", 0)
        ag   = g.get("away_goals", 0)
        gid  = g.get("game_id")
        game_type = g.get("game_type", 2)

        if not home or not away or home not in ratings or away not in ratings:
            continue

        h_pre = ratings[home]
        a_pre = ratings[away]
        prob  = elo_probability(h_pre, a_pre)

        snapshots[gid] = {
            "home_elo_pre" : round(h_pre, 2),
            "away_elo_pre" : round(a_pre, 2),
            "elo_prob_home": round(prob, 4),
            "elo_diff"     : round(h_pre - a_pre, 2),
        }

        k   = ELO_K_PLAYOFF if game_type == 3 else ELO_K_REGULAR
        ot  = g.get("ot_flag", False)
        k_e = k * (0.75 if ot else 1.0)
        m   = abs(hg - ag)

        if hg > ag:
            ratings[home], ratings[away] = update_elo(ratings[home], ratings[away], k_e, m)
        elif ag > hg:
            ratings[away], ratings[home] = update_elo(ratings[away], ratings[home], k_e, m)

    return snapshots
