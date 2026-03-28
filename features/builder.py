"""
Master feature matrix builder.

Two modes:
  1. Training matrix — for every completed game, build the feature vector
     using only information available *before* that game was played.
  2. Prediction features — for upcoming games, build feature vectors
     using the current state of all data.

Feature vector structure (per game):
  [home_features] + [away_features] + [delta_features] + [matchup_features]
  + [goalie_features] + [elo_features] + [context_features]
"""
import numpy as np
import pandas as pd
from datetime import date
from typing import Dict, List, Optional, Tuple

from features.elo import build_elo_trajectory, build_elo_ratings, elo_probability
from features.team_stats import build_team_stat_features
from features.form_features import build_team_game_logs, get_pregame_form, get_current_form
from features.goalie_features import build_goalie_features, apply_confirmed_starters
from features.h2h_features import get_h2h_features, build_team_cover_stats
from config import CURRENT_SEASON, GAME_TYPE_REGULAR, MIN_GAMES_FOR_FEAT


# ─── Feature prefix helpers ───────────────────────────────────────────────────

def _prefix(d: Dict, prefix: str) -> Dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def _delta_features(home: Dict, away: Dict, keys: List[str]) -> Dict:
    """Compute home - away delta for numeric features."""
    deltas = {}
    for k in keys:
        h = home.get(k, 0) or 0
        a = away.get(k, 0) or 0
        try:
            deltas[f"delta_{k}"] = round(float(h) - float(a), 4)
        except (TypeError, ValueError):
            deltas[f"delta_{k}"] = 0.0
    return deltas


# Keys used for delta computation
DELTA_KEYS = [
    "gf_pg", "ga_pg", "gf_ga_ratio",
    "shots_for_pg", "shots_against_pg",
    "pp_pct", "pk_pct", "net_pp_advantage",
    "corsi_pct_5v5", "fenwick_pct_5v5",
    "pdo_5v5", "shooting_pct_5v5", "save_pct_5v5",
    "win_pct", "regulation_win_pct",
    "l10_win_pct", "streak_value",
    "blowout_index",
    "home_win_pct",        # home team's home advantage metric
    "road_win_pct",        # away team's road performance
    "scoring_trend_l10",
    "days_rest",
    "starter_save_pct", "starter_gaa", "starter_gsax_pg",
    "starter_l5_sv_pct", "starter_l5_gaa", "starter_l5_vs_season",
    "cover_rate_minus1_5",
    "avg_margin_when_win",
]


# ─── Single-game feature vector ───────────────────────────────────────────────

def build_game_feature_vector(
    home_team: str,
    away_team: str,
    game_date: str,
    team_stats: Dict[str, Dict],     # from team_stats.build_team_stat_features
    form: Dict[str, Dict],           # from form_features.get_current_form or pregame
    goalie_feats: Dict[str, Dict],   # from goalie_features (with confirmed applied)
    cover_stats: Dict[str, Dict],    # from h2h_features.build_team_cover_stats
    h2h_results: List[Dict],         # current season game results (for H2H lookup)
    elo_ratings: Dict[str, float],   # current Elo ratings
    ou_line: float = 5.5,
    is_playoff: int = 0,
) -> pd.Series:
    """Build a single feature vector for one game."""

    h_stats  = team_stats.get(home_team, {})
    a_stats  = team_stats.get(away_team, {})
    h_form   = form.get(home_team, {})
    a_form   = form.get(away_team, {})
    h_goalie = goalie_feats.get(home_team, {})
    a_goalie = goalie_feats.get(away_team, {})
    h_cover  = cover_stats.get(home_team, {})
    a_cover  = cover_stats.get(away_team, {})

    # Merge team stats + form + goalie + cover for each side
    h_all = {**h_stats, **h_form, **h_goalie,
              "cover_rate_minus1_5": h_cover.get("cover_rate_minus1_5", 0.4),
              "avg_margin_when_win" : h_cover.get("avg_margin_when_win", 1.5),
              "blowout_rate"        : h_cover.get("blowout_rate", 0.15),
              "one_goal_game_rate"  : h_cover.get("one_goal_game_rate", 0.35),
              "ot_game_rate"        : h_cover.get("ot_game_rate", 0.25),
             }
    a_all = {**a_stats, **a_form, **a_goalie,
              "cover_rate_minus1_5": a_cover.get("cover_rate_minus1_5", 0.4),
              "avg_margin_when_win" : a_cover.get("avg_margin_when_win", 1.5),
              "blowout_rate"        : a_cover.get("blowout_rate", 0.15),
              "one_goal_game_rate"  : a_cover.get("one_goal_game_rate", 0.35),
              "ot_game_rate"        : a_cover.get("ot_game_rate", 0.25),
             }

    # H2H features
    h2h = get_h2h_features(home_team, away_team, h2h_results, ou_line)

    # Elo features
    h_elo = elo_ratings.get(home_team, 1500)
    a_elo = elo_ratings.get(away_team, 1500)
    elo_prob = elo_probability(h_elo, a_elo)

    # Delta features
    deltas = _delta_features(h_all, a_all, DELTA_KEYS)

    # Totals-specific combined features
    combined_gf = (h_stats.get("gf_pg", 0) + a_stats.get("ga_pg", 0)) / 2
    combined_ga = (a_stats.get("gf_pg", 0) + h_stats.get("ga_pg", 0)) / 2
    combined_goalie_sv = (
        h_goalie.get("starter_save_pct", 0.900) +
        a_goalie.get("starter_save_pct", 0.900)
    ) / 2
    combined_pp = (h_stats.get("pp_pct", 20) + a_stats.get("pp_pct", 20)) / 2

    # Puck-line: matchup cover likelihood
    # If home is a blowout team vs a team that gives up blowouts, cover is likely
    pl_matchup_score = (
        h_cover.get("cover_rate_minus1_5", 0.4) -
        a_cover.get("cover_rate_plus1_5", 0.4)
    )

    feats = {
        # ── Home team features ──────────────────────────────────────────────
        **_prefix(h_all, "h"),
        # ── Away team features ──────────────────────────────────────────────
        **_prefix(a_all, "a"),
        # ── Delta features (most predictive for ML) ──────────────────────────
        **deltas,
        # ── Elo ───────────────────────────────────────────────────────────────
        "home_elo"          : round(h_elo, 2),
        "away_elo"          : round(a_elo, 2),
        "elo_diff"          : round(h_elo - a_elo, 2),
        "elo_prob_home"     : round(elo_prob, 4),
        # ── H2H ───────────────────────────────────────────────────────────────
        **h2h,
        # ── Totals features ───────────────────────────────────────────────────
        "combined_gf_pg"    : round(combined_gf, 4),
        "combined_ga_pg"    : round(combined_ga, 4),
        "combined_goalie_sv": round(combined_goalie_sv, 4),
        "combined_pp_pct"   : round(combined_pp, 4),
        # ── Puck line features ────────────────────────────────────────────────
        "pl_matchup_score"  : round(pl_matchup_score, 4),
        "h_en_cover_proxy"  : round(
            h_cover.get("cover_rate_minus1_5", 0.4) *
            (1 - h_cover.get("ot_game_rate", 0.25)), 4),
        # ── Context ───────────────────────────────────────────────────────────
        "home_advantage"    : 1,
        "is_playoff"        : is_playoff,
        "ou_line"           : ou_line,
        # ── Back-to-back combined flag ────────────────────────────────────────
        "either_b2b"        : int(
            h_form.get("is_back_to_back", 0) or
            a_form.get("is_back_to_back", 0)
        ),
        "both_b2b"          : int(
            h_form.get("is_back_to_back", 0) and
            a_form.get("is_back_to_back", 0)
        ),
    }

    # Drop string columns (goalie names, streak codes)
    clean = {k: v for k, v in feats.items()
             if not isinstance(v, str)}

    return pd.Series(clean)


# ─── Training matrix ──────────────────────────────────────────────────────────

def build_training_matrix(
    game_results: List[Dict],
    prev_results: List[Dict],
    team_stats: Dict[str, Dict],
    goalie_feats: Dict[str, Dict],
    ou_line_default: float = 5.5,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series,
           pd.Series, pd.Series]:
    """
    Build the full training feature matrix.

    Returns
    -------
    X         : feature DataFrame
    y_ml      : binary target — 1 if home team won (regulation or OT)
    y_pl_home : binary — 1 if home won by 2+ goals (covers -1.5)
    y_pl_away : binary — 1 if away covered +1.5 (lost by ≤1 or won)
    y_goals_h : actual home goals (Poisson target)
    y_goals_a : actual away goals (Poisson target)
    """
    print("Building Elo trajectory...")
    elo_snapshots = build_elo_trajectory(game_results, prev_results)

    print("Building team game logs...")
    logs = build_team_game_logs(game_results + prev_results)

    print("Building cover stats...")
    cover_stats = build_team_cover_stats(game_results)

    rows, y_ml, y_pl_h, y_pl_a, y_gh, y_ga = [], [], [], [], [], []
    skipped = 0

    for g in sorted(game_results, key=lambda x: x.get("date", "")):
        home = g.get("home_team")
        away = g.get("away_team")
        gid  = g.get("game_id")
        d    = g.get("date", "")
        hg   = g.get("home_goals", 0)
        ag   = g.get("away_goals", 0)
        game_type = g.get("game_type", GAME_TYPE_REGULAR)

        if not home or not away:
            continue

        # Build pre-game log snapshots
        h_log_pre = [x for x in logs.get(home, [])
                     if x["date"] < d or (x["date"] == d and x["game_id"] != gid)]
        a_log_pre = [x for x in logs.get(away, [])
                     if x["date"] < d or (x["date"] == d and x["game_id"] != gid)]

        # Skip if not enough games for reliable features
        if len(h_log_pre) < MIN_GAMES_FOR_FEAT or len(a_log_pre) < MIN_GAMES_FOR_FEAT:
            skipped += 1
            continue

        h_form = get_pregame_form(home, h_log_pre, d)
        a_form = get_pregame_form(away, a_log_pre, d)
        form   = {home: h_form, away: a_form}

        elo_snap = elo_snapshots.get(gid, {})
        elo_ratings = {
            home: elo_snap.get("home_elo_pre", 1500),
            away: elo_snap.get("away_elo_pre", 1500),
        }

        # H2H from games strictly before this game
        h2h_hist = [x for x in game_results if x.get("date", "") < d]

        is_playoff = int(game_type == 3)

        try:
            vec = build_game_feature_vector(
                home, away, d,
                team_stats, form, goalie_feats, cover_stats,
                h2h_hist, elo_ratings, ou_line_default, is_playoff,
            )
        except Exception as e:
            skipped += 1
            continue

        rows.append(vec)
        y_ml.append(int(hg > ag))
        y_pl_h.append(int(hg >= ag + 2))    # Home covers -1.5
        y_pl_a.append(int(ag >= hg - 1))    # Away covers +1.5
        y_gh.append(hg)
        y_ga.append(ag)

    print(f"  Built {len(rows)} training samples ({skipped} skipped — insufficient history).")
    X = pd.DataFrame(rows).fillna(0)
    return (
        X,
        pd.Series(y_ml,  name="y_ml"),
        pd.Series(y_pl_h, name="y_pl_home"),
        pd.Series(y_pl_a, name="y_pl_away"),
        pd.Series(y_gh,   name="y_goals_home"),
        pd.Series(y_ga,   name="y_goals_away"),
    )


# ─── Prediction features ──────────────────────────────────────────────────────

def build_prediction_features(
    upcoming_games: List[Dict],
    game_results: List[Dict],
    prev_results: List[Dict],
    confirmed_starters: Dict[str, Dict],
    ou_lines: Dict[str, float] = None,   # game_key → ou_line
    is_playoff: int = 0,
) -> pd.DataFrame:
    """
    Build feature vectors for all upcoming games using current data.
    """
    from data.nhl_api import get_all_team_stats
    from features.goalie_features import build_goalie_features, apply_confirmed_starters

    print("Fetching current team stats...")
    team_stats_raw = build_team_stat_features()

    print("Building goalie features...")
    goalie_feats = build_goalie_features()
    goalie_feats = apply_confirmed_starters(goalie_feats, confirmed_starters)

    print("Building cover stats...")
    cover_stats = build_team_cover_stats(game_results)

    print("Building form features...")
    all_results = prev_results + game_results
    logs = build_team_game_logs(all_results)
    today = date.today().isoformat()
    form = get_current_form(logs, today)

    print("Building Elo ratings...")
    elo_ratings = build_elo_ratings(game_results, prev_results)

    rows = []
    game_keys = []

    for g in upcoming_games:
        home = g.get("homeTeam", {}).get("abbrev") or g.get("home_team")
        away = g.get("awayTeam", {}).get("abbrev") or g.get("away_team")
        if not home or not away:
            continue

        gd   = g.get("gameDate", today)
        ou   = (ou_lines or {}).get(f"{home}_vs_{away}", 5.5)

        try:
            vec = build_game_feature_vector(
                home, away, gd,
                team_stats_raw, form, goalie_feats, cover_stats,
                game_results, elo_ratings, ou, is_playoff,
            )
        except Exception as e:
            print(f"  Warning: could not build features for {home} vs {away}: {e}")
            continue

        rows.append(vec)
        game_keys.append(f"{home}_vs_{away}")

    X = pd.DataFrame(rows).fillna(0)
    X.index = game_keys
    return X
