"""
Master feature matrix builder.

Training rows and future-game rows now share the same design principle:
team-state features must reflect only information available before puck drop.
"""
import pandas as pd
from datetime import date
from typing import Dict, List, Tuple

from features.elo import build_elo_trajectory, build_elo_ratings, elo_probability
from features.form_features import build_team_game_logs, build_pregame_team_snapshot
from features.h2h_features import get_h2h_features, build_team_cover_stats
from data.nhl_api import get_team_en_stats
from config import CURRENT_SEASON, GAME_TYPE_REGULAR, MIN_GAMES_FOR_FEAT


def _prefix(d: Dict, prefix: str) -> Dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def _delta_features(home: Dict, away: Dict, keys: List[str]) -> Dict:
    deltas = {}
    for k in keys:
        h = home.get(k, 0) or 0
        a = away.get(k, 0) or 0
        try:
            deltas[f"delta_{k}"] = round(float(h) - float(a), 4)
        except (TypeError, ValueError):
            deltas[f"delta_{k}"] = 0.0
    return deltas


DELTA_KEYS = [
    "gf_pg", "ga_pg", "gf_ga_ratio",
    "shots_for_pg", "shots_against_pg",
    "goal_diff_pg", "shot_diff_pg",
    "win_pct", "regulation_win_pct",
    "l10_win_pct", "streak_value",
    "home_win_pct", "road_win_pct",
    "home_form_l10", "road_form_l10",
    "scoring_trend_l10",
    "goal_diff_l5", "goal_diff_l10",
    "shot_diff_l5", "shot_diff_l10",
    "save_pct_l5", "save_pct_l10",
    "shooting_pct_l5", "shooting_pct_l10",
    "one_goal_rate_l10", "ot_rate_l10",
    "regulation_win_pct_l10",
    "days_rest",
    "cover_rate_minus1_5",
    "avg_margin_when_win",
    "en_goals_for_pg",
    "en_goals_against_pg",
]


def _build_team_snapshot_map(
    logs: Dict[str, List[Dict]],
    home_team: str,
    away_team: str,
    game_date: str,
    game_id: str = None,
) -> Dict[str, Dict]:
    h_log_pre = [
        x for x in logs.get(home_team, [])
        if x["date"] < game_date or (x["date"] == game_date and x.get("game_id") != game_id)
    ]
    a_log_pre = [
        x for x in logs.get(away_team, [])
        if x["date"] < game_date or (x["date"] == game_date and x.get("game_id") != game_id)
    ]
    return {
        home_team: build_pregame_team_snapshot(h_log_pre, game_date),
        away_team: build_pregame_team_snapshot(a_log_pre, game_date),
    }


def build_game_feature_vector(
    home_team: str,
    away_team: str,
    game_date: str,
    team_stats: Dict[str, Dict],
    cover_stats: Dict[str, Dict],
    h2h_results: List[Dict],
    elo_ratings: Dict[str, float],
    ou_line: float = 5.5,
    is_playoff: int = 0,
    en_stats: Dict[str, Dict] = None,
    goalie_feats: Dict[str, Dict] = None,
) -> pd.Series:
    h_stats = team_stats.get(home_team, {})
    a_stats = team_stats.get(away_team, {})
    h_cover = cover_stats.get(home_team, {})
    a_cover = cover_stats.get(away_team, {})
    h_en = (en_stats or {}).get(home_team, {})
    a_en = (en_stats or {}).get(away_team, {})
    h_g = (goalie_feats or {}).get(home_team, {})
    a_g = (goalie_feats or {}).get(away_team, {})

    h_all = {
        **h_stats,
        "cover_rate_minus1_5": h_cover.get("cover_rate_minus1_5", 0.4),
        "avg_margin_when_win": h_cover.get("avg_margin_when_win", 1.5),
        "blowout_rate": h_cover.get("blowout_rate", 0.15),
        "one_goal_game_rate": h_cover.get("one_goal_game_rate", 0.35),
        "ot_game_rate": h_cover.get("ot_game_rate", 0.25),
        "en_goals_for_pg":     h_en.get("en_goals_for_pg", 0.0),
        "en_goals_against_pg": h_en.get("en_goals_against_pg", 0.0),
    }
    a_all = {
        **a_stats,
        "cover_rate_minus1_5": a_cover.get("cover_rate_minus1_5", 0.4),
        "avg_margin_when_win": a_cover.get("avg_margin_when_win", 1.5),
        "blowout_rate": a_cover.get("blowout_rate", 0.15),
        "one_goal_game_rate": a_cover.get("one_goal_game_rate", 0.35),
        "ot_game_rate": a_cover.get("ot_game_rate", 0.25),
        "en_goals_for_pg":     a_en.get("en_goals_for_pg", 0.0),
        "en_goals_against_pg": a_en.get("en_goals_against_pg", 0.0),
    }

    h2h = get_h2h_features(home_team, away_team, h2h_results, ou_line)
    h_elo = elo_ratings.get(home_team, 1500)
    a_elo = elo_ratings.get(away_team, 1500)
    elo_prob = elo_probability(h_elo, a_elo)
    deltas = _delta_features(h_all, a_all, DELTA_KEYS)

    combined_gf = (h_stats.get("gf_pg", 0) + a_stats.get("ga_pg", 0)) / 2
    combined_ga = (a_stats.get("gf_pg", 0) + h_stats.get("ga_pg", 0)) / 2
    combined_shots = (h_stats.get("shots_for_pg", 0) + a_stats.get("shots_against_pg", 0)) / 2
    combined_goal_diff_l10 = (h_stats.get("goal_diff_l10", 0) - a_stats.get("goal_diff_l10", 0)) / 2
    combined_shot_diff_l10 = (h_stats.get("shot_diff_l10", 0) - a_stats.get("shot_diff_l10", 0)) / 2
    combined_save_pct_l10 = (h_stats.get("save_pct_l10", 0.9) + a_stats.get("save_pct_l10", 0.9)) / 2
    combined_shooting_pct_l10 = (h_stats.get("shooting_pct_l10", 0.1) + a_stats.get("shooting_pct_l10", 0.1)) / 2
    pl_matchup_score = (
        h_cover.get("cover_rate_minus1_5", 0.4) -
        a_cover.get("cover_rate_plus1_5", 0.4)
    )

    feats = {
        **_prefix(h_all, "h"),
        **_prefix(a_all, "a"),
        **deltas,
        "home_elo": round(h_elo, 2),
        "away_elo": round(a_elo, 2),
        "elo_diff": round(h_elo - a_elo, 2),
        "elo_prob_home": round(elo_prob, 4),
        **h2h,
        "combined_gf_pg": round(combined_gf, 4),
        "combined_ga_pg": round(combined_ga, 4),
        "combined_shots_pg": round(combined_shots, 4),
        "combined_goal_diff_l10": round(combined_goal_diff_l10, 4),
        "combined_shot_diff_l10": round(combined_shot_diff_l10, 4),
        "combined_save_pct_l10": round(combined_save_pct_l10, 4),
        "combined_shooting_pct_l10": round(combined_shooting_pct_l10, 4),
        "pl_matchup_score": round(pl_matchup_score, 4),
        "h_en_cover_proxy": round(
            h_cover.get("cover_rate_minus1_5", 0.4) *
            (1 - h_cover.get("ot_game_rate", 0.25)), 4
        ),
        "en_goals_matchup_score": round(
            h_en.get("en_goals_for_pg", 0.0) -
            a_en.get("en_goals_for_pg", 0.0) +
            a_en.get("en_goals_against_pg", 0.0) -
            h_en.get("en_goals_against_pg", 0.0), 4
        ),
        # ── Goalie form ───────────────────────────────────────────────────────
        "h_starter_sv_pct":       round(h_g.get("starter_save_pct", 0.900), 4),
        "h_starter_l5_sv_pct":    round(h_g.get("starter_l5_sv_pct", 0.900), 4),
        "h_starter_l5_gaa":       round(h_g.get("starter_l5_gaa", 2.85), 4),
        "h_starter_gsax_pg":      round(h_g.get("starter_gsax_pg", 0.0), 4),
        "h_starter_l5_vs_season": round(h_g.get("starter_l5_vs_season", 0.0), 4),
        "h_starter_confirmed":    int(h_g.get("starter_confirmed", 0)),
        "a_starter_sv_pct":       round(a_g.get("starter_save_pct", 0.900), 4),
        "a_starter_l5_sv_pct":    round(a_g.get("starter_l5_sv_pct", 0.900), 4),
        "a_starter_l5_gaa":       round(a_g.get("starter_l5_gaa", 2.85), 4),
        "a_starter_gsax_pg":      round(a_g.get("starter_gsax_pg", 0.0), 4),
        "a_starter_l5_vs_season": round(a_g.get("starter_l5_vs_season", 0.0), 4),
        "a_starter_confirmed":    int(a_g.get("starter_confirmed", 0)),
        "delta_starter_sv_pct":   round(
            h_g.get("starter_save_pct", 0.900) - a_g.get("starter_save_pct", 0.900), 4),
        "delta_starter_l5_sv_pct": round(
            h_g.get("starter_l5_sv_pct", 0.900) - a_g.get("starter_l5_sv_pct", 0.900), 4),
        "delta_starter_l5_gaa":   round(
            a_g.get("starter_l5_gaa", 2.85) - h_g.get("starter_l5_gaa", 2.85), 4),
        "combined_goalie_l5_sv":  round(
            (h_g.get("starter_l5_sv_pct", 0.900) + a_g.get("starter_l5_sv_pct", 0.900)) / 2, 4),
        "home_advantage": 1,
        "is_playoff": is_playoff,
        "ou_line": ou_line,
        "either_b2b": int(
            h_stats.get("is_back_to_back", 0) or
            a_stats.get("is_back_to_back", 0)
        ),
        "both_b2b": int(
            h_stats.get("is_back_to_back", 0) and
            a_stats.get("is_back_to_back", 0)
        ),
        "home_b2b_away_fresh": int(
            h_stats.get("is_back_to_back", 0) == 1 and
            a_stats.get("is_back_to_back", 0) == 0
        ),
        "away_b2b_home_fresh": int(
            a_stats.get("is_back_to_back", 0) == 1 and
            h_stats.get("is_back_to_back", 0) == 0
        ),
        "rest_advantage": int(h_stats.get("days_rest", 3)) - int(a_stats.get("days_rest", 3)),
        "games_played_diff": int(h_stats.get("games_played_season", 0)) - int(a_stats.get("games_played_season", 0)),
    }

    return pd.Series({k: v for k, v in feats.items() if not isinstance(v, str)})


def build_training_matrix(
    game_results: List[Dict],
    prev_results: List[Dict],
    team_stats: Dict[str, Dict],
    goalie_feats: Dict[str, Dict],
    ou_line_default: float = 5.5,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    print("Building Elo trajectory...")
    elo_snapshots = build_elo_trajectory(game_results, prev_results)

    print("Building team game logs...")
    all_training_games = sorted(prev_results + game_results, key=lambda x: x.get("date", ""))
    logs = build_team_game_logs(all_training_games)

    print("Building cover stats...")
    cover_stats = build_team_cover_stats(all_training_games)

    print("Fetching empty-net goal stats...")
    en_stats = get_team_en_stats(game_results)

    rows, y_ml, y_pl_h, y_pl_a, y_gh, y_ga, meta_rows = [], [], [], [], [], [], []
    skipped = 0

    for g in all_training_games:
        home = g.get("home_team")
        away = g.get("away_team")
        gid = g.get("game_id")
        d = g.get("date", "")
        hg = g.get("home_goals", 0)
        ag = g.get("away_goals", 0)
        game_type = g.get("game_type", GAME_TYPE_REGULAR)

        if not home or not away:
            continue

        team_snapshots = _build_team_snapshot_map(logs, home, away, d, gid)
        if (
            team_snapshots[home].get("games_played_season", 0) < MIN_GAMES_FOR_FEAT or
            team_snapshots[away].get("games_played_season", 0) < MIN_GAMES_FOR_FEAT
        ):
            skipped += 1
            continue

        elo_snap = elo_snapshots.get(gid, {})
        elo_ratings = {
            home: elo_snap.get("home_elo_pre", 1500),
            away: elo_snap.get("away_elo_pre", 1500),
        }
        h2h_hist = [x for x in all_training_games if x.get("date", "") < d]

        game_season = g.get("season", CURRENT_SEASON)
        game_goalie_feats = (goalie_feats or {}).get(
            game_season, (goalie_feats or {}).get(CURRENT_SEASON, {}))

        try:
            vec = build_game_feature_vector(
                home, away, d,
                team_snapshots, cover_stats,
                h2h_hist, elo_ratings, ou_line_default, int(game_type == 3),
                en_stats=en_stats,
                goalie_feats=game_goalie_feats,
            )
        except Exception:
            skipped += 1
            continue

        rows.append(vec)
        y_ml.append(int(hg > ag))
        y_pl_h.append(int(hg >= ag + 2))
        y_pl_a.append(int(ag >= hg - 1))
        y_gh.append(hg)
        y_ga.append(ag)
        meta_rows.append({
            "date": d,
            "season": g.get("season", CURRENT_SEASON),
            "game_type": game_type,
            "game_id": gid,
        })

    print(f"  Built {len(rows)} training samples ({skipped} skipped - insufficient history).")
    X = pd.DataFrame(rows).fillna(0)
    return (
        X,
        pd.Series(y_ml, name="y_ml"),
        pd.Series(y_pl_h, name="y_pl_home"),
        pd.Series(y_pl_a, name="y_pl_away"),
        pd.Series(y_gh, name="y_goals_home"),
        pd.Series(y_ga, name="y_goals_away"),
        pd.DataFrame(meta_rows),
    )


def build_prediction_features(
    upcoming_games: List[Dict],
    game_results: List[Dict],
    prev_results: List[Dict],
    confirmed_starters: Dict[str, Dict],
    ou_lines: Dict[str, float] = None,
    is_playoff: int = 0,
    goalie_feats: Dict[str, Dict] = None,
) -> pd.DataFrame:
    print("Building cover stats...")
    all_results = sorted(prev_results + game_results, key=lambda x: x.get("date", ""))
    cover_stats = build_team_cover_stats(all_results)

    print("Building pregame team snapshots...")
    logs = build_team_game_logs(all_results)
    today = date.today().isoformat()

    print("Building Elo ratings...")
    elo_ratings = build_elo_ratings(game_results, prev_results)

    print("Fetching empty-net goal stats...")
    en_stats = get_team_en_stats(game_results)

    rows = []
    game_keys = []

    for g in upcoming_games:
        home = g.get("homeTeam", {}).get("abbrev") or g.get("home_team")
        away = g.get("awayTeam", {}).get("abbrev") or g.get("away_team")
        if not home or not away:
            continue

        gd = g.get("gameDate", today)
        ou = (ou_lines or {}).get(f"{home}_vs_{away}", 5.5)
        team_snapshots = _build_team_snapshot_map(logs, home, away, gd)

        try:
            vec = build_game_feature_vector(
                home, away, gd,
                team_snapshots, cover_stats,
                all_results, elo_ratings, ou, is_playoff,
                en_stats=en_stats,
                goalie_feats=goalie_feats,
            )
        except Exception as e:
            print(f"  Warning: could not build features for {home} vs {away}: {e}")
            continue

        rows.append(vec)
        game_keys.append(f"{home}_vs_{away}")

    X = pd.DataFrame(rows).fillna(0)
    X.index = game_keys
    return X
