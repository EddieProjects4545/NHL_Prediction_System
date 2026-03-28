"""
Head-to-head feature builder.

Computes from game result history:
  - H2H win% (current season)
  - H2H average GF/GA
  - H2H goal margin trend
  - H2H sample size (used to penalise confidence when small)
  - Puck-line specific: H2H cover rate (won by 2+) and O/U history
"""
from typing import Dict, List, Tuple


def build_h2h_lookup(game_results: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    """
    Build a lookup dict: (home_team, away_team) → list of historical matchup results.
    Only regular season games (same season) considered.
    """
    lookup: Dict[Tuple[str, str], List[Dict]] = {}
    for g in game_results:
        key = (g.get("home_team"), g.get("away_team"))
        lookup.setdefault(key, []).append(g)
    return lookup


def get_h2h_features(home: str, away: str,
                     game_results: List[Dict],
                     ou_line: float = 5.5) -> Dict:
    """
    Compute H2H features for a specific home/away matchup.

    Parameters
    ----------
    home         : home team abbreviation
    away         : away team abbreviation
    game_results : list of completed game results (all teams, this season)
    ou_line      : O/U line to use for historical over/under calculation

    Returns
    -------
    Dict of H2H features
    """
    # Gather all matchups between the two teams (both directions)
    matchups = []
    for g in game_results:
        ht = g.get("home_team")
        at = g.get("away_team")
        if (ht == home and at == away) or (ht == away and at == home):
            matchups.append(g)

    n = len(matchups)
    if n == 0:
        return _empty_h2h()

    # From home team's perspective
    home_wins  = 0
    home_gf    = []
    home_ga    = []
    home_covers_minus1_5 = 0
    overs      = 0

    for g in matchups:
        ht = g.get("home_team")
        hg = g.get("home_goals", 0)
        ag = g.get("away_goals", 0)

        if ht == home:
            gf, ga = hg, ag
        else:
            gf, ga = ag, hg

        home_gf.append(gf)
        home_ga.append(ga)
        if gf > ga:
            home_wins += 1
        if gf >= ga + 2:   # home covered -1.5
            home_covers_minus1_5 += 1
        if (gf + ga) > ou_line:
            overs += 1

    avg_gf   = sum(home_gf) / n
    avg_ga   = sum(home_ga) / n
    avg_total = (avg_gf + avg_ga)

    return {
        "h2h_sample"            : n,
        "h2h_win_pct"           : round(home_wins / n, 4),
        "h2h_gf_avg"            : round(avg_gf, 4),
        "h2h_ga_avg"            : round(avg_ga, 4),
        "h2h_goal_diff_avg"     : round(avg_gf - avg_ga, 4),
        "h2h_avg_total"         : round(avg_total, 4),
        "h2h_over_rate"         : round(overs / n, 4),
        "h2h_cover_rate_minus1_5": round(home_covers_minus1_5 / n, 4),
    }


def build_team_cover_stats(game_results: List[Dict]) -> Dict[str, Dict]:
    """
    Compute per-team puck-line cover statistics from all season results.

    For each team:
      - cover_rate_minus1_5  : % of wins that were by 2+ goals
      - cover_rate_plus1_5   : % of losses that were by 1 goal (covered +1.5)
      - avg_margin_when_win  : average goal margin in wins
      - blowout_rate         : % of all games won by 3+ goals
      - one_goal_win_rate    : % of all games decided by exactly 1 goal
      - en_inflation_proxy   : % of wins that went to EN (from OT flag logic)

    These stats are the primary puck-line differentiators.
    """
    stats: Dict[str, Dict] = {}

    team_games: Dict[str, List] = {}
    for g in game_results:
        ht = g.get("home_team")
        at = g.get("away_team")
        hg = g.get("home_goals", 0)
        ag = g.get("away_goals", 0)
        ot = g.get("ot_flag", False)

        for team, gf, ga in [(ht, hg, ag), (at, ag, hg)]:
            if team:
                team_games.setdefault(team, []).append({
                    "gf": gf, "ga": ga, "ot": ot,
                    "win": gf > ga,
                    "margin": gf - ga,
                })

    for team, games in team_games.items():
        n = len(games) or 1
        wins = [g for g in games if g["win"]]
        losses = [g for g in games if not g["win"]]
        n_wins = len(wins) or 1
        n_losses = len(losses) or 1

        # Covered -1.5: won by 2 or more
        covers_minus = sum(1 for g in wins if g["margin"] >= 2)
        # Covered +1.5: lost by 1 or less (i.e. margin > -2)
        covers_plus  = sum(1 for g in losses if g["margin"] >= -1)
        # Blowout: won by 3+
        blowouts = sum(1 for g in wins if g["margin"] >= 3)
        # One-goal games
        one_goal = sum(1 for g in games if abs(g["margin"]) == 1)
        # Close finish OT (EN goal inflates margin)
        # Approximation: games that went to OT are by definition 1-goal margin at end of reg
        ot_games = sum(1 for g in games if g["ot"])

        avg_margin_win = (
            sum(g["margin"] for g in wins) / n_wins
        )

        stats[team] = {
            "cover_rate_minus1_5" : round(covers_minus / n_wins, 4),
            "cover_rate_plus1_5"  : round(covers_plus  / n_losses, 4),
            "avg_margin_when_win" : round(avg_margin_win, 4),
            "blowout_rate"        : round(blowouts / n, 4),
            "one_goal_game_rate"  : round(one_goal / n, 4),
            "ot_game_rate"        : round(ot_games / n, 4),
            "games_played"        : n,
        }

    return stats


def _empty_h2h() -> Dict:
    return {
        "h2h_sample"             : 0,
        "h2h_win_pct"            : 0.5,
        "h2h_gf_avg"             : 0,
        "h2h_ga_avg"             : 0,
        "h2h_goal_diff_avg"      : 0,
        "h2h_avg_total"          : 5.5,
        "h2h_over_rate"          : 0.5,
        "h2h_cover_rate_minus1_5": 0.4,
    }
