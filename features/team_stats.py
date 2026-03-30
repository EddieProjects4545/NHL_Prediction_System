"""
Season-aggregate team feature builder.
Merges all stat endpoints into a normalised per-team feature dict.
"""
from typing import Dict

from data.nhl_api import get_all_team_stats, parse_standings
from config import CURRENT_SEASON, GAME_TYPE_REGULAR


def build_team_stat_features(season: str = CURRENT_SEASON,
                              game_type: int = GAME_TYPE_REGULAR) -> Dict[str, Dict]:
    """
    Combine all team stats and standings into a single feature dict.
    Also computes derived 'blowout' and cover-rate proxies from standing splits.

    Returns: {team_abbrev: {feature_name: value, ...}}
    """
    stats    = get_all_team_stats(season, game_type)
    standings = parse_standings() if season == CURRENT_SEASON else {}

    merged: Dict[str, Dict] = {}

    for abbrev, s in stats.items():
        st = standings.get(abbrev, {})

        # ── Core offensive / defensive rates ──────────────────────────────────
        gp  = s.get("gp", 1) or 1
        gf  = s.get("gf_pg", 0)
        ga  = s.get("ga_pg", 0)

        # ── Special teams ─────────────────────────────────────────────────────
        pp_pct  = s.get("pp_pct", 0)
        pk_pct  = s.get("pk_pct", 0)
        pp_opps = s.get("pp_opportunities_pg", 0)
        pk_opps = s.get("pk_opportunities_pg", 0)

        # ── Possession / shot quality ──────────────────────────────────────────
        corsi   = s.get("corsi_pct_5v5", 50)
        fenwick = s.get("fenwick_pct_5v5", 50)
        sh_pct  = s.get("shooting_pct_5v5", 0)
        sv_pct  = s.get("save_pct_5v5", 0)
        pdo     = s.get("pdo_5v5", sh_pct + sv_pct)
        zone_o  = s.get("zone_start_off_pct", 50)

        # ── Physical / puck management ─────────────────────────────────────────
        hits      = s.get("hits_pg", 0)
        blocks    = s.get("blocks_pg", 0)
        giveaways = s.get("giveaways_pg", 0)
        takeaways = s.get("takeaways_pg", 0)
        tk_ratio  = s.get("takeaway_ratio", 1)

        # ── Home / road splits (from standings) ───────────────────────────────
        home_gf_pg = st.get("home_gf_pg", gf)
        home_ga_pg = st.get("home_ga_pg", ga)
        road_gf_pg = st.get("road_gf_pg", gf)
        road_ga_pg = st.get("road_ga_pg", ga)
        home_win   = st.get("home_win_pct", 0.5)
        road_win   = st.get("road_win_pct", 0.5)

        # ── Recent form (L10 from standings) ──────────────────────────────────
        l10_wins   = st.get("l10_wins", 5)
        l10_losses = st.get("l10_losses", 5)
        l10_win_pct = l10_wins / 10 if (l10_wins + l10_losses) > 0 else 0.5

        # ── Streak encoding ────────────────────────────────────────────────────
        streak_code  = st.get("streak_code", "")
        streak_count = st.get("streak_count", 0)
        if streak_code.startswith("W"):
            streak_value = streak_count
        elif streak_code.startswith("L"):
            streak_value = -streak_count
        else:
            streak_value = 0

        # ── Puck-line specific: goal-margin proxies ────────────────────────────
        # Teams with high GF and high Corsi tend to win by larger margins.
        # We create a "blowout index" as a composite.
        blowout_index = (
            0.4 * (gf / max(ga, 0.1))     +   # scoring dominance
            0.3 * ((corsi - 50) / 10)     +   # possession advantage (normalised)
            0.2 * (pp_pct / 25)           +   # PP efficiency
            0.1 * (l10_win_pct - 0.5) * 2     # recent form momentum
        )

        # PDO regression flag: teams with PDO > 102 or < 98 are likely to regress
        pdo_regression_flag = 1 if pdo > 102 else (-1 if pdo < 98 else 0)

        merged[abbrev] = {
            # ── Goals ─────────────────────────────────────────────────────────
            "gf_pg"              : round(gf, 4),
            "ga_pg"              : round(ga, 4),
            "gf_ga_ratio"        : round(gf / max(ga, 0.1), 4),
            "shots_for_pg"       : round(s.get("shots_for_pg", 0), 4),
            "shots_against_pg"   : round(s.get("shots_against_pg", 0), 4),
            # ── Special teams ─────────────────────────────────────────────────
            "pp_pct"             : round(pp_pct, 4),
            "pk_pct"             : round(pk_pct, 4),
            "pp_opportunities_pg": round(pp_opps, 4),
            "pk_opportunities_pg": round(pk_opps, 4),
            "net_pp_advantage"   : round(pp_pct - (100 - pk_pct), 4),
            # ── Possession ────────────────────────────────────────────────────
            "corsi_pct_5v5"      : round(corsi, 4),
            "fenwick_pct_5v5"    : round(fenwick, 4),
            "shooting_pct_5v5"   : round(sh_pct, 4),
            "save_pct_5v5"       : round(sv_pct, 4),
            "pdo_5v5"            : round(pdo, 4),
            "pdo_regression_flag": pdo_regression_flag,
            "zone_start_off_pct" : round(zone_o, 4),
            # ── Physical ──────────────────────────────────────────────────────
            "hits_pg"            : round(hits, 4),
            "blocks_pg"          : round(blocks, 4),
            "giveaways_pg"       : round(giveaways, 4),
            "takeaways_pg"       : round(takeaways, 4),
            "takeaway_ratio"     : round(tk_ratio, 4),
            # ── Winning / standings ───────────────────────────────────────────
            "win_pct"            : round(s.get("win_pct", 0), 4),
            "regulation_win_pct" : round(st.get("regulation_win_pct", 0), 4),
            "point_pct"          : round(st.get("point_pct", 0), 4),
            # ── Home / road ───────────────────────────────────────────────────
            "home_win_pct"       : round(home_win, 4),
            "road_win_pct"       : round(road_win, 4),
            "home_gf_pg"         : round(home_gf_pg, 4),
            "home_ga_pg"         : round(home_ga_pg, 4),
            "road_gf_pg"         : round(road_gf_pg, 4),
            "road_ga_pg"         : round(road_ga_pg, 4),
            "home_road_win_diff" : round(home_win - road_win, 4),
            # ── Form ──────────────────────────────────────────────────────────
            "l10_win_pct"        : round(l10_win_pct, 4),
            "streak_value"       : streak_value,
            # ── Puck-line ─────────────────────────────────────────────────────
            "blowout_index"      : round(blowout_index, 4),
            "faceoff_win_pct"    : round(s.get("faceoff_win_pct", 50), 4),
            # ── Meta ──────────────────────────────────────────────────────────
            "gp"                 : gp,
        }

    return merged
