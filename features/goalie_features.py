"""
Goalie feature builder.

Goalie is the single highest-leverage variable in NHL prediction.
We track:
  - Season SV%, GAA, Quality Start %, games started
  - Starter vs backup differential (teams with deep goaltending are more
    resilient when their starter is unexpectedly rested)
  - Confirmed vs estimated starter flag (confidence modifier)
  - GSAX proxy: (team_save_pct - league_avg_save_pct) × shots_against
"""
from typing import Dict, List, Optional

from data.nhl_api import get_goalies_by_team, get_goalie_game_log
from config import CURRENT_SEASON, GAME_TYPE_REGULAR

# 2025-26 approximate league-average SV%
LEAGUE_AVG_SV_PCT = 0.900
LEAGUE_AVG_GAA    = 2.85


def build_goalie_features(season: str = CURRENT_SEASON,
                          game_type: int = GAME_TYPE_REGULAR) -> Dict[str, Dict]:
    """
    Build per-team goalie feature set.
    Returns: {team_abbrev: {goalie features}}
    """
    by_team = get_goalies_by_team(season, game_type)
    result: Dict[str, Dict] = {}

    for team, goalies in by_team.items():
        if not goalies:
            result[team] = _empty_goalie_features()
            continue

        # Starter = highest games started
        starter = goalies[0]
        backup  = goalies[1] if len(goalies) > 1 else None

        starter_sv  = starter.get("save_pct", LEAGUE_AVG_SV_PCT) or LEAGUE_AVG_SV_PCT
        starter_gaa = starter.get("gaa", LEAGUE_AVG_GAA) or LEAGUE_AVG_GAA
        starter_gs  = starter.get("gs", 0)
        starter_qsp = starter.get("quality_starts_pct", 0) or 0

        backup_sv   = (backup.get("save_pct", LEAGUE_AVG_SV_PCT)
                       if backup else LEAGUE_AVG_SV_PCT) or LEAGUE_AVG_SV_PCT
        backup_gs   = backup.get("gs", 0) if backup else 0

        # GSAX proxy: how many extra goals the goalie saves per game vs average
        gsax_pg = (starter_sv - LEAGUE_AVG_SV_PCT) * 30  # ~30 shots/game avg

        # Starter workload: high workload means more fatigue but also consistency
        workload_pct = starter_gs / max(starter_gs + backup_gs, 1)

        # Depth differential: difference between starter and backup SV%
        depth_diff = starter_sv - backup_sv

        # ── Recent form (last 5 starts) ──────────────────────────────────────
        player_id = starter.get("player_id")
        l5_sv_pct  = starter_sv   # default to season average
        l5_gaa     = starter_gaa
        l5_vs_season = 0.0
        if player_id:
            try:
                logs = get_goalie_game_log(player_id, season=season, n=5)
                sv_vals  = [g["sv_pct"] for g in logs if g.get("sv_pct", 0) > 0]
                gaa_vals = [g["gaa"]    for g in logs if g.get("gaa",    0) > 0]
                if len(sv_vals) >= 3:
                    l5_sv_pct    = round(sum(sv_vals) / len(sv_vals), 4)
                    l5_vs_season = round(l5_sv_pct - starter_sv, 4)
                if len(gaa_vals) >= 3:
                    l5_gaa = round(sum(gaa_vals) / len(gaa_vals), 4)
            except Exception:
                pass   # silently fall back to season stats

        result[team] = {
            "player_id"           : player_id,
            "starter_save_pct"    : round(starter_sv, 4),
            "starter_gaa"         : round(starter_gaa, 4),
            "starter_gs"          : starter_gs,
            "starter_qs_pct"      : round(starter_qsp, 4),
            "starter_gsax_pg"     : round(gsax_pg, 4),
            "starter_sv_vs_avg"   : round(starter_sv - LEAGUE_AVG_SV_PCT, 4),
            "starter_l5_sv_pct"   : l5_sv_pct,
            "starter_l5_gaa"      : l5_gaa,
            "starter_l5_vs_season": l5_vs_season,
            "backup_save_pct"     : round(backup_sv, 4),
            "backup_gs"           : backup_gs,
            "goalie_depth_diff"   : round(depth_diff, 4),
            "starter_workload_pct": round(workload_pct, 4),
            "starter_name"        : starter.get("name", ""),
            # Default confirmed=1: season leader IS the primary starter for historical games.
            # apply_confirmed_starters() overrides this with explicit 0/1 on game day.
            "starter_confirmed"   : 1,
        }

    return result


def apply_confirmed_starters(goalie_feats: Dict[str, Dict],
                              confirmed: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Override starter stats with confirmed starter data.
    `confirmed` is the output of goalie_scraper.get_confirmed_starters().
    Updates: starter_save_pct, starter_gaa, starter_confirmed flag.
    """
    updated = {}
    for team, feats in goalie_feats.items():
        f = dict(feats)
        conf = confirmed.get(team, {})
        if conf:
            f["starter_name"]      = conf.get("name", f.get("starter_name", ""))
            f["starter_confirmed"] = int(conf.get("confirmed", False))
            # Override with confirmed goalie's season stats if available
            sv = conf.get("save_pct")
            ga = conf.get("gaa")
            if sv:
                f["starter_save_pct"]  = round(sv, 4)
                f["starter_sv_vs_avg"] = round(sv - LEAGUE_AVG_SV_PCT, 4)
                f["starter_gsax_pg"]   = round((sv - LEAGUE_AVG_SV_PCT) * 30, 4)
            if ga:
                f["starter_gaa"] = round(ga, 4)
        else:
            f["starter_confirmed"] = 0
        updated[team] = f
    return updated


def _empty_goalie_features() -> Dict:
    return {
        "player_id"           : None,
        "starter_save_pct"    : LEAGUE_AVG_SV_PCT,
        "starter_gaa"         : LEAGUE_AVG_GAA,
        "starter_gs"          : 0,
        "starter_qs_pct"      : 0,
        "starter_gsax_pg"     : 0,
        "starter_sv_vs_avg"   : 0,
        "starter_l5_sv_pct"   : LEAGUE_AVG_SV_PCT,
        "starter_l5_gaa"      : LEAGUE_AVG_GAA,
        "starter_l5_vs_season": 0.0,
        "backup_save_pct"     : LEAGUE_AVG_SV_PCT,
        "backup_gs"           : 0,
        "goalie_depth_diff"   : 0,
        "starter_workload_pct": 0.5,
        "starter_confirmed"   : 0,
        "starter_name"        : "Unknown",
    }
