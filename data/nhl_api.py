"""
NHL Stats API wrapper.
Covers both the modern web API (api-web.nhle.com) and the legacy
stats REST API (api.nhle.com/stats/rest/en).

All calls are cached via data/cache.py.
"""
import time
from datetime import date, timedelta, datetime
from typing import List, Dict, Optional

import requests

from config import (
    NHL_WEB_API_BASE, NHL_STATS_API_BASE,
    CURRENT_SEASON, PREV_SEASON,
    GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFF,
    SEASON_START_DATE, PREV_SEASON_START, PREV_SEASON_END,
    TEAM_ABBREVS, TEAM_FULL_NAMES, CACHE_TTL_SECONDS,
)
from data.cache import cached_request, cache_get, cache_set

# Reverse map: "Carolina Hurricanes" → "CAR"
_FULL_TO_ABBREV: Dict[str, str] = {v: k for k, v in TEAM_FULL_NAMES.items()}


def _team_abbrev(row: Dict) -> str:
    """
    Extract team abbreviation from any NHL Stats REST API row.
    Different endpoints use different field names — try all of them.
    """
    abbrev = row.get("teamAbbrevs") or row.get("teamAbbrev") or ""
    if not abbrev:
        full = row.get("teamFullName", "")
        abbrev = _FULL_TO_ABBREV.get(full, "")
    return abbrev

_session = requests.Session()
_session.headers.update({"User-Agent": "NHLBettingModel/1.0"})

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _web(endpoint: str, params: dict = None, ttl: int = CACHE_TTL_SECONDS):
    url = f"{NHL_WEB_API_BASE}/{endpoint.lstrip('/')}"
    return cached_request(_session, url, params, ttl)


def _stats(endpoint: str, params: dict = None, ttl: int = CACHE_TTL_SECONDS):
    url = f"{NHL_STATS_API_BASE}/{endpoint.lstrip('/')}"
    return cached_request(_session, url, params, ttl)


def _date_range(start: str, end: str) -> List[str]:
    """Return every date string YYYY-MM-DD from start to end inclusive."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out = []
    while s <= e:
        out.append(s.isoformat())
        s += timedelta(days=1)
    return out


# ─── Schedule / Results ───────────────────────────────────────────────────────

def get_schedule(game_date: str) -> List[Dict]:
    """Return list of games scheduled on a given date."""
    data = _web(f"schedule/{game_date}", ttl=900)
    games = []
    for week in data.get("gameWeek", []):
        week_date = week.get("date", "")
        if week_date == game_date:
            for g in week.get("games", []):
                g["gameDate"] = week_date   # inject date — API omits it per game
                games.append(g)
    return games


def get_upcoming_games(days_ahead: int = 7) -> List[Dict]:
    """Return all upcoming games for the next N days (single API call)."""
    today = date.today().isoformat()
    end_d = (date.today() + timedelta(days=days_ahead)).isoformat()
    data  = _web(f"schedule/{today}", ttl=900)
    games = []
    for week in data.get("gameWeek", []):
        week_date = week.get("date", "")
        if week_date < today or week_date > end_d:
            continue
        for g in week.get("games", []):
            state = g.get("gameState", "")
            if state in ("OFF", "FINAL", "OVER"):
                continue
            g["gameDate"] = week_date
            games.append(g)
    return games


def get_game_results_range(start_date: str, end_date: str,
                           game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    """
    Return completed game results between two dates.
    Each result dict contains:
        game_id, date, home_team, away_team,
        home_goals, away_goals, ot_flag, shootout_flag,
        home_sog, away_sog
    """
    cache_key = f"game_results_{start_date}_{end_date}_{game_type}"
    cached = cache_get(cache_key, ttl=CACHE_TTL_SECONDS * 24)
    if cached is not None:
        return cached

    results = []
    for d in _date_range(start_date, end_date):
        try:
            data = _web(f"score/{d}", ttl=CACHE_TTL_SECONDS * 24)
        except Exception:
            continue
        for g in data.get("games", []):
            if g.get("gameType") != game_type:
                continue
            state = g.get("gameState", "")
            if state not in ("FINAL", "OFF", "OVER"):
                continue
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            period = g.get("periodDescriptor", {}).get("periodType", "REG")
            results.append({
                "game_id"       : g.get("id"),
                "date"          : d,
                "home_team"     : home.get("abbrev"),
                "away_team"     : away.get("abbrev"),
                "home_goals"    : home.get("score", 0),
                "away_goals"    : away.get("score", 0),
                "home_sog"      : home.get("sog", 0),
                "away_sog"      : away.get("sog", 0),
                "ot_flag"       : period in ("OT", "SO"),
                "shootout_flag" : period == "SO",
                "game_type"     : game_type,
            })

    cache_set(cache_key, results)
    return results


def get_season_results(season: str = CURRENT_SEASON,
                       game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    """All completed results for an entire season."""
    if season == CURRENT_SEASON:
        start = SEASON_START_DATE
        end   = (date.today() - timedelta(days=1)).isoformat()
    else:
        start = PREV_SEASON_START
        end   = PREV_SEASON_END
    results = get_game_results_range(start, end, game_type)
    for row in results:
        row["season"] = season
    return results


# ─── Standings ────────────────────────────────────────────────────────────────

def get_standings() -> List[Dict]:
    """Current standings with full home/road splits."""
    data = _web("standings/now", ttl=3600)
    return data.get("standings", [])


def parse_standings() -> Dict[str, Dict]:
    """
    Return dict keyed by team abbreviation with pre-processed split stats.
    """
    raw = get_standings()
    out = {}
    for row in raw:
        abbrev = row.get("teamAbbrev", {})
        if isinstance(abbrev, dict):
            abbrev = abbrev.get("default", "")
        gp   = row.get("gamesPlayed", 1) or 1
        hgp  = row.get("homeGamesPlayed", 1) or 1
        rgp  = row.get("roadGamesPlayed", 1) or 1
        out[abbrev] = {
            "gp"                  : gp,
            "wins"                : row.get("wins", 0),
            "losses"              : row.get("losses", 0),
            "ot_losses"           : row.get("otLosses", 0),
            "points"              : row.get("points", 0),
            "regulation_wins"     : row.get("regulationWins", 0),
            "regulation_win_pct"  : row.get("regulationWins", 0) / gp,
            "point_pct"           : row.get("pointPctg", 0),
            "streak_code"         : row.get("streakCode", ""),
            "streak_count"        : row.get("streakCount", 0),
            "l10_wins"            : row.get("l10Wins", 0),
            "l10_losses"          : row.get("l10Losses", 0),
            "l10_ot"              : row.get("l10OtLosses", 0),
            "home_gp"             : hgp,
            "home_wins"           : row.get("homeWins", 0),
            "home_losses"         : row.get("homeLosses", 0),
            "home_ot"             : row.get("homeOtLosses", 0),
            "home_win_pct"        : row.get("homeWins", 0) / hgp,
            "home_gf"             : row.get("homeGoalsFor", 0),
            "home_ga"             : row.get("homeGoalsAgainst", 0),
            "home_gf_pg"          : row.get("homeGoalsFor", 0) / hgp,
            "home_ga_pg"          : row.get("homeGoalsAgainst", 0) / hgp,
            "road_gp"             : rgp,
            "road_wins"           : row.get("roadWins", 0),
            "road_losses"         : row.get("roadLosses", 0),
            "road_ot"             : row.get("roadOtLosses", 0),
            "road_win_pct"        : row.get("roadWins", 0) / rgp,
            "road_gf"             : row.get("roadGoalsFor", 0),
            "road_ga"             : row.get("roadGoalsAgainst", 0),
            "road_gf_pg"          : row.get("roadGoalsFor", 0) / rgp,
            "road_ga_pg"          : row.get("roadGoalsAgainst", 0) / rgp,
        }
    return out


# ─── Team Season Stats (Legacy Stats REST API) ────────────────────────────────

def _stats_cayenne(endpoint: str, season: str, game_type: int,
                   extra: str = "") -> List[Dict]:
    expr = f"seasonId={season} and gameTypeId={game_type}"
    if extra:
        expr += f" and {extra}"
    data = _stats(endpoint, {"cayenneExp": expr, "limit": -1})
    return data.get("data", [])


def get_team_summary(season: str = CURRENT_SEASON,
                     game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    return _stats_cayenne("team/summary", season, game_type)


def get_team_realtime(season: str = CURRENT_SEASON,
                      game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    return _stats_cayenne("team/realtime", season, game_type)


def get_team_percentages(season: str = CURRENT_SEASON,
                         game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    return _stats_cayenne("team/percentages", season, game_type)


def get_team_powerplay(season: str = CURRENT_SEASON,
                       game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    return _stats_cayenne("team/powerplay", season, game_type)


def get_team_faceoff(season: str = CURRENT_SEASON,
                     game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    return _stats_cayenne("team/faceoffpercentages", season, game_type)


def get_goalie_stats(season: str = CURRENT_SEASON,
                     game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    return _stats_cayenne("goalie/summary", season, game_type)


def get_skater_stats(season: str = CURRENT_SEASON,
                     game_type: int = GAME_TYPE_REGULAR) -> List[Dict]:
    return _stats_cayenne("skater/summary", season, game_type)


# ─── Merged Team Stats ────────────────────────────────────────────────────────

def get_all_team_stats(season: str = CURRENT_SEASON,
                       game_type: int = GAME_TYPE_REGULAR) -> Dict[str, Dict]:
    """
    Merge summary + realtime + percentages + powerplay + faceoff
    into one dict keyed by team abbreviation.
    """
    cache_key = f"all_team_stats_{season}_{game_type}"
    cached = cache_get(cache_key, ttl=3600)
    if cached:
        return cached

    merged: Dict[str, Dict] = {}

    for row in get_team_summary(season, game_type):
        abbrev = _team_abbrev(row)
        if not abbrev:
            continue
        gp = row.get("gamesPlayed", 1) or 1
        merged[abbrev] = {
            "gp"              : gp,
            "gf_pg"           : row.get("goalsForPerGame", 0),
            "ga_pg"           : row.get("goalsAgainstPerGame", 0),
            "gf_total"        : row.get("goalsFor", 0),
            "ga_total"        : row.get("goalsAgainst", 0),
            "shots_for_pg"    : row.get("shotsForPerGame", 0),
            "shots_against_pg": row.get("shotsAgainstPerGame", 0),
            "pp_pct"          : row.get("powerPlayPct", 0) or 0,
            "pk_pct"          : row.get("penaltyKillPct", 0) or 0,
            "wins"            : row.get("wins", 0),
            "losses"          : row.get("losses", 0),
            "ot_losses"       : row.get("otLosses", 0),
            "win_pct"         : row.get("wins", 0) / gp,
        }

    for row in get_team_realtime(season, game_type):
        abbrev = _team_abbrev(row)
        if abbrev not in merged:
            continue
        gp = merged[abbrev]["gp"] or 1
        merged[abbrev].update({
            "hits_pg"       : row.get("hits", 0) / gp,
            "blocks_pg"     : row.get("blockedShots", 0) / gp,
            "giveaways_pg"  : row.get("giveaways", 0) / gp,
            "takeaways_pg"  : row.get("takeaways", 0) / gp,
            "takeaway_ratio": (row.get("takeaways", 0) /
                               max(row.get("giveaways", 1), 1)),
        })

    for row in get_team_percentages(season, game_type):
        abbrev = _team_abbrev(row)
        if abbrev not in merged:
            continue
        sp  = row.get("shootingPctg5v5", 0) or 0
        svp = row.get("savePctg5v5", 0) or 0
        merged[abbrev].update({
            "corsi_pct_5v5"    : row.get("satPctg", row.get("corsiPctg", 50)) or 50,
            "fenwick_pct_5v5"  : row.get("usatPctg", row.get("fenwickPctg", 50)) or 50,
            "shooting_pct_5v5" : sp,
            "save_pct_5v5"     : svp,
            "pdo_5v5"          : round(sp + svp, 4),
            "zone_start_off_pct": row.get("offZoneStartPctg", 50) or 50,
        })

    for row in get_team_powerplay(season, game_type):
        abbrev = _team_abbrev(row)
        if abbrev not in merged:
            continue
        gp = merged[abbrev]["gp"] or 1
        merged[abbrev].update({
            "pp_opportunities_pg": row.get("ppOpportunities", 0) / gp,
            "pp_goals_pg"        : row.get("ppGoalsFor", 0) / gp,
            "pk_opportunities_pg": row.get("shOpportunities", 0) / gp,
        })

    for row in get_team_faceoff(season, game_type):
        abbrev = _team_abbrev(row)
        if abbrev not in merged:
            continue
        merged[abbrev].update({
            "faceoff_win_pct": row.get("totalFaceoffWinPct",
                               row.get("faceoffWinPct", 50)) or 50,
        })

    cache_set(cache_key, merged)
    return merged


# ─── Goalies ──────────────────────────────────────────────────────────────────

def get_goalies_by_team(season: str = CURRENT_SEASON,
                        game_type: int = GAME_TYPE_REGULAR) -> Dict[str, List[Dict]]:
    """Return dict: team_abbrev → list of goalie stat dicts, sorted by GP desc."""
    cache_key = f"goalies_by_team_{season}_{game_type}"
    cached = cache_get(cache_key, ttl=3600)
    if cached:
        return cached

    raw = get_goalie_stats(season, game_type)
    by_team: Dict[str, List] = {}
    for g in raw:
        team = g.get("teamAbbrevs", g.get("teamAbbrev", ""))
        by_team.setdefault(team, []).append({
            "player_id"  : g.get("playerId"),
            "name"       : g.get("goalieFullName", ""),
            "gp"         : g.get("gamesPlayed", 0),
            "gs"         : g.get("gamesStarted", g.get("gamesPlayed", 0)),
            "wins"       : g.get("wins", 0),
            "losses"     : g.get("losses", 0),
            "gaa"        : g.get("goalsAgainstAverage", 0) or 0,
            "save_pct"   : g.get("savePct", 0) or 0,
            "shutouts"   : g.get("shutouts", 0),
            "quality_starts_pct": g.get("qualityStartsPct", 0) or 0,
        })

    for team in by_team:
        by_team[team].sort(key=lambda x: x["gs"], reverse=True)

    cache_set(cache_key, by_team)
    return by_team


def get_goalie_game_log(player_id: int,
                        season: str = CURRENT_SEASON,
                        n: int = 10) -> List[Dict]:
    """
    Return the last N game entries for a goalie from the NHL Stats REST API.
    Each entry: {date, sv_pct, gaa, toi, decision, shots_against}
    Cached for 1 hour (stats finalize after each game).
    """
    cache_key = f"goalie_log_{player_id}_{season}_{n}"
    cached = cache_get(cache_key, ttl=CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        data = _stats(
            "goalie/gamebyGameRegularAndPlayoffs",
            params={
                "limit"       : n,
                "sort"        : '[{"property":"gameDate","direction":"DESC"}]',
                "cayenneExp"  : f"seasonId={season} and playerId={player_id}",
            },
            ttl=CACHE_TTL_SECONDS,
        )
        rows = data.get("data", [])
    except Exception:
        rows = []

    logs = []
    for r in rows:
        sv  = r.get("savePcnt") or r.get("savePct") or 0
        gaa = r.get("goalsAgainstAverage") or r.get("gaa") or 0
        logs.append({
            "date"           : r.get("gameDate", ""),
            "sv_pct"         : float(sv),
            "gaa"            : float(gaa),
            "shots_against"  : int(r.get("shotsAgainst", 0)),
            "decision"       : r.get("decision", ""),   # W/L/OTL/ND
        })

    cache_set(cache_key, logs)
    return logs


# ─── Club-level stats & roster ────────────────────────────────────────────────

def get_club_stats(team_abbrev: str) -> Dict:
    """Per-team endpoint with skater + goalie breakdowns."""
    return _web(f"club-stats/{team_abbrev}/now", ttl=3600)


def get_roster(team_abbrev: str, season: str = CURRENT_SEASON) -> List[Dict]:
    data = _web(f"roster/{team_abbrev}/{season}", ttl=86400)
    players = []
    for pos in ("forwards", "defensemen", "goalies"):
        for p in data.get(pos, []):
            players.append(p)
    return players


# ─── Play-by-play (empty net goals) ───────────────────────────────────────────

def get_play_by_play(game_id: int) -> List[Dict]:
    """Raw play-by-play events for a single game."""
    return _web(f"gamecenter/{game_id}/play-by-play",
                ttl=CACHE_TTL_SECONDS * 72).get("plays", [])


def extract_empty_net_goals(plays: List[Dict]) -> Dict:
    """
    From a play-by-play list, return counts of empty net goals
    for and against each team.
    Returns dict: {home_en_goals, away_en_goals}
    """
    home_en = 0
    away_en = 0
    for p in plays:
        if p.get("typeDescKey") != "goal":
            continue
        sc = str(p.get("situationCode", "0000"))
        # situationCode: [away_skaters][away_goalie][home_skaters][home_goalie]
        # e.g. "1051" = 5v4 with home goalie present
        if len(sc) == 4:
            away_goalie = sc[1]
            home_goalie = sc[3]
            team = p.get("details", {}).get("eventOwnerTeamId")
            home_id = p.get("homeTeamId")
            if team == home_id and away_goalie == "0":
                home_en += 1   # Home scored into empty net
            elif team != home_id and home_goalie == "0":
                away_en += 1   # Away scored into empty net
    return {"home_en_goals": home_en, "away_en_goals": away_en}


def get_team_en_stats(game_results: List[Dict],
                      sample_games: int = 60) -> Dict[str, Dict]:
    """
    Compute per-team empty-net goal rates by sampling recent game PBP.
    Returns: {team_abbrev: {en_goals_for_pg, en_goals_against_pg,
                            lead_hold_rate (estimated)}}
    """
    cache_key = f"en_stats_{sample_games}"
    cached = cache_get(cache_key, ttl=CACHE_TTL_SECONDS * 6)
    if cached:
        return cached

    # Take the most recent games to sample
    recent = sorted(game_results, key=lambda x: x["date"], reverse=True)
    recent = recent[:sample_games]

    team_data: Dict[str, Dict] = {
        t: {"en_for": 0, "en_against": 0, "gp": 0} for t in TEAM_ABBREVS
    }

    for g in recent:
        gid = g.get("game_id")
        if not gid:
            continue
        try:
            plays = get_play_by_play(gid)
            en = extract_empty_net_goals(plays)
            ht = g["home_team"]
            at = g["away_team"]
            if ht in team_data:
                team_data[ht]["en_for"]     += en["home_en_goals"]
                team_data[ht]["en_against"] += en["away_en_goals"]
                team_data[ht]["gp"]         += 1
            if at in team_data:
                team_data[at]["en_for"]     += en["away_en_goals"]
                team_data[at]["en_against"] += en["home_en_goals"]
                team_data[at]["gp"]         += 1
            time.sleep(0.05)   # Gentle throttle
        except Exception:
            continue

    result = {}
    for team, d in team_data.items():
        gp = d["gp"] or 1
        result[team] = {
            "en_goals_for_pg"    : d["en_for"]     / gp,
            "en_goals_against_pg": d["en_against"] / gp,
        }

    cache_set(cache_key, result)
    return result
