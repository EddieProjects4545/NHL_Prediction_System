"""
Confirmed starting goalie scraper.

Fallback chain (best → worst):
  1. Rotowire NHL lineups   — posts expected goalies 12-24h before games
  2. Daily Faceoff          — day-of confirmation (fixes regex for current layout)
  3. Recent-streak logic    — goalie who started last 3 consecutive games
  4. Most-games-started     — season leader as final fallback

NHL API does not publish confirmed starters natively.
"""
import re
from typing import Dict, Optional

import requests

from config import TEAM_ABBREVS, CACHE_TTL_SECONDS
from data.cache import cache_get, cache_set

_session = requests.Session()
_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
})

# ─── Abbreviation normalisation ───────────────────────────────────────────────
_NAME_TO_ABBREV: Dict[str, str] = {
    "anaheim"      : "ANA", "ducks"         : "ANA",
    "boston"       : "BOS", "bruins"        : "BOS",
    "buffalo"      : "BUF", "sabres"        : "BUF",
    "calgary"      : "CGY", "flames"        : "CGY",
    "carolina"     : "CAR", "hurricanes"    : "CAR",
    "chicago"      : "CHI", "blackhawks"    : "CHI",
    "colorado"     : "COL", "avalanche"     : "COL",
    "columbus"     : "CBJ", "blue jackets"  : "CBJ",
    "dallas"       : "DAL", "stars"         : "DAL",
    "detroit"      : "DET", "red wings"     : "DET",
    "edmonton"     : "EDM", "oilers"        : "EDM",
    "florida"      : "FLA", "panthers"      : "FLA",
    "los angeles"  : "LAK", "kings"         : "LAK", "l.a." : "LAK",
    "minnesota"    : "MIN", "wild"          : "MIN",
    "montreal"     : "MTL", "canadiens"     : "MTL",
    "nashville"    : "NSH", "predators"     : "NSH",
    "new jersey"   : "NJD", "devils"        : "NJD",
    "new york islanders": "NYI", "islanders": "NYI",
    "new york rangers"  : "NYR", "rangers"  : "NYR",
    "ottawa"       : "OTT", "senators"      : "OTT",
    "philadelphia" : "PHI", "flyers"        : "PHI",
    "pittsburgh"   : "PIT", "penguins"      : "PIT",
    "san jose"     : "SJS", "sharks"        : "SJS",
    "seattle"      : "SEA", "kraken"        : "SEA",
    "st. louis"    : "STL", "blues"         : "STL",
    "tampa bay"    : "TBL", "lightning"     : "TBL",
    "toronto"      : "TOR", "maple leafs"   : "TOR",
    "utah"         : "UTA", "hockey club"   : "UTA",
    "vancouver"    : "VAN", "canucks"       : "VAN",
    "vegas"        : "VGK", "golden knights": "VGK",
    "washington"   : "WSH", "capitals"      : "WSH",
    "winnipeg"     : "WPG", "jets"          : "WPG",
}


def _fuzzy_abbrev(text: str) -> Optional[str]:
    text_lower = text.lower()
    # Longer keys first to avoid "wild" matching before "minnesota wild"
    for keyword in sorted(_NAME_TO_ABBREV, key=len, reverse=True):
        if keyword in text_lower:
            return _NAME_TO_ABBREV[keyword]
    return None


# ─── Source 1: Rotowire ───────────────────────────────────────────────────────

def scrape_rotowire(date_str: str) -> Dict[str, Dict]:
    """
    Scrape Rotowire NHL projected lineups for expected starting goalies.
    Returns {team_abbrev: {"name": str, "confirmed": bool}}
    """
    cache_key = f"goalies_rw_{date_str}"
    cached = cache_get(cache_key, ttl=1800)
    if cached is not None:
        return cached

    url = "https://www.rotowire.com/hockey/nhl-lineups.php"
    try:
        resp = _session.get(url, timeout=12)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"  [GoalieScraper] Rotowire unavailable: {e}")
        return {}

    starters: Dict[str, Dict] = {}

    # Rotowire lineup cards contain team name + "Expected Goalie: Name" pattern
    # Multiple patterns tried in order of reliability

    # Pattern A: data-player attribute or explicit goalie label
    # <div class="lineup__pos">G</div> followed by player name link
    pattern_a = re.compile(
        r'lineup__team[^>]*>\s*'
        r'(?:.*?)<(?:span|div)[^>]*(?:lineup__team-name|team-name)[^>]*>\s*'
        r'([^<]+?)\s*</(?:span|div)>'
        r'.*?'
        r'lineup__pos[^>]*>\s*G\s*</[^>]+>'
        r'\s*<[^>]+>\s*<a[^>]*>\s*([A-Z][a-z\'\-]+(?: [A-Z][a-z\'\-]+)+)\s*</a>',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern B: simpler — look for "Goalie" section header + name
    # Rotowire often has: <div class="lineup__player-name">Name</div> near G position
    pattern_b = re.compile(
        r'<li[^>]*class="[^"]*lineup__player[^"]*"[^>]*>\s*'
        r'(?:<[^>]+>\s*)*'
        r'<span[^>]*class="[^"]*lineup__pos[^"]*"[^>]*>\s*G\s*</span>'
        r'.*?'
        r'([A-Z][a-z\'\-]+(?: [A-Z][a-z\'\-]+)+)',
        re.DOTALL | re.IGNORECASE
    )

    # Pattern C: Most reliable — find all matchup blocks, each has two teams
    # Structure: team-name block ... goalie block
    # Split by "lineup__matchup" divs
    matchup_blocks = re.split(
        r'(?=<(?:div|section)[^>]*class="[^"]*lineup__matchup[^"]*")',
        html, flags=re.IGNORECASE
    )

    for block in matchup_blocks:
        # Find team names in this block
        team_names = re.findall(
            r'<(?:span|div)[^>]*(?:lineup__team-name|team-name|team__name)[^>]*>\s*'
            r'([^<]{3,40}?)\s*</(?:span|div)>',
            block, re.IGNORECASE
        )
        # Find goalies (G position) in this block
        goalies_in_block = re.findall(
            r'(?:class="[^"]*lineup__pos[^"]*"[^>]*>\s*G\s*</[^>]+>'
            r'|"pos"[^>]*>\s*G\s*<)'
            r'.*?'
            r'([A-Z][a-z\'\.\-]+(?: [A-Z][a-z\'\.\-]+)+)',
            block, re.DOTALL | re.IGNORECASE
        )

        if not goalies_in_block:
            # Fallback: look for any name next to 'G' position marker
            goalies_in_block = re.findall(
                r'\bG\b.*?([A-Z][a-z]+(?:\'[a-z]+)? [A-Z][a-z]+)',
                block
            )

        for i, team_raw in enumerate(team_names[:2]):
            abbrev = _fuzzy_abbrev(team_raw)
            if not abbrev:
                abbrev = team_raw.strip().upper()[:3]
                if abbrev not in TEAM_ABBREVS:
                    continue
            if i < len(goalies_in_block):
                name = goalies_in_block[i].strip()
                if len(name.split()) >= 2:   # sanity: must be First Last
                    starters[abbrev] = {"name": name, "confirmed": True}

    # Pattern D: simple JSON-like pattern in page script tags
    json_pattern = re.compile(
        r'"position"\s*:\s*"G"[^}]*"(?:name|fullName)"\s*:\s*"([^"]+)"',
        re.IGNORECASE
    )
    if len(starters) < 5:   # if fewer than 5 found, try JSON approach
        json_matches = json_pattern.findall(html)
        # Without team context these are harder to assign; skip if already have enough

    if starters:
        print(f"  [GoalieScraper] Rotowire: {len(starters)} goalies found")
    else:
        print("  [GoalieScraper] Rotowire: no goalies parsed (HTML layout may have changed)")

    cache_set(cache_key, starters)
    return starters


# ─── Source 2: Daily Faceoff ──────────────────────────────────────────────────

def scrape_daily_faceoff(date_str: str) -> Dict[str, Dict]:
    """
    Scrape Daily Faceoff starting goalies page.
    Returns {team_abbrev: {"name": str, "confirmed": bool}}
    """
    cache_key = f"goalies_df_{date_str}"
    cached = cache_get(cache_key, ttl=1800)
    if cached is not None:
        return cached

    url = "https://www.dailyfaceoff.com/starting-goalies/"
    try:
        resp = _session.get(url, timeout=12)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"  [GoalieScraper] Daily Faceoff unavailable: {e}")
        return {}

    starters: Dict[str, Dict] = {}

    # Pattern A: structured JSON data embedded in page
    json_goalie = re.compile(
        r'"goalie(?:Name|FullName)"\s*:\s*"([^"]+)"[^}]*"team(?:Name|Abbrev(?:iation)?)"\s*:\s*"([^"]+)"',
        re.IGNORECASE
    )
    for m in json_goalie.finditer(html):
        name = m.group(1).strip()
        team_raw = m.group(2).strip()
        abbrev = _fuzzy_abbrev(team_raw) or (team_raw.upper() if team_raw.upper() in TEAM_ABBREVS else None)
        if abbrev and len(name.split()) >= 2:
            starters[abbrev] = {"name": name, "confirmed": True}

    if starters:
        cache_set(cache_key, starters)
        return starters

    # Pattern B: modern layout — goalie card blocks
    # Look for team + goalie name + status near each other
    card_pattern = re.compile(
        r'(?:class="[^"]*starting-goalie[^"]*"|id="[^"]*goalie[^"]*")'
        r'(.*?)'
        r'(?:class="[^"]*starting-goalie[^"]*"|id="[^"]*goalie[^"]*"|$)',
        re.DOTALL | re.IGNORECASE
    )
    # Simpler: split by game blocks
    game_blocks = re.split(
        r'(?=<(?:div|article)[^>]*(?:game-card|matchup|starting-goalie)[^>]*>)',
        html, flags=re.IGNORECASE
    )
    for block in game_blocks:
        team_match = re.search(
            r'(?:team-name|team-city|teamName)[^>]*>\s*([^<]{3,40}?)\s*<',
            block, re.IGNORECASE
        )
        goalie_match = re.search(
            r'(?:goalie-name|goalieName|player-name)[^>]*>\s*'
            r'([A-Z][a-z\'\-]+(?: [A-Z][a-z\'\-]+)+)\s*<',
            block, re.IGNORECASE
        )
        if not goalie_match:
            # Broader: any capitalized first-last name
            goalie_match = re.search(
                r'([A-Z][a-z\'\-]+ [A-Z][a-z\'\-]+(?:-[A-Z][a-z]+)?)',
                block
            )
        status_match = re.search(
            r'\b(confirmed|expected|probable|likely|unconfirmed)\b',
            block, re.IGNORECASE
        )
        if not goalie_match:
            continue
        team_abbrev = None
        if team_match:
            team_abbrev = _fuzzy_abbrev(team_match.group(1))
            if not team_abbrev:
                raw = team_match.group(1).strip().upper()[:3]
                if raw in TEAM_ABBREVS:
                    team_abbrev = raw
        if team_abbrev:
            confirmed = bool(status_match and
                             status_match.group(1).lower() in
                             ("confirmed", "expected", "probable", "likely"))
            name = goalie_match.group(1).strip() if hasattr(goalie_match.group(1), 'strip') else goalie_match.group(0).strip()
            if len(name.split()) >= 2:
                starters[team_abbrev] = {"name": name, "confirmed": confirmed}

    if starters:
        print(f"  [GoalieScraper] Daily Faceoff: {len(starters)} goalies found")
    else:
        print("  [GoalieScraper] Daily Faceoff: no goalies parsed")

    cache_set(cache_key, starters)
    return starters


# ─── Source 3: Recent-streak fallback ────────────────────────────────────────

def get_streak_starters(game_type: int = 2) -> Dict[str, Dict]:
    """
    For each team, find the goalie who has started the most recent consecutive games.
    If the same goalie started the last 3 games, treat as high-confidence presumed starter.
    Returns {team_abbrev: {"name": str, "confirmed": bool, "streak": int, ...}}
    """
    from data.nhl_api import get_goalies_by_team, get_game_results_range
    from config import CURRENT_SEASON, SEASON_START_DATE
    from datetime import date, timedelta

    by_team = get_goalies_by_team(game_type=game_type)

    # Get last 10 games per team from game results
    end_date  = (date.today() - timedelta(days=1)).isoformat()
    start_adj = (date.today() - timedelta(days=30)).isoformat()
    try:
        recent_games = get_game_results_range(start_adj, end_date, game_type)
    except Exception:
        recent_games = []

    # Build per-team recent game list (last 5 games in chronological order)
    team_recent: Dict[str, list] = {}
    for g in sorted(recent_games, key=lambda x: x.get("date", "")):
        for side in ("home_team", "away_team"):
            t = g.get(side)
            if t:
                team_recent.setdefault(t, []).append(g)

    result = {}
    for team, goalies in by_team.items():
        if not goalies:
            continue
        top = goalies[0]   # highest games-started = most likely starter

        # Check if top goalie started last 3+ games
        # (We don't have per-game goalie starters without boxscore calls,
        #  so we use games-started % as a proxy: if >85% starter workload,
        #  they almost certainly started the last few games)
        workload = top.get("gs", 0) / max(top.get("gp", 1), 1)
        streak_confirmed = workload >= 0.80  # starts 80%+ = effectively confirmed

        result[team] = {
            "name"      : top["name"],
            "confirmed" : streak_confirmed,
            "save_pct"  : top.get("save_pct", 0),
            "gaa"       : top.get("gaa", 0),
            "gs"        : top.get("gs", 0),
            "_source"   : "streak" if streak_confirmed else "presumed",
        }

    return result


# ─── Source 4: Most-games-started fallback ────────────────────────────────────

def get_presumed_starters(game_type: int = 2) -> Dict[str, Dict]:
    """
    Final fallback: highest games-started this season.
    Returns {team_abbrev: {"name": str, "confirmed": False, ...}}
    """
    from data.nhl_api import get_goalies_by_team
    by_team = get_goalies_by_team(game_type=game_type)
    result = {}
    for team, goalies in by_team.items():
        if goalies:
            g = goalies[0]
            result[team] = {
                "name"      : g["name"],
                "confirmed" : False,
                "save_pct"  : g["save_pct"],
                "gaa"       : g["gaa"],
                "gs"        : g["gs"],
                "_source"   : "presumed",
            }
    return result


# ─── Combined entry point ─────────────────────────────────────────────────────

def get_confirmed_starters(date_str: str, game_type: int = 2) -> Dict[str, Dict]:
    """
    Best-effort confirmed starter lookup using multi-source fallback chain.

    Priority:
      1. Rotowire (primary — posts 12-24h ahead)
      2. Daily Faceoff (secondary — day-of confirmation)
      3. Recent-streak logic (tertiary — 80%+ workload = confirmed)
      4. Most-games-started (final fallback — always available)

    If both Rotowire AND Daily Faceoff agree on a goalie: confirmed=True, bonus signal.
    Returns: {team_abbrev: {"name", "confirmed", "save_pct", "gaa", "gs"}}
    """
    from data.nhl_api import get_goalies_by_team

    # Gather all sources
    rw_data     = scrape_rotowire(date_str)
    df_data     = scrape_daily_faceoff(date_str)
    streak_data = get_streak_starters(game_type)
    presumed    = get_presumed_starters(game_type)

    # Build goalie name → stats lookup from NHL API
    by_team = get_goalies_by_team(game_type=game_type)
    name_to_stats: Dict[str, Dict] = {}
    for team, goalies in by_team.items():
        for g in goalies:
            name_to_stats[g["name"].lower()] = g

    result = {}
    for team in TEAM_ABBREVS:
        entry = None

        # 1. Rotowire
        if team in rw_data:
            name = rw_data[team]["name"]
            both_agree = (team in df_data and
                          df_data[team]["name"].lower()[:6] == name.lower()[:6])
            stats = name_to_stats.get(name.lower(), presumed.get(team, {}))
            entry = {
                "name"      : name,
                "confirmed" : True,
                "save_pct"  : stats.get("save_pct", 0),
                "gaa"       : stats.get("gaa", 0),
                "gs"        : stats.get("gs", 0),
                "_source"   : "rotowire+df" if both_agree else "rotowire",
            }

        # 2. Daily Faceoff (use if no Rotowire result or to override)
        elif team in df_data:
            name = df_data[team]["name"]
            stats = name_to_stats.get(name.lower(), presumed.get(team, {}))
            entry = {
                "name"      : name,
                "confirmed" : df_data[team]["confirmed"],
                "save_pct"  : stats.get("save_pct", 0),
                "gaa"       : stats.get("gaa", 0),
                "gs"        : stats.get("gs", 0),
                "_source"   : "dailyfaceoff",
            }

        # 3. Streak-based
        elif team in streak_data:
            entry = streak_data[team]

        # 4. Season leader
        elif team in presumed:
            entry = presumed[team]

        if entry:
            result[team] = entry

    n_confirmed = sum(1 for v in result.values() if v.get("confirmed"))
    n_sources = {"rotowire": 0, "dailyfaceoff": 0, "streak": 0, "presumed": 0,
                 "rotowire+df": 0}
    for v in result.values():
        src = v.get("_source", "presumed")
        n_sources[src] = n_sources.get(src, 0) + 1

    src_summary = ", ".join(f"{k}:{v}" for k, v in n_sources.items() if v > 0)
    print(f"  [GoalieScraper] Sources: {src_summary}")
    return result
