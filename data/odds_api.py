"""
The Odds API wrapper.
Fetches h2h (ML), spreads (puck line ±1.5), and totals (O/U) for NHL games.
Priority books: FanDuel, DraftKings, then best available.
Free tier = 500 credits/month; caching protects this limit.
"""
import requests
from typing import Dict, List, Optional, Tuple

from config import (
    ODDS_API_KEY, ODDS_API_BASE, ODDS_SPORT,
    ODDS_REGIONS, ODDS_MARKETS, ODDS_PRIORITY_BOOKS,
    CACHE_TTL_SECONDS,
)
from data.cache import cache_get, cache_set

_session = requests.Session()


# ─── Raw Fetch ────────────────────────────────────────────────────────────────

def fetch_odds(markets: str = ODDS_MARKETS) -> List[Dict]:
    """
    Fetch live NHL odds from The Odds API.
    Returns raw list of game odds objects.
    Cached for 1 hour to protect credit limit.
    """
    if not ODDS_API_KEY:
        return []

    cache_key = f"odds_{markets}"
    cached = cache_get(cache_key, ttl=CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    url = f"{ODDS_API_BASE}/sports/{ODDS_SPORT}/odds"
    params = {
        "apiKey"         : ODDS_API_KEY,
        "regions"        : ODDS_REGIONS,
        "markets"        : markets,
        "oddsFormat"     : "american",
        "dateFormat"     : "iso",
    }
    try:
        resp = _session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Log remaining credits from response headers
        remaining = resp.headers.get("x-requests-remaining", "?")
        used      = resp.headers.get("x-requests-used", "?")
        print(f"  [Odds API] Credits used: {used} | Remaining: {remaining}")
        cache_set(cache_key, data)
        return data
    except Exception as e:
        print(f"  [Odds API] Error fetching odds: {e}")
        return []


# ─── Parsing Helpers ──────────────────────────────────────────────────────────

def _american_to_decimal(american: int) -> float:
    if american > 0:
        return american / 100 + 1
    else:
        return 100 / abs(american) + 1


def _best_line(bookmakers: List[Dict], market_key: str,
               team_name: str, is_over: bool = False,
               point_sign: int = None) -> Optional[Tuple[int, str]]:
    """
    Return (best_american_odds, book_name, point) for a given team/side.
    Priority books are checked first; fallback to best odds across all books.
    For totals, team_name should be "Over" or "Under" and is_over used to filter.
    For spreads, point_sign=-1 restricts to negative-point outcomes (-1.5),
    point_sign=+1 restricts to positive-point outcomes (+1.5), None = no filter.
    """
    candidates = []
    for book in bookmakers:
        book_name = book.get("key", "")
        for mkt in book.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            for outcome in mkt.get("outcomes", []):
                name  = outcome.get("name", "")
                price = outcome.get("price")
                point = outcome.get("point")
                if price is None:
                    continue
                if market_key == "totals":
                    if is_over and name == "Over":
                        candidates.append((price, book_name, point))
                    elif not is_over and name == "Under":
                        candidates.append((price, book_name, point))
                else:
                    if team_name.lower() in name.lower():
                        # Filter by point sign for spread markets
                        if point_sign is not None and market_key == "spreads":
                            if point is None:
                                continue
                            if point_sign == -1 and point >= 0:
                                continue
                            if point_sign == 1 and point <= 0:
                                continue
                        candidates.append((price, book_name, point))

    if not candidates:
        return None

    # Priority books first
    for priority in ODDS_PRIORITY_BOOKS:
        for c in candidates:
            if c[1] == priority:
                return (c[0], c[1], c[2])

    # Best odds (highest decimal return)
    candidates.sort(key=lambda x: _american_to_decimal(x[0]), reverse=True)
    return candidates[0]


def _consensus_line(bookmakers: List[Dict], market_key: str,
                    team_name: str, is_over: bool = False) -> Optional[float]:
    """Return average american odds across all available books."""
    prices = []
    for book in bookmakers:
        for mkt in book.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            for outcome in mkt.get("outcomes", []):
                name  = outcome.get("name", "")
                price = outcome.get("price")
                if price is None:
                    continue
                if market_key == "totals":
                    match = (is_over and name == "Over") or \
                            (not is_over and name == "Under")
                else:
                    match = team_name.lower() in name.lower()
                if match:
                    prices.append(price)
    return round(sum(prices) / len(prices)) if prices else None


def _get_ou_line(bookmakers: List[Dict]) -> Optional[float]:
    """Return the most common O/U line across books."""
    lines = []
    for book in bookmakers:
        for mkt in book.get("markets", []):
            if mkt.get("key") != "totals":
                continue
            for outcome in mkt.get("outcomes", []):
                pt = outcome.get("point")
                if pt is not None:
                    lines.append(pt)
    if not lines:
        return None
    # Return modal line
    from collections import Counter
    return Counter(lines).most_common(1)[0][0]


# ─── Structured Game Odds ─────────────────────────────────────────────────────

def parse_game_odds(raw: Dict) -> Dict:
    """
    Parse a single game from the API response into a structured dict:
    {
        game_id, home_team, away_team, commence_time,
        ml: {home: {odds, book, implied}, away: {odds, book, implied},
             consensus_home, consensus_away},
        pl: {home_ml1_5: {...}, away_pl1_5: {...}, line},
        ou: {over: {...}, under: {...}, line},
    }
    """
    home = raw.get("home_team", "")
    away = raw.get("away_team", "")
    books = raw.get("bookmakers", [])

    def implied(american_odds: Optional[int]) -> float:
        if american_odds is None:
            return 0.5
        if american_odds < 0:
            return abs(american_odds) / (abs(american_odds) + 100)
        else:
            return 100 / (american_odds + 100)

    # Moneyline
    ml_home = _best_line(books, "h2h", home)
    ml_away = _best_line(books, "h2h", away)
    c_home  = _consensus_line(books, "h2h", home)
    c_away  = _consensus_line(books, "h2h", away)

    # Puck line (spreads) — filter by actual point value to avoid picking up
    # the wrong side (e.g. away team at -1.5 being mislabeled as away_plus1_5)
    pl_home_minus = _best_line(books, "spreads", home, point_sign=-1)  # home -1.5
    pl_away_plus  = _best_line(books, "spreads", away, point_sign=+1)  # away +1.5
    pl_away_minus = _best_line(books, "spreads", away, point_sign=-1)  # away -1.5
    pl_home_plus  = _best_line(books, "spreads", home, point_sign=+1)  # home +1.5

    # O/U
    ou_line = _get_ou_line(books)
    ou_over  = _best_line(books, "totals", "Over",  is_over=True)
    ou_under = _best_line(books, "totals", "Under", is_over=False)

    return {
        "game_id"       : raw.get("id"),
        "home_team"     : home,
        "away_team"     : away,
        "commence_time" : raw.get("commence_time"),
        "ml": {
            "home"           : {"odds": ml_home[0] if ml_home else None,
                                "book": ml_home[1] if ml_home else None,
                                "implied": implied(ml_home[0] if ml_home else None)},
            "away"           : {"odds": ml_away[0] if ml_away else None,
                                "book": ml_away[1] if ml_away else None,
                                "implied": implied(ml_away[0] if ml_away else None)},
            "consensus_home" : c_home,
            "consensus_away" : c_away,
        },
        "pl": {
            "home_minus1_5"  : {"odds": pl_home_minus[0] if pl_home_minus else None,
                                "book": pl_home_minus[1] if pl_home_minus else None,
                                "implied": implied(pl_home_minus[0] if pl_home_minus else None)},
            "away_plus1_5"   : {"odds": pl_away_plus[0]  if pl_away_plus  else None,
                                "book": pl_away_plus[1]  if pl_away_plus  else None,
                                "implied": implied(pl_away_plus[0]  if pl_away_plus  else None)},
            "away_minus1_5"  : {"odds": pl_away_minus[0] if pl_away_minus else None,
                                "book": pl_away_minus[1] if pl_away_minus else None,
                                "implied": implied(pl_away_minus[0] if pl_away_minus else None)},
            "home_plus1_5"   : {"odds": pl_home_plus[0]  if pl_home_plus  else None,
                                "book": pl_home_plus[1]  if pl_home_plus  else None,
                                "implied": implied(pl_home_plus[0]  if pl_home_plus  else None)},
        },
        "ou": {
            "line"  : ou_line,
            "over"  : {"odds": ou_over[0]  if ou_over  else None,
                       "book": ou_over[1]  if ou_over  else None,
                       "implied": implied(ou_over[0] if ou_over else None)},
            "under" : {"odds": ou_under[0] if ou_under else None,
                       "book": ou_under[1] if ou_under else None,
                       "implied": implied(ou_under[0] if ou_under else None)},
        },
    }


def get_all_game_odds() -> Dict[str, Dict]:
    """
    Return dict keyed by "{home_team} vs {away_team}" → structured odds.
    Falls back to empty dict if no API key or request fails.
    """
    raw_list = fetch_odds()
    result = {}
    for raw in raw_list:
        parsed = parse_game_odds(raw)
        key = f"{parsed['home_team']} vs {parsed['away_team']}"
        result[key] = parsed
    return result


def match_odds_to_game(game: Dict, all_odds: Dict) -> Optional[Dict]:
    """
    Match a game dict (from NHL API) to odds dict by team names.
    NHL API uses abbreviations; Odds API uses full names — fuzzy match.
    """
    home_abbrev = game.get("homeTeam", {}).get("abbrev", "")
    away_abbrev = game.get("awayTeam", {}).get("abbrev", "")

    for key, odds in all_odds.items():
        oh = odds["home_team"].lower()
        oa = odds["away_team"].lower()
        if (home_abbrev.lower() in oh or oh in home_abbrev.lower()) and \
           (away_abbrev.lower() in oa or oa in away_abbrev.lower()):
            return odds
    return None
