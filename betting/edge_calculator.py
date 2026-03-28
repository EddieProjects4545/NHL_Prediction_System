"""
Betting mathematics: implied probability, vig removal, edge, EV, Kelly.

All functions operate on American odds format (integers).
"""
from typing import Optional, Tuple


# ─── Odds conversions ─────────────────────────────────────────────────────────

def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal (European) format."""
    if odds > 0:
        return odds / 100 + 1.0
    else:
        return 100 / abs(odds) + 1.0


def american_to_implied(odds: int) -> float:
    """Raw implied probability from American odds (includes vig)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American format."""
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    else:
        return round(-100 / (decimal_odds - 1))


# ─── Vig removal (two-way markets) ───────────────────────────────────────────

def remove_vig_two_way(odds_a: int, odds_b: int) -> Tuple[float, float]:
    """
    Remove bookmaker margin from a two-way market.
    Returns vig-adjusted true probabilities (sum to 1.0).
    """
    imp_a = american_to_implied(odds_a)
    imp_b = american_to_implied(odds_b)
    total = imp_a + imp_b
    return imp_a / total, imp_b / total


def remove_vig_single(odds: int, other_odds: int) -> float:
    """Return vig-adjusted implied prob for one side of a two-way market."""
    true_a, _ = remove_vig_two_way(odds, other_odds)
    return true_a


def overround(odds_a: int, odds_b: int) -> float:
    """Return the bookmaker's overround (margin) as a %."""
    imp_a = american_to_implied(odds_a)
    imp_b = american_to_implied(odds_b)
    return round((imp_a + imp_b - 1.0) * 100, 2)


# ─── Edge & Expected Value ────────────────────────────────────────────────────

def calculate_edge(model_prob: float, market_prob: float) -> float:
    """
    Edge = model probability - vig-adjusted market implied probability.
    Positive → model believes this side is underpriced.
    """
    return round(model_prob - market_prob, 4)


def calculate_ev(edge: float, odds: int, stake: float = 100.0) -> float:
    """
    Expected value per unit stake.
    EV = (model_prob × payout) - (model_fail_prob × stake)
    """
    decimal = american_to_decimal(odds)
    payout  = (decimal - 1) * stake   # Net profit on a win
    model_prob  = min(max(edge + american_to_implied(odds), 0.001), 0.999)
    ev = (model_prob * payout) - ((1 - model_prob) * stake)
    return round(ev, 2)


def calculate_ev_pct(model_prob: float, odds: int) -> float:
    """EV as a percentage of stake risked. Cleaner for ranking."""
    decimal  = american_to_decimal(odds)
    ev_pct   = model_prob * decimal - 1.0
    return round(ev_pct * 100, 2)


# ─── Kelly Criterion ──────────────────────────────────────────────────────────

def kelly_fraction(model_prob: float, odds: int,
                   fraction: float = 0.25) -> float:
    """
    Fractional Kelly bet size as % of bankroll.
    fraction=0.25 → Quarter-Kelly (recommended for noisy sports models).

    Kelly formula: f* = (b·p - q) / b
    where b = net odds (decimal - 1), p = model probability, q = 1 - p
    """
    b = american_to_decimal(odds) - 1.0
    p = model_prob
    q = 1.0 - p
    if b <= 0 or p <= 0:
        return 0.0
    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0
    return round(full_kelly * fraction * 100, 2)   # as % of bankroll


# ─── Full game odds analysis ──────────────────────────────────────────────────

def analyse_ml(model_prob_home: float, ml_home_odds: int,
               ml_away_odds: int, kelly_frac: float = 0.25) -> dict:
    """
    Full moneyline analysis for one game.
    Returns analysis for both sides.
    """
    if ml_home_odds is None or ml_away_odds is None:
        return {}

    market_home, market_away = remove_vig_two_way(ml_home_odds, ml_away_odds)
    model_prob_away = 1.0 - model_prob_home

    edge_home = calculate_edge(model_prob_home, market_home)
    edge_away = calculate_edge(model_prob_away, market_away)

    return {
        "home": {
            "model_prob"   : round(model_prob_home, 4),
            "market_prob"  : round(market_home, 4),
            "edge"         : round(edge_home * 100, 2),   # as %
            "ev_pct"       : calculate_ev_pct(model_prob_home, ml_home_odds),
            "kelly_pct"    : kelly_fraction(model_prob_home, ml_home_odds, kelly_frac),
            "odds"         : ml_home_odds,
        },
        "away": {
            "model_prob"   : round(model_prob_away, 4),
            "market_prob"  : round(market_away, 4),
            "edge"         : round(edge_away * 100, 2),
            "ev_pct"       : calculate_ev_pct(model_prob_away, ml_away_odds),
            "kelly_pct"    : kelly_fraction(model_prob_away, ml_away_odds, kelly_frac),
            "odds"         : ml_away_odds,
        },
        "overround_pct": overround(ml_home_odds, ml_away_odds),
    }


def analyse_puckline(prob_home_minus1_5: float,
                     prob_away_plus1_5: float,
                     pl_home_odds: int,
                     pl_away_odds: int,
                     kelly_frac: float = 0.25) -> dict:
    """Puck line edge analysis."""
    if pl_home_odds is None or pl_away_odds is None:
        return {}

    # For spreads the two sides are: home -1.5 and away +1.5
    # They don't always sum to 1 in model space (independent classifiers)
    market_h, market_a = remove_vig_two_way(pl_home_odds, pl_away_odds)

    return {
        "home_minus1_5": {
            "model_prob"   : round(prob_home_minus1_5, 4),
            "market_prob"  : round(market_h, 4),
            "edge"         : round(calculate_edge(prob_home_minus1_5, market_h) * 100, 2),
            "ev_pct"       : calculate_ev_pct(prob_home_minus1_5, pl_home_odds),
            "kelly_pct"    : kelly_fraction(prob_home_minus1_5, pl_home_odds, kelly_frac),
            "odds"         : pl_home_odds,
        },
        "away_plus1_5": {
            "model_prob"   : round(prob_away_plus1_5, 4),
            "market_prob"  : round(market_a, 4),
            "edge"         : round(calculate_edge(prob_away_plus1_5, market_a) * 100, 2),
            "ev_pct"       : calculate_ev_pct(prob_away_plus1_5, pl_away_odds),
            "kelly_pct"    : kelly_fraction(prob_away_plus1_5, pl_away_odds, kelly_frac),
            "odds"         : pl_away_odds,
        },
    }


def analyse_totals(prob_over: float, prob_under: float,
                   ou_line: float,
                   over_odds: int, under_odds: int,
                   kelly_frac: float = 0.25) -> dict:
    """Over/Under edge analysis."""
    if over_odds is None or under_odds is None:
        return {}

    market_over, market_under = remove_vig_two_way(over_odds, under_odds)

    return {
        "over": {
            "model_prob"   : round(prob_over, 4),
            "market_prob"  : round(market_over, 4),
            "edge"         : round(calculate_edge(prob_over, market_over) * 100, 2),
            "ev_pct"       : calculate_ev_pct(prob_over, over_odds),
            "kelly_pct"    : kelly_fraction(prob_over, over_odds, kelly_frac),
            "odds"         : over_odds,
        },
        "under": {
            "model_prob"   : round(prob_under, 4),
            "market_prob"  : round(market_under, 4),
            "edge"         : round(calculate_edge(prob_under, market_under) * 100, 2),
            "ev_pct"       : calculate_ev_pct(prob_under, under_odds),
            "kelly_pct"    : kelly_fraction(prob_under, under_odds, kelly_frac),
            "odds"         : under_odds,
        },
        "line"  : ou_line,
    }
