"""
Confidence scorer.

Confidence is a 0–95 score reflecting how certain the model is about
a prediction — SEPARATE from edge/EV. A high-edge bet can have low
confidence if the data is thin or uncertain.

Components
──────────
Base                           : 50
+ Goalie confirmed (both)      : +10
+ Both teams 60+ games         : +10
+ Model agreement (std < 0.04) : +10
- Goalie unconfirmed (either)  : -10
- Back-to-back (either team)   : -15
- High model disagreement      : -10   (std > 0.08)
- Low training sample (<500)   : -10

Cap: max 95, min 0
"""
from typing import Dict


def score_confidence(
    goalie_confirmed: bool,
    home_gp: int,
    away_gp: int,
    model_std: float,          # std across logistic/xgboost/elo predictions
    is_back_to_back: bool,     # either team
    n_training_samples: int,
    home_b2b: bool = False,
    away_b2b: bool = False,
) -> int:
    """Return integer confidence score 0–95."""
    score = 50

    # ── Positive factors ──────────────────────────────────────────────────────
    if goalie_confirmed:
        score += 10

    if home_gp >= 60 and away_gp >= 60:
        score += 10

    if model_std < 0.04:
        score += 10    # Strong model agreement

    # ── Negative factors ──────────────────────────────────────────────────────
    if not goalie_confirmed:
        score -= 10

    if is_back_to_back or home_b2b or away_b2b:
        score -= 15

    if model_std > 0.08:
        score -= 10   # Models significantly disagree

    if n_training_samples < 500:
        score -= 10

    return max(0, min(95, score))


def score_game_confidence(
    game_features: Dict,
    model_components: Dict,
    goalie_feats_home: Dict,
    goalie_feats_away: Dict,
    n_training_samples: int,
) -> Dict:
    """
    Compute confidence for all three markets of one game.

    Returns:
        {
            "ml"      : int,
            "pl_home" : int,
            "pl_away" : int,
            "ou"      : int,
            "factors" : list[str],   # Human-readable reasons
        }
    """
    # Treat goalie certainty as a two-team input, not just the home side.
    gc_home = bool(goalie_feats_home.get("starter_confirmed", False) or
                   goalie_feats_home.get("starter_save_pct", 0) > 0)
    gc_away = bool(goalie_feats_away.get("starter_confirmed", False) or
                   goalie_feats_away.get("starter_save_pct", 0) > 0)
    gc = gc_home and gc_away

    h_gp  = int(game_features.get("h_gp", 0))
    a_gp  = int(game_features.get("a_gp", 0))
    std   = float(model_components.get("std", 0.05))
    b2b   = bool(game_features.get("either_b2b", 0))
    h_b2b = bool(game_features.get("h_is_back_to_back", 0))
    a_b2b = bool(game_features.get("a_is_back_to_back", 0))

    base_conf = score_confidence(
        goalie_confirmed    = gc,
        home_gp             = h_gp,
        away_gp             = a_gp,
        model_std           = std,
        is_back_to_back     = b2b,
        n_training_samples  = n_training_samples,
        home_b2b            = h_b2b,
        away_b2b            = a_b2b,
    )

    # Build factor list
    factors = []
    if gc:
        factors.append("Both goalies confirmed")
    else:
        if gc_home or gc_away:
            factors.append("One goalie unconfirmed")
        else:
            factors.append("GOALIES UNCONFIRMED")
    if b2b or h_b2b or a_b2b:
        factors.append("Back-to-back game")
    if std < 0.04:
        factors.append("Strong model agreement")
    elif std > 0.08:
        factors.append("High model disagreement")

    return {
        "ml"      : base_conf,
        "pl_home" : base_conf,
        "pl_away" : base_conf,
        "ou"      : base_conf,
        "factors" : factors,
    }
