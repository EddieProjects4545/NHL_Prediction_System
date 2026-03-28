"""
Confidence scorer.

Confidence is a 0–95 score reflecting how certain the model is about
a prediction — SEPARATE from edge/EV. A high-edge bet can have low
confidence if the data is thin or uncertain.

Components
──────────
Base                           : 50
+ Goalie confirmed             : +10
+ Both teams 60+ games         : +10
+ Model agreement (std < 0.04) : +10   (all three models agree)
+ Strong recent form (L10>60%) : +5
+ Home Elo advantage >75pts    : +5
- Goalie unconfirmed           : -10
- Back-to-back (either team)   : -15
- H2H sample < 2 games         : -10
- High model disagreement      : -10   (std > 0.08)
- Playoff game                 : -5    (higher variance)
- Low training sample (<500)   : -10

Cap: max 95, min 0
"""
from typing import Dict


def score_confidence(
    goalie_confirmed: bool,
    home_gp: int,
    away_gp: int,
    model_std: float,          # std across logistic/xgboost/elo predictions
    home_l10_win_pct: float,
    away_l10_win_pct: float,
    elo_diff: float,           # |home_elo - away_elo|
    h2h_sample: int,
    is_back_to_back: bool,     # either team
    is_playoff: bool,
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
    elif home_gp >= 40 and away_gp >= 40:
        score += 5

    if model_std < 0.04:
        score += 10    # Strong model agreement
    elif model_std < 0.06:
        score += 5

    if home_l10_win_pct > 0.60 or away_l10_win_pct < 0.40:
        score += 5    # Home team in strong form / away in poor form

    if abs(elo_diff) > 75:
        score += 5    # Clear Elo favourite

    # ── Negative factors ──────────────────────────────────────────────────────
    if not goalie_confirmed:
        score -= 10

    if is_back_to_back or home_b2b or away_b2b:
        score -= 15

    if h2h_sample < 2:
        score -= 10
    elif h2h_sample < 4:
        score -= 5

    if model_std > 0.08:
        score -= 10   # Models significantly disagree
    elif model_std > 0.06:
        score -= 5

    if is_playoff:
        score -= 5    # Higher variance in playoffs

    if n_training_samples < 500:
        score -= 10

    return max(0, min(95, score))


def score_game_confidence(
    game_features: Dict,
    model_components: Dict,
    goalie_feats_home: Dict,
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
    # Confirmed if either scraper confirmed it OR if we have a name+stats (presumed starter)
    gc = bool(goalie_feats_home.get("starter_confirmed", False) or
              goalie_feats_home.get("starter_save_pct", 0) > 0)

    h_gp  = int(game_features.get("h_gp", 0))
    a_gp  = int(game_features.get("a_gp", 0))
    std   = float(model_components.get("std", 0.05))
    h_l10 = float(game_features.get("h_l10_win_pct", 0.5))
    a_l10 = float(game_features.get("a_l10_win_pct", 0.5))
    elo_d = abs(float(game_features.get("elo_diff", 0)))
    h2h   = int(game_features.get("h2h_sample", 0))
    b2b   = bool(game_features.get("either_b2b", 0))
    h_b2b = bool(game_features.get("h_is_back_to_back", 0))
    a_b2b = bool(game_features.get("a_is_back_to_back", 0))
    playoff = bool(game_features.get("is_playoff", 0))

    base_conf = score_confidence(
        goalie_confirmed    = gc,
        home_gp             = h_gp,
        away_gp             = a_gp,
        model_std           = std,
        home_l10_win_pct    = h_l10,
        away_l10_win_pct    = a_l10,
        elo_diff            = elo_d,
        h2h_sample          = h2h,
        is_back_to_back     = b2b,
        is_playoff          = playoff,
        n_training_samples  = n_training_samples,
        home_b2b            = h_b2b,
        away_b2b            = a_b2b,
    )

    # Puck line gets extra penalty if team is a one-goal-win specialist
    h_ogr = float(game_features.get("h_one_goal_game_rate", 0.35))
    a_ogr = float(game_features.get("a_one_goal_game_rate", 0.35))
    pl_penalty = 0
    if h_ogr > 0.45:   # Home tends to play tight — bad for -1.5
        pl_penalty -= 5
    if a_ogr > 0.45:   # Away tends to play tight — bad for +1.5 fading
        pl_penalty -= 3

    # O/U gets extra penalty if both goalies are unconfirmed
    ou_penalty = 0
    if not gc:
        ou_penalty -= 5   # Goalie uncertainty affects totals prediction too

    # Build factor list
    factors = []
    if gc:
        factors.append("Goalie confirmed")
    else:
        factors.append("GOALIE UNCONFIRMED")
    if b2b or h_b2b or a_b2b:
        factors.append("Back-to-back game")
    if h2h < 2:
        factors.append("Limited H2H history")
    if std < 0.04:
        factors.append("Strong model agreement")
    elif std > 0.08:
        factors.append("High model disagreement")
    if elo_d > 75:
        factors.append(f"Clear Elo favourite (Δ{elo_d:.0f})")
    if playoff:
        factors.append("Playoff game")

    return {
        "ml"      : base_conf,
        "pl_home" : max(0, min(95, base_conf + pl_penalty)),
        "pl_away" : max(0, min(95, base_conf + pl_penalty)),
        "ou"      : max(0, min(95, base_conf + ou_penalty)),
        "factors" : factors,
    }
