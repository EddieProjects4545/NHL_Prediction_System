"""
Recommendation engine.

Takes model predictions + odds data + confidence scores and produces
a ranked list of Recommendation objects filtered by minimum edge/confidence.

Each Recommendation covers exactly ONE bet (one side, one market).
Multiple recommendations can exist per game (e.g., ML home + Over).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from betting.edge_calculator import (
    analyse_ml, analyse_puckline, analyse_totals, american_to_decimal,
)
from betting.confidence_scorer import score_game_confidence
from config import (
    MIN_EDGE_PCT, MIN_CONFIDENCE, MIN_ODDS, MAX_ODDS, KELLY_FRACTION,
    TEAM_FULL_NAMES,
)


@dataclass
class Recommendation:
    # Identity
    game_key    : str         # "CAR_vs_FLA"
    home_team   : str
    away_team   : str
    game_time   : str
    market      : str         # "ML", "PL", "OU"
    side        : str         # "home", "away", "over", "under"

    # Core metrics
    model_prob  : float
    market_prob : float
    edge_pct    : float       # positive = value
    ev_pct      : float
    units       : float       # unit stake (flat 1u)
    odds        : int
    book        : str

    # Puck line specific
    pl_line     : Optional[float] = None   # -1.5 or +1.5

    # Totals specific
    ou_line     : Optional[float] = None
    exp_total   : Optional[float] = None   # model expected total goals

    # Confidence
    confidence  : int = 50
    conf_factors: List[str] = field(default_factory=list)

    # Model breakdown (for display)
    logistic_prob : float = 0.0
    xgboost_prob  : float = 0.0
    elo_prob      : float = 0.0
    model_std     : float = 0.0

    # Goalie info
    home_goalie  : str = ""
    away_goalie  : str = ""
    home_goalie_sv: float = 0.0
    away_goalie_sv: float = 0.0

    # Key edge reason (generated)
    key_edges    : List[str] = field(default_factory=list)

    @property
    def bet_label(self) -> str:
        home_full = TEAM_FULL_NAMES.get(self.home_team, self.home_team)
        away_full = TEAM_FULL_NAMES.get(self.away_team, self.away_team)
        if self.market == "ML":
            team = home_full if self.side == "home" else away_full
            return f"{team} ML"
        elif self.market == "PL":
            if self.side == "home":
                return f"{home_full} -1.5 (Puck Line)"
            else:
                return f"{away_full} +1.5 (Puck Line)"
        elif self.market == "OU":
            return f"{'OVER' if self.side == 'over' else 'UNDER'} {self.ou_line} ({home_full} vs {away_full})"
        return ""

    @property
    def rank_score(self) -> float:
        """Primary sort key: combination of EV and confidence."""
        return self.ev_pct * (self.confidence / 95)


def _generate_key_edges(game_feats: Dict, home: str, away: str,
                        market: str, side: str) -> List[str]:
    """Generate human-readable edge reasons for a recommendation."""
    reasons = []
    hf = TEAM_FULL_NAMES.get(home, home)
    af = TEAM_FULL_NAMES.get(away, away)

    corsi_d = game_feats.get("delta_corsi_pct_5v5", 0)
    pp_d    = game_feats.get("delta_net_pp_advantage", 0)
    gf_d    = game_feats.get("delta_gf_pg", 0)
    pdo_h   = game_feats.get("h_pdo_5v5", 100)
    pdo_a   = game_feats.get("a_pdo_5v5", 100)
    l10_d   = game_feats.get("delta_l10_win_pct", 0)
    elo_d   = game_feats.get("elo_diff", 0)
    sv_d    = game_feats.get("delta_starter_save_pct", 0)
    b2b_h   = game_feats.get("h_is_back_to_back", 0)
    b2b_a   = game_feats.get("a_is_back_to_back", 0)
    cover_h = game_feats.get("h_cover_rate_minus1_5", 0)
    cover_a = game_feats.get("a_cover_rate_minus1_5", 0)
    ogr_h   = game_feats.get("h_one_goal_game_rate", 0)
    comb_sv = game_feats.get("combined_goalie_sv", 0.9)
    h2h_ov  = game_feats.get("h2h_over_rate", 0.5)

    team = hf if side in ("home", "over") else af

    if market == "ML":
        if abs(corsi_d) > 3:
            leader = hf if corsi_d > 0 else af
            reasons.append(f"{leader} Corsi advantage ({corsi_d:+.1f}%)")
        if abs(pp_d) > 3:
            reasons.append(f"Special teams edge ({pp_d:+.1f}% net)")
        if abs(gf_d) > 0.3:
            reasons.append(f"GF/game gap ({gf_d:+.2f})")
        if pdo_h > 102:
            reasons.append(f"{hf} PDO regression risk ({pdo_h:.3f} — above 102)")
        if pdo_a < 98 and pdo_a > 0:
            reasons.append(f"{af} PDO regression candidate ({pdo_a:.3f} — below 98)")
        if abs(l10_d) > 0.2:
            leader = hf if l10_d > 0 else af
            reasons.append(f"{leader} stronger L10 form ({l10_d:+.0%})")
        if abs(elo_d) > 50:
            reasons.append(f"Elo strength gap ({elo_d:+.0f} pts)")
        if abs(sv_d) > 0.008:
            better = hf if sv_d > 0 else af
            reasons.append(f"{better} goalie advantage (SV% Δ{sv_d:+.3f})")
        if b2b_h:
            reasons.append(f"{hf} on back-to-back (fatigue risk)")
        if b2b_a:
            reasons.append(f"{af} on back-to-back (fatigue risk)")

    elif market == "PL":
        if side == "home":
            if cover_h > 0.50:
                reasons.append(f"{hf} covers -1.5 {cover_h:.0%} of wins")
            if ogr_h < 0.30:
                reasons.append(f"{hf} rarely plays 1-goal games ({ogr_h:.0%})")
            if corsi_d > 5:
                reasons.append(f"Dominant possession → score inflation")
            if pp_d > 5:
                reasons.append(f"Large PP advantage → margin builder")
        else:
            if ogr_h > 0.45:
                reasons.append(f"{hf} specialises in 1-goal wins (bad -1.5)")
            if cover_h < 0.35:
                reasons.append(f"{hf} rarely covers -1.5 ({cover_h:.0%} of wins)")
            if game_feats.get("h_ot_game_rate", 0) > 0.35:
                reasons.append(f"{hf} high OT-game rate → +1.5 value")

    elif market == "OU":
        if side == "over":
            if comb_sv < 0.895:
                reasons.append(f"Both goalies below avg (combined SV {comb_sv:.3f})")
            if h2h_ov > 0.60:
                reasons.append(f"H2H goes over {h2h_ov:.0%} of the time")
            if game_feats.get("combined_pp_pct", 20) > 24:
                reasons.append(f"High combined PP% → more goals")
        else:
            if comb_sv > 0.915:
                reasons.append(f"Elite goaltending matchup (combined SV {comb_sv:.3f})")
            if h2h_ov < 0.40:
                reasons.append(f"H2H goes under {1-h2h_ov:.0%} of the time")
            if game_feats.get("is_playoff", 0):
                reasons.append("Playoff games trend lower-scoring")

    return reasons[:3]   # Cap at 3 reasons


def generate_recommendations(
    upcoming_games: List[Dict],
    X_pred: pd.DataFrame,
    ensemble_probs: np.ndarray,
    ensemble_components: Dict[str, np.ndarray],
    pl_prob_home: np.ndarray,
    pl_prob_away: np.ndarray,
    poisson_mu_home: np.ndarray,
    poisson_mu_away: np.ndarray,
    all_odds: Dict,
    goalie_feats: Dict[str, Dict],
    confidence_scores: List[Dict],
    n_training_samples: int,
) -> List[Recommendation]:
    """
    Generate all Recommendation objects for upcoming games.
    Filters by min edge and min confidence.
    """
    recs: List[Recommendation] = []

    for i, g in enumerate(upcoming_games):
        home = g.get("homeTeam", {}).get("abbrev") or g.get("home_team", "")
        away = g.get("awayTeam", {}).get("abbrev") or g.get("away_team", "")
        if not home or not away:
            continue

        game_key  = f"{home}_vs_{away}"
        game_time = g.get("startTimeUTC", g.get("gameDate", "TBD"))

        # Odds
        odds_key = f"{TEAM_FULL_NAMES.get(home, home)} vs {TEAM_FULL_NAMES.get(away, away)}"
        game_odds = all_odds.get(odds_key, {})

        # Model outputs
        p_ml_home = float(ensemble_probs[i])
        comp      = {k: float(v[i]) for k, v in ensemble_components.items()}
        p_pl_h    = float(pl_prob_home[i])
        p_pl_a    = float(pl_prob_away[i])
        mu_h      = float(poisson_mu_home[i])
        mu_a      = float(poisson_mu_away[i])
        exp_total = round(mu_h + mu_a, 2)

        conf = confidence_scores[i] if i < len(confidence_scores) else {}
        game_feats = X_pred.iloc[i].to_dict() if i < len(X_pred) else {}
        ou_line = game_feats.get("ou_line", 5.5)

        # Goalie info
        hg = goalie_feats.get(home, {})
        ag = goalie_feats.get(away, {})

        def _add_rec(market, side, model_prob, market_prob, edge_pct,
                     ev_pct, odds, book,
                     pl_line=None, ou_line_v=None):
            if odds is None:
                return
            if not (MIN_ODDS <= odds <= MAX_ODDS):
                return
            if edge_pct < MIN_EDGE_PCT:
                return
            conf_val = conf.get(
                "ml" if market == "ML" else
                ("pl_home" if side == "home" else
                 ("pl_away" if side == "away" else "ou")), 50
            )
            if conf_val < MIN_CONFIDENCE:
                return

            recs.append(Recommendation(
                game_key     = game_key,
                home_team    = home,
                away_team    = away,
                game_time    = game_time,
                market       = market,
                side         = side,
                model_prob   = round(model_prob, 4),
                market_prob  = round(market_prob, 4),
                edge_pct     = round(edge_pct, 2),
                ev_pct       = round(ev_pct, 2),
                units        = 1.0,
                odds         = odds,
                book         = book or "best",
                pl_line      = pl_line,
                ou_line      = ou_line_v,
                exp_total    = exp_total if market == "OU" else None,
                confidence   = conf_val,
                conf_factors = conf.get("factors", []),
                logistic_prob= round(comp.get("logistic", 0), 4),
                xgboost_prob = round(comp.get("xgboost", 0), 4),
                elo_prob     = round(comp.get("elo", 0), 4),
                model_std    = round(comp.get("std", 0), 4),
                home_goalie  = hg.get("starter_name", ""),
                away_goalie  = ag.get("starter_name", ""),
                home_goalie_sv= hg.get("starter_save_pct", 0),
                away_goalie_sv= ag.get("starter_save_pct", 0),
                key_edges    = _generate_key_edges(game_feats, home, away, market, side),
            ))

        # ── Moneyline ────────────────────────────────────────────────────────
        ml_data = game_odds.get("ml", {})
        ml_h_odds = (ml_data.get("home") or {}).get("odds")
        ml_a_odds = (ml_data.get("away") or {}).get("odds")

        if ml_h_odds and ml_a_odds:
            ml_analysis = analyse_ml(p_ml_home, ml_h_odds, ml_a_odds, KELLY_FRACTION)
            h = ml_analysis.get("home", {})
            a = ml_analysis.get("away", {})
            _add_rec("ML", "home", h["model_prob"], h["market_prob"],
                     h["edge"], h["ev_pct"], h["odds"],
                     (ml_data.get("home") or {}).get("book"))
            _add_rec("ML", "away", a["model_prob"], a["market_prob"],
                     a["edge"], a["ev_pct"], a["odds"],
                     (ml_data.get("away") or {}).get("book"))

        # ── Puck Line ─────────────────────────────────────────────────────────
        pl_data = game_odds.get("pl", {})
        pl_h_odds = (pl_data.get("home_minus1_5") or {}).get("odds")
        pl_a_odds = (pl_data.get("away_plus1_5")  or {}).get("odds")

        if pl_h_odds and pl_a_odds:
            pl_analysis = analyse_puckline(p_pl_h, p_pl_a, pl_h_odds, pl_a_odds, KELLY_FRACTION)
            ph = pl_analysis.get("home_minus1_5", {})
            pa = pl_analysis.get("away_plus1_5",  {})
            _add_rec("PL", "home", ph["model_prob"], ph["market_prob"],
                     ph["edge"], ph["ev_pct"], ph["odds"],
                     (pl_data.get("home_minus1_5") or {}).get("book"), pl_line=-1.5)
            _add_rec("PL", "away", pa["model_prob"], pa["market_prob"],
                     pa["edge"], pa["ev_pct"], pa["odds"],
                     (pl_data.get("away_plus1_5") or {}).get("book"),  pl_line=+1.5)

        # ── Over / Under ──────────────────────────────────────────────────────
        ou_data  = game_odds.get("ou", {})
        ou_v     = ou_data.get("line", ou_line)
        ov_odds  = (ou_data.get("over")  or {}).get("odds")
        un_odds  = (ou_data.get("under") or {}).get("odds")

        if ov_odds and un_odds and ou_v:
            from models.poisson_model import PoissonModel as _PM
            # Use pre-computed Poisson probs (passed from run.py via caller)
            # Here we recalculate for the specific line from the odds
            from scipy import stats as _stats
            MAX_G = 15
            p_total = [0.0] * (MAX_G * 2 + 1)
            for hh in range(MAX_G + 1):
                ph = _stats.poisson.pmf(hh, mu_h)
                for aa in range(MAX_G + 1):
                    pa = _stats.poisson.pmf(aa, mu_a)
                    p_total[hh + aa] += ph * pa
            p_ov = sum(p for k, p in enumerate(p_total) if k > ou_v)
            p_un = sum(p for k, p in enumerate(p_total) if k < ou_v)

            ou_analysis = analyse_totals(p_ov, p_un, ou_v, ov_odds, un_odds, KELLY_FRACTION)
            ov = ou_analysis.get("over",  {})
            un = ou_analysis.get("under", {})
            _add_rec("OU", "over",  ov["model_prob"], ov["market_prob"],
                     ov["edge"], ov["ev_pct"], ov["odds"],
                     (ou_data.get("over")  or {}).get("book"), ou_line_v=ou_v)
            _add_rec("OU", "under", un["model_prob"], un["market_prob"],
                     un["edge"], un["ev_pct"], un["odds"],
                     (ou_data.get("under") or {}).get("book"), ou_line_v=ou_v)

    # Sort by rank_score descending
    recs.sort(key=lambda r: r.rank_score, reverse=True)
    return recs
