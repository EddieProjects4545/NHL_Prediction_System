"""
CSV, JSON, and Excel export of betting recommendations.
"""
import csv
import json
import os
from datetime import date
from typing import Dict, List, Optional

import numpy as np

from betting.recommender import Recommendation
from config import OUTPUT_DIR, TEAM_FULL_NAMES


def _rec_to_dict(rec: Recommendation, rank: int) -> dict:
    return {
        "rank"           : rank,
        "date"           : date.today().isoformat(),
        "game"           : f"{rec.away_team} @ {rec.home_team}",
        "game_time"      : rec.game_time,
        "market"         : rec.market,
        "side"           : rec.side,
        "bet_label"      : rec.bet_label,
        "odds"           : rec.odds,
        "book"           : rec.book,
        "model_prob"     : rec.model_prob,
        "market_prob"    : rec.market_prob,
        "edge_pct"       : rec.edge_pct,
        "ev_pct"         : rec.ev_pct,
        "kelly_pct"      : rec.kelly_pct,
        "confidence"     : rec.confidence,
        "logistic_prob"  : rec.logistic_prob,
        "xgboost_prob"   : rec.xgboost_prob,
        "elo_prob"       : rec.elo_prob,
        "model_std"      : rec.model_std,
        "home_goalie"    : rec.home_goalie,
        "away_goalie"    : rec.away_goalie,
        "home_sv_pct"    : rec.home_goalie_sv,
        "away_sv_pct"    : rec.away_goalie_sv,
        "ou_line"        : rec.ou_line or "",
        "exp_total"      : rec.exp_total or "",
        "pl_line"        : rec.pl_line or "",
        "key_edges"      : " | ".join(rec.key_edges),
        "conf_factors"   : ", ".join(rec.conf_factors),
    }


def export_csv(recs: List[Recommendation],
               filename: str = None) -> str:
    if not recs:
        return ""
    today = date.today().isoformat()
    filename = filename or os.path.join(OUTPUT_DIR,
                                        f"recommendations_{today}.csv")
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    rows = [_rec_to_dict(r, i + 1) for i, r in enumerate(recs)]
    fieldnames = list(rows[0].keys())

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return filename


def export_json(recs: List[Recommendation],
                filename: str = None) -> str:
    if not recs:
        return ""
    today = date.today().isoformat()
    filename = filename or os.path.join(OUTPUT_DIR,
                                        f"recommendations_{today}.json")
    rows = [_rec_to_dict(r, i + 1) for i, r in enumerate(recs)]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return filename


# ---------------------------------------------------------------------------
# Excel export
# ---------------------------------------------------------------------------

def export_excel(
    recs: List[Recommendation],
    upcoming: List[Dict],
    ensemble_probs: np.ndarray,
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    goalie_feats: Dict[str, Dict],
    game_date: str,
    filename: Optional[str] = None,
) -> str:
    """
    Build and save a color-coded Excel workbook for the given game date.
    Returns the path to the saved file, or "" on error.
    """
    from output.excel_writer import build_daily_workbook, save_workbook

    game_results = _build_game_results(
        recs, upcoming, ensemble_probs, mu_home, mu_away, goalie_feats
    )

    wb   = build_daily_workbook(game_date, game_results)
    path = save_workbook(wb, game_date)
    return path


def _build_game_results(
    recs: List[Recommendation],
    upcoming: List[Dict],
    ensemble_probs: np.ndarray,
    mu_home: np.ndarray,
    mu_away: np.ndarray,
    goalie_feats: Dict[str, Dict],
) -> List[Dict]:
    """
    Convert a flat list of Recommendation objects + raw model arrays into a
    per-game list of dicts that excel_writer.build_daily_workbook() expects.

    Each dict has keys:
        game_label, home_team, away_team,
        home_goalie, home_goalie_sv, away_goalie, away_goalie_sv,
        pred_home, pred_away, pred_total, home_win_prob,
        ml, pl, ou   — each is a rec sub-dict or None
    """
    # Index upcoming games by (home_abbrev, away_abbrev)
    game_index: Dict[str, int] = {}
    for i, g in enumerate(upcoming):
        h = (g.get("homeTeam") or {}).get("abbrev", "")
        a = (g.get("awayTeam") or {}).get("abbrev", "")
        if h and a:
            game_index[f"{h}_{a}"] = i

    # Group recommendations by game_key → best rec per market
    game_recs: Dict[str, Dict[str, Optional[Recommendation]]] = {}
    for rec in recs:
        gk = rec.game_key  # e.g. "CAR_vs_FLA"
        if gk not in game_recs:
            game_recs[gk] = {"ml": None, "pl": None, "ou": None,
                              "home": rec.home_team, "away": rec.away_team}
        mkt = rec.market.lower()  # "ml", "pl", "ou"
        if mkt in game_recs[gk]:
            existing = game_recs[gk][mkt]
            if existing is None or rec.rank_score > existing.rank_score:
                game_recs[gk][mkt] = rec

    # Build ordered game list: ALL upcoming games, even those with no recs
    seen_keys = set()
    ordered_games: List[Dict] = []

    # First add games that appear in upcoming (preserves schedule order)
    for i, g in enumerate(upcoming):
        h = (g.get("homeTeam") or {}).get("abbrev", "")
        a = (g.get("awayTeam") or {}).get("abbrev", "")
        if not h or not a:
            continue
        gk = f"{h}_vs_{a}"
        if gk in seen_keys:
            continue
        seen_keys.add(gk)

        rec_group = game_recs.get(gk, {})
        ml_rec = rec_group.get("ml")
        pl_rec = rec_group.get("pl")
        ou_rec = rec_group.get("ou")

        # Model outputs
        mu_h  = float(mu_home[i])  if i < len(mu_home)        else 0.0
        mu_a  = float(mu_away[i])  if i < len(mu_away)        else 0.0
        ens_p = float(ensemble_probs[i]) if i < len(ensemble_probs) else 0.5

        # Goalie info
        hg = goalie_feats.get(h, {})
        ag = goalie_feats.get(a, {})

        ordered_games.append({
            "game_label":     f"{a} @ {h}",
            "home_team":      h,
            "away_team":      a,
            "home_goalie":    hg.get("starter_name", "TBD"),
            "home_goalie_sv": hg.get("starter_save_pct", 0.0),
            "away_goalie":    ag.get("starter_name", "TBD"),
            "away_goalie_sv": ag.get("starter_save_pct", 0.0),
            "pred_home":      round(mu_h, 2),
            "pred_away":      round(mu_a, 2),
            "pred_total":     round(mu_h + mu_a, 2),
            "home_win_prob":  round(ens_p, 4),
            "ml": _rec_to_bet_dict(ml_rec, h, a, "ML"),
            "pl": _rec_to_bet_dict(pl_rec, h, a, "PL"),
            "ou": _rec_to_bet_dict(ou_rec, h, a, "OU"),
        })

    return ordered_games


def _rec_to_bet_dict(rec: Optional[Recommendation],
                     home: str, away: str,
                     market: str) -> Optional[Dict]:
    """Convert a Recommendation to the compact sub-dict used by excel_writer."""
    if rec is None:
        return None

    # Build pick string using ABBREVIATIONS (required for write-back team matching)
    if market == "ML":
        team_abbr = home if rec.side == "home" else away
        pick = f"{team_abbr} ML"
    elif market == "PL":
        team_abbr = home if rec.side == "home" else away
        line = "-1.5" if rec.side == "home" else "+1.5"
        pick = f"{team_abbr} {line}"
    else:  # OU
        side = "OVER" if rec.side == "over" else "UNDER"
        pick = f"{side} {rec.ou_line}"

    return {
        "pick":        pick,
        "confidence":  rec.confidence,
        "model_prob":  rec.model_prob,
        "market_prob": rec.market_prob,
        "edge_pct":    rec.edge_pct,
        "ev_pct":      rec.ev_pct,
        "kelly_pct":   rec.kelly_pct,
        "odds":        rec.odds,
    }
