"""
CSV and JSON export of betting recommendations.
"""
import csv
import json
import os
from datetime import date
from typing import List

from betting.recommender import Recommendation
from config import OUTPUT_DIR


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
