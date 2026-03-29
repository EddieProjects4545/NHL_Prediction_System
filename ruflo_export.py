"""
ruflo_export.py — NHL System
Reads existing NHL outputs and prints a structured JSON summary to stdout.
Run AFTER run.py.

Usage:
    python ruflo_export.py                # today's date
    python ruflo_export.py --date 2026-04-01
"""
import argparse
import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "outputs"

def _load_recommendations_json(date_str: str) -> list:
    path = OUTPUT_DIR / f"recommendations_{date_str}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []

def _load_recommendations_csv(date_str: str) -> list:
    path = OUTPUT_DIR / f"recommendations_{date_str}.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Coerce numeric fields
            for k in ("model_prob", "market_prob", "edge_pct", "ev_pct", "confidence",
                      "units", "logistic_prob", "xgboost_prob", "elo_prob"):
                if k in row and row[k] not in ("", None):
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        pass
            rows.append(row)
    return rows

def _load_backtest_metrics() -> dict:
    path = ROOT / "backtest_results.xlsx"
    if not path.exists():
        return {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        metrics = {}
        for row in ws.iter_rows(values_only=True):
            if row[0] is None:
                continue
            label = str(row[0]).strip().lower()
            val = row[1] if len(row) > 1 else None
            if "accuracy" in label and val is not None:
                metrics["accuracy"] = round(float(val), 4) if isinstance(val, float) else val
            elif "brier" in label and val is not None:
                metrics["brier_score"] = round(float(val), 4) if isinstance(val, float) else val
            elif "auc" in label and val is not None:
                metrics["roc_auc"] = round(float(val), 4) if isinstance(val, float) else val
            elif "log loss" in label and val is not None:
                metrics["log_loss"] = round(float(val), 4) if isinstance(val, float) else val
            elif "units" in label and val is not None:
                metrics["units_profit"] = round(float(val), 2) if isinstance(val, float) else val
        return metrics
    except Exception as e:
        return {"error": str(e)}

def _summarize_recs(recs: list) -> dict:
    if not recs:
        return {"count": 0}
    markets = {}
    total_edge = 0.0
    games = set()
    for r in recs:
        market = r.get("market", "unknown")
        markets[market] = markets.get(market, 0) + 1
        game = r.get("game", "")
        if game:
            games.add(game)
        try:
            total_edge += float(r.get("edge_pct", 0) or 0)
        except (ValueError, TypeError):
            pass
    return {
        "count": len(recs),
        "games_with_picks": len(games),
        "by_market": markets,
        "avg_edge_pct": round(total_edge / len(recs), 2) if recs else 0,
        "top_picks": recs[:3],  # top 3 by rank
    }

def main():
    parser = argparse.ArgumentParser(description="Export NHL prediction summary for ruflo")
    parser.add_argument("--date", default=date.today().isoformat(), help="Date (YYYY-MM-DD)")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args()

    # Try JSON first, then CSV
    recs = _load_recommendations_json(args.date)
    source = "json"
    if not recs:
        recs = _load_recommendations_csv(args.date)
        source = "csv"
    if not recs:
        source = "none"

    backtest = _load_backtest_metrics()
    rec_summary = _summarize_recs(recs)

    summary = {
        "sport": "NHL",
        "export_date": args.date,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "recommendations_source": source,
        "recommendations_summary": rec_summary,
        "model_metrics": backtest,
        "paths": {
            "outputs_dir": str(OUTPUT_DIR),
            "backtest_results": str(ROOT / "backtest_results.xlsx"),
        }
    }

    indent = 2 if args.pretty else None
    print(json.dumps(summary, indent=indent))

if __name__ == "__main__":
    main()
