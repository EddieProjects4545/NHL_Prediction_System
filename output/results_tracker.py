"""
Results Tracker.

Loads a prior day's recommendations CSV, fetches final NHL scores,
and determines WIN / LOSS / PUSH for each recommendation.

Appends outcomes to results_log.csv and prints a P&L summary.

Usage (via run.py):
    python run.py --track                    # track yesterday
    python run.py --track --date 2026-03-27  # track specific date
"""
import csv
import os
from datetime import date, timedelta
from typing import Dict, List, Optional

from colorama import Fore, Style

from config import OUTPUT_DIR


# ─── Outcome Logic ────────────────────────────────────────────────────────────

def determine_outcome(rec: Dict, score: Dict) -> str:
    """
    Return 'WIN', 'LOSS', or 'PUSH' for one recommendation given final score.
    rec  : row dict from recommendations CSV
    score: {home_team, away_team, home_goals, away_goals}
    """
    h = int(score.get("home_goals", 0))
    a = int(score.get("away_goals", 0))
    margin = h - a   # positive = home won

    market = rec.get("market", "")
    side   = rec.get("side", "")

    if market == "ML":
        if side == "home":
            return "WIN" if margin > 0 else "LOSS"
        else:
            return "WIN" if margin < 0 else "LOSS"

    elif market == "PL":
        pl_line = rec.get("pl_line", "")
        try:
            line = float(pl_line) if pl_line else None
        except ValueError:
            line = None

        if line is None:
            # Infer from side
            line = -1.5 if side == "home" else 1.5

        if line < 0:          # favourite side (-1.5): must win by 2+
            if side == "home":
                return "WIN" if margin >= 2 else "LOSS"
            else:
                return "WIN" if margin <= -2 else "LOSS"
        else:                 # underdog side (+1.5): must not lose by 2+
            if side == "home":
                return "WIN" if margin >= -1 else "LOSS"
            else:
                return "WIN" if margin <= 1 else "LOSS"

    elif market == "OU":
        total = h + a
        ou_line = rec.get("ou_line", "")
        try:
            line = float(ou_line) if ou_line else 5.5
        except ValueError:
            line = 5.5

        if abs(total - line) < 0.01:
            return "PUSH"
        if side == "over":
            return "WIN" if total > line else "LOSS"
        else:
            return "WIN" if total < line else "LOSS"

    return "UNKNOWN"


def calc_pnl(outcome: str, odds: int, unit: float = 100.0) -> float:
    """Return P&L in dollars for a $unit bet at American odds."""
    if outcome == "WIN":
        if odds > 0:
            return round(unit * odds / 100, 2)
        else:
            return round(unit * 100 / abs(odds), 2)
    elif outcome == "LOSS":
        return -unit
    return 0.0  # PUSH


# ─── Score Lookup ─────────────────────────────────────────────────────────────

def get_scores_for_date(game_date: str) -> Dict[str, Dict]:
    """
    Return {'{away}@{home}': score_dict} for all completed games on game_date.
    Uses the NHL web API score endpoint (already in nhl_api.py).
    """
    from data.nhl_api import get_game_results_range
    results = get_game_results_range(game_date, game_date)
    index = {}
    for r in results:
        key = f"{r['away_team']}@{r['home_team']}"
        index[key] = r
    return index


# ─── CSV helpers ──────────────────────────────────────────────────────────────

def load_recommendations(rec_date: str) -> List[Dict]:
    """Load the recommendations CSV for a given date."""
    path = os.path.join(OUTPUT_DIR, f"recommendations_{rec_date}.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_results_log(rows: List[Dict]) -> str:
    """Append tracked outcomes to results_log.csv. Returns path."""
    path = os.path.join(OUTPUT_DIR, "results_log.csv")
    fieldnames = [
        "date", "game", "market", "side", "bet_label",
        "odds", "model_prob", "edge_pct", "confidence",
        "outcome", "pnl",
        "actual_home_goals", "actual_away_goals",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return path


# ─── Main entry point ─────────────────────────────────────────────────────────

def track_results(rec_date: Optional[str] = None) -> None:
    """
    Load recommendations for rec_date, fetch scores, compute outcomes,
    print P&L summary, and append to results_log.csv.
    """
    if rec_date is None:
        rec_date = (date.today() - timedelta(days=1)).isoformat()

    print(Fore.CYAN + Style.BRIGHT + f"\n  RESULTS TRACKING — {rec_date}")
    print(Fore.WHITE + "  " + "─" * 61)

    recs = load_recommendations(rec_date)
    if not recs:
        print(Fore.YELLOW +
              f"  No recommendations file found for {rec_date}.")
        print(Fore.WHITE +
              f"  Expected: {os.path.join(OUTPUT_DIR, f'recommendations_{rec_date}.csv')}\n")
        return

    scores = get_scores_for_date(rec_date)
    if not scores:
        print(Fore.YELLOW +
              f"  No completed game scores found for {rec_date}. "
              "Games may not have finished yet.\n")
        return

    tracked_rows = []
    wins = losses = pushes = skipped = 0
    total_pnl = 0.0
    total_bet = 0.0

    # Header
    print(Fore.WHITE + Style.BRIGHT +
          f"  {'#':<4} {'Bet':<35} {'Odds':>6} {'Result':>6} {'P&L':>9}")
    print(Fore.WHITE + "  " + "─" * 61)

    for rec in recs:
        game_str = rec.get("game", "")             # "ANA @ EDM"
        parts = game_str.split(" @ ")
        if len(parts) != 2:
            skipped += 1
            continue
        away_abbr, home_abbr = parts[0].strip(), parts[1].strip()
        score_key = f"{away_abbr}@{home_abbr}"

        if score_key not in scores:
            skipped += 1
            continue

        score   = scores[score_key]
        outcome = determine_outcome(rec, score)
        try:
            odds = int(rec.get("odds", 0))
        except (ValueError, TypeError):
            skipped += 1
            continue

        pnl = calc_pnl(outcome, odds)
        total_pnl += pnl
        total_bet += 100.0

        if outcome == "WIN":
            wins += 1
            result_color = Fore.GREEN
        elif outcome == "LOSS":
            losses += 1
            result_color = Fore.RED
        elif outcome == "PUSH":
            pushes += 1
            result_color = Fore.YELLOW
        else:
            skipped += 1
            continue

        rank     = rec.get("rank", "?")
        bet_lbl  = rec.get("bet_label", rec.get("market", ""))[:34]
        odds_str = f"+{odds}" if odds > 0 else str(odds)
        pnl_str  = f"{pnl:+.2f}"

        print(
            Fore.WHITE + f"  #{rank:<3} {bet_lbl:<35} {odds_str:>6} " +
            result_color + f"{outcome:>6}" +
            (Fore.GREEN if pnl >= 0 else Fore.RED) + f" ${pnl_str:>8}"
        )

        tracked_rows.append({
            "date"              : rec_date,
            "game"              : game_str,
            "market"            : rec.get("market", ""),
            "side"              : rec.get("side", ""),
            "bet_label"         : rec.get("bet_label", ""),
            "odds"              : odds,
            "model_prob"        : rec.get("model_prob", ""),
            "edge_pct"          : rec.get("edge_pct", ""),
            "confidence"        : rec.get("confidence", ""),
            "outcome"           : outcome,
            "pnl"               : pnl,
            "actual_home_goals" : score["home_goals"],
            "actual_away_goals" : score["away_goals"],
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    total_decided = wins + losses + pushes
    print(Fore.WHITE + "  " + "─" * 61)

    if total_decided == 0:
        print(Fore.YELLOW + "  No completed games matched recommendations.\n")
        return

    roi = (total_pnl / total_bet * 100) if total_bet > 0 else 0.0
    acc = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0

    pnl_color = Fore.GREEN if total_pnl >= 0 else Fore.RED
    print(
        Fore.WHITE + Style.BRIGHT +
        f"  Bets: {total_decided}  |  "
        f"Won: {wins}  Lost: {losses}  Push: {pushes}  Skipped: {skipped}"
    )
    print(
        pnl_color + Style.BRIGHT +
        f"  P&L (@ $100/bet): ${total_pnl:+.2f}  |  "
        f"ROI: {roi:+.1f}%  |  Win rate: {acc:.1f}%"
    )

    if tracked_rows:
        log_path = append_results_log(tracked_rows)
        print(Fore.WHITE + f"  Appended {len(tracked_rows)} records → {log_path}")

    print()
