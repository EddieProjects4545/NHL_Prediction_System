"""
Terminal output formatter using colorama.
Produces a color-coded, ranked display of betting recommendations.
"""
from typing import List, Dict
from datetime import datetime, timezone

from colorama import init, Fore, Back, Style
init(autoreset=True)

from betting.recommender import Recommendation
from config import TEAM_FULL_NAMES

# ─── Colour helpers ───────────────────────────────────────────────────────────

def _edge_colour(edge: float) -> str:
    if edge >= 8:
        return Fore.GREEN + Style.BRIGHT
    elif edge >= 5:
        return Fore.GREEN
    elif edge >= 3:
        return Fore.YELLOW
    else:
        return Fore.WHITE


def _conf_colour(conf: int) -> str:
    if conf >= 75:
        return Fore.CYAN + Style.BRIGHT
    elif conf >= 60:
        return Fore.CYAN
    elif conf >= 45:
        return Fore.WHITE
    else:
        return Fore.RED


def _market_badge(market: str) -> str:
    badges = {
        "ML" : Fore.BLUE  + Style.BRIGHT + " ML  " + Style.RESET_ALL,
        "PL" : Fore.MAGENTA + Style.BRIGHT + " PL  " + Style.RESET_ALL,
        "OU" : Fore.YELLOW  + Style.BRIGHT + " O/U " + Style.RESET_ALL,
    }
    return badges.get(market, f" {market} ")


def _fmt_odds(odds: int) -> str:
    if odds is None:
        return "N/A"
    return f"+{odds}" if odds > 0 else str(odds)


def _fmt_time(utc_str: str) -> str:
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        local = dt.astimezone()
        return local.strftime("%I:%M %p %Z")
    except Exception:
        return utc_str


# ─── Header ──────────────────────────────────────────────────────────────────

def print_header(game_date: str, n_games: int, metrics: Dict,
                 n_samples: int) -> None:
    brier = metrics.get("ensemble_val_brier", "N/A")
    auc   = metrics.get("ensemble_val_auc", "N/A")
    acc   = metrics.get("ensemble_val_accuracy", "N/A")

    print()
    print(Fore.WHITE + Style.BRIGHT + "=" * 65)
    print(Fore.CYAN  + Style.BRIGHT +
          f"  NHL BETTING RECOMMENDATIONS — {game_date}")
    print(Fore.WHITE +
          f"  {n_games} upcoming game(s) | {n_samples:,} training samples")
    print(Fore.WHITE +
          f"  Model: Brier {brier} | AUC {auc} | Accuracy {acc}")
    print(Fore.WHITE + Style.BRIGHT + "=" * 65)
    print()


# ─── Single Recommendation Block ──────────────────────────────────────────────

def print_recommendation(rank: int, rec: Recommendation) -> None:
    ec = _edge_colour(rec.edge_pct)
    cc = _conf_colour(rec.confidence)

    home_full = TEAM_FULL_NAMES.get(rec.home_team, rec.home_team)
    away_full = TEAM_FULL_NAMES.get(rec.away_team, rec.away_team)

    # ── Header bar ────────────────────────────────────────────────────────────
    print(
        Fore.WHITE + Style.BRIGHT +
        f"RANK #{rank}  " +
        _market_badge(rec.market) +
        ec + f"  EDGE: {rec.edge_pct:+.1f}%  " +
        Fore.WHITE + f"| EV: {rec.ev_pct:+.1f}%  " +
        cc + f"| CONFIDENCE: {rec.confidence}/95"
    )

    # ── Bet label ─────────────────────────────────────────────────────────────
    print(Fore.WHITE + Style.BRIGHT + f"  Bet:    {rec.bet_label}")

    # ── Matchup ───────────────────────────────────────────────────────────────
    print(Fore.WHITE +
          f"  Game:   {away_full} @ {home_full}  —  {_fmt_time(rec.game_time)}")

    # ── Odds ──────────────────────────────────────────────────────────────────
    print(Fore.WHITE +
          f"  Odds:   {_fmt_odds(rec.odds)} ({rec.book.upper()})  |  "
          f"Market implied: {rec.market_prob:.1%}  |  "
          f"Model: {rec.model_prob:.1%}")

    # ── Model breakdown ───────────────────────────────────────────────────────
    print(Fore.WHITE +
          f"  Models: LogReg {rec.logistic_prob:.1%} | "
          f"XGBoost {rec.xgboost_prob:.1%} | "
          f"Elo {rec.elo_prob:.1%}  "
          f"[std {rec.model_std:.3f}]")

    # ── Totals: expected goals ─────────────────────────────────────────────────
    if rec.market == "OU" and rec.exp_total:
        print(Fore.WHITE +
              f"  Total:  Expected {rec.exp_total:.1f} goals vs line {rec.ou_line}")

    # ── Puck line: line ────────────────────────────────────────────────────────
    if rec.market == "PL":
        line_str = f"{rec.pl_line:+.1f}" if rec.pl_line else ""
        print(Fore.WHITE + f"  Line:   {line_str}")

    # ── Goalies ───────────────────────────────────────────────────────────────
    hg_conf = "(confirmed)" if rec.home_goalie else "(unconfirmed)"
    ag_conf = "(confirmed)" if rec.away_goalie else "(unconfirmed)"
    if rec.home_goalie or rec.away_goalie:
        print(Fore.WHITE +
              f"  Goalies: {home_full}: "
              f"{rec.home_goalie or 'TBD'} {hg_conf} "
              f"SV%{rec.home_goalie_sv:.3f}  |  "
              f"{away_full}: "
              f"{rec.away_goalie or 'TBD'} {ag_conf} "
              f"SV%{rec.away_goalie_sv:.3f}")

    # ── Key edges ─────────────────────────────────────────────────────────────
    if rec.key_edges:
        print(Fore.YELLOW + f"  Edge:   " + " | ".join(rec.key_edges))

    # ── Confidence factors ────────────────────────────────────────────────────
    if rec.conf_factors:
        print(Fore.WHITE + f"  Flags:  " + ", ".join(rec.conf_factors))

    # ── Kelly sizing ──────────────────────────────────────────────────────────
    print(Fore.WHITE + Style.BRIGHT +
          f"  Size:   {rec.kelly_pct:.1f}% of bankroll (quarter-Kelly)")

    print(Fore.WHITE + "  " + "─" * 61)


# ─── Full output ──────────────────────────────────────────────────────────────

def print_recommendations(recs: List[Recommendation],
                          game_date: str,
                          n_games: int,
                          metrics: Dict,
                          n_samples: int) -> None:
    print_header(game_date, n_games, metrics, n_samples)

    if not recs:
        print(Fore.RED + Style.BRIGHT +
              "  No recommendations met the edge/confidence thresholds today.")
        print(Fore.WHITE +
              "  Consider lowering --min-edge or checking back closer to game time.")
        print()
        return

    for i, rec in enumerate(recs, 1):
        print_recommendation(i, rec)

    print()
    print(Fore.GREEN + Style.BRIGHT +
          f"  {len(recs)} bet(s) above threshold  |  "
          f"Sorted by EV × Confidence")
    print()


def print_model_summary(metrics: Dict, n_samples: int) -> None:
    """Print a standalone model performance summary."""
    print()
    print(Fore.WHITE + Style.BRIGHT + "─" * 65)
    print(Fore.CYAN + Style.BRIGHT + "  MODEL PERFORMANCE SUMMARY")
    print(Fore.WHITE + Style.BRIGHT + "─" * 65)
    print(Fore.WHITE + f"  Training samples : {n_samples:,}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(Fore.WHITE + f"  {k:<35}: {v:.4f}")
    print()


def print_no_odds_warning(no_odds_flag: bool = False) -> None:
    if no_odds_flag:
        print(Fore.YELLOW + "\n  [--no-odds] Skipping odds fetch — model probabilities only.\n")
    else:
        print(Fore.YELLOW + Style.BRIGHT +
              "\n  [WARNING] Odds API key not set — running in model-only mode.")
        print(Fore.WHITE +
              "  Add ODDS_API_KEY to config.py to see edge/EV calculations.\n")
