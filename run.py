"""
NHL Betting Model — Main Entry Point

Usage
─────
  python run.py                          # Today's games (default)
  python run.py --date 2026-03-29        # Specific date
  python run.py --retrain                # Force model retrain
  python run.py --no-odds                # Model-only mode (no API key needed)
  python run.py --min-edge 5.0           # Override minimum edge threshold
  python run.py --market ml              # Only ML recommendations
  python run.py --market pl              # Only Puck Line recommendations
  python run.py --market ou              # Only Over/Under recommendations
  python run.py --export csv             # Export as CSV instead of Excel (default is Excel)
  python run.py --export json            # Export as JSON
  python run.py --playoff                # Playoff mode
  python run.py --summary                # Show model summary only
"""
import argparse
import sys
import os

# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from datetime import date

# ─── Path setup ───────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    CURRENT_SEASON, PREV_SEASON, GAME_TYPE_REGULAR, GAME_TYPE_PLAYOFF,
    ODDS_API_KEY, MIN_EDGE_PCT, MIN_CONFIDENCE, TEAM_FULL_NAMES,
)


def parse_args():
    p = argparse.ArgumentParser(description="NHL Betting Model")
    p.add_argument("--date",      type=str, default=date.today().isoformat(),
                   help="Game date YYYY-MM-DD")
    p.add_argument("--retrain",   action="store_true",
                   help="Force model retrain even if today's cache exists")
    p.add_argument("--no-odds",   action="store_true",
                   help="Skip odds fetching (model-only mode)")
    p.add_argument("--min-edge",  type=float, default=MIN_EDGE_PCT,
                   help="Minimum edge %% to show a recommendation")
    p.add_argument("--market",    type=str, default="all",
                   choices=["all", "ml", "pl", "ou"],
                   help="Filter to a specific market")
    p.add_argument("--export",    type=str, default="excel",
                   choices=["csv", "json", "excel", "none"],
                   help="Export format (default: excel)")
    p.add_argument("--playoff",   action="store_true",
                   help="Playoff mode (different K-factor, context features)")
    p.add_argument("--summary",   action="store_true",
                   help="Show model performance summary only")
    p.add_argument("--track",     action="store_true",
                   help="Track results for --date (default: yesterday). "
                        "Loads prior recommendations CSV and compares to final scores.")
    return p.parse_args()


def main():
    args = parse_args()
    game_date  = args.date
    is_playoff = int(args.playoff)
    game_type  = GAME_TYPE_PLAYOFF if args.playoff else GAME_TYPE_REGULAR

    # Patch config thresholds from CLI args
    import config
    config.MIN_EDGE_PCT   = args.min_edge

    from colorama import Fore, Style, init
    init(autoreset=True)

    print(Fore.CYAN + Style.BRIGHT + "\n NHL Betting Model v1.0")
    print(Fore.WHITE + f" Date: {game_date}  |  Mode: {'Playoff' if args.playoff else 'Regular Season'}")
    print(Fore.WHITE + "─" * 45 + "\n")

    # ── Results tracking (early exit — no model needed) ───────────────────────
    if args.track:
        if args.export == "excel":
            from output.results_tracker import track_and_update_excel
            track_and_update_excel(game_date)
        else:
            from output.results_tracker import track_results
            track_results(game_date)
        return

    # ── 1. Fetch Schedule ──────────────────────────────────────────────────────
    print("[1/8] Fetching upcoming games...")
    from data.nhl_api import get_upcoming_games, get_season_results
    upcoming = get_upcoming_games(days_ahead=1)
    # Filter to games on the requested date that have NOT yet started
    _IN_PROGRESS = {"LIVE", "CRIT", "IN_PROGRESS", "ON_ICE"}
    upcoming = [g for g in upcoming
                if (g.get("gameDate", "")[:10] == game_date or
                    g.get("startTimeUTC", "")[:10] == game_date)
                and g.get("gameState", "PRE") not in _IN_PROGRESS]

    if not upcoming:
        # Try schedule endpoint directly
        from data.nhl_api import get_schedule
        raw_sched = get_schedule(game_date)
        upcoming = raw_sched

    if not upcoming:
        print(Fore.YELLOW + f"  No games found for {game_date}. "
              "Check the date or try --date YYYY-MM-DD.\n")
        return

    print(f"  Found {len(upcoming)} game(s) on {game_date}")

    # ── 2. Fetch Historical Results ────────────────────────────────────────────
    print("\n[2/8] Fetching historical game results...")
    curr_results = get_season_results(CURRENT_SEASON, game_type)
    prev_results = get_season_results(PREV_SEASON, GAME_TYPE_REGULAR)
    print(f"  Current season: {len(curr_results)} games | "
          f"Prior season: {len(prev_results)} games")

    # ── 3. Build Features ──────────────────────────────────────────────────────
    print("\n[3/8] Building feature matrices...")
    from features.team_stats  import build_team_stat_features
    from features.goalie_features import build_goalie_features, apply_confirmed_starters
    from features.form_features   import build_team_game_logs
    from features.h2h_features    import build_team_cover_stats
    from features.builder         import build_training_matrix, build_prediction_features

    team_stats   = build_team_stat_features(CURRENT_SEASON, game_type)
    goalie_feats = build_goalie_features(CURRENT_SEASON, game_type)

    # ── 4. Fetch Confirmed Starters ────────────────────────────────────────────
    print("\n[4/8] Fetching confirmed starting goalies...")
    from data.goalie_scraper import get_confirmed_starters
    confirmed = get_confirmed_starters(game_date, game_type)
    goalie_feats = apply_confirmed_starters(goalie_feats, confirmed)
    n_confirmed = sum(1 for v in confirmed.values() if v.get("confirmed"))
    print(f"  {n_confirmed} confirmed starters | "
          f"{len(confirmed) - n_confirmed} estimated")

    # ── 5. Build Training Matrix ────────────────────────────────────────────────
    print("\n[5/8] Building training matrix...")
    (X_train, y_ml, y_pl_home, y_pl_away,
     y_goals_home, y_goals_away) = build_training_matrix(
        curr_results, prev_results,
        team_stats, goalie_feats,
    )

    # Attach dates for time-decay weights
    all_dated = sorted(curr_results + prev_results, key=lambda x: x.get("date", ""))
    # Only keep games that made it into training (after MIN_GAMES filter)
    game_dates   = pd.Series([g["date"]           for g in all_dated[:len(X_train)]])
    game_seasons = pd.Series([g.get("game_type", GAME_TYPE_REGULAR)
                               for g in all_dated[:len(X_train)]])

    # ── 6. Train Models ────────────────────────────────────────────────────────
    print("\n[6/8] Training / loading models...")
    from models.trainer import train_all_models
    model_bundle = train_all_models(
        X_train, y_ml, y_pl_home, y_pl_away, y_goals_home, y_goals_away,
        game_dates=game_dates, game_seasons=game_seasons,
        force_retrain=args.retrain,
    )
    ensemble  = model_bundle["ensemble"]
    puckline  = model_bundle["puckline"]
    poisson   = model_bundle["poisson"]
    metrics   = model_bundle["metrics"]
    n_samples = model_bundle["n_samples"]

    if args.summary:
        from output.formatter import print_model_summary
        print_model_summary(metrics, n_samples)
        return

    if args.track:
        from output.results_tracker import track_results
        track_date = game_date  # use --date arg (defaults to today, but tracking yesterday is common)
        track_results(track_date)
        return

    # ── 7. Build Prediction Features ───────────────────────────────────────────
    print("\n[7/8] Building prediction features for upcoming games...")

    # Get O/U lines from odds to use as features
    ou_lines_map = {}
    if not args.no_odds and ODDS_API_KEY:
        from data.odds_api import get_all_game_odds
        all_odds = get_all_game_odds()
        for key, odd in all_odds.items():
            ou_lines_map[key] = odd.get("ou", {}).get("line", 5.5)
    else:
        all_odds = {}

    X_pred = build_prediction_features(
        upcoming, curr_results, prev_results,
        confirmed, ou_lines_map, is_playoff,
    )

    if X_pred.empty:
        print(Fore.YELLOW + "  Could not build prediction features. "
              "Check NHL API connectivity.\n")
        return

    # Align columns with training matrix
    missing_cols = set(X_train.columns) - set(X_pred.columns)
    for col in missing_cols:
        X_pred[col] = 0.0
    X_pred = X_pred[X_train.columns]

    # ── 8. Generate Predictions ────────────────────────────────────────────────
    print("\n[8/8] Generating predictions and recommendations...")

    # Moneyline ensemble
    ensemble_probs, comp = ensemble.predict_proba(X_pred)

    # Puck line
    pl_prob_home = puckline.predict_proba_home_minus1_5(X_pred)
    pl_prob_away = puckline.predict_proba_away_plus1_5(X_pred)

    # Totals (Poisson)
    ou_lines_arr = X_pred["ou_line"].fillna(5.5).values
    mu_home, mu_away = poisson.predict_goals(X_pred)

    # Confidence scores
    from betting.confidence_scorer import score_game_confidence
    confidence_scores = []
    for i in range(len(X_pred)):
        gf = X_pred.iloc[i].to_dict()
        hg = goalie_feats.get(
            upcoming[i].get("homeTeam", {}).get("abbrev", ""), {}
        ) if i < len(upcoming) else {}
        comp_i = {k: float(v[i]) if hasattr(v, "__len__") else v
                  for k, v in comp.items()}
        cs = score_game_confidence(gf, comp_i, hg, n_samples)
        confidence_scores.append(cs)

    # Recommendations
    if not args.no_odds and ODDS_API_KEY:
        from betting.recommender import generate_recommendations
        recs = generate_recommendations(
            upcoming, X_pred,
            ensemble_probs, comp,
            pl_prob_home, pl_prob_away,
            mu_home, mu_away,
            all_odds, goalie_feats, confidence_scores, n_samples,
        )
        # Filter by market
        if args.market != "all":
            market_map = {"ml": "ML", "pl": "PL", "ou": "OU"}
            recs = [r for r in recs if r.market == market_map[args.market]]
    else:
        from output.formatter import print_no_odds_warning
        print_no_odds_warning(no_odds_flag=args.no_odds)
        recs = []

    # ── Output ────────────────────────────────────────────────────────────────
    from output.formatter import print_recommendations
    print_recommendations(recs, game_date, len(upcoming), metrics, n_samples)

    # If model-only mode, still print raw probabilities
    if args.no_odds or not ODDS_API_KEY:
        _print_model_only(upcoming, ensemble_probs, comp,
                          pl_prob_home, pl_prob_away,
                          mu_home, mu_away, ou_lines_arr,
                          goalie_feats, confirmed)

    # ── Export ────────────────────────────────────────────────────────────────
    if args.export != "none":
        from colorama import Fore
        if args.export == "excel":
            from output.export import export_excel
            path = export_excel(
                recs         = recs,
                upcoming     = upcoming,
                ensemble_probs = ensemble_probs,
                mu_home      = mu_home,
                mu_away      = mu_away,
                goalie_feats = goalie_feats,
                game_date    = game_date,
            )
            from output.excel_writer import _conf_tier
            n_high = sum(1 for r in recs if _conf_tier(r.model_prob, r.edge_pct) == "HIGH")
            n_med  = sum(1 for r in recs if _conf_tier(r.model_prob, r.edge_pct) == "MEDIUM")
            print(Fore.WHITE + f"  Exported to: {path}")
            print(Fore.GREEN + f"  Picks: {n_high} HIGH | {n_med} MEDIUM confidence\n")
        elif args.export == "csv":
            from output.export import export_csv
            path = export_csv(recs)
            print(Fore.WHITE + f"  Exported to: {path}\n")
        elif args.export == "json":
            from output.export import export_json
            path = export_json(recs)
            print(Fore.WHITE + f"  Exported to: {path}\n")


def _print_model_only(upcoming, ensemble_probs, comp,
                      pl_prob_home, pl_prob_away,
                      mu_home, mu_away, ou_lines,
                      goalie_feats, confirmed):
    """Print raw model probabilities when no odds are available."""
    from colorama import Fore, Style
    print(Fore.WHITE + Style.BRIGHT + "\n  RAW MODEL PROBABILITIES (no odds available)")
    print(Fore.WHITE + "  " + "-" * 71)
    header = (f"  {'Game':<28} {'Ens%':>5} {'LR%':>5} {'XGB%':>5} "
              f"{'Elo%':>5} {'PL-H':>5} {'PL-A':>5} {'Exp':>4}")
    print(Fore.WHITE + Style.BRIGHT + header)
    print(Fore.WHITE + "  " + "-" * 71)

    for i, g in enumerate(upcoming):
        home = (g.get("homeTeam", {}) or {}).get("abbrev", "")
        away = (g.get("awayTeam", {}) or {}).get("abbrev", "")
        if not home or not away or i >= len(ensemble_probs):
            continue
        ens = ensemble_probs[i]
        lr  = comp["logistic"][i]
        xg  = comp["xgboost"][i]
        el  = comp["elo"][i]
        plh = pl_prob_home[i]
        pla = pl_prob_away[i]
        exp = mu_home[i] + mu_away[i]
        game_str = f"{away} @ {home}"
        print(Fore.WHITE +
              f"  {game_str:<28} {ens:>5.1%} {lr:>5.1%} {xg:>5.1%} "
              f"{el:>5.1%} {plh:>5.1%} {pla:>5.1%} {exp:>4.1f}")
    print()


if __name__ == "__main__":
    main()
