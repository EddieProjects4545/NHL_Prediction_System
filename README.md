# NHL Betting Model

A machine learning system for NHL game prediction and betting recommendation generation. Combines an ensemble ML model (Logistic Regression + XGBoost + Elo), a dedicated Puck Line model, and a Poisson regression totals model. Pulls live odds from The Odds API and goalie data from Rotowire, Daily Faceoff, and the NHL Stats API.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API key

Open `config.py` and set your Odds API key:

```python
ODDS_API_KEY = "your_key_here"
```

Get a free key at [the-odds-api.com](https://the-odds-api.com). The free tier allows 500 requests/month. Each full model run uses approximately 3–5 requests.

### 3. Verify Python version

Python 3.10+ required. The model was built and tested on Python 3.13.

---

## Daily Workflow

### Morning — train and get picks

```bash
python run.py --retrain
```

Retrains all models with the latest season data, then generates recommendations for today's games. Run this once per day in the morning. Retraining takes 2–4 minutes.

### Throughout the day — refresh picks without retraining

```bash
python run.py
```

Uses cached models. Run this 30–60 minutes before puck drop to get the latest confirmed goalies and odds. The model will use whichever confirmed starters are available at that point.

### Next morning — track yesterday's results

```bash
python run.py --track --date 2026-03-28
```

Loads the recommendations CSV from the specified date, fetches final scores, and prints a W/L/P breakdown with P&L per $100 unit. Also appends results to `results_log.csv`.

---

## All CLI Flags

| Flag | Description |
|------|-------------|
| `--date YYYY-MM-DD` | Target date for games (default: today) |
| `--retrain` | Force model retrain even if a cached model exists |
| `--no-odds` | Skip odds fetching — prints raw model probabilities only |
| `--min-edge 5.0` | Override the minimum edge % threshold (default: 3.0) |
| `--market ml` | Show only Moneyline recommendations |
| `--market pl` | Show only Puck Line recommendations |
| `--market ou` | Show only Over/Under recommendations |
| `--export json` | Export recommendations as JSON instead of CSV |
| `--export none` | Skip file export entirely |
| `--playoff` | Enable Playoff mode (higher Elo K-factor, adjusted context) |
| `--summary` | Print model performance metrics only, no predictions |
| `--track` | Track results for `--date` (default: today) |

### Examples

```bash
# Get picks for a specific date (past or future)
python run.py --date 2026-03-29

# Only show moneyline picks with at least 6% edge
python run.py --market ml --min-edge 6.0

# Model-only mode — no API key needed, prints raw probabilities
python run.py --no-odds

# Check how yesterday's picks did
python run.py --track --date 2026-03-27

# Playoff picks
python run.py --playoff --retrain

# View model accuracy summary
python run.py --summary
```

---

## System Architecture

```
NHL Stats API  ──────────────────────────────────────┐
                                                      │
Rotowire / Daily Faceoff (goalie starters)            │
         │                                            │
         v                                            v
  features/                                    data/
  ├── team_stats.py     (Corsi, PP%, GF/GA)   ├── nhl_api.py
  ├── goalie_features.py (SV%, GAA, L5 form)  ├── odds_api.py
  ├── form_features.py   (L10 streak, B2B)    └── goalie_scraper.py
  ├── h2h_features.py    (head-to-head record)
  ├── elo.py             (per-game Elo ratings)
  └── builder.py         (assembles feature matrix)
         │
         v
  models/
  ├── ensemble.py        (LR 25% + XGB 50% + Elo 25%)
  ├── puckline_model.py  (home -1.5 / away +1.5)
  └── poisson_model.py   (expected goals → O/U prob)
         │
         v
  betting/
  ├── edge_calculator.py  (model prob → edge vs. book)
  ├── recommender.py      (rank + filter by edge/confidence)
  └── confidence_scorer.py (composite 0–95 score)
         │
         v
  output/
  ├── formatter.py        (color console output)
  ├── export.py           (CSV / JSON)
  └── results_tracker.py  (W/L/P + P&L tracking)
```

---

## Data Sources

| Source | What it provides | Frequency |
|--------|-----------------|-----------|
| NHL Web API (`api-web.nhle.com`) | Schedule, scores, rosters | Per run |
| NHL Stats API (`api.nhle.com/stats/rest`) | Team stats, goalie game logs, Corsi/Fenwick | Per run |
| The Odds API | Live ML, PL, and O/U odds from 7 books | Per run |
| Rotowire | Projected starting goalies (primary source) | Per run |
| Daily Faceoff | Starting goalie confirmation (secondary) | Per run |

### Odds API books (priority order)

FanDuel → DraftKings → BetMGM → William Hill → Caesars → PointsBet → Bovada

The best available line is used for edge calculation.

---

## Models

### Moneyline Ensemble

Three sub-models whose outputs are weighted together:

| Model | Weight | Notes |
|-------|--------|-------|
| Logistic Regression | 25% | L2 regularized; calibrated probabilities |
| XGBoost | 50% | 300 estimators, depth 4; calibrated with isotonic regression |
| Elo | 25% | Per-game trajectory with home bonus (+35) |

Training uses `TimeSeriesSplit` (5 folds) to prevent lookahead bias. Sample weights apply time decay (`exp(-0.003 × days_ago)`) so recent games matter more. Prior season games receive an additional 0.55x multiplier.

### Puck Line Model

Two independent XGBoost classifiers:
- `home_model`: P(home covers −1.5)
- `away_model`: P(away covers +1.5)

Key features: cover rate history, blowout rate, one-goal-game rate, OT rate, Corsi differential, PP/PK differential, starter SV%, L5 goalie form, H2H cover rate, Elo differential.

### Poisson Totals Model

Two Poisson GLMs (home goals, away goals) using `sklearn.PoissonRegressor`. Expected goals are computed per team, then the joint Poisson distribution is integrated to get P(over) and P(under) for any line.

---

## Feature Engineering

### Team features (~40 per team)

- Goals for/against per game, GF/GA ratio
- Shots for/against, Corsi%, Fenwick% (5v5)
- PDO (shooting% + save% at 5v5)
- PP%, PK%, net PP advantage
- Win%, regulation win%, road win%, home win%
- L10 form, scoring trend L10, current streak
- Back-to-back flag, days rest

### Goalie features (~12 per team)

- Season SV%, GAA, GSAX per game proxy
- Last 5 starts SV%, GAA, momentum vs. season average
- Starter workload %, goalie depth differential
- Starter confirmed flag (True = Rotowire/Daily Faceoff sourced)

### Confirmation fallback chain

1. **Rotowire** projected lineups (primary — posts 12–24h before games)
2. **Daily Faceoff** starting goalies page (secondary — day-of confirmation)
3. **Workload streak**: goalie with ≥80% workload marked as presumed starter
4. **Season leader**: goalie with most games started

### Context features

- Is playoff, home advantage flag, O/U line, H2H record, H2H avg total

---

## Output

### Console output

Recommendations are printed in ranked order (by edge × confidence). Each row shows:

```
Market  Matchup              Side           Odds    Model%  Book%   Edge    Conf   Bet/$100
ML      ANA @ BOS            BOS ML         -145    64.5%   59.2%   +5.3%   72     $68.97
PL      ANA @ BOS            BOS -1.5       +115    44.1%   46.5%   ...
OU      ANA @ BOS            Under 5.5      -110    54.2%   52.4%   +1.8%   58     ...
```

Confidence score is on a 0–95 scale. Bet size is Quarter-Kelly (25% of full Kelly) per $100 bankroll unit.

### Files written

| File | Contents |
|------|----------|
| `recommendations_YYYY-MM-DD.csv` | All recommendations for that date |
| `results_log.csv` | Cumulative W/L/P results with P&L after tracking |

---

## Betting Thresholds (config.py)

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_EDGE_PCT` | 3.0% | Minimum model edge over book implied probability |
| `MIN_CONFIDENCE` | 45 | Minimum confidence score (0–95) |
| `MIN_ODDS` | −300 | Ignore heavy favorites (too much juice) |
| `MAX_ODDS` | +400 | Ignore extreme longshots |
| `KELLY_FRACTION` | 0.25 | Quarter-Kelly bet sizing |

Override edge threshold at runtime with `--min-edge`.

---

## Performance Metrics

Run `python run.py --summary` to see current model metrics:

- **Brier Score** — measures calibration (lower is better; random = 0.25)
- **AUC-ROC** — discrimination ability (0.5 = random, 1.0 = perfect)
- **Accuracy** — % of games where model picked the correct winner
- **Log Loss** — probabilistic accuracy (lower is better)

After the L5 goalie form improvement was added, metrics improved to approximately:
- Brier: 0.2421
- AUC: 0.6468
- Accuracy: 57.2%

---

## Notes

- The Odds API free tier provides **500 requests/month**. Each full model run uses ~3–5. Running `--no-odds` uses zero requests.
- Models are cached in `saved_models/` and reused across runs on the same day. Use `--retrain` to force refresh.
- API responses are cached in `cache/` with a 1-hour TTL. Goalie game logs are cached separately per player.
- Games already in progress (LIVE/CRIT states) are automatically filtered out to avoid live-odds artifacts.
- The puck line model only recommends **home −1.5** and **away +1.5** bets. Away −1.5 requires a separate model not yet implemented.
