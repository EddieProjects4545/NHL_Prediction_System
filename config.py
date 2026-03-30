"""
NHL Betting Model — Central Configuration
"""

# ─── API Keys ─────────────────────────────────────────────────────────────────
ODDS_API_KEY = "564c068ce8911fb0eac9b46c4539e831"

# ─── NHL API Base URLs ────────────────────────────────────────────────────────
NHL_WEB_API_BASE  = "https://api-web.nhle.com/v1"
NHL_STATS_API_BASE = "https://api.nhle.com/stats/rest/en"

# ─── The Odds API ─────────────────────────────────────────────────────────────
ODDS_API_BASE      = "https://api.the-odds-api.com/v4"
ODDS_SPORT         = "icehockey_nhl"
ODDS_REGIONS       = "us"
ODDS_MARKETS       = "h2h,spreads,totals"
# Books ranked by sharpness / line quality; first available book is used
ODDS_PRIORITY_BOOKS = [
    "fanduel", "draftkings", "betmgm", "williamhill_us",
    "caesars", "pointsbetus", "bovada",
]

# ─── Season ───────────────────────────────────────────────────────────────────
CURRENT_SEASON     = "20252026"
PREV_SEASON        = "20242025"
GAME_TYPE_REGULAR  = 2
GAME_TYPE_PLAYOFF  = 3
SEASON_START_DATE  = "2025-10-04"   # First game of 2025-26 season
PREV_SEASON_START  = "2024-10-08"   # First game of 2024-25 season
PREV_SEASON_END    = "2025-04-18"   # Last regular-season game 2024-25

# ─── Elo Rating ───────────────────────────────────────────────────────────────
ELO_INITIAL       = 1500.0
ELO_K_REGULAR     = 4.0
ELO_K_PLAYOFF     = 6.0    # Higher K in playoffs — results matter more
ELO_HOME_BONUS    = 35.0   # Added to home team before probability calculation
ELO_RESET_FACTOR  = 0.35   # Pull ratings toward 1500 at season start (regression)

# ─── Training ─────────────────────────────────────────────────────────────────
TIME_DECAY_LAMBDA   = 0.008   # exp(-λ·days_ago); 180d ago → ~23% weight
PREV_SEASON_WEIGHT  = 0.35    # Additional multiplier for prior-season games
ROLLING_WINDOWS     = [5, 10, 20]
MIN_GAMES_FOR_FEAT  = 5       # Games needed before rolling features are meaningful
CV_SPLITS           = 5       # TimeSeriesSplit folds

XGBOOST_ML_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : 4,
    "learning_rate"    : 0.04,
    "subsample"        : 0.80,
    "colsample_bytree" : 0.75,
    "min_child_weight" : 3,
    "reg_alpha"        : 0.10,
    "reg_lambda"       : 1.50,
    "eval_metric"      : "logloss",
    "random_state"     : 42,
    "n_jobs"           : -1,
}

XGBOOST_PL_PARAMS = {          # Puck-line model — slightly more regularised
    "n_estimators"     : 250,
    "max_depth"        : 4,
    "learning_rate"    : 0.04,
    "subsample"        : 0.80,
    "colsample_bytree" : 0.75,
    "min_child_weight" : 4,
    "reg_alpha"        : 0.15,
    "reg_lambda"       : 2.00,
    "eval_metric"      : "logloss",
    "random_state"     : 42,
    "n_jobs"           : -1,
}

LOGISTIC_PARAMS = {
    "C"          : 0.5,
    "penalty"    : "l2",
    "solver"     : "lbfgs",
    "max_iter"   : 1000,
    "random_state": 42,
}

# Ensemble weights for moneyline prediction
ENSEMBLE_WEIGHTS = {
    "logistic" : 0.25,
    "xgboost"  : 0.50,
    "elo"      : 0.25,
}

# ─── Betting Thresholds ───────────────────────────────────────────────────────
MIN_EDGE_PCT             = 3.0   # Default minimum edge % (favourite ML, PL)
MIN_EDGE_PCT_UNDERDOG_ML = 5.0   # Positive-odds ML (away underdog) — slightly higher bar
MIN_EDGE_PCT_OU          = 5.0   # Over/under both directions
MIN_MODEL_PROB_UNDERDOG_ML = 0.30  # Model must give underdog at least 30% win prob
MIN_CONFIDENCE  = 45      # Out of 95
MIN_ODDS        = -300    # Ignore massive favourites (too much juice)
MAX_ODDS        = 400     # Ignore extreme underdogs
KELLY_FRACTION  = 0.25    # Quarter-Kelly sizing

# ─── Cache / Storage ──────────────────────────────────────────────────────────
CACHE_DIR           = "cache"
CACHE_TTL_SECONDS   = 3600   # 1 hour
SAVED_MODELS_DIR    = "saved_models"
OUTPUT_DIR          = "./outputs"    # Recommendations CSV written here

# ─── All 32 NHL Team Abbreviations (canonical NHL API format) ─────────────────
TEAM_ABBREVS = [
    "ANA","BOS","BUF","CGY","CAR","CHI","COL","CBJ","DAL","DET",
    "EDM","FLA","LAK","MIN","MTL","NSH","NJD","NYI","NYR","OTT",
    "PHI","PIT","SJS","SEA","STL","TBL","TOR","UTA","VAN","VGK",
    "WSH","WPG",
]

TEAM_FULL_NAMES = {
    "ANA":"Anaheim Ducks",      "BOS":"Boston Bruins",
    "BUF":"Buffalo Sabres",     "CGY":"Calgary Flames",
    "CAR":"Carolina Hurricanes","CHI":"Chicago Blackhawks",
    "COL":"Colorado Avalanche", "CBJ":"Columbus Blue Jackets",
    "DAL":"Dallas Stars",       "DET":"Detroit Red Wings",
    "EDM":"Edmonton Oilers",    "FLA":"Florida Panthers",
    "LAK":"LA Kings",           "MIN":"Minnesota Wild",
    "MTL":"Montreal Canadiens", "NSH":"Nashville Predators",
    "NJD":"New Jersey Devils",  "NYI":"New York Islanders",
    "NYR":"New York Rangers",   "OTT":"Ottawa Senators",
    "PHI":"Philadelphia Flyers","PIT":"Pittsburgh Penguins",
    "SJS":"San Jose Sharks",    "SEA":"Seattle Kraken",
    "STL":"St. Louis Blues",    "TBL":"Tampa Bay Lightning",
    "TOR":"Toronto Maple Leafs","UTA":"Utah Hockey Club",
    "VAN":"Vancouver Canucks",  "VGK":"Vegas Golden Knights",
    "WSH":"Washington Capitals","WPG":"Winnipeg Jets",
}
