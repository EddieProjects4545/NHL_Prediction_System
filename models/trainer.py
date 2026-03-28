"""
Training pipeline.

Handles:
  - Time-decay sample weighting (recent games matter more)
  - Prior-season discount weighting
  - TimeSeriesSplit cross-validation (no data leakage)
  - Model performance reporting (Brier score, log-loss, AUC)
  - Same-day model caching (avoid retraining on every run)
"""
import os
from typing import Dict
import numpy as np
import pandas as pd
import joblib
from datetime import date
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score,
    accuracy_score,
)

from config import (
    TIME_DECAY_LAMBDA, PREV_SEASON_WEIGHT, CV_SPLITS, SAVED_MODELS_DIR,
)
from models.logistic_model  import LogisticModel
from models.xgboost_model   import XGBoostModel
from models.elo_model        import EloModel
from models.ensemble         import EnsembleModel
from models.poisson_model    import PoissonModel
from models.puckline_model   import PuckLineModel


# ─── Sample Weights ────────────────────────────────────────────────────────────

def compute_sample_weights(game_dates: pd.Series,
                            game_seasons: pd.Series = None) -> np.ndarray:
    """
    Time-decay weights: exp(-λ · days_ago)
    Prior-season games get an additional PREV_SEASON_WEIGHT multiplier.
    """
    today = date.today()
    weights = []
    current_season_str = str(today.year)[:4]

    for i, d in enumerate(game_dates):
        try:
            game_date = date.fromisoformat(str(d))
        except Exception:
            weights.append(1.0)
            continue
        days_ago = (today - game_date).days
        w = np.exp(-TIME_DECAY_LAMBDA * days_ago)
        # Discount prior season
        if game_seasons is not None:
            season = str(game_seasons.iloc[i])
            if current_season_str not in season:
                w *= PREV_SEASON_WEIGHT
        weights.append(w)
    return np.array(weights)


# ─── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate_model(model_class, X: pd.DataFrame, y: pd.Series,
                          sample_weight: np.ndarray = None,
                          n_splits: int = CV_SPLITS) -> Dict:
    """
    TimeSeriesSplit cross-validation. Returns dict of metric lists.
    """
    from typing import Dict
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {"brier": [], "logloss": [], "auc": [], "accuracy": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_va = X.iloc[val_idx]
        y_va = y.iloc[val_idx]
        sw   = sample_weight[train_idx] if sample_weight is not None else None

        model = model_class()
        if hasattr(model, "fit"):
            try:
                model.fit(X_tr, y_tr, sample_weight=sw)
            except TypeError:
                model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_va)
        metrics["brier"].append(brier_score_loss(y_va, probs))
        metrics["logloss"].append(log_loss(y_va, probs))
        metrics["auc"].append(roc_auc_score(y_va, probs))
        metrics["accuracy"].append(accuracy_score(y_va, (probs >= 0.5).astype(int)))

    return {k: round(np.mean(v), 4) for k, v in metrics.items()}


# ─── Main Training Entry Point ────────────────────────────────────────────────

def train_all_models(
    X: pd.DataFrame,
    y_ml: pd.Series,
    y_pl_home: pd.Series,
    y_pl_away: pd.Series,
    y_goals_home: pd.Series,
    y_goals_away: pd.Series,
    game_dates: pd.Series = None,
    game_seasons: pd.Series = None,
    force_retrain: bool = False,
) -> dict:
    """
    Train all models and return them in a dict.
    Caches models on disk; reuses same-day cache unless force_retrain=True.

    Returns
    -------
    {
        "ensemble"  : EnsembleModel,
        "puckline"  : PuckLineModel,
        "poisson"   : PoissonModel,
        "metrics"   : dict of CV metrics,
        "n_samples" : int,
    }
    """
    today = date.today().isoformat()
    stamp_path = os.path.join(SAVED_MODELS_DIR, "train_date.txt")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # Check if today's models are cached
    if not force_retrain and os.path.exists(stamp_path):
        with open(stamp_path) as f:
            cached_date = f.read().strip()
        if cached_date == today:
            print("Loading cached models (trained today)...")
            try:
                return {
                    "ensemble"  : joblib.load(os.path.join(SAVED_MODELS_DIR, "ensemble.joblib")),
                    "puckline"  : joblib.load(os.path.join(SAVED_MODELS_DIR, "puckline.joblib")),
                    "poisson"   : joblib.load(os.path.join(SAVED_MODELS_DIR, "poisson_totals.joblib")),
                    "metrics"   : joblib.load(os.path.join(SAVED_MODELS_DIR, "metrics.joblib")),
                    "n_samples" : len(X),
                }
            except Exception:
                print("Cache load failed — retraining...")

    print(f"\nTraining models on {len(X)} samples...")

    # Sample weights
    sw = compute_sample_weights(
        game_dates if game_dates is not None else pd.Series(["2026-01-01"] * len(X)),
        game_seasons,
    )

    # ── Ensemble (ML) ──────────────────────────────────────────────────────────
    print("\n[1/3] Training Moneyline Ensemble...")
    # Time-split validation set (most recent 15%)
    split_idx = int(len(X) * 0.85)
    X_tr, X_va = X.iloc[:split_idx], X.iloc[split_idx:]
    y_tr, y_va = y_ml.iloc[:split_idx], y_ml.iloc[split_idx:]
    sw_tr      = sw[:split_idx]

    ensemble = EnsembleModel()
    ensemble.fit(X_tr, y_tr, sample_weight=sw_tr, X_val=X_va, y_val=y_va)

    # CV metrics
    print("  Running cross-validation...")
    lr_cv  = cross_validate_model(LogisticModel, X, y_ml, sw)
    xgb_cv = cross_validate_model(XGBoostModel,  X, y_ml, sw)

    # Val set metrics for full ensemble
    val_probs, _ = ensemble.predict_proba(X_va)
    val_metrics = {
        "ensemble_val_brier"   : round(brier_score_loss(y_va, val_probs), 4),
        "ensemble_val_logloss" : round(log_loss(y_va, val_probs), 4),
        "ensemble_val_auc"     : round(roc_auc_score(y_va, val_probs), 4),
        "ensemble_val_accuracy": round(accuracy_score(y_va, (val_probs >= 0.5).astype(int)), 4),
        "logistic_cv_brier"    : lr_cv["brier"],
        "xgboost_cv_brier"     : xgb_cv["brier"],
        "n_train"              : split_idx,
        "n_val"                : len(X_va),
    }
    print(f"  Val Brier: {val_metrics['ensemble_val_brier']:.4f} | "
          f"AUC: {val_metrics['ensemble_val_auc']:.4f} | "
          f"Acc: {val_metrics['ensemble_val_accuracy']:.4f}")

    # Refit on all data
    ensemble.fit(X, y_ml, sample_weight=sw)

    # ── Puck Line ──────────────────────────────────────────────────────────────
    print("\n[2/3] Training Puck Line Model...")
    puckline = PuckLineModel()
    puckline.fit(X, y_pl_home, y_pl_away, sample_weight=sw)
    pl_probs_h = puckline.predict_proba_home_minus1_5(X_va)
    pl_probs_a = puckline.predict_proba_away_plus1_5(X_va)
    val_metrics["puckline_home_brier"] = round(
        brier_score_loss(y_pl_home.iloc[split_idx:], pl_probs_h), 4)
    val_metrics["puckline_away_brier"] = round(
        brier_score_loss(y_pl_away.iloc[split_idx:], pl_probs_a), 4)
    print(f"  Puck Line Home Brier: {val_metrics['puckline_home_brier']:.4f}")

    # ── Poisson Totals ─────────────────────────────────────────────────────────
    print("\n[3/3] Training Poisson Totals Model...")
    poisson = PoissonModel()
    poisson.fit(X, y_goals_home, y_goals_away, sample_weight=sw)
    print("  Poisson model fitted.")

    # ── Save ───────────────────────────────────────────────────────────────────
    joblib.dump(ensemble,  os.path.join(SAVED_MODELS_DIR, "ensemble.joblib"))
    joblib.dump(puckline,  os.path.join(SAVED_MODELS_DIR, "puckline.joblib"))
    joblib.dump(poisson,   os.path.join(SAVED_MODELS_DIR, "poisson_totals.joblib"))
    joblib.dump(val_metrics, os.path.join(SAVED_MODELS_DIR, "metrics.joblib"))
    with open(stamp_path, "w") as f:
        f.write(today)

    print(f"\nAll models saved to {SAVED_MODELS_DIR}/")

    return {
        "ensemble"  : ensemble,
        "puckline"  : puckline,
        "poisson"   : poisson,
        "metrics"   : val_metrics,
        "n_samples" : len(X),
    }


