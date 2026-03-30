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
from typing import Dict, List
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


def compute_ensemble_weights(component_briers: Dict[str, float]) -> Dict[str, float]:
    """
    Convert validation Brier scores into normalized ensemble weights.
    Lower Brier -> higher weight, with a small floor to avoid collapse.
    """
    raw = {}
    for name, brier in component_briers.items():
        safe_brier = max(float(brier), 1e-6)
        raw[name] = 1.0 / safe_brier

    total = sum(raw.values()) or 1.0
    weights = {k: round(v / total, 4) for k, v in raw.items()}

    # Normalization after rounding.
    norm = sum(weights.values()) or 1.0
    return {k: v / norm for k, v in weights.items()}


def build_reliability_report(
    probs: np.ndarray,
    y_true: pd.Series,
    bucket_edges: List[float] = None,
) -> List[Dict]:
    """Bucketed calibration report for quick inspection."""
    bucket_edges = bucket_edges or [0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
    probs = np.asarray(probs, dtype=float)
    y_arr = np.asarray(y_true, dtype=float)
    rows: List[Dict] = []

    for low, high in zip(bucket_edges[:-1], bucket_edges[1:]):
        if high >= 1.0:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        count = int(mask.sum())
        if count == 0:
            continue
        bucket_probs = probs[mask]
        bucket_actual = y_arr[mask]
        rows.append({
            "bucket": f"{low:.2f}-{high:.2f}",
            "count": count,
            "avg_pred": round(float(bucket_probs.mean()), 4),
            "actual_win_rate": round(float(bucket_actual.mean()), 4),
            "gap": round(float(bucket_probs.mean() - bucket_actual.mean()), 4),
        })

    return rows


def build_training_diagnostics(
    X: pd.DataFrame,
    sample_weight: np.ndarray,
    ensemble: EnsembleModel,
    puckline: PuckLineModel,
    X_val: pd.DataFrame,
    val_probs: np.ndarray,
) -> Dict:
    """Compact diagnostics to surface dead features/models quickly."""
    nunique = X.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    low_variance_cols = nunique[(nunique > 1) & (nunique <= 3)].index.tolist()

    pl_home_val = puckline.predict_proba_home_minus1_5(X_val) if len(X_val) else np.array([])
    pl_away_val = puckline.predict_proba_away_plus1_5(X_val) if len(X_val) else np.array([])

    diagnostics = {
        "n_features": int(X.shape[1]),
        "constant_feature_count": int(len(constant_cols)),
        "constant_feature_examples": constant_cols[:10],
        "low_variance_feature_count": int(len(low_variance_cols)),
        "low_variance_feature_examples": low_variance_cols[:10],
        "sample_weight_sum": round(float(sample_weight.sum()), 4),
        "sample_weight_mean": round(float(sample_weight.mean()), 4),
        "sample_weight_min": round(float(sample_weight.min()), 4),
        "sample_weight_max": round(float(sample_weight.max()), 4),
        "ensemble_val_prob_mean": round(float(np.mean(val_probs)), 4),
        "ensemble_val_prob_std": round(float(np.std(val_probs)), 4),
        "pl_home_val_prob_mean": round(float(np.mean(pl_home_val)), 4) if pl_home_val.size else 0.0,
        "pl_home_val_prob_std": round(float(np.std(pl_home_val)), 4) if pl_home_val.size else 0.0,
        "pl_away_val_prob_mean": round(float(np.mean(pl_away_val)), 4) if pl_away_val.size else 0.0,
        "pl_away_val_prob_std": round(float(np.std(pl_away_val)), 4) if pl_away_val.size else 0.0,
        "xgboost_top_features": ensemble.xgboost.feature_importance(10),
        "puckline_home_top_features": puckline.feature_importance_home(10),
        "puckline_away_top_features": puckline.feature_importance_away(10),
    }
    return diagnostics


def print_training_diagnostics(diagnostics: Dict) -> None:
    """Human-readable trainer diagnostics."""
    print("  Diagnostics:")
    print(
        f"    Features: {diagnostics['n_features']} | "
        f"Constant: {diagnostics['constant_feature_count']} | "
        f"Low-variance: {diagnostics['low_variance_feature_count']}"
    )
    print(
        f"    Sample weights sum/mean/min/max: "
        f"{diagnostics['sample_weight_sum']:.2f} / "
        f"{diagnostics['sample_weight_mean']:.4f} / "
        f"{diagnostics['sample_weight_min']:.4f} / "
        f"{diagnostics['sample_weight_max']:.4f}"
    )
    print(
        f"    Val prob std: ensemble {diagnostics['ensemble_val_prob_std']:.4f} | "
        f"PL home {diagnostics['pl_home_val_prob_std']:.4f} | "
        f"PL away {diagnostics['pl_away_val_prob_std']:.4f}"
    )
    print(f"    XGB top features: {diagnostics['xgboost_top_features']}")
    if diagnostics.get("reliability_report"):
        print("    Reliability buckets:")
        for row in diagnostics["reliability_report"][:8]:
            print(
                f"      {row['bucket']} | n={row['count']} | "
                f"pred={row['avg_pred']:.3f} | actual={row['actual_win_rate']:.3f} | "
                f"gap={row['gap']:+.3f}"
            )


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
    # Time-split train/calibration/eval sets to keep calibration honest.
    train_end = int(len(X) * 0.70)
    calib_end = int(len(X) * 0.85)
    X_tr, X_cal, X_va = X.iloc[:train_end], X.iloc[train_end:calib_end], X.iloc[calib_end:]
    y_tr, y_cal, y_va = y_ml.iloc[:train_end], y_ml.iloc[train_end:calib_end], y_ml.iloc[calib_end:]
    sw_tr = sw[:train_end]

    # Class imbalance: compute ratio of away wins to home wins
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    scale_pos_weight = round(n_neg / max(n_pos, 1), 3)
    print(f"  Class balance: {n_pos} home wins / {n_neg} away wins → scale_pos_weight={scale_pos_weight:.3f}")

    # CV metrics
    print("  Running cross-validation...")
    lr_cv  = cross_validate_model(LogisticModel, X, y_ml, sw)
    xgb_cv = cross_validate_model(XGBoostModel,  X, y_ml, sw)
    elo_cv = cross_validate_model(EloModel,      X, y_ml, sw)

    learned_weights = compute_ensemble_weights({
        "logistic": lr_cv["brier"],
        "xgboost": xgb_cv["brier"],
        "elo": elo_cv["brier"],
    })
    print(f"  Learned ensemble weights: {learned_weights}")

    ensemble = EnsembleModel(weights=learned_weights)
    ensemble.fit(X_tr, y_tr, sample_weight=sw_tr, X_val=X_cal, y_val=y_cal,
                 scale_pos_weight=scale_pos_weight)

    # Calibrate on a separate holdout, then evaluate on the most recent slice.
    cal_probs, cal_components = ensemble.predict_proba(X_cal)
    raw_cal_probs = cal_components["raw_ensemble"]
    ensemble.fit_calibrator(raw_cal_probs, y_cal)
    val_probs, val_components = ensemble.predict_proba(X_va)
    val_metrics = {
        "ensemble_val_brier"   : round(brier_score_loss(y_va, val_probs), 4),
        "ensemble_val_logloss" : round(log_loss(y_va, val_probs), 4),
        "ensemble_val_auc"     : round(roc_auc_score(y_va, val_probs), 4),
        "ensemble_val_accuracy": round(accuracy_score(y_va, (val_probs >= 0.5).astype(int)), 4),
        "logistic_cv_brier"    : lr_cv["brier"],
        "xgboost_cv_brier"     : xgb_cv["brier"],
        "elo_cv_brier"         : elo_cv["brier"],
        "ensemble_weights"     : learned_weights,
        "n_train"              : train_end,
        "n_calibration"        : len(X_cal),
        "n_val"                : len(X_va),
        "reliability_report"   : build_reliability_report(val_probs, y_va),
        "underdog_reliability" : build_reliability_report(
            val_probs,
            y_va,
            bucket_edges=[0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
        ),
    }
    print(f"  Val Brier: {val_metrics['ensemble_val_brier']:.4f} | "
          f"AUC: {val_metrics['ensemble_val_auc']:.4f} | "
          f"Acc: {val_metrics['ensemble_val_accuracy']:.4f}")

    # Refit on all data
    n_pos_all = int(y_ml.sum())
    n_neg_all = len(y_ml) - n_pos_all
    spw_all = round(n_neg_all / max(n_pos_all, 1), 3)
    ensemble = EnsembleModel(weights=learned_weights)
    ensemble.fit(X, y_ml, sample_weight=sw, scale_pos_weight=spw_all)
    ensemble.fit_calibrator(raw_cal_probs, y_cal)

    # ── Puck Line ──────────────────────────────────────────────────────────────
    print("\n[2/3] Training Puck Line Model...")
    puckline = PuckLineModel()
    puckline.fit(X, y_pl_home, y_pl_away, sample_weight=sw)
    pl_probs_h = puckline.predict_proba_home_minus1_5(X_va)
    pl_probs_a = puckline.predict_proba_away_plus1_5(X_va)
    val_metrics["puckline_home_brier"] = round(
        brier_score_loss(y_pl_home.iloc[calib_end:], pl_probs_h), 4)
    val_metrics["puckline_away_brier"] = round(
        brier_score_loss(y_pl_away.iloc[calib_end:], pl_probs_a), 4)
    print(f"  Puck Line Home Brier: {val_metrics['puckline_home_brier']:.4f}")
    diagnostics = build_training_diagnostics(X, sw, ensemble, puckline, X_va, val_probs)
    val_metrics["training_diagnostics"] = diagnostics
    diagnostics["reliability_report"] = val_metrics["reliability_report"]
    diagnostics["underdog_reliability"] = val_metrics["underdog_reliability"]
    print_training_diagnostics(diagnostics)

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
