"""Train LightGBM quantile regressors for P50/P90 commute duration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

from src.config import (
    DATA_DIR,
    DUCKDB_PATH,
    MIN_TRAINING_ROWS,
    MODEL_FEATURES,
    QUANTILES,
    RANDOM_STATE,
    REAL_SOURCE_WEIGHT,
    REAL_TRAINING_MIN_ROWS,
    SYNTHETIC_SOURCE_WEIGHT,
    TARGET_COLUMN,
    TEST_DAYS,
)
from src.features.build_features import build_feature_frame, get_training_matrix


MODELS_DIR = DATA_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainResult:
    route_id: str
    model_paths: dict[str, str]
    metrics: dict[str, float]
    rows: int
    train_rows: int
    test_rows: int


def _time_split(frame: pd.DataFrame, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, frame
    ts = pd.to_datetime(frame["departure_ts"])
    cutoff = ts.max() - pd.Timedelta(days=test_days)
    train = frame[ts <= cutoff].copy()
    test = frame[ts > cutoff].copy()
    # Ensure we have both sides; fallback to ratio split for small datasets.
    if len(train) < 100 or len(test) < 50:
        idx = int(len(frame) * 0.8)
        train = frame.iloc[:idx].copy()
        test = frame.iloc[idx:].copy()
    return train, test


def _model_path(route_id: str, quantile_name: str) -> Path:
    return MODELS_DIR / f"{route_id}_{quantile_name}_lgbm.joblib"


def _metrics_path(route_id: str) -> Path:
    return MODELS_DIR / f"{route_id}_metrics.json"


def _is_real_source(source: pd.Series) -> pd.Series:
    return source.astype(str).isin(["real_api", "user_log", "tomtom", "live_api"])


def _select_training_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    real_mask = _is_real_source(frame.get("source", pd.Series(dtype=str)))
    real_count = int(real_mask.sum())
    synthetic_count = int((~real_mask).sum())
    if real_count >= REAL_TRAINING_MIN_ROWS:
        selected = frame[real_mask].copy()
        strategy = "real_only"
    else:
        selected = frame.copy()
        strategy = "hybrid_weighted"
    meta = {
        "real_rows": float(real_count),
        "synthetic_rows": float(synthetic_count),
        "training_strategy_real_only": 1.0 if strategy == "real_only" else 0.0,
    }
    return selected, meta


def train_route_models(
    route_id: str,
    db_path: Path = DUCKDB_PATH,
    force_synthetic_if_small: bool = True,
) -> TrainResult:
    frame = build_feature_frame(db_path=db_path, route_id=route_id)
    if frame.empty:
        raise ValueError(f"No training data found for route {route_id}. Run ingestion pipeline first.")

    if len(frame) < MIN_TRAINING_ROWS and force_synthetic_if_small:
        # Caller should have bootstrapped; we still train but warn through metrics.
        pass

    training_frame, strategy_meta = _select_training_frame(frame)
    train_df, test_df = _time_split(training_frame)
    X_train, y_train = get_training_matrix(train_df)
    X_test, y_test = get_training_matrix(test_df)
    train_weights = np.where(
        _is_real_source(train_df.get("source", pd.Series(["synthetic"] * len(train_df)))),
        REAL_SOURCE_WEIGHT,
        SYNTHETIC_SOURCE_WEIGHT,
    )

    model_paths: dict[str, str] = {}
    metrics: dict[str, float] = {
        "rows": float(len(frame)),
        "selected_rows": float(len(training_frame)),
        "train_rows": float(len(train_df)),
        "test_rows": float(len(test_df)),
        **strategy_meta,
    }

    predictions: dict[str, np.ndarray] = {}
    for name, alpha in QUANTILES.items():
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=350,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_samples=20,
            random_state=RANDOM_STATE,
        )
        model.fit(X_train, y_train, sample_weight=train_weights)
        path = _model_path(route_id, name)
        joblib.dump({"model": model, "features": MODEL_FEATURES, "quantile": alpha}, path)
        model_paths[name] = str(path)

        if len(X_test) > 0:
            pred = model.predict(X_test)
            predictions[name] = np.asarray(pred, dtype=float)
            metrics[f"{name}_mae"] = float(mean_absolute_error(y_test, pred))

    if "p50" in predictions and len(X_test) > 0:
        metrics["p50_pinball"] = _pinball_loss(y_test.to_numpy(), predictions["p50"], 0.5)
    if "p90" in predictions and len(X_test) > 0:
        metrics["p90_pinball"] = _pinball_loss(y_test.to_numpy(), predictions["p90"], 0.9)
        metrics["p90_coverage"] = float(np.mean(y_test.to_numpy() <= predictions["p90"]))

    # Simple baseline: median duration by departure hour from training data.
    if len(X_test) > 0:
        hourly_median = train_df.groupby("departure_hour")[TARGET_COLUMN].median()
        baseline_pred = test_df["departure_hour"].map(hourly_median).fillna(train_df[TARGET_COLUMN].median())
        metrics["baseline_hour_median_mae"] = float(mean_absolute_error(y_test, baseline_pred))

    _metrics_path(route_id).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainResult(
        route_id=route_id,
        model_paths=model_paths,
        metrics=metrics,
        rows=len(frame),
        train_rows=len(train_df),
        test_rows=len(test_df),
    )


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def load_route_models(route_id: str) -> dict[str, dict]:
    models = {}
    for name in QUANTILES:
        path = _model_path(route_id, name)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing model {path}. Train models first with train_route_models('{route_id}')."
            )
        models[name] = joblib.load(path)
    return models


def load_metrics(route_id: str) -> dict:
    path = _metrics_path(route_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train commute duration quantile models")
    parser.add_argument("--route-id", default="bsd_jakarta")
    args = parser.parse_args()
    result = train_route_models(args.route_id)
    print(json.dumps(result.metrics, indent=2))
