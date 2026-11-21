"""Tests for feature engineering and zone detection."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from src.config import DUCKDB_PATH, MODEL_FEATURES
from src.features.build_features import add_time_features_for_inference, get_training_matrix
from src.ingestion.pipeline import get_connection, run_pipeline
from src.models.predict import detect_zones


def _make_curve(rows):
    base = datetime(2026, 1, 6, 6, 0)
    data = []
    for i, duration in enumerate(rows):
        ts = base + timedelta(minutes=15 * i)
        data.append(
            {
                "departure_ts": ts,
                "duration_p50_min": float(duration),
                "duration_p90_min": float(duration) + 10.0,
                "risk_spread_min": 10.0,
            }
        )
    return pd.DataFrame(data)


def test_detect_zones_flags_delay_cliff():
    curve = _make_curve([40, 41, 42, 60, 62, 63])
    out = detect_zones(curve)
    assert "delay_cliff" in out["zone"].tolist()
    cliff_rows = out[out["zone"] == "delay_cliff"]
    assert cliff_rows.iloc[0]["departure_ts"].strftime("%H:%M") == "06:45"


def test_detect_zones_flags_diminishing_returns():
    # Early departures are similarly slow/flat; leaving 45 min earlier saves little.
    curve = _make_curve([58, 58, 58, 57, 50, 45])
    out = detect_zones(curve)
    assert "diminishing_return" in out["zone"].tolist()


def test_feature_inference_columns_present():
    frame = pd.DataFrame(
        {
            "departure_ts": pd.date_range("2026-01-06 06:00", periods=4, freq="15min"),
            "base_duration_min": [45, 45, 45, 45],
            "precipitation_mm": [0.0, 1.0, 0.0, 5.0],
            "rain_mm": [0.0, 0.5, 0.0, 4.0],
            "weathercode": [1, 61, 1, 63],
            "windspeed_10m": [5, 6, 5, 8],
        }
    )
    out = add_time_features_for_inference(frame)
    for col in MODEL_FEATURES:
        assert col in out.columns
    assert out["rain_x_rush"].dtype.kind in "fc"


def test_pipeline_and_feature_frame_smoke(tmp_path):
    db_path = tmp_path / "test_commute.duckdb"
    summary = run_pipeline(route_ids=["bsd_jakarta"], db_path=db_path, bootstrap_days=14)
    assert summary["weather_rows"] >= 0
    with get_connection(db_path) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM trip_observations")
        trips = cur.fetchone()[0]
        cur.fetchall()
    assert trips > 0

    from src.features.build_features import build_feature_frame

    frame = build_feature_frame(db_path=db_path, route_id="bsd_jakarta")
    assert not frame.empty
    X, y = get_training_matrix(frame)
    assert len(X) == len(y)
    assert set(MODEL_FEATURES).issubset(X.columns)
