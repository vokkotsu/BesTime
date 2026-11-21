"""Feature engineering for commute duration modeling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DUCKDB_PATH, MODEL_FEATURES, RUSH_WINDOWS, TARGET_COLUMN
from src.ingestion.pipeline import get_connection
from src.ingestion.routing import rush_intensity


def _cyclical(value: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * value.astype(float) / period
    return np.sin(radians), np.cos(radians)


def _flag_rush(hour: pd.Series, window: dict) -> pd.Series:
    return ((hour >= window["start_hour"]) & (hour <= window["end_hour"])).astype(int)


def build_feature_frame(
    db_path: Path = DUCKDB_PATH,
    route_id: str | None = None,
) -> pd.DataFrame:
    """Join trips to hourly weather and compute model features."""
    query = """
        SELECT
            t.trip_id,
            t.route_id,
            t.departure_ts,
            t.observed_duration_min,
            t.source,
            COALESCE(t.base_duration_min, r.base_duration_min) AS base_duration_min,
            COALESCE(t.distance_km, r.distance_km) AS distance_km,
            w.precipitation AS precipitation_mm,
            w.rain AS rain_mm,
            w.weathercode,
            w.windspeed_10m
        FROM trip_observations t
        LEFT JOIN routes r ON t.route_id = r.route_id
        LEFT JOIN weather_hourly w
          ON t.route_id = w.route_id
         AND date_trunc('hour', t.departure_ts) = w.timestamp_local
    """
    params: list = []
    if route_id:
        query += " WHERE t.route_id = ?"
        params.append(route_id)
    query += " ORDER BY t.departure_ts"

    with get_connection(db_path) as conn:
        cur = conn.execute(query, params)
        frame = cur.df()

    if frame.empty:
        return frame

    frame["departure_ts"] = pd.to_datetime(frame["departure_ts"])
    frame = frame.sort_values(["route_id", "departure_ts"]).reset_index(drop=True)

    # Fill missing weather with route-local rolling median then zero.
    weather_cols = ["precipitation_mm", "rain_mm", "weathercode", "windspeed_10m"]
    for col in weather_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame[weather_cols] = (
        frame.groupby("route_id")[weather_cols]
        .transform(lambda s: s.fillna(s.median()))
        .fillna(0.0)
    )

    frame["base_duration_min"] = pd.to_numeric(frame["base_duration_min"], errors="coerce")
    frame["base_duration_min"] = frame.groupby("route_id")["base_duration_min"].transform(
        lambda s: s.fillna(s.median())
    )
    frame["base_duration_min"] = frame["base_duration_min"].fillna(frame["observed_duration_min"].median())

    frame["departure_hour"] = frame["departure_ts"].dt.hour
    frame["departure_minute"] = frame["departure_ts"].dt.minute
    frame["hour_decimal"] = frame["departure_hour"] + frame["departure_minute"] / 60.0
    frame["day_of_week"] = frame["departure_ts"].dt.day_of_week
    frame["is_weekday"] = (frame["day_of_week"] < 5).astype(int)

    frame["hour_sin"], frame["hour_cos"] = _cyclical(frame["hour_decimal"], 24.0)
    frame["dow_sin"], frame["dow_cos"] = _cyclical(frame["day_of_week"], 7.0)

    morning = RUSH_WINDOWS[0]
    evening = RUSH_WINDOWS[1]
    frame["is_morning_rush"] = _flag_rush(frame["hour_decimal"], morning)
    frame["is_evening_rush"] = _flag_rush(frame["hour_decimal"], evening)
    frame["rush_intensity"] = frame["hour_decimal"].apply(rush_intensity).astype(float)

    # Rolling rain features use recent departures in same route; shift prevents leakage.
    frame["precip_rolling_3h"] = (
        frame.groupby("route_id")["precipitation_mm"]
        .transform(lambda s: s.shift(1).rolling(window=12, min_periods=1).mean())
        .fillna(0.0)
    )
    frame["rain_rolling_3h"] = (
        frame.groupby("route_id")["rain_mm"]
        .transform(lambda s: s.shift(1).rolling(window=12, min_periods=1).mean())
        .fillna(0.0)
    )
    frame["is_raining"] = ((frame["precipitation_mm"] > 0.1) | (frame["rain_mm"] > 0.1)).astype(int)
    frame["heavy_rain"] = (frame["precipitation_mm"] >= 5.0).astype(int)

    frame["rain_x_rush"] = frame["precipitation_mm"] * frame["rush_intensity"]
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    frame = frame[frame[TARGET_COLUMN] > 0]

    return frame


def get_training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in MODEL_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Feature frame missing required columns: {missing}")
    X = frame[MODEL_FEATURES].copy()
    y = frame[TARGET_COLUMN].astype(float).copy()
    return X, y


def add_time_features_for_inference(frame: pd.DataFrame) -> pd.DataFrame:
    """Add model features to a future departure/weather grid."""
    out = frame.copy()
    out["departure_ts"] = pd.to_datetime(out["departure_ts"])
    out["departure_hour"] = out["departure_ts"].dt.hour
    out["departure_minute"] = out["departure_ts"].dt.minute
    out["hour_decimal"] = out["departure_hour"] + out["departure_minute"] / 60.0
    out["day_of_week"] = out["departure_ts"].dt.day_of_week
    out["is_weekday"] = (out["day_of_week"] < 5).astype(int)
    out["hour_sin"], out["hour_cos"] = _cyclical(out["hour_decimal"], 24.0)
    out["dow_sin"], out["dow_cos"] = _cyclical(out["day_of_week"], 7.0)
    morning = RUSH_WINDOWS[0]
    evening = RUSH_WINDOWS[1]
    out["is_morning_rush"] = _flag_rush(out["hour_decimal"], morning)
    out["is_evening_rush"] = _flag_rush(out["hour_decimal"], evening)
    out["rush_intensity"] = out["hour_decimal"].apply(rush_intensity).astype(float)

    for col in ["precipitation_mm", "rain_mm", "weathercode", "windspeed_10m"]:
        out[col] = pd.to_numeric(out.get(col, 0.0), errors="coerce").fillna(0.0)

    out["precip_rolling_3h"] = (
        out["precipitation_mm"].rolling(window=12, min_periods=1).mean().fillna(0.0)
    )
    out["rain_rolling_3h"] = (
        out["rain_mm"].rolling(window=12, min_periods=1).mean().fillna(0.0)
    )
    out["is_raining"] = ((out["precipitation_mm"] > 0.1) | (out["rain_mm"] > 0.1)).astype(int)
    out["heavy_rain"] = (out["precipitation_mm"] >= 5.0).astype(int)
    out["rain_x_rush"] = out["precipitation_mm"] * out["rush_intensity"]
    return out
