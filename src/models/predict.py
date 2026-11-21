"""Generate departure-vs-arrival curves and detect commute risk zones."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    CLIFF_MIN_INCREASE_MINUTES,
    CLIFF_MIN_INCREASE_PCT,
    DEFAULT_TIMEZONE,
    DEPARTURE_INTERVAL_MINUTES,
    DIMINISHING_LOOKBACK_STEPS,
    DIMINISHING_MAX_SAVING_MINUTES,
    DUCKDB_PATH,
    MODEL_FEATURES,
)
from src.features.build_features import add_time_features_for_inference
from src.ingestion.pipeline import get_connection
from src.models.train import load_route_models


def get_route_baseline(route_id: str, db_path=DUCKDB_PATH) -> dict:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT base_duration_min, distance_km, name FROM routes WHERE route_id = ?",
            [route_id],
        ).fetchone()
    if not row:
        raise ValueError(f"Route {route_id} not found in DB. Run pipeline first.")
    return {"base_duration_min": float(row[0]), "distance_km": float(row[1] or 0.0), "name": row[2]}


def get_weather_for_day(route_id: str, target_date: date, db_path=DUCKDB_PATH) -> pd.DataFrame:
    start = datetime.combine(target_date, datetime.min.time())
    end = start + timedelta(days=1)
    with get_connection(db_path) as conn:
        weather = conn.execute(
            """
            SELECT timestamp_local, precipitation, rain, weathercode, windspeed_10m
            FROM weather_hourly
            WHERE route_id = ?
              AND timestamp_local >= ?
              AND timestamp_local < ?
            ORDER BY timestamp_local
            """,
            [route_id, start, end],
        ).df()
    if weather.empty:
        # fallback: use latest available weather profile for the route
        with get_connection(db_path) as conn:
            weather = conn.execute(
                """
                SELECT timestamp_local, precipitation, rain, weathercode, windspeed_10m
                FROM weather_hourly
                WHERE route_id = ?
                ORDER BY timestamp_local DESC
                LIMIT 24
                """,
                [route_id],
            ).df()
        if weather.empty:
            # absolute fallback: dry weather
            hours = pd.date_range(start=start, periods=24, freq="h")
            weather = pd.DataFrame(
                {
                    "timestamp_local": hours,
                    "precipitation": 0.0,
                    "rain": 0.0,
                    "weathercode": 0,
                    "windspeed_10m": 5.0,
                }
            )
        else:
            weather = weather.sort_values("timestamp_local")
            # Shift timestamps to target date while preserving hour profile.
            weather["timestamp_local"] = weather["timestamp_local"].dt.floor("h")
            weather["hour"] = weather["timestamp_local"].dt.hour
            weather = weather.drop_duplicates(subset=["hour"]).set_index("hour")
            hours = pd.date_range(start=start, periods=24, freq="h")
            weather = (
                weather.reindex(range(24))
                .fillna({"precipitation": 0.0, "rain": 0.0, "weathercode": 0.0, "windspeed_10m": 5.0})
                .reset_index(drop=True)
            )
            weather["timestamp_local"] = hours
    weather["timestamp_local"] = pd.to_datetime(weather["timestamp_local"])
    return weather


def build_departure_grid(
    route_id: str,
    target_date: date,
    db_path=DUCKDB_PATH,
    interval_minutes: int = DEPARTURE_INTERVAL_MINUTES,
    start_hour: int = 5,
    end_hour: int = 22,
) -> pd.DataFrame:
    baseline = get_route_baseline(route_id, db_path=db_path)
    weather = get_weather_for_day(route_id, target_date, db_path=db_path)
    weather = weather.copy()
    weather["hour"] = pd.to_datetime(weather["timestamp_local"]).dt.hour
    weather_by_hour = weather.set_index("hour")

    start_dt = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=start_hour)
    end_dt = datetime.combine(target_date, datetime.min.time()) + timedelta(hours=end_hour)
    departures = pd.date_range(start=start_dt, end=end_dt, freq=f"{interval_minutes}min")

    rows = []
    for ts in departures:
        hour = ts.hour
        if hour in weather_by_hour.index:
            w = weather_by_hour.loc[hour]
            if isinstance(w, pd.DataFrame):
                w = w.iloc[0]
        else:
            w = {"precipitation": 0.0, "rain": 0.0, "weathercode": 0, "windspeed_10m": 5.0}
        rows.append(
            {
                "route_id": route_id,
                "departure_ts": ts,
                "base_duration_min": baseline["base_duration_min"],
                "precipitation_mm": float(w.get("precipitation", 0.0)),
                "rain_mm": float(w.get("rain", 0.0)),
                "weathercode": float(w.get("weathercode", 0)),
                "windspeed_10m": float(w.get("windspeed_10m", 0.0)),
            }
        )
    grid = pd.DataFrame(rows)
    grid = add_time_features_for_inference(grid)
    return grid


def predict_departure_curve(
    route_id: str,
    target_date: date,
    db_path=DUCKDB_PATH,
    interval_minutes: int = DEPARTURE_INTERVAL_MINUTES,
) -> pd.DataFrame:
    models = load_route_models(route_id)
    grid = build_departure_grid(route_id, target_date, db_path=db_path, interval_minutes=interval_minutes)
    X = grid[MODEL_FEATURES].copy()

    for name, bundle in models.items():
        model = bundle["model"]
        pred = np.asarray(model.predict(X), dtype=float)
        grid[f"duration_{name}_min"] = np.clip(pred, 5.0, None)

    # Enforce quantile ordering.
    if "duration_p50_min" in grid and "duration_p90_min" in grid:
        grid["duration_p90_min"] = np.maximum(grid["duration_p90_min"], grid["duration_p50_min"] + 1.0)

    grid["arrival_p50_ts"] = grid["departure_ts"] + pd.to_timedelta(grid["duration_p50_min"], unit="m")
    grid["arrival_p90_ts"] = grid["departure_ts"] + pd.to_timedelta(grid["duration_p90_min"], unit="m")
    grid["risk_spread_min"] = grid["duration_p90_min"] - grid["duration_p50_min"]

    grid = detect_zones(grid)
    return grid


def detect_zones(curve: pd.DataFrame) -> pd.DataFrame:
    """Flag diminishing returns and delay cliff zones on a departure curve."""
    out = curve.sort_values("departure_ts").reset_index(drop=True).copy()
    out["zone"] = "normal"
    out["zone_reason"] = ""

    # Delay cliff: large increase compared to previous departure slot.
    prev = out["duration_p50_min"].shift(1)
    delta_min = out["duration_p50_min"] - prev
    delta_pct = delta_min / prev.replace(0, np.nan)
    cliff_mask = (delta_min >= CLIFF_MIN_INCREASE_MINUTES) | (delta_pct >= CLIFF_MIN_INCREASE_PCT)
    cliff_mask = cliff_mask.fillna(False)
    out.loc[cliff_mask, "zone"] = "delay_cliff"
    out.loc[cliff_mask, "zone_reason"] = (
        "Delay cliff: leaving now adds ~"
        + delta_min[cliff_mask].round(0).astype(int).astype(str)
        + " min vs previous slot"
    )

    # Diminishing returns: leaving earlier by lookback window saves very little.
    # For each point, compare to point LOOKBACK steps earlier (earlier departure).
    earlier = out["duration_p50_min"].shift(-DIMINISHING_LOOKBACK_STEPS)
    saving_from_earlier = earlier - out["duration_p50_min"]  # positive if earlier is slower
    # We want points where departing much earlier yields little/no saving.
    # Mark earlier slot as diminishing if it doesn't save much compared to current.
    diminishing_earlier_mask = saving_from_earlier.notna() & (
        saving_from_earlier <= DIMINISHING_MAX_SAVING_MINUTES
    )
    # Apply to earlier indices.
    earlier_indices = out.index[diminishing_earlier_mask]
    target_indices = earlier_indices  # mark the earlier departure as diminishing return
    # Only mark as diminishing if not already a cliff.
    for idx in target_indices:
        if out.loc[idx, "zone"] == "normal":
            save_val = saving_from_earlier.loc[idx]
            later_idx = idx + DIMINISHING_LOOKBACK_STEPS
            later_ts = out.loc[later_idx, "departure_ts"] if later_idx < len(out) else None
            if later_ts is not None:
                mins = int(DIMINISHING_LOOKBACK_STEPS * DEPARTURE_INTERVAL_MINUTES)
                out.loc[idx, "zone"] = "diminishing_return"
                out.loc[idx, "zone_reason"] = (
                    f"Diminishing return: leaving {mins} min earlier saves <= {DIMINISHING_MAX_SAVING_MINUTES:.0f} min"
                )

    # Also flag points where P90 spread is very high as elevated tail risk.
    spread = out["risk_spread_min"]
    high_spread = spread >= spread.quantile(0.85) if len(out) >= 5 else spread >= 20
    out.loc[high_spread & (out["zone"] == "normal"), "zone"] = "high_tail_risk"
    out.loc[high_spread & (out["zone"] == "normal"), "zone_reason"] = "High tail risk: P90 spread is elevated"

    return out


def summarize_zones(curve: pd.DataFrame) -> dict:
    summary = {
        "delay_cliff_count": int((curve["zone"] == "delay_cliff").sum()),
        "diminishing_return_count": int((curve["zone"] == "diminishing_return").sum()),
        "high_tail_risk_count": int((curve["zone"] == "high_tail_risk").sum()),
    }
    cliffs = curve[curve["zone"] == "delay_cliff"]["departure_ts"]
    if not cliffs.empty:
        summary["first_cliff_departure"] = cliffs.iloc[0].strftime("%H:%M")
    return summary
