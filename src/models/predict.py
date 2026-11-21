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
    DIMINISHING_ARRIVAL_SAVING_MINUTES,
    DIMINISHING_ELEVATED_RATIO,
    DIMINISHING_LOOKBACK_STEPS,
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

    # Ensure arrival columns exist for callers that pass duration-only curves.
    if "arrival_p50_ts" not in out.columns and "duration_p50_min" in out.columns:
        out["arrival_p50_ts"] = out["departure_ts"] + pd.to_timedelta(out["duration_p50_min"], unit="m")
    if "arrival_p90_ts" not in out.columns and "duration_p90_min" in out.columns:
        out["arrival_p90_ts"] = out["departure_ts"] + pd.to_timedelta(out["duration_p90_min"], unit="m")
    if "risk_spread_min" not in out.columns and {"duration_p90_min", "duration_p50_min"}.issubset(out.columns):
        out["risk_spread_min"] = out["duration_p90_min"] - out["duration_p50_min"]

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

    # Diminishing returns: only when congestion is elevated above free-flow baseline,
    # and leaving earlier by lookback window saves little in actual arrival clock time
    # because the earlier slot sits on a steeper part of the peak.
    free_flow_baseline = _free_flow_baseline(out)
    lookback_steps = _lookback_steps(out)
    earlier_arrival = out["arrival_p50_ts"].shift(lookback_steps)
    arrival_saving_min = (out["arrival_p50_ts"] - earlier_arrival).dt.total_seconds() / 60.0
    elevated = out["duration_p50_min"] > (free_flow_baseline * DIMINISHING_ELEVATED_RATIO)

    # Flag the earlier slot as diminishing: it is elevated, and leaving it (earlier)
    # only saves < threshold minutes in arrival clock time vs departing lookback later.
    for idx in out.index[earlier_arrival.notna() & arrival_saving_min.notna()]:
        earlier_idx = idx - lookback_steps
        if earlier_idx < 0:
            continue
        if out.loc[earlier_idx, "zone"] != "normal":
            continue
        if not bool(elevated.loc[earlier_idx]):
            continue
        saving = float(arrival_saving_min.loc[idx])
        if saving < DIMINISHING_ARRIVAL_SAVING_MINUTES:
            mins = int(lookback_steps * _infer_interval_minutes(out))
            out.loc[earlier_idx, "zone"] = "diminishing_return"
            out.loc[earlier_idx, "zone_reason"] = (
                f"Diminishing return: leaving {mins} min earlier saves only {saving:.0f} min "
                "in arrival time (earlier slot hits steeper congestion)"
            )

    # High tail risk: elevated P90 spread, but avoid masking cliffs/diminishing returns.
    spread = pd.to_numeric(out.get("risk_spread_min", 0.0), errors="coerce").fillna(0.0)
    if len(out) >= 5:
        q85 = float(spread.quantile(0.85))
        high_spread = (spread >= q85) & (spread >= 15.0)
    else:
        high_spread = spread >= 20.0
    out.loc[high_spread & (out["zone"] == "normal"), "zone"] = "high_tail_risk"
    out.loc[high_spread & (out["zone"] == "normal"), "zone_reason"] = "High tail risk: P90 spread is elevated"

    return out


def _free_flow_baseline(curve: pd.DataFrame) -> float:
    """Estimate free-flow baseline from the lower tail of durations."""
    durations = pd.to_numeric(curve.get("duration_p50_min"), errors="coerce").dropna()
    if durations.empty:
        return 35.0
    # Use a robust lower-tail estimate; avoid being skewed by a single outlier.
    q20 = float(durations.quantile(0.2))
    median = float(durations.median())
    return max(5.0, min(q20, median))


def _infer_interval_minutes(curve: pd.DataFrame) -> int:
    ts = pd.to_datetime(curve["departure_ts"]).sort_values()
    if len(ts) < 2:
        return DEPARTURE_INTERVAL_MINUTES
    delta = ts.diff().dropna().dt.total_seconds() / 60.0
    if delta.empty:
        return DEPARTURE_INTERVAL_MINUTES
    return max(1, int(round(float(delta.median()))))


def _lookback_steps(curve: pd.DataFrame) -> int:
    interval = _infer_interval_minutes(curve)
    target_minutes = DIMINISHING_LOOKBACK_STEPS * DEPARTURE_INTERVAL_MINUTES
    return max(1, int(round(target_minutes / interval)))


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
