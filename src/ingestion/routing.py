"""Route baseline retrieval and synthetic rush-hour/rain travel-time simulation."""

from __future__ import annotations

import math
from dataclasses import asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from src.config import (
    OSRM_ROUTE_URL,
    RAIN_PENALTY_CAP_MINUTES,
    RAIN_PENALTY_PER_MM,
    RUSH_WINDOWS,
    RouteConfig,
)


class RoutingError(RuntimeError):
    """Raised when baseline routing cannot be fetched."""


def fetch_osrm_baseline(route: RouteConfig, timeout: int = 20) -> dict:
    """Fetch baseline route distance/duration from OSRM public demo API."""
    url = OSRM_ROUTE_URL.format(
        lon1=route.origin_lon,
        lat1=route.origin_lat,
        lon2=route.dest_lon,
        lat2=route.dest_lat,
    )
    params = {"overview": "false", "alternatives": "false", "steps": "false"}
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    routes = payload.get("routes", [])
    if not routes:
        raise RoutingError("OSRM returned no routes")
    best = routes[0]
    return {
        "route_id": route.route_id,
        "distance_km": float(best["distance"]) / 1000.0,
        "base_duration_min": float(best["duration"]) / 60.0,
        "source": "osrm",
    }


def estimate_baseline(route: RouteConfig) -> dict:
    """Get OSRM baseline, falling back to a distance-based estimate."""
    try:
        return fetch_osrm_baseline(route)
    except Exception:
        # Haversine fallback keeps the app usable when OSRM demo is unavailable/rate-limited.
        km = _haversine_km(route.origin_lat, route.origin_lon, route.dest_lat, route.dest_lon)
        avg_speed_kmh = 28.0
        return {
            "route_id": route.route_id,
            "distance_km": km,
            "base_duration_min": (km / avg_speed_kmh) * 60.0,
            "source": "estimated",
        }


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def rush_intensity(hour_decimal: float) -> float:
    """Smooth rush-hour intensity in [0, ~1.1] from configured rush windows."""
    intensity = 0.0
    for window in RUSH_WINDOWS:
        start = window["start_hour"]
        peak = window["peak_hour"]
        end = window["end_hour"]
        amp = window["amplitude"]
        if hour_decimal < start or hour_decimal > end:
            continue
        if hour_decimal <= peak:
            # rising edge
            x = (hour_decimal - start) / max(peak - start, 1e-6)
        else:
            # falling edge
            x = (end - hour_decimal) / max(end - peak, 1e-6)
        # Smoothstep-like curve with sharper peak for phase transition behavior.
        intensity += amp * (x**1.3)
    return float(np.clip(intensity, 0.0, 1.2))


def simulate_observed_duration(
    base_duration_min: float,
    departure_dt: datetime,
    precipitation_mm: float,
    rain_mm: float,
    rng: np.random.Generator,
) -> float:
    """Generate realistic non-linear travel duration with rush-hour and rain effects."""
    hour_decimal = departure_dt.hour + departure_dt.minute / 60.0
    rush = rush_intensity(hour_decimal)
    weekday = departure_dt.weekday() < 5
    weekday_factor = 1.0 if weekday else 0.72

    # Non-linear congestion: modest early, steep near peak saturation.
    congestion_multiplier = 1.0 + weekday_factor * (0.55 * rush + 1.25 * (rush**2.2))

    # Rain penalty: amplified during rush to create delay cliffs.
    rain_signal = max(float(precipitation_mm), float(rain_mm))
    rain_penalty = min(RAIN_PENALTY_CAP_MINUTES, RAIN_PENALTY_PER_MM * rain_signal)
    rain_multiplier = 1.0 + (rain_penalty / max(base_duration_min, 1.0)) * (0.65 + 1.1 * rush)

    # Incident tail-risk: rare but more likely in heavy rain + peak rush.
    incident_prob = 0.015 + 0.08 * rush + 0.02 * min(rain_signal, 10.0) / 10.0
    incident_minutes = float(rng.gamma(shape=2.0, scale=8.0)) if rng.random() < incident_prob else 0.0

    noise = float(rng.normal(0.0, 2.5 + 4.0 * rush))
    duration = base_duration_min * congestion_multiplier * rain_multiplier + incident_minutes + noise
    return float(np.clip(duration, base_duration_min * 0.85, base_duration_min * 3.5))


def generate_synthetic_trips(
    route: RouteConfig,
    weather_df: pd.DataFrame,
    base_duration_min: float,
    days: int = 56,
    departures_per_day: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap synthetic trip observations from weather and route baseline.

    This is used when real user logs are sparse/unavailable, and can be replaced over time
    by actual logged trips via the pipeline/user logger.
    """
    rng = np.random.default_rng(seed)
    weather = weather_df.copy()
    weather["timestamp_local"] = pd.to_datetime(weather["timestamp_local"])
    weather = weather.set_index("timestamp_local").sort_index()

    end_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=days)
    timestamps = pd.date_range(start=start_dt, end=end_dt - timedelta(minutes=15), freq="15min")

    # Focus synthetic sampling around commute windows to make model useful with less data.
    commute_hours = set(list(range(5, 11)) + list(range(15, 22)))
    timestamps = [ts for ts in timestamps if ts.hour in commute_hours]
    if departures_per_day is not None:
        # Optional downsampling; keep deterministic-ish by stride.
        stride = max(1, len(timestamps) // max(days * departures_per_day, 1))
        timestamps = timestamps[::stride]

    rows = []
    for ts in timestamps:
        weather_row = _lookup_weather(weather, ts)
        observed = simulate_observed_duration(
            base_duration_min=base_duration_min,
            departure_dt=ts.to_pydatetime(),
            precipitation_mm=weather_row["precipitation"],
            rain_mm=weather_row["rain"],
            rng=rng,
        )
        rows.append(
            {
                "route_id": route.route_id,
                "departure_ts": ts,
                "observed_duration_min": round(observed, 2),
                "source": "synthetic_bootstrap",
                "base_duration_min": base_duration_min,
                "distance_km": None,
            }
        )

    trips = pd.DataFrame(rows)
    if trips.empty:
        return trips
    trips["departure_date"] = trips["departure_ts"].dt.date
    return trips


def _lookup_weather(weather_indexed: pd.DataFrame, ts: pd.Timestamp) -> dict:
    if weather_indexed.empty:
        return {"precipitation": 0.0, "rain": 0.0, "weathercode": 0, "windspeed_10m": 0.0}
    # Weather is hourly; align departure timestamp to hour.
    hour_ts = ts.floor("h")
    if hour_ts in weather_indexed.index:
        row = weather_indexed.loc[hour_ts]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        return {
            "precipitation": float(row.get("precipitation", 0.0)),
            "rain": float(row.get("rain", 0.0)),
            "weathercode": float(row.get("weathercode", 0.0)),
            "windspeed_10m": float(row.get("windspeed_10m", 0.0)),
        }
    # nearest fallback
    idx = weather_indexed.index.get_indexer([hour_ts], method="nearest")[0]
    row = weather_indexed.iloc[idx]
    return {
        "precipitation": float(row.get("precipitation", 0.0)),
        "rain": float(row.get("rain", 0.0)),
        "weathercode": float(row.get("weathercode", 0.0)),
        "windspeed_10m": float(row.get("windspeed_10m", 0.0)),
    }


def route_to_record(route: RouteConfig) -> dict:
    data = asdict(route)
    data["typical_departure"] = route.typical_departure.strftime("%H:%M")
    return data
