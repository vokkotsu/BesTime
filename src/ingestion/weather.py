"""Open-Meteo weather ingestion with local caching."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import requests

from src.config import (
    DEFAULT_TIMEZONE,
    FORECAST_HORIZON_DAYS,
    HISTORICAL_BOOTSTRAP_DAYS,
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    WEATHER_HOURLY_FIELDS,
)


class WeatherIngestionError(RuntimeError):
    """Raised when weather data cannot be fetched or parsed."""


def _request_json(url: str, params: dict, timeout: int = 30) -> dict:
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if "hourly" not in data:
        raise WeatherIngestionError(f"Unexpected weather response shape from {url}")
    return data


def _hourly_to_frame(payload: dict, route_id: str, source: str) -> pd.DataFrame:
    hourly = payload["hourly"]
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()

    frame = pd.DataFrame({"timestamp_local": pd.to_datetime(times)})
    for field in WEATHER_HOURLY_FIELDS:
        frame[field] = pd.to_numeric(hourly.get(field, [0.0] * len(times)), errors="coerce").fillna(0.0)
    frame["route_id"] = route_id
    frame["source"] = source
    frame["latitude"] = payload.get("latitude")
    frame["longitude"] = payload.get("longitude")
    frame["timezone"] = payload.get("timezone", DEFAULT_TIMEZONE)
    frame["fetched_at_utc"] = pd.Timestamp.now("UTC")
    return frame


def fetch_forecast_weather(
    route_id: str,
    latitude: float,
    longitude: float,
    start_date: date | None = None,
    end_date: date | None = None,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Fetch hourly forecast weather from Open-Meteo."""
    start = start_date or date.today()
    end = end_date or (start + timedelta(days=FORECAST_HORIZON_DAYS - 1))
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_HOURLY_FIELDS),
        "timezone": timezone,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    payload = _request_json(OPEN_METEO_FORECAST_URL, params)
    return _hourly_to_frame(payload, route_id=route_id, source="forecast")


def fetch_historical_weather(
    route_id: str,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str = DEFAULT_TIMEZONE,
) -> pd.DataFrame:
    """Fetch hourly historical/archive weather from Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_HOURLY_FIELDS),
        "timezone": timezone,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }
    payload = _request_json(OPEN_METEO_ARCHIVE_URL, params)
    return _hourly_to_frame(payload, route_id=route_id, source="archive")


def fetch_weather_for_route(
    route_id: str,
    latitude: float,
    longitude: float,
    timezone: str = DEFAULT_TIMEZONE,
    historical_days: int = HISTORICAL_BOOTSTRAP_DAYS,
    include_forecast: bool = True,
) -> pd.DataFrame:
    """Fetch historical + forecast weather for a route and return deduped hourly rows."""
    frames: list[pd.DataFrame] = []
    today = date.today()
    hist_start = today - timedelta(days=historical_days)
    hist_end = today - timedelta(days=1)
    try:
        frames.append(
            fetch_historical_weather(
                route_id=route_id,
                latitude=latitude,
                longitude=longitude,
                start_date=hist_start,
                end_date=hist_end,
                timezone=timezone,
            )
        )
    except Exception:
        # Historical archive can fail transiently; forecast/bootstrap can still proceed.
        pass

    if include_forecast:
        frames.append(
            fetch_forecast_weather(
                route_id=route_id,
                latitude=latitude,
                longitude=longitude,
                start_date=today,
                end_date=today + timedelta(days=FORECAST_HORIZON_DAYS - 1),
                timezone=timezone,
            )
        )

    if not frames:
        raise WeatherIngestionError("No weather data could be fetched.")

    weather = pd.concat(frames, ignore_index=True)
    weather = weather.drop_duplicates(subset=["route_id", "timestamp_local"], keep="last")
    weather = weather.sort_values("timestamp_local").reset_index(drop=True)
    return weather


def normalize_weather_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure weather frame has required columns and dtypes."""
    required = {"route_id", "timestamp_local", *WEATHER_HOURLY_FIELDS}
    missing = required - set(frame.columns)
    if missing:
        raise WeatherIngestionError(f"Weather frame missing columns: {sorted(missing)}")
    out = frame.copy()
    out["timestamp_local"] = pd.to_datetime(out["timestamp_local"])
    for field in WEATHER_HOURLY_FIELDS:
        out[field] = pd.to_numeric(out[field], errors="coerce").fillna(0.0)
    return out
