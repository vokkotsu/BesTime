"""Central configuration for the commute optimizer MVP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DUCKDB_PATH = DATA_DIR / "commute.duckdb"
RAW_WEATHER_PATH = DATA_DIR / "weather_raw.parquet"
RAW_TRIPS_PATH = DATA_DIR / "trips_raw.parquet"
PROCESSED_FEATURES_PATH = DATA_DIR / "features.parquet"

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"

DEFAULT_TIMEZONE = "Asia/Jakarta"
DEPARTURE_INTERVAL_MINUTES = 15
FORECAST_HORIZON_DAYS = 3
HISTORICAL_BOOTSTRAP_DAYS = 56


@dataclass(frozen=True)
class RouteConfig:
    """A named commute route."""

    route_id: str
    name: str
    origin_label: str
    destination_label: str
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    typical_departure: time


ROUTES: dict[str, RouteConfig] = {
    "bsd_jakarta": RouteConfig(
        route_id="bsd_jakarta",
        name="BSD Serpong -> Sudirman, Jakarta",
        origin_label="BSD Serpong",
        destination_label="Sudirman, Jakarta",
        origin_lat=-6.3024,
        origin_lon=106.6527,
        dest_lat=-6.2088,
        dest_lon=106.8226,
        typical_departure=time(7, 0),
    ),
    "bekasi_jakarta": RouteConfig(
        route_id="bekasi_jakarta",
        name="Bekasi -> Kuningan, Jakarta",
        origin_label="Bekasi",
        destination_label="Kuningan, Jakarta",
        origin_lat=-6.2383,
        origin_lon=106.9756,
        dest_lat=-6.2297,
        dest_lon=106.8295,
        typical_departure=time(6, 45),
    ),
}

DEFAULT_ROUTE_ID = "bsd_jakarta"

# Rush-hour shape parameters used by synthetic routing fallback and feature heuristics.
# Calibrated for Jakarta-style commutes: free-flow ~35 min, rush peak ~65-80 min,
# heavy rain during rush can push P90 tail risk to ~95-115 min.
RUSH_WINDOWS = [
    {"name": "morning", "start_hour": 5.5, "peak_hour": 7.25, "end_hour": 9.5, "amplitude": 0.78},
    {"name": "evening", "start_hour": 16.0, "peak_hour": 17.75, "end_hour": 20.0, "amplitude": 0.74},
]

WEATHER_HOURLY_FIELDS = [
    "precipitation",
    "rain",
    "weathercode",
    "windspeed_10m",
]

# Rain effects: amplified during rush to create non-linear delay cliffs and P90 tail risk.
RAIN_PENALTY_PER_MM = 2.8
RAIN_PENALTY_CAP_MINUTES = 45.0

MODEL_FEATURES = [
    "departure_hour",
    "departure_minute",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekday",
    "is_morning_rush",
    "is_evening_rush",
    "rush_intensity",
    "precipitation_mm",
    "rain_mm",
    "precip_rolling_3h",
    "rain_rolling_3h",
    "is_raining",
    "heavy_rain",
    "weathercode",
    "windspeed_10m",
    "base_duration_min",
    "rain_x_rush",
]

TARGET_COLUMN = "observed_duration_min"
QUANTILES = {"p50": 0.5, "p90": 0.9}
RANDOM_STATE = 42
TEST_DAYS = 7
MIN_TRAINING_ROWS = 300
CLIFF_MIN_INCREASE_MINUTES = 12.0
CLIFF_MIN_INCREASE_PCT = 0.15
DIMINISHING_ELEVATED_RATIO = 1.15  # only flag diminishing returns when above free-flow baseline
DIMINISHING_ARRIVAL_SAVING_MINUTES = 10.0  # earlier departure must save < this in arrival clock time
DIMINISHING_LOOKBACK_STEPS = 3  # 45 minutes when interval is 15 min
