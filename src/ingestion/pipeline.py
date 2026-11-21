"""Idempotent ETL pipeline: weather + baseline + synthetic/user trips into DuckDB."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DATA_DIR, DUCKDB_PATH, ROUTES, RouteConfig
from src.ingestion.routing import estimate_baseline, generate_synthetic_trips, get_live_travel_duration
from src.ingestion.weather import fetch_weather_for_route, normalize_weather_columns


def _fallback_weather_frame(route_id: str, days: int = 7) -> pd.DataFrame:
    """Deterministic local weather fallback when Open-Meteo is unavailable."""
    start = pd.Timestamp.now().floor("h") - pd.Timedelta(days=days)
    hours = pd.date_range(start=start, periods=24 * (days + 3), freq="h")
    # Simple synthetic weather profile: light afternoon rain on some days.
    precipitation = []
    for ts in hours:
        base = 0.0
        if ts.hour in (15, 16, 17, 18) and ts.day % 3 == 0:
            base = 2.5
        if ts.hour in (17, 18) and ts.day % 5 == 0:
            base = 7.0
        precipitation.append(base)
    return pd.DataFrame(
        {
            "route_id": route_id,
            "timestamp_local": hours,
            "precipitation": precipitation,
            "rain": precipitation,
            "weathercode": [61 if p > 0 else 1 for p in precipitation],
            "windspeed_10m": 6.0,
            "source": "fallback_synthetic",
            "latitude": None,
            "longitude": None,
            "timezone": "local",
            "fetched_at_utc": pd.Timestamp.now("UTC"),
        }
    )


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS routes (
    route_id VARCHAR PRIMARY KEY,
    name VARCHAR,
    origin_label VARCHAR,
    destination_label VARCHAR,
    origin_lat DOUBLE,
    origin_lon DOUBLE,
    dest_lat DOUBLE,
    dest_lon DOUBLE,
    typical_departure VARCHAR,
    distance_km DOUBLE,
    base_duration_min DOUBLE,
    baseline_source VARCHAR,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS weather_hourly (
    route_id VARCHAR,
    timestamp_local TIMESTAMP,
    precipitation DOUBLE,
    rain DOUBLE,
    weathercode DOUBLE,
    windspeed_10m DOUBLE,
    source VARCHAR,
    latitude DOUBLE,
    longitude DOUBLE,
    timezone VARCHAR,
    fetched_at_utc TIMESTAMP,
    PRIMARY KEY (route_id, timestamp_local)
);

CREATE TABLE IF NOT EXISTS trip_observations (
    trip_id VARCHAR PRIMARY KEY,
    route_id VARCHAR,
    departure_ts TIMESTAMP,
    observed_duration_min DOUBLE,
    source VARCHAR,
    base_duration_min DOUBLE,
    distance_km DOUBLE,
    created_at TIMESTAMP
);
"""


def _execute(conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    """Execute SQL and consume any pending result to keep the connection reusable."""
    cur = conn.execute(sql) if params is None else conn.execute(sql, params)
    try:
        cur.fetchall()
    except Exception:
        # Some statements may not have fetchable results; ignore.
        pass
    return cur


def _fetchone(conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None):
    cur = conn.execute(sql) if params is None else conn.execute(sql, params)
    row = cur.fetchone()
    # Drain any remaining rows to avoid unfinished-result errors on subsequent commands.
    try:
        cur.fetchall()
    except Exception:
        pass
    return row


def _fetchdf(conn: duckdb.DuckDBPyConnection, sql: str, params: list | None = None) -> pd.DataFrame:
    cur = conn.execute(sql) if params is None else conn.execute(sql, params)
    return cur.df()


def get_connection(db_path: Path = DUCKDB_PATH) -> duckdb.DuckDBPyConnection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    _execute(conn, "PRAGMA threads=4")
    for statement in SCHEMA_SQL.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            _execute(conn, stmt)
    return conn


def upsert_routes(conn: duckdb.DuckDBPyConnection, baselines: list[dict], routes: dict[str, RouteConfig]) -> None:
    rows = []
    now = datetime.now(timezone.utc)
    for baseline in baselines:
        route = routes[baseline["route_id"]]
        rows.append(
            {
                "route_id": route.route_id,
                "name": route.name,
                "origin_label": route.origin_label,
                "destination_label": route.destination_label,
                "origin_lat": route.origin_lat,
                "origin_lon": route.origin_lon,
                "dest_lat": route.dest_lat,
                "dest_lon": route.dest_lon,
                "typical_departure": route.typical_departure.strftime("%H:%M"),
                "distance_km": baseline.get("distance_km"),
                "base_duration_min": baseline.get("base_duration_min"),
                "baseline_source": baseline.get("source"),
                "updated_at": now,
            }
        )
    frame = pd.DataFrame(rows)
    conn.register("routes_df", frame)
    try:
        _execute(conn, "DELETE FROM routes WHERE route_id IN (SELECT route_id FROM routes_df)")
        _execute(
            conn,
            """
            INSERT INTO routes
            SELECT * FROM routes_df
            """,
        )
    finally:
        try:
            conn.unregister("routes_df")
        except Exception:
            pass


def upsert_weather(conn: duckdb.DuckDBPyConnection, weather_df: pd.DataFrame) -> int:
    if weather_df.empty:
        return 0
    weather_df = normalize_weather_columns(weather_df)
    conn.register("weather_df", weather_df)
    try:
        _execute(
            conn,
            """
            DELETE FROM weather_hourly
            WHERE EXISTS (
                SELECT 1 FROM weather_df w
                WHERE w.route_id = weather_hourly.route_id
                  AND w.timestamp_local = weather_hourly.timestamp_local
            )
            """,
        )
        _execute(
            conn,
            """
            INSERT INTO weather_hourly (
                route_id, timestamp_local, precipitation, rain, weathercode, windspeed_10m,
                source, latitude, longitude, timezone, fetched_at_utc
            )
            SELECT route_id, timestamp_local, precipitation, rain, weathercode, windspeed_10m,
                   source, latitude, longitude, timezone, fetched_at_utc
            FROM weather_df
            """,
        )
    finally:
        try:
            conn.unregister("weather_df")
        except Exception:
            pass
    return len(weather_df)


def _make_trip_id(route_id: str, departure_ts: pd.Timestamp, source: str) -> str:
    ts = pd.Timestamp(departure_ts).strftime("%Y%m%d%H%M")
    return f"{route_id}:{ts}:{source}"


def upsert_trips(conn: duckdb.DuckDBPyConnection, trips_df: pd.DataFrame) -> int:
    if trips_df.empty:
        return 0
    frame = trips_df.copy()
    frame["departure_ts"] = pd.to_datetime(frame["departure_ts"])
    frame["trip_id"] = frame.apply(
        lambda r: _make_trip_id(r["route_id"], r["departure_ts"], r.get("source", "unknown")), axis=1
    )
    frame["created_at"] = datetime.now(timezone.utc)
    cols = [
        "trip_id",
        "route_id",
        "departure_ts",
        "observed_duration_min",
        "source",
        "base_duration_min",
        "distance_km",
        "created_at",
    ]
    for col in cols:
        if col not in frame.columns:
            frame[col] = None
    frame = frame[cols]
    conn.register("trips_df", frame)
    try:
        _execute(
            conn,
            """
            DELETE FROM trip_observations
            WHERE trip_id IN (SELECT trip_id FROM trips_df)
            """,
        )
        _execute(conn, "INSERT INTO trip_observations SELECT * FROM trips_df")
    finally:
        try:
            conn.unregister("trips_df")
        except Exception:
            pass
    return len(frame)


def log_user_trip(
    route_id: str,
    departure_ts: datetime | str | pd.Timestamp,
    observed_duration_min: float,
    source: str = "user_log",
    db_path: Path = DUCKDB_PATH,
) -> str:
    """Log an actual commute duration. Used by UI or CLI to improve the model over time."""
    if route_id not in ROUTES:
        raise ValueError(f"Unknown route_id: {route_id}")
    departure_ts = pd.Timestamp(departure_ts)
    if pd.isna(departure_ts):
        raise ValueError("departure_ts is invalid")
    if observed_duration_min <= 0:
        raise ValueError("observed_duration_min must be positive")

    with get_connection(db_path) as conn:
        route_row = _fetchone(
            conn,
            "SELECT base_duration_min, distance_km FROM routes WHERE route_id = ?",
            [route_id],
        )
        base_duration_min = route_row[0] if route_row else None
        distance_km = route_row[1] if route_row else None
        frame = pd.DataFrame(
            [
                {
                    "route_id": route_id,
                    "departure_ts": departure_ts,
                    "observed_duration_min": float(observed_duration_min),
                    "source": source,
                    "base_duration_min": base_duration_min,
                    "distance_km": distance_km,
                }
            ]
        )
        upsert_trips(conn, frame)
        return _make_trip_id(route_id, departure_ts, source)


def run_pipeline(
    route_ids: list[str] | None = None,
    db_path: Path = DUCKDB_PATH,
    include_forecast: bool = True,
    bootstrap_days: int = 56,
) -> dict:
    """Run idempotent ingestion for selected routes."""
    selected_ids = route_ids or list(ROUTES.keys())
    selected_routes = [ROUTES[rid] for rid in selected_ids if rid in ROUTES]
    if not selected_routes:
        raise ValueError("No valid routes selected")

    summary = {"routes": [], "weather_rows": 0, "trip_rows": 0}
    with get_connection(db_path) as conn:
        baselines = []
        for route in selected_routes:
            baseline = estimate_baseline(route)
            baselines.append(baseline)
        upsert_routes(conn, baselines, ROUTES)

        for route, baseline in zip(selected_routes, baselines):
            try:
                weather_df = fetch_weather_for_route(
                    route_id=route.route_id,
                    latitude=route.origin_lat,
                    longitude=route.origin_lon,
                    include_forecast=include_forecast,
                )
            except Exception:
                weather_df = _fallback_weather_frame(route.route_id)
            weather_count = upsert_weather(conn, weather_df)

            existing_row = _fetchone(
                conn,
                "SELECT COUNT(*) FROM trip_observations WHERE route_id = ?",
                [route.route_id],
            )
            existing = int(existing_row[0]) if existing_row else 0
            trip_count = 0
            if existing == 0:
                trips_df = generate_synthetic_trips(
                    route=route,
                    weather_df=weather_df,
                    base_duration_min=float(baseline["base_duration_min"]),
                    days=bootstrap_days,
                )
                trip_count = upsert_trips(conn, trips_df)

            summary["routes"].append(
                {
                    "route_id": route.route_id,
                    "baseline_source": baseline["source"],
                    "base_duration_min": baseline["base_duration_min"],
                    "weather_rows_upserted": weather_count,
                    "trip_rows_inserted": trip_count,
                }
            )
            summary["weather_rows"] += weather_count
            summary["trip_rows"] += trip_count

    return summary


def _fetch_live_weather_row(route_id: str, lat: float, lon: float, departure_ts: pd.Timestamp) -> dict:
    """Fetch current-hour weather for live observation; fallback to zeros on failure."""
    try:
        df = fetch_weather_for_route(
            route_id=route_id,
            latitude=lat,
            longitude=lon,
            include_forecast=True,
        )
    except Exception:
        return {"precipitation": 0.0, "rain": 0.0, "weathercode": 0.0, "windspeed_10m": 0.0}

    if df.empty:
        return {"precipitation": 0.0, "rain": 0.0, "weathercode": 0.0, "windspeed_10m": 0.0}
    df = df.copy()
    df["timestamp_local"] = pd.to_datetime(df["timestamp_local"])
    hour_ts = departure_ts.floor("h")
    exact = df[df["timestamp_local"] == hour_ts]
    row = exact.iloc[-1] if not exact.empty else df.iloc[(df["timestamp_local"] - hour_ts).abs().argsort().iloc[0]]
    return {
        "precipitation": float(row.get("precipitation", 0.0)),
        "rain": float(row.get("rain", 0.0)),
        "weathercode": float(row.get("weathercode", 0.0)),
        "windspeed_10m": float(row.get("windspeed_10m", 0.0)),
    }


def collect_live_observation(
    route_id: str,
    db_path: Path = DUCKDB_PATH,
    depart_at: datetime | None = None,
) -> dict:
    """Fetch live weather + live traffic duration and upsert into DuckDB."""
    if route_id not in ROUTES:
        raise ValueError(f"Unknown route_id: {route_id}")

    route = ROUTES[route_id]
    departure_ts = pd.Timestamp(depart_at or datetime.now()).floor("min")
    weather_row = _fetch_live_weather_row(route_id, route.origin_lat, route.origin_lon, departure_ts)

    traffic = get_live_travel_duration(
        route,
        depart_at=departure_ts.to_pydatetime(),
        precipitation_mm=weather_row["precipitation"],
        rain_mm=weather_row["rain"],
    )

    with get_connection(db_path) as conn:
        weather_frame = pd.DataFrame(
            [
                {
                    "route_id": route_id,
                    "timestamp_local": departure_ts.floor("h"),
                    "precipitation": weather_row["precipitation"],
                    "rain": weather_row["rain"],
                    "weathercode": weather_row["weathercode"],
                    "windspeed_10m": weather_row["windspeed_10m"],
                    "source": "open_meteo_live",
                    "latitude": route.origin_lat,
                    "longitude": route.origin_lon,
                    "timezone": "auto",
                    "fetched_at_utc": pd.Timestamp.now("UTC"),
                }
            ]
        )
        upsert_weather(conn, weather_frame)

        route_row = _fetchone(
            conn,
            "SELECT base_duration_min, distance_km FROM routes WHERE route_id = ?",
            [route_id],
        )
        base_duration_min = route_row[0] if route_row else None
        distance_km = route_row[1] if route_row else traffic.get("distance_km")

        trip_frame = pd.DataFrame(
            [
                {
                    "route_id": route_id,
                    "departure_ts": departure_ts,
                    "observed_duration_min": float(traffic["travel_time_min"]),
                    "source": traffic.get("source", "synthetic"),
                    "base_duration_min": base_duration_min,
                    "distance_km": distance_km,
                }
            ]
        )
        upsert_trips(conn, trip_frame)

    return {
        "route_id": route_id,
        "departure_ts": departure_ts.isoformat(),
        "observed_duration_min": float(traffic["travel_time_min"]),
        "source": traffic.get("source", "synthetic"),
        "precipitation_mm": float(weather_row["precipitation"]),
        "rain_mm": float(weather_row["rain"]),
    }


def get_observation_counts(route_id: str | None = None, db_path: Path = DUCKDB_PATH) -> dict:
    with get_connection(db_path) as conn:
        params: list = []
        where = ""
        if route_id:
            where = "WHERE route_id = ?"
            params.append(route_id)
        rows = _fetchdf(
            conn,
            f"""
            SELECT
                SUM(CASE WHEN source IN ('real_api','tomtom','live_api') THEN 1 ELSE 0 END) AS real_api,
                SUM(CASE WHEN source = 'user_log' THEN 1 ELSE 0 END) AS user_log,
                SUM(CASE WHEN source IN ('synthetic','synthetic_bootstrap') THEN 1 ELSE 0 END) AS synthetic,
                COUNT(*) AS total
            FROM trip_observations
            {where}
            """,
            params,
        )
    if rows.empty:
        return {"real_api": 0, "user_log": 0, "synthetic": 0, "total": 0}
    r = rows.iloc[0].fillna(0)
    return {
        "real_api": int(r.get("real_api", 0)),
        "user_log": int(r.get("user_log", 0)),
        "synthetic": int(r.get("synthetic", 0)),
        "total": int(r.get("total", 0)),
    }


def read_table(table: str, db_path: Path = DUCKDB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return _fetchdf(conn, f"SELECT * FROM {table}")


if __name__ == "__main__":
    result = run_pipeline()
    print(result)
