"""Kafka producers for commute traffic and weather telemetry.

Publishes JSON events to:
- commute.traffic.raw
- commute.weather.raw

Supports:
- --mode live      : poll TomTom/Open-Meteo (or simulator fallback) and publish
- --mode replay    : replay historical/synthetic observations from DuckDB
- --mode synthetic : generate synthetic events without external APIs

Examples:
  python -m src.streaming.kafka_producer --mode live --route-id bsd_jakarta --interval-seconds 60
  python -m src.streaming.kafka_producer --mode replay --route-id bsd_jakarta --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from src.config import DEFAULT_ROUTE_ID, DUCKDB_PATH, ROUTES
from src.ingestion.pipeline import collect_live_observation, get_connection

LOGGER = logging.getLogger("commute.kafka_producer")

TRAFFIC_TOPIC_DEFAULT = "commute.traffic.raw"
WEATHER_TOPIC_DEFAULT = "commute.weather.raw"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_serializer(value: Any) -> bytes:
    return json.dumps(value, default=str, separators=(",", ":")).encode("utf-8")


def build_kafka_producer(bootstrap_servers: str):
    """Create a KafkaProducer with lazy import so module import stays lightweight."""
    try:
        from kafka import KafkaProducer  # type: ignore
    except Exception as exc:  # pragma: no cover - exercised in integration env
        raise RuntimeError(
            "kafka-python is required for Kafka publishing. Install with `pip install kafka-python`."
        ) from exc

    return KafkaProducer(
        bootstrap_servers=[s.strip() for s in bootstrap_servers.split(",") if s.strip()],
        value_serializer=_json_serializer,
        key_serializer=lambda k: k.encode("utf-8") if isinstance(k, str) else k,
        linger_ms=50,
        retries=5,
        acks="all",
        client_id="commute-producer",
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def build_traffic_event(
    *,
    route_id: str,
    departure_ts: datetime,
    observed_duration_min: float,
    base_duration_min: float | None,
    distance_km: float | None,
    source: str,
    precipitation_mm: float = 0.0,
    rain_mm: float = 0.0,
    weathercode: float = 0.0,
    windspeed_10m: float = 0.0,
) -> dict[str, Any]:
    route = ROUTES[route_id]
    event_ts = departure_ts.astimezone(timezone.utc) if departure_ts.tzinfo else departure_ts.replace(tzinfo=timezone.utc)
    return {
        "event_id": str(uuid.uuid4()),
        "event_ts": event_ts.isoformat(),
        "route_id": route_id,
        "origin": route.origin_label,
        "destination": route.destination_label,
        "origin_lat": float(route.origin_lat),
        "origin_lon": float(route.origin_lon),
        "dest_lat": float(route.dest_lat),
        "dest_lon": float(route.dest_lon),
        "departure_ts": departure_ts.isoformat(),
        "observed_duration_min": _safe_float(observed_duration_min),
        "base_duration_min": _safe_float(base_duration_min, default=_safe_float(observed_duration_min)),
        "distance_km": _safe_float(distance_km, default=0.0),
        "source": str(source),
        "precipitation_mm": _safe_float(precipitation_mm),
        "rain_mm": _safe_float(rain_mm),
        "weathercode": _safe_float(weathercode),
        "windspeed_10m": _safe_float(windspeed_10m),
        "ingested_at_utc": _utc_now_iso(),
        "schema_version": 1,
    }


def build_weather_event(
    *,
    route_id: str,
    timestamp_local: datetime,
    precipitation_mm: float,
    rain_mm: float,
    weathercode: float,
    windspeed_10m: float,
    source: str,
) -> dict[str, Any]:
    route = ROUTES[route_id]
    event_ts = (
        timestamp_local.astimezone(timezone.utc)
        if timestamp_local.tzinfo
        else timestamp_local.replace(tzinfo=timezone.utc)
    )
    return {
        "event_id": str(uuid.uuid4()),
        "event_ts": event_ts.isoformat(),
        "route_id": route_id,
        "timestamp_local": timestamp_local.isoformat(),
        "precipitation_mm": _safe_float(precipitation_mm),
        "rain_mm": _safe_float(rain_mm),
        "weathercode": _safe_float(weathercode),
        "windspeed_10m": _safe_float(windspeed_10m),
        "source": str(source),
        "latitude": float(route.origin_lat),
        "longitude": float(route.origin_lon),
        "ingested_at_utc": _utc_now_iso(),
        "schema_version": 1,
    }


def publish_event(producer: Any, topic: str, event: dict[str, Any], key: str | None = None) -> None:
    producer.send(topic, key=key or event.get("route_id"), value=event)


def _query_replay_rows(route_id: str, limit: int) -> list[dict[str, Any]]:
    sql = """
        SELECT
            t.route_id,
            t.departure_ts,
            t.observed_duration_min,
            t.source,
            COALESCE(t.base_duration_min, r.base_duration_min) AS base_duration_min,
            COALESCE(t.distance_km, r.distance_km) AS distance_km,
            w.precipitation AS precipitation_mm,
            w.rain AS rain_mm,
            w.weathercode,
            w.windspeed_10m,
            w.timestamp_local AS weather_ts
        FROM trip_observations t
        LEFT JOIN routes r ON t.route_id = r.route_id
        LEFT JOIN weather_hourly w
          ON t.route_id = w.route_id
         AND date_trunc('hour', t.departure_ts) = w.timestamp_local
        WHERE t.route_id = ?
        ORDER BY t.departure_ts DESC
        LIMIT ?
    """
    with get_connection(DUCKDB_PATH) as conn:
        df = conn.execute(sql, [route_id, limit]).df()
    if df.empty:
        return []
    df = df.sort_values("departure_ts")
    return df.to_dict(orient="records")


def _synthetic_row(route_id: str, depart_at: datetime, rng: random.Random) -> dict[str, Any]:
    # Lightweight synthetic generator; for production-like sim use routing simulator in live mode.
    rush = 1.0 if depart_at.hour in {6, 7, 8, 17, 18, 19} else 0.2
    rain = 6.0 if rng.random() < (0.25 * rush + 0.05) else (1.5 if rng.random() < 0.2 else 0.0)
    base = 38.0 + (5.0 if route_id == "bekasi_jakarta" else 0.0)
    duration = base * (1.0 + 0.35 * rush) + rain * (1.5 + 1.2 * rush) + rng.uniform(-2.0, 4.0)
    return {
        "route_id": route_id,
        "departure_ts": depart_at,
        "observed_duration_min": max(15.0, duration),
        "source": "synthetic_stream",
        "base_duration_min": base,
        "distance_km": 28.0,
        "precipitation_mm": rain,
        "rain_mm": rain,
        "weathercode": 61.0 if rain > 0 else 1.0,
        "windspeed_10m": 8.0 + rain,
        "weather_ts": depart_at.replace(minute=0, second=0, microsecond=0),
    }


def iter_events(
    *,
    mode: str,
    route_ids: list[str],
    limit: int,
    interval_seconds: int,
) -> Iterable[tuple[dict[str, Any], dict[str, Any] | None]]:
    """Yield (traffic_event, weather_event_or_none)."""
    rng = random.Random(42)
    if mode == "replay":
        for route_id in route_ids:
            rows = _query_replay_rows(route_id, limit=limit)
            for row in rows:
                dep = pd_to_datetime(row["departure_ts"])
                traffic = build_traffic_event(
                    route_id=route_id,
                    departure_ts=dep,
                    observed_duration_min=row.get("observed_duration_min", 0.0),
                    base_duration_min=row.get("base_duration_min"),
                    distance_km=row.get("distance_km"),
                    source=row.get("source", "replay"),
                    precipitation_mm=row.get("precipitation_mm", 0.0) or 0.0,
                    rain_mm=row.get("rain_mm", 0.0) or 0.0,
                    weathercode=row.get("weathercode", 0.0) or 0.0,
                    windspeed_10m=row.get("windspeed_10m", 0.0) or 0.0,
                )
                weather = None
                if row.get("weather_ts") is not None:
                    wts = pd_to_datetime(row["weather_ts"])
                    weather = build_weather_event(
                        route_id=route_id,
                        timestamp_local=wts,
                        precipitation_mm=row.get("precipitation_mm", 0.0) or 0.0,
                        rain_mm=row.get("rain_mm", 0.0) or 0.0,
                        weathercode=row.get("weathercode", 0.0) or 0.0,
                        windspeed_10m=row.get("windspeed_10m", 0.0) or 0.0,
                        source="replay_weather",
                    )
                yield traffic, weather
        return

    if mode == "synthetic":
        for i in range(limit):
            route_id = route_ids[i % len(route_ids)]
            depart_at = datetime.now() - timedelta(minutes=15 * (limit - i))
            row = _synthetic_row(route_id, depart_at, rng)
            traffic = build_traffic_event(
                route_id=route_id,
                departure_ts=row["departure_ts"],
                observed_duration_min=row["observed_duration_min"],
                base_duration_min=row["base_duration_min"],
                distance_km=row["distance_km"],
                source=row["source"],
                precipitation_mm=row["precipitation_mm"],
                rain_mm=row["rain_mm"],
                weathercode=row["weathercode"],
                windspeed_10m=row["windspeed_10m"],
            )
            weather = build_weather_event(
                route_id=route_id,
                timestamp_local=row["weather_ts"],
                precipitation_mm=row["precipitation_mm"],
                rain_mm=row["rain_mm"],
                weathercode=row["weathercode"],
                windspeed_10m=row["windspeed_10m"],
                source="synthetic_weather",
            )
            yield traffic, weather
        return

    if mode == "live":
        while True:
            for route_id in route_ids:
                obs = collect_live_observation(route_id)
                dep = pd_to_datetime(obs["departure_ts"])
                traffic = build_traffic_event(
                    route_id=route_id,
                    departure_ts=dep,
                    observed_duration_min=obs["observed_duration_min"],
                    base_duration_min=None,
                    distance_km=None,
                    source=obs.get("source", "live_api"),
                    precipitation_mm=obs.get("precipitation_mm", 0.0),
                    rain_mm=obs.get("rain_mm", 0.0),
                )
                weather = build_weather_event(
                    route_id=route_id,
                    timestamp_local=dep.replace(minute=0, second=0, microsecond=0),
                    precipitation_mm=obs.get("precipitation_mm", 0.0),
                    rain_mm=obs.get("rain_mm", 0.0),
                    weathercode=0.0,
                    windspeed_10m=0.0,
                    source="open_meteo_live",
                )
                yield traffic, weather
            time.sleep(max(1, interval_seconds))
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def pd_to_datetime(value: Any) -> datetime:
    """Convert pandas.Timestamp/datetime/string to datetime."""
    if isinstance(value, datetime):
        return value
    try:
        import pandas as pd  # local import to keep module import cheap

        ts = pd.Timestamp(value)
        if pd.isna(ts):
            raise ValueError("NaT")
        return ts.to_pydatetime()
    except Exception as exc:
        raise ValueError(f"Invalid datetime value: {value!r}") from exc


def run_producer(args: argparse.Namespace) -> int:
    route_ids = args.route_ids or [args.route_id]
    for rid in route_ids:
        if rid not in ROUTES:
            raise ValueError(f"Unknown route_id: {rid}")

    producer = build_kafka_producer(args.bootstrap_servers)
    sent_traffic = 0
    sent_weather = 0
    try:
        for traffic_event, weather_event in iter_events(
            mode=args.mode,
            route_ids=route_ids,
            limit=args.limit,
            interval_seconds=args.interval_seconds,
        ):
            publish_event(producer, args.traffic_topic, traffic_event, key=traffic_event["route_id"])
            sent_traffic += 1
            if weather_event is not None:
                publish_event(producer, args.weather_topic, weather_event, key=weather_event["route_id"])
                sent_weather += 1

            if sent_traffic % 25 == 0 or args.mode != "replay":
                producer.flush()
                LOGGER.info(
                    "Published events traffic=%s weather=%s topic_traffic=%s topic_weather=%s",
                    sent_traffic,
                    sent_weather,
                    args.traffic_topic,
                    args.weather_topic,
                )

            if args.mode in {"replay", "synthetic"} and sent_traffic >= args.limit * len(route_ids):
                break
    finally:
        producer.flush(timeout=10)
        producer.close(timeout=10)

    LOGGER.info("Done. traffic=%s weather=%s", sent_traffic, sent_weather)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish commute telemetry events to Kafka")
    parser.add_argument("--bootstrap-servers", default="localhost:29092", help="Kafka bootstrap servers")
    parser.add_argument("--traffic-topic", default=TRAFFIC_TOPIC_DEFAULT)
    parser.add_argument("--weather-topic", default=WEATHER_TOPIC_DEFAULT)
    parser.add_argument("--mode", choices=["live", "replay", "synthetic"], default="replay")
    parser.add_argument("--route-id", default=DEFAULT_ROUTE_ID)
    parser.add_argument("--route-ids", nargs="*", default=None, help="Optional multiple route IDs")
    parser.add_argument("--limit", type=int, default=200, help="Per-route limit for replay/synthetic")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Polling interval for live mode")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return run_producer(args)


if __name__ == "__main__":
    raise SystemExit(main())
