"""Continuous live traffic/weather collector for commute observations."""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from src.config import COLLECTOR_ACTIVE_END_HOUR, COLLECTOR_ACTIVE_START_HOUR, ROUTES
from src.ingestion.pipeline import collect_live_observation


def _in_active_window(now: datetime, start_hour: int, end_hour: int) -> bool:
    return start_hour <= now.hour < end_hour


def collect_once(route_ids: list[str] | None = None) -> list[dict]:
    selected = route_ids or list(ROUTES.keys())
    results = []
    for route_id in selected:
        try:
            result = collect_live_observation(route_id)
            results.append(result)
            print(
                f"[collector] {result['departure_ts']} {route_id} "
                f"{result['observed_duration_min']:.1f}min source={result['source']}"
            )
        except Exception as exc:
            print(f"[collector] ERROR {route_id}: {exc}")
    return results


def run_collector(
    interval_minutes: int = 15,
    route_ids: list[str] | None = None,
    once: bool = False,
    active_start_hour: int = COLLECTOR_ACTIVE_START_HOUR,
    active_end_hour: int = COLLECTOR_ACTIVE_END_HOUR,
) -> None:
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be > 0")
    selected = route_ids or list(ROUTES.keys())

    print(
        f"[collector] starting routes={selected} interval={interval_minutes}m "
        f"active_window={active_start_hour:02d}:00-{active_end_hour:02d}:00"
    )

    while True:
        now = datetime.now()
        if _in_active_window(now, active_start_hour, active_end_hour):
            collect_once(selected)
        else:
            print(f"[collector] {now.isoformat()} outside active window; skipping")

        if once:
            break
        time.sleep(interval_minutes * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll live traffic + weather into DuckDB")
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--route-id", action="append", dest="route_ids")
    parser.add_argument("--once", action="store_true", help="Collect one cycle and exit")
    parser.add_argument("--start-hour", type=int, default=COLLECTOR_ACTIVE_START_HOUR)
    parser.add_argument("--end-hour", type=int, default=COLLECTOR_ACTIVE_END_HOUR)
    args = parser.parse_args()
    run_collector(
        interval_minutes=args.interval_minutes,
        route_ids=args.route_ids,
        once=args.once,
        active_start_hour=args.start_hour,
        active_end_hour=args.end_hour,
    )


if __name__ == "__main__":
    main()
