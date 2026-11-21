"""Streamlit dashboard for Commute Departure & Delay Risk Optimizer."""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Ensure project root is importable when running `streamlit run src/app/dashboard.py`
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_ROUTE_ID, DUCKDB_PATH, ROUTES  # noqa: E402
from src.ingestion.pipeline import get_connection, log_user_trip, run_pipeline  # noqa: E402
from src.models.predict import predict_departure_curve, summarize_zones  # noqa: E402
from src.models.train import load_metrics, train_route_models  # noqa: E402


st.set_page_config(page_title="Commute Departure & Delay Risk Optimizer", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def cached_curve(route_id: str, target_date: date, db_mtime: float) -> pd.DataFrame:
    # db_mtime is included to invalidate cache when DB changes.
    return predict_departure_curve(route_id=route_id, target_date=target_date, db_path=DUCKDB_PATH)


def db_mtime() -> float:
    return DUCKDB_PATH.stat().st_mtime if DUCKDB_PATH.exists() else 0.0


def ensure_data(route_id: str) -> None:
    if not DUCKDB_PATH.exists():
        with st.spinner("Bootstrapping data (weather + synthetic trips)..."):
            run_pipeline(route_ids=[route_id])
            train_route_models(route_id)
        return

    with get_connection(DUCKDB_PATH) as conn:
        route_exists_cur = conn.execute("SELECT 1 FROM routes WHERE route_id = ?", [route_id])
        route_exists = route_exists_cur.fetchone()
        route_exists_cur.fetchall()
        trip_count_cur = conn.execute(
            "SELECT COUNT(*) FROM trip_observations WHERE route_id = ?", [route_id]
        )
        trip_count_row = trip_count_cur.fetchone()
        trip_count_cur.fetchall()
        trip_count = int(trip_count_row[0]) if trip_count_row else 0
    if not route_exists or trip_count == 0:
        with st.spinner("Preparing route data and model..."):
            run_pipeline(route_ids=[route_id])
            train_route_models(route_id)


def apply_weather_scenario(curve: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Apply a simple what-if weather adjustment to predicted durations.

    This keeps the MVP interactive even when forecast data is dry. The adjustment mirrors
    the model's rain/rush interaction intuition; for production, retrain/predict with scenario features.
    """
    out = curve.copy()
    if scenario == "Forecast":
        return out

    multipliers = {
        "Dry": 0.95,
        "Light rain": 1.08,
        "Heavy rain": 1.20,
    }
    mult = multipliers.get(scenario, 1.0)
    rush = out["rush_intensity"].fillna(0.0)
    rain_boost = 1.0 + (mult - 1.0) * (0.6 + 0.8 * rush)
    out["duration_p50_min"] = out["duration_p50_min"] * rain_boost
    out["duration_p90_min"] = out["duration_p90_min"] * (rain_boost + 0.03 * rush)
    out["duration_p90_min"] = np_max(out["duration_p90_min"], out["duration_p50_min"] + 1.0)
    out["arrival_p50_ts"] = out["departure_ts"] + pd.to_timedelta(out["duration_p50_min"], unit="m")
    out["arrival_p90_ts"] = out["departure_ts"] + pd.to_timedelta(out["duration_p90_min"], unit="m")
    out["risk_spread_min"] = out["duration_p90_min"] - out["duration_p50_min"]
    from src.models.predict import detect_zones

    return detect_zones(out)


def np_max(a, b):
    import numpy as np

    return np.maximum(a, b)


def zone_color(zone: str) -> str:
    return {
        "normal": "#4C78A8",
        "diminishing_return": "#F58518",
        "delay_cliff": "#E45756",
        "high_tail_risk": "#B279A2",
    }.get(zone, "#4C78A8")


def build_chart(curve: pd.DataFrame) -> alt.Chart:
    plot_df = curve.copy()
    plot_df["departure_label"] = plot_df["departure_ts"].dt.strftime("%H:%M")
    plot_df["arrival_p50_label"] = plot_df["arrival_p50_ts"].dt.strftime("%H:%M")
    plot_df["arrival_p90_label"] = plot_df["arrival_p90_ts"].dt.strftime("%H:%M")

    base = alt.Chart(plot_df).encode(
        x=alt.X("departure_ts:T", title="Departure time"),
        tooltip=[
            alt.Tooltip("departure_label:N", title="Depart"),
            alt.Tooltip("duration_p50_min:Q", title="P50 min", format=".1f"),
            alt.Tooltip("duration_p90_min:Q", title="P90 min", format=".1f"),
            alt.Tooltip("arrival_p50_label:N", title="Arrive P50"),
            alt.Tooltip("arrival_p90_label:N", title="Arrive P90"),
            alt.Tooltip("zone:N", title="Zone"),
            alt.Tooltip("zone_reason:N", title="Reason"),
        ],
    )

    band = base.mark_area(opacity=0.2, color="#72B7B2").encode(
        y=alt.Y("duration_p50_min:Q", title="Estimated travel duration (minutes)"),
        y2="duration_p90_min:Q",
    )
    p50 = base.mark_line(color="#1f77b4", strokeWidth=2).encode(y="duration_p50_min:Q")
    p90 = base.mark_line(color="#2ca02c", strokeDash=[4, 4], strokeWidth=2).encode(y="duration_p90_min:Q")

    points = base.mark_circle(size=55).encode(
        y="duration_p50_min:Q",
        color=alt.Color(
            "zone:N",
            scale=alt.Scale(
                domain=["normal", "diminishing_return", "delay_cliff", "high_tail_risk"],
                range=["#4C78A8", "#F58518", "#E45756", "#B279A2"],
            ),
            legend=alt.Legend(title="Zone"),
        ),
    )
    return (band + p50 + p90 + points).properties(height=420).interactive()


def render_alerts(curve: pd.DataFrame) -> None:
    cliffs = curve[curve["zone"] == "delay_cliff"]
    diminishing = curve[curve["zone"] == "diminishing_return"]
    tail = curve[curve["zone"] == "high_tail_risk"]

    if not cliffs.empty:
        first = cliffs.iloc[0]
        st.error(
            f"Delay cliff detected around {first['departure_ts'].strftime('%H:%M')}: "
            f"P50 duration jumps to {first['duration_p50_min']:.0f} min. {first['zone_reason']}"
        )
    if not diminishing.empty:
        sample = diminishing.iloc[0]
        st.warning(
            f"Diminishing returns: {sample['departure_ts'].strftime('%H:%M')} shows limited benefit from leaving earlier. "
            f"{sample['zone_reason']}"
        )
    if not tail.empty:
        st.info(
            f"High tail-risk slots: {len(tail)} intervals have elevated P90 spread. "
            "Consider buffer time if you must depart in these windows."
        )
    if cliffs.empty and diminishing.empty and tail.empty:
        st.success("No major delay cliffs or diminishing-return zones detected for this scenario.")


def main() -> None:
    st.title("Commute Departure & Delay Risk Optimizer")
    st.caption(
        "P50/P90 departure curve with diminishing-return and delay-cliff detection. "
        "Uses Open-Meteo weather, OSRM baseline (or fallback), and DuckDB-backed trip logs."
    )

    with st.sidebar:
        st.header("Scenario")
        route_id = st.selectbox(
            "Route",
            options=list(ROUTES.keys()),
            format_func=lambda rid: ROUTES[rid].name,
            index=list(ROUTES.keys()).index(DEFAULT_ROUTE_ID),
        )
        target_date = st.date_input("Date", value=date.today())
        scenario = st.selectbox("Weather scenario", ["Forecast", "Dry", "Light rain", "Heavy rain"])
        interval = st.select_slider("Interval (minutes)", options=[10, 15, 20, 30], value=15)

        st.divider()
        st.header("Data / Model")
        if st.button("Refresh data pipeline", use_container_width=True):
            with st.spinner("Running ingestion pipeline..."):
                result = run_pipeline(route_ids=[route_id])
            st.success(f"Ingested weather rows: {result['weather_rows']}, trips: {result['trip_rows']}")
            st.cache_data.clear()

        if st.button("Train / retrain model", use_container_width=True):
            with st.spinner("Training quantile models..."):
                res = train_route_models(route_id)
            st.success(f"Trained {route_id}: {int(res.rows)} rows")
            st.cache_data.clear()

        metrics = load_metrics(route_id)
        if metrics:
            st.caption("Model metrics")
            st.json({k: round(v, 3) for k, v in metrics.items() if isinstance(v, (int, float))})

    ensure_data(route_id)
    curve = cached_curve(route_id, target_date, db_mtime())
    if interval != 15:
        # simple resample by selecting nearest matching interval from 15-min grid
        curve = curve[curve["departure_ts"].dt.minute % interval == 0].copy()
    curve = apply_weather_scenario(curve, scenario)

    summary = summarize_zones(curve)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Delay cliffs", summary.get("delay_cliff_count", 0))
    c2.metric("Diminishing returns", summary.get("diminishing_return_count", 0))
    c3.metric("High tail-risk slots", summary.get("high_tail_risk_count", 0))
    best = curve.loc[curve["duration_p50_min"].idxmin()]
    c4.metric("Best P50 departure", best["departure_ts"].strftime("%H:%M"), f"{best['duration_p50_min']:.0f} min")

    st.subheader("Departure vs. Estimated Duration (P50/P90)")
    st.altair_chart(build_chart(curve), use_container_width=True)
    render_alerts(curve)

    with st.expander("Departure table", expanded=False):
        display = curve[
            [
                "departure_ts",
                "duration_p50_min",
                "duration_p90_min",
                "arrival_p50_ts",
                "arrival_p90_ts",
                "risk_spread_min",
                "precipitation_mm",
                "zone",
                "zone_reason",
            ]
        ].copy()
        for col in ["departure_ts", "arrival_p50_ts", "arrival_p90_ts"]:
            display[col] = display[col].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Log actual commute")
    st.caption("Logging real trips improves future model quality. Entries are stored in DuckDB.")
    with st.form("log_trip"):
        log_date = st.date_input("Trip date", value=date.today(), key="log_date")
        log_time = st.time_input("Departure time", value=datetime.now().time().replace(second=0, microsecond=0))
        duration = st.number_input("Observed duration (minutes)", min_value=1.0, max_value=300.0, value=60.0, step=1.0)
        submitted = st.form_submit_button("Save trip")
        if submitted:
            departure_ts = datetime.combine(log_date, log_time)
            trip_id = log_user_trip(route_id=route_id, departure_ts=departure_ts, observed_duration_min=duration)
            st.success(f"Saved trip {trip_id}. Retrain model to incorporate it.")
            st.cache_data.clear()


if __name__ == "__main__":
    main()
