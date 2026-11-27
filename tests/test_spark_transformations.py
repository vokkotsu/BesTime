"""Parity tests between pandas feature pipeline and PySpark Gold ETL transforms."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.config import MODEL_FEATURES
from src.features.build_features import add_time_features_for_inference
from src.ingestion.routing import rush_intensity

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from src.batch.spark_gold_etl import add_rolling_rain_features, add_time_and_interaction_features  # noqa: E402


@pytest.fixture(scope="session")
def spark():
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("commute-gold-etl-tests")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    yield spark
    spark.stop()


def _sample_departures() -> pd.DataFrame:
    base = datetime(2026, 1, 6, 6, 0)
    rows = []
    for i in range(20):
        ts = base + timedelta(minutes=15 * i)
        precip = [0.0, 0.2, 1.0, 2.5, 6.0, 0.0][i % 6]
        rain = precip * 0.8
        rows.append(
            {
                "route_id": "bsd_jakarta",
                "departure_ts": ts,
                "base_duration_min": 45.0,
                "precipitation_mm": float(precip),
                "rain_mm": float(rain),
                "weathercode": 61.0 if precip > 0 else 1.0,
                "windspeed_10m": 6.0 + precip,
                "observed_duration_min": 40.0 + i + precip * 2.0,
                "source": "test",
            }
        )
    return pd.DataFrame(rows)


def _pandas_reference_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_time_features_for_inference(frame.copy())
    # Match Spark/pandas build_features.py rolling definition used in training:
    # shift(1).rolling(window=12, min_periods=1).mean() per route
    out = out.sort_values(["route_id", "departure_ts"]).reset_index(drop=True)
    out["precip_rolling_3h"] = (
        out.groupby("route_id")["precipitation_mm"]
        .transform(lambda s: s.shift(1).rolling(window=12, min_periods=1).mean())
        .fillna(0.0)
    )
    out["rain_rolling_3h"] = (
        out.groupby("route_id")["rain_mm"]
        .transform(lambda s: s.shift(1).rolling(window=12, min_periods=1).mean())
        .fillna(0.0)
    )
    out["rain_x_rush"] = out["precipitation_mm"] * out["rush_intensity"]
    return out


def test_spark_time_and_interaction_features_match_pandas(spark):
    pdf = _sample_departures()
    ref = _pandas_reference_features(pdf)

    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("departure_ts_parsed", F.to_timestamp("departure_ts"))
    sdf = add_time_and_interaction_features(sdf)
    got = sdf.select(
        "departure_ts",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
        "is_weekday",
        "is_morning_rush",
        "is_evening_rush",
        "rush_intensity",
        "rain_x_rush",
        "is_raining",
        "heavy_rain",
    ).toPandas()
    got["departure_ts"] = pd.to_datetime(got["departure_ts"])
    got = got.sort_values("departure_ts").reset_index(drop=True)
    ref = ref.sort_values("departure_ts").reset_index(drop=True)

    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "rush_intensity", "rain_x_rush"]:
        np.testing.assert_allclose(got[col].to_numpy(dtype=float), ref[col].to_numpy(dtype=float), rtol=1e-6, atol=1e-6)
    for col in ["is_weekday", "is_morning_rush", "is_evening_rush", "is_raining", "heavy_rain"]:
        np.testing.assert_array_equal(got[col].astype(int).to_numpy(), ref[col].astype(int).to_numpy())


def test_spark_rolling_rain_3h_matches_pandas_shift_rolling(spark):
    pdf = _sample_departures()
    ref = _pandas_reference_features(pdf)

    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("departure_ts_parsed", F.to_timestamp("departure_ts"))
    sdf = add_rolling_rain_features(sdf)
    got = sdf.select("departure_ts", "precip_rolling_3h", "rain_rolling_3h").toPandas()
    got["departure_ts"] = pd.to_datetime(got["departure_ts"])
    got = got.sort_values("departure_ts").reset_index(drop=True)
    ref = ref.sort_values("departure_ts").reset_index(drop=True)

    np.testing.assert_allclose(
        got["precip_rolling_3h"].to_numpy(dtype=float),
        ref["precip_rolling_3h"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        got["rain_rolling_3h"].to_numpy(dtype=float),
        ref["rain_rolling_3h"].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )
    # Leakage guard: first row has no prior rows.
    assert got.loc[0, "precip_rolling_3h"] == 0.0
    assert got.loc[0, "rain_rolling_3h"] == 0.0


def test_rush_intensity_parity_spot_checks():
    for hour in [5.5, 7.25, 8.0, 12.0, 17.75, 21.0]:
        # Spark expression parity is covered above; keep a direct sanity check on reference.
        val = rush_intensity(hour)
        assert 0.0 <= val <= 1.2


def test_model_feature_columns_present_after_gold_transform(spark):
    pdf = _sample_departures()
    sdf = spark.createDataFrame(pdf)
    sdf = sdf.withColumn("departure_ts_parsed", F.to_timestamp("departure_ts"))
    sdf = add_time_and_interaction_features(sdf)
    sdf = add_rolling_rain_features(sdf)
    cols = set(sdf.columns)
    missing = [c for c in MODEL_FEATURES if c not in cols]
    assert not missing, f"Missing MODEL_FEATURES in Spark transform: {missing}"