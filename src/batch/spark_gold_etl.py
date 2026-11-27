"""PySpark batch ETL: Silver -> Gold feature store -> DuckDB analytical snapshot.

Reads:
- Silver enriched commute events: <hdfs-base>/silver/commute_enriched
- Silver weather rolling aggregates: <hdfs-base>/silver/weather_rolling_3h (optional enrichment/audit)

Writes:
- Gold feature store Parquet: <hdfs-base>/gold/feature_store partitioned by route_id,event_date
- DuckDB snapshot: routes, weather_hourly, trip_observations, gold_features (upserts)

Feature parity target: src/features/build_features.py + src/config.py MODEL_FEATURES.

Examples:
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 \
    src/batch/spark_gold_etl.py --hdfs-base hdfs://namenode:9000/datalake

  python -m src.batch.spark_gold_etl --local-mode --hdfs-base ./data/lakehouse
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Iterable

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

# Ensure project root is importable when run as spark-submit script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DUCKDB_PATH, MODEL_FEATURES, RUSH_WINDOWS, TARGET_COLUMN  # noqa: E402

LOGGER = logging.getLogger("commute.spark_gold_etl")

GOLD_SCHEMA = T.StructType(
    [
        T.StructField("trip_id", T.StringType(), False),
        T.StructField("route_id", T.StringType(), False),
        T.StructField("departure_ts", T.TimestampType(), False),
        T.StructField("event_date", T.DateType(), False),
        T.StructField("observed_duration_min", T.DoubleType(), False),
        T.StructField("source", T.StringType(), True),
        T.StructField("base_duration_min", T.DoubleType(), True),
        T.StructField("distance_km", T.DoubleType(), True),
        T.StructField("precipitation_mm", T.DoubleType(), True),
        T.StructField("rain_mm", T.DoubleType(), True),
        T.StructField("weathercode", T.DoubleType(), True),
        T.StructField("windspeed_10m", T.DoubleType(), True),
        T.StructField("departure_hour", T.IntegerType(), True),
        T.StructField("departure_minute", T.IntegerType(), True),
        T.StructField("hour_sin", T.DoubleType(), True),
        T.StructField("hour_cos", T.DoubleType(), True),
        T.StructField("dow_sin", T.DoubleType(), True),
        T.StructField("dow_cos", T.DoubleType(), True),
        T.StructField("is_weekday", T.IntegerType(), True),
        T.StructField("is_morning_rush", T.IntegerType(), True),
        T.StructField("is_evening_rush", T.IntegerType(), True),
        T.StructField("rush_intensity", T.DoubleType(), True),
        T.StructField("precip_rolling_3h", T.DoubleType(), True),
        T.StructField("rain_rolling_3h", T.DoubleType(), True),
        T.StructField("is_raining", T.IntegerType(), True),
        T.StructField("heavy_rain", T.IntegerType(), True),
        T.StructField("rain_x_rush", T.DoubleType(), True),
    ]
)


def build_spark(app_name: str, local_mode: bool) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if local_mode:
        builder = builder.master("local[*]")
    builder = (
        builder.config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _safe_col(df: DataFrame, name: str, default: str | float | int | None = None, dtype: str = "double") -> F.Column:
    if name in df.columns:
        return F.col(name).cast(dtype)
    return F.lit(default).cast(dtype)


def read_silver_enriched(spark: SparkSession, path: str) -> DataFrame:
    df = spark.read.parquet(path)
    # Normalize expected columns from stream processor.
    if "event_ts_parsed" in df.columns and "departure_ts_parsed" not in df.columns:
        df = df.withColumn("departure_ts_parsed", F.col("event_ts_parsed"))
    if "departure_ts_parsed" not in df.columns and "departure_ts" in df.columns:
        df = df.withColumn("departure_ts_parsed", F.to_timestamp("departure_ts"))
    if "event_date" not in df.columns:
        df = df.withColumn("event_date", F.to_date(F.col("departure_ts_parsed")))

    # Ensure observed/base/distance columns exist.
    df = df.withColumn("observed_duration_min", _safe_col(df, "observed_duration_min", dtype="double"))
    df = df.withColumn("base_duration_min", _safe_col(df, "base_duration_min", dtype="double"))
    df = df.withColumn("distance_km", _safe_col(df, "distance_km", dtype="double"))

    # Weather columns may be present directly or from enrichment names.
    for col, aliases in {
        "precipitation_mm": ["precipitation_mm", "precipitation"],
        "rain_mm": ["rain_mm", "rain"],
        "weathercode": ["weathercode"],
        "windspeed_10m": ["windspeed_10m"],
    }.items():
        found = next((a for a in aliases if a in df.columns), None)
        if found and found != col:
            df = df.withColumn(col, F.col(found).cast("double"))
        elif not found:
            df = df.withColumn(col, F.lit(0.0).cast("double"))
        else:
            df = df.withColumn(col, F.coalesce(F.col(col).cast("double"), F.lit(0.0)))

    if "source" not in df.columns:
        df = df.withColumn("source", F.lit("stream_silver"))
    df = df.withColumn("source", F.coalesce(F.col("source").cast("string"), F.lit("stream_silver")))

    # Required identity columns.
    df = df.filter(F.col("route_id").isNotNull() & F.col("departure_ts_parsed").isNotNull())
    df = df.filter(F.col("observed_duration_min").isNotNull() & (F.col("observed_duration_min") > 0))
    return df


def read_weather_rolling(spark: SparkSession, path: str) -> DataFrame | None:
    try:
        df = spark.read.parquet(path)
    except Exception as exc:
        LOGGER.warning("Weather rolling silver not available at %s (%s)", path, exc)
        return None
    # Expected columns: route_id, window_start, window_end, rolling_precip_3h_mm, rolling_rain_3h_mm
    needed = {"route_id", "window_end"}
    if not needed.issubset(set(df.columns)):
        LOGGER.warning("Weather rolling silver missing expected columns: %s", df.columns)
        return None
    return df


def make_trip_id(df: DataFrame) -> DataFrame:
    if "trip_id" in df.columns:
        return df.withColumn("trip_id", F.col("trip_id").cast("string"))
    # Match pipeline.py: route_id:YYYYMMDDHHMM:source
    ts_key = F.date_format(F.col("departure_ts_parsed"), "yyyyMMddHHmm")
    return df.withColumn("trip_id", F.concat_ws(":", F.col("route_id"), ts_key, F.col("source")))


def rush_intensity_expr(hour_decimal_col: str = "hour_decimal") -> F.Column:
    """PySpark expression mirroring src.ingestion.routing.rush_intensity."""
    h = F.col(hour_decimal_col).cast("double")
    intensity = F.lit(0.0)
    for w in RUSH_WINDOWS:
        start = float(w["start_hour"])
        peak = float(w["peak_hour"])
        end = float(w["end_hour"])
        amp = float(w["amplitude"])
        in_window = (h >= F.lit(start)) & (h <= F.lit(end))
        rising = h <= F.lit(peak)
        x_rise = (h - F.lit(start)) / F.lit(max(peak - start, 1e-6))
        x_fall = (F.lit(end) - h) / F.lit(max(end - peak, 1e-6))
        x = F.when(rising, x_rise).otherwise(x_fall)
        contrib = F.when(in_window, F.lit(amp) * F.pow(F.greatest(x, F.lit(0.0)), F.lit(1.3))).otherwise(F.lit(0.0))
        intensity = intensity + contrib
    return F.least(F.lit(1.2), F.greatest(F.lit(0.0), intensity))


def add_time_and_interaction_features(df: DataFrame) -> DataFrame:
    out = df
    out = out.withColumn("departure_hour", F.hour("departure_ts_parsed").cast("int"))
    out = out.withColumn("departure_minute", F.minute("departure_ts_parsed").cast("int"))
    out = out.withColumn("hour_decimal", F.col("departure_hour") + F.col("departure_minute") / F.lit(60.0))
    out = out.withColumn("day_of_week", F.dayofweek("departure_ts_parsed") - F.lit(2))  # Spark: 1=Sun..7=Sat ->  -1=Sun? easier use date_format
    # Use pandas-compatible dayofweek: Monday=0..Sunday=6
    out = out.withColumn("day_of_week", ((F.dayofweek("departure_ts_parsed") + F.lit(5)) % F.lit(7)).cast("int"))
    out = out.withColumn("is_weekday", (F.col("day_of_week") < F.lit(5)).cast("int"))

    out = out.withColumn("hour_sin", F.sin(F.lit(2 * math.pi) * F.col("hour_decimal") / F.lit(24.0)))
    out = out.withColumn("hour_cos", F.cos(F.lit(2 * math.pi) * F.col("hour_decimal") / F.lit(24.0)))
    out = out.withColumn("dow_sin", F.sin(F.lit(2 * math.pi) * F.col("day_of_week") / F.lit(7.0)))
    out = out.withColumn("dow_cos", F.cos(F.lit(2 * math.pi) * F.col("day_of_week") / F.lit(7.0)))

    morning = RUSH_WINDOWS[0]
    evening = RUSH_WINDOWS[1]
    out = out.withColumn(
        "is_morning_rush",
        ((F.col("hour_decimal") >= F.lit(float(morning["start_hour"]))) & (F.col("hour_decimal") <= F.lit(float(morning["end_hour"])))).cast("int"),
    )
    out = out.withColumn(
        "is_evening_rush",
        ((F.col("hour_decimal") >= F.lit(float(evening["start_hour"]))) & (F.col("hour_decimal") <= F.lit(float(evening["end_hour"])))).cast("int"),
    )
    out = out.withColumn("rush_intensity", rush_intensity_expr("hour_decimal").cast("double"))

    out = out.withColumn("is_raining", ((F.col("precipitation_mm") > F.lit(0.1)) | (F.col("rain_mm") > F.lit(0.1))).cast("int"))
    out = out.withColumn("heavy_rain", (F.col("precipitation_mm") >= F.lit(5.0)).cast("int"))
    out = out.withColumn("rain_x_rush", (F.col("precipitation_mm") * F.col("rush_intensity")).cast("double"))
    return out


def add_rolling_rain_features(df: DataFrame) -> DataFrame:
    """Leakage-safe rolling 3h rain features matching pandas build_features.py.

    Pandas logic: groupby(route_id)[col].shift(1).rolling(window=12, min_periods=1).mean()
    where observations are 15-min departure grid-like series. We approximate by taking
    previous 12 rows per route ordered by departure_ts (excluding current row) using a
    window frame. For irregular timestamps this is a close approximation; for regular
    15-min data it matches.
    """
    w = Window.partitionBy("route_id").orderBy(F.col("departure_ts_parsed").cast("long")).rowsBetween(-12, -1)
    out = df.withColumn("precip_rolling_3h", F.avg("precipitation_mm").over(w))
    out = out.withColumn("rain_rolling_3h", F.avg("rain_mm").over(w))
    out = out.withColumn("precip_rolling_3h", F.coalesce(F.col("precip_rolling_3h"), F.lit(0.0)).cast("double"))
    out = out.withColumn("rain_rolling_3h", F.coalesce(F.col("rain_rolling_3h"), F.lit(0.0)).cast("double"))
    return out


def compute_gold_features(silver_df: DataFrame) -> DataFrame:
    df = make_trip_id(silver_df)
    # Deduplicate by trip identity; keep latest ingest if duplicates exist.
    order_cols = [F.col("departure_ts_parsed").desc()]
    if "kafka_ts" in df.columns:
        order_cols.insert(0, F.col("kafka_ts").desc())
    if "ingested_at_utc" in df.columns:
        order_cols.insert(0, F.col("ingested_at_utc").desc())
    win = Window.partitionBy("trip_id").orderBy(*order_cols)
    df = df.withColumn("_rn", F.row_number().over(win)).filter(F.col("_rn") == 1).drop("_rn")

    # Fill base_duration_min per route median fallback.
    route_med = Window.partitionBy("route_id")
    df = df.withColumn("base_duration_min", F.col("base_duration_min").cast("double"))
    df = df.withColumn(
        "base_duration_min",
        F.when(F.col("base_duration_min").isNull() | (F.col("base_duration_min") <= 0), F.expr("percentile_approx(base_duration_min, 0.5) OVER (PARTITION BY route_id)")).otherwise(F.col("base_duration_min")),
    )
    global_med = df.select(F.expr("percentile_approx(base_duration_min, 0.5)").alias("m")).collect()[0]["m"]
    global_med = float(global_med) if global_med is not None else 45.0
    df = df.withColumn("base_duration_min", F.coalesce(F.col("base_duration_min"), F.lit(global_med)))

    df = df.withColumn("departure_ts", F.col("departure_ts_parsed"))
    df = df.withColumn("event_date", F.to_date("departure_ts"))
    df = add_time_and_interaction_features(df)
    df = add_rolling_rain_features(df)

    # Ensure required columns exist and types are clean.
    out = df.select(
        F.col("trip_id").cast("string").alias("trip_id"),
        F.col("route_id").cast("string").alias("route_id"),
        F.col("departure_ts").cast("timestamp").alias("departure_ts"),
        F.col("event_date").cast("date").alias("event_date"),
        F.col("observed_duration_min").cast("double").alias("observed_duration_min"),
        F.col("source").cast("string").alias("source"),
        F.col("base_duration_min").cast("double").alias("base_duration_min"),
        F.col("distance_km").cast("double").alias("distance_km"),
        F.col("precipitation_mm").cast("double").alias("precipitation_mm"),
        F.col("rain_mm").cast("double").alias("rain_mm"),
        F.col("weathercode").cast("double").alias("weathercode"),
        F.col("windspeed_10m").cast("double").alias("windspeed_10m"),
        F.col("departure_hour").cast("int").alias("departure_hour"),
        F.col("departure_minute").cast("int").alias("departure_minute"),
        F.col("hour_sin").cast("double").alias("hour_sin"),
        F.col("hour_cos").cast("double").alias("hour_cos"),
        F.col("dow_sin").cast("double").alias("dow_sin"),
        F.col("dow_cos").cast("double").alias("dow_cos"),
        F.col("is_weekday").cast("int").alias("is_weekday"),
        F.col("is_morning_rush").cast("int").alias("is_morning_rush"),
        F.col("is_evening_rush").cast("int").alias("is_evening_rush"),
        F.col("rush_intensity").cast("double").alias("rush_intensity"),
        F.col("precip_rolling_3h").cast("double").alias("precip_rolling_3h"),
        F.col("rain_rolling_3h").cast("double").alias("rain_rolling_3h"),
        F.col("is_raining").cast("int").alias("is_raining"),
        F.col("heavy_rain").cast("int").alias("heavy_rain"),
        F.col("rain_x_rush").cast("double").alias("rain_x_rush"),
    )
    return out


def write_gold_parquet(gold_df: DataFrame, output_path: str) -> None:
    (
        gold_df.write.mode("overwrite")
        .format("parquet")
        .partitionBy("route_id", "event_date")
        .save(output_path)
    )


def _gold_df_to_pandas(gold_df: DataFrame):
    """Convert Spark DF to pandas without requiring pyspark pandas extra."""
    import pandas as pd

    rows = [r.asDict(recursive=True) for r in gold_df.collect()]
    return pd.DataFrame(rows)


def _duckdb_connect(db_path: Path):
    import duckdb  # local import; allows spark-only envs to run transform/write parquet

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path))
    conn.execute("PRAGMA threads=4")
    return conn


def sync_gold_to_duckdb(gold_df: DataFrame, db_path: Path) -> dict[str, int]:
    """Upsert Gold snapshot into DuckDB for existing train.py/dashboard compatibility.

    Writes:
    - gold_features: engineered feature snapshot (MODEL_FEATURES + target + ids)
    - trip_observations: upsert for compatibility with existing feature builder
    - weather_hourly: upsert hourly weather derived from gold rows
    - routes: update base_duration_min/distance_km when missing
    """
    pdf = _gold_df_to_pandas(gold_df)
    if pdf.empty:
        return {"gold_features": 0, "trip_observations": 0, "weather_hourly": 0}

    # Ensure datetime types
    import pandas as pd

    pdf["departure_ts"] = pd.to_datetime(pdf["departure_ts"])
    pdf["event_date"] = pd.to_datetime(pdf["event_date"]).dt.date

    conn = _duckdb_connect(db_path)
    try:
        from src.ingestion.pipeline import get_connection as get_pipeline_connection  # reuse schema init

        conn.close()
        conn = get_pipeline_connection(db_path)

        conn.register("gold_df", pdf)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gold_features (
                trip_id VARCHAR,
                route_id VARCHAR,
                departure_ts TIMESTAMP,
                event_date DATE,
                observed_duration_min DOUBLE,
                source VARCHAR,
                base_duration_min DOUBLE,
                distance_km DOUBLE,
                precipitation_mm DOUBLE,
                rain_mm DOUBLE,
                weathercode DOUBLE,
                windspeed_10m DOUBLE,
                departure_hour INTEGER,
                departure_minute INTEGER,
                hour_sin DOUBLE,
                hour_cos DOUBLE,
                dow_sin DOUBLE,
                dow_cos DOUBLE,
                is_weekday INTEGER,
                is_morning_rush INTEGER,
                is_evening_rush INTEGER,
                rush_intensity DOUBLE,
                precip_rolling_3h DOUBLE,
                rain_rolling_3h DOUBLE,
                is_raining INTEGER,
                heavy_rain INTEGER,
                rain_x_rush DOUBLE
            )
            """
        )
        # Upsert gold_features by trip_id
        conn.execute("DELETE FROM gold_features WHERE trip_id IN (SELECT trip_id FROM gold_df)")
        conn.execute(
            """
            INSERT INTO gold_features
            SELECT trip_id, route_id, departure_ts, event_date, observed_duration_min, source,
                   base_duration_min, distance_km, precipitation_mm, rain_mm, weathercode, windspeed_10m,
                   departure_hour, departure_minute, hour_sin, hour_cos, dow_sin, dow_cos,
                   is_weekday, is_morning_rush, is_evening_rush, rush_intensity,
                   precip_rolling_3h, rain_rolling_3h, is_raining, heavy_rain, rain_x_rush
            FROM gold_df
            """
        )

        # Upsert trip_observations for existing training pipeline compatibility.
        conn.execute(
            """
            DELETE FROM trip_observations WHERE trip_id IN (SELECT trip_id FROM gold_df)
            """
        )
        conn.execute(
            """
            INSERT INTO trip_observations (
                trip_id, route_id, departure_ts, observed_duration_min, source,
                base_duration_min, distance_km, created_at
            )
            SELECT
                trip_id, route_id, departure_ts, observed_duration_min,
                COALESCE(source, 'spark_gold'),
                base_duration_min, distance_km, CURRENT_TIMESTAMP
            FROM gold_df
            """
        )

        # Upsert weather_hourly from gold rows (hourly grain).
        conn.execute(
            """
            CREATE TEMP TABLE _gold_weather AS
            SELECT
                route_id,
                date_trunc('hour', departure_ts) AS timestamp_local,
                avg(precipitation_mm) AS precipitation,
                avg(rain_mm) AS rain,
                max(weathercode) AS weathercode,
                avg(windspeed_10m) AS windspeed_10m,
                'spark_gold' AS source,
                NULL::DOUBLE AS latitude,
                NULL::DOUBLE AS longitude,
                'UTC' AS timezone,
                CURRENT_TIMESTAMP AS fetched_at_utc
            FROM gold_df
            GROUP BY 1,2
            """
        )
        conn.execute(
            """
            DELETE FROM weather_hourly
            WHERE EXISTS (
                SELECT 1 FROM _gold_weather gw
                WHERE gw.route_id = weather_hourly.route_id
                  AND gw.timestamp_local = weather_hourly.timestamp_local
            )
            """
        )
        conn.execute(
            """
            INSERT INTO weather_hourly (
                route_id, timestamp_local, precipitation, rain, weathercode, windspeed_10m,
                source, latitude, longitude, timezone, fetched_at_utc
            )
            SELECT * FROM _gold_weather
            """
        )
        weather_rows = conn.execute("SELECT COUNT(*) FROM _gold_weather").fetchone()[0]
        conn.execute("DROP TABLE _gold_weather")

        # Update routes baseline/distance when missing.
        conn.execute(
            """
            MERGE INTO routes r
            USING (
                SELECT route_id,
                       avg(base_duration_min) AS base_duration_min,
                       avg(distance_km) AS distance_km
                FROM gold_df
                GROUP BY route_id
            ) g
            ON r.route_id = g.route_id
            WHEN MATCHED THEN UPDATE SET
                base_duration_min = COALESCE(r.base_duration_min, g.base_duration_min),
                distance_km = COALESCE(r.distance_km, g.distance_km),
                updated_at = CURRENT_TIMESTAMP
            """
        )

        gold_count = conn.execute("SELECT COUNT(*) FROM gold_df").fetchone()[0]
        trip_count = gold_count
    finally:
        try:
            conn.unregister("gold_df")
        except Exception:
            pass
        conn.close()

    return {"gold_features": int(gold_count), "trip_observations": int(trip_count), "weather_hourly": int(weather_rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark batch ETL: Silver -> Gold Feature Store -> DuckDB")
    parser.add_argument("--hdfs-base", default="hdfs://namenode:9000/datalake", help="Lakehouse base path")
    parser.add_argument("--duckdb-path", default=str(DUCKDB_PATH), help="Local DuckDB path for snapshot sync")
    parser.add_argument("--local-mode", action="store_true", help="Use Spark local[*] and local filesystem paths")
    parser.add_argument("--skip-duckdb-sync", action="store_true", help="Only write Gold Parquet, skip DuckDB sync")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    hdfs_base = args.hdfs_base.rstrip("/")
    silver_enriched_path = f"{hdfs_base}/silver/commute_enriched"
    silver_weather_rolling_path = f"{hdfs_base}/silver/weather_rolling_3h"
    gold_output_path = f"{hdfs_base}/gold/feature_store"

    spark = build_spark("commute-gold-etl", local_mode=args.local_mode)
    LOGGER.info("Reading silver enriched: %s", silver_enriched_path)
    silver_df = read_silver_enriched(spark, silver_enriched_path)
    rolling_df = read_weather_rolling(spark, silver_weather_rolling_path)
    if rolling_df is not None:
        LOGGER.info("Weather rolling silver rows available: %s", rolling_df.count())

    gold_df = compute_gold_features(silver_df)
    # Validate expected feature columns are present
    missing = [c for c in MODEL_FEATURES + [TARGET_COLUMN, "trip_id", "route_id", "departure_ts"] if c not in gold_df.columns]
    if missing:
        raise ValueError(f"Gold dataframe missing required columns: {missing}")

    LOGGER.info("Writing gold parquet: %s", gold_output_path)
    write_gold_parquet(gold_df, gold_output_path)

    stats = {"gold_rows": gold_df.count()}
    if not args.skip_duckdb_sync:
        db_path = Path(args.duckdb_path)
        if not db_path.is_absolute():
            db_path = (PROJECT_ROOT / db_path).resolve()
        LOGGER.info("Syncing gold snapshot to DuckDB: %s", db_path)
        sync_stats = sync_gold_to_duckdb(gold_df, db_path)
        stats.update(sync_stats)

    LOGGER.info("Gold ETL complete: %s", json.dumps(stats))
    print(json.dumps(stats))
    spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
