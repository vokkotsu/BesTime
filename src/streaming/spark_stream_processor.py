"""PySpark Structured Streaming job for commute lakehouse ingestion.

Consumes:
- commute.traffic.raw
- commute.weather.raw

Writes:
- Bronze Parquet (raw append-only) to HDFS/local
- Silver Parquet (enriched traffic + weather, deduped) to HDFS/local
- Real-time alerts to Kafka topic commute.alerts.realtime

Notes:
- Uses explicit StructType schemas (no inferSchema in streaming).
- Uses checkpointing for fault-tolerant restarts.
- Supports --local-mode for local filesystem testing without Docker/HDFS.

Example (cluster/HDFS):
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 \
    src/streaming/spark_stream_processor.py \
    --kafka-bootstrap kafka:9092 \
    --hdfs-base hdfs://namenode:9000/datalake

Example (local mode):
  spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 \
    src/streaming/spark_stream_processor.py --local-mode --hdfs-base ./data/lakehouse
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

LOGGER = logging.getLogger("commute.spark_stream")

TRAFFIC_TOPIC = "commute.traffic.raw"
WEATHER_TOPIC = "commute.weather.raw"
ALERTS_TOPIC = "commute.alerts.realtime"


def traffic_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("event_id", T.StringType(), False),
            T.StructField("event_ts", T.StringType(), False),
            T.StructField("route_id", T.StringType(), False),
            T.StructField("origin", T.StringType(), True),
            T.StructField("destination", T.StringType(), True),
            T.StructField("origin_lat", T.DoubleType(), True),
            T.StructField("origin_lon", T.DoubleType(), True),
            T.StructField("dest_lat", T.DoubleType(), True),
            T.StructField("dest_lon", T.DoubleType(), True),
            T.StructField("departure_ts", T.StringType(), True),
            T.StructField("observed_duration_min", T.DoubleType(), True),
            T.StructField("base_duration_min", T.DoubleType(), True),
            T.StructField("distance_km", T.DoubleType(), True),
            T.StructField("source", T.StringType(), True),
            T.StructField("precipitation_mm", T.DoubleType(), True),
            T.StructField("rain_mm", T.DoubleType(), True),
            T.StructField("weathercode", T.DoubleType(), True),
            T.StructField("windspeed_10m", T.DoubleType(), True),
            T.StructField("ingested_at_utc", T.StringType(), True),
            T.StructField("schema_version", T.IntegerType(), True),
        ]
    )


def weather_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("event_id", T.StringType(), False),
            T.StructField("event_ts", T.StringType(), False),
            T.StructField("route_id", T.StringType(), False),
            T.StructField("timestamp_local", T.StringType(), True),
            T.StructField("precipitation_mm", T.DoubleType(), True),
            T.StructField("rain_mm", T.DoubleType(), True),
            T.StructField("weathercode", T.DoubleType(), True),
            T.StructField("windspeed_10m", T.DoubleType(), True),
            T.StructField("source", T.StringType(), True),
            T.StructField("latitude", T.DoubleType(), True),
            T.StructField("longitude", T.DoubleType(), True),
            T.StructField("ingested_at_utc", T.StringType(), True),
            T.StructField("schema_version", T.IntegerType(), True),
        ]
    )


def alert_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("alert_id", T.StringType(), False),
            T.StructField("alert_ts", T.TimestampType(), False),
            T.StructField("route_id", T.StringType(), False),
            T.StructField("alert_type", T.StringType(), False),
            T.StructField("severity", T.StringType(), False),
            T.StructField("message", T.StringType(), False),
            T.StructField("observed_duration_min", T.DoubleType(), True),
            T.StructField("baseline_duration_min", T.DoubleType(), True),
            T.StructField("precipitation_mm", T.DoubleType(), True),
            T.StructField("rain_mm", T.DoubleType(), True),
            T.StructField("event_ts", T.TimestampType(), True),
        ]
    )


def build_spark(app_name: str, local_mode: bool) -> SparkSession:
    builder = SparkSession.builder.appName(app_name)
    if local_mode:
        builder = builder.master("local[*]")
    # Reasonable local defaults; cluster submit can override via spark-submit --conf.
    builder = (
        builder.config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.streaming.schemaInference", "false")
        .config("spark.sql.parquet.compression.codec", "snappy")
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def _parse_kafka_df(df: DataFrame, schema: T.StructType) -> DataFrame:
    parsed = (
        df.select(
            F.col("key").cast("string").alias("kafka_key"),
            F.col("value").cast("string").alias("raw_json"),
            F.col("topic"),
            F.col("partition"),
            F.col("offset"),
            F.col("timestamp").alias("kafka_ts"),
        )
        .withColumn("payload", F.from_json(F.col("raw_json"), schema))
        .select("kafka_key", "topic", "partition", "offset", "kafka_ts", "raw_json", "payload.*")
    )
    return parsed


def read_kafka_stream(spark: SparkSession, bootstrap: str, topic: str, starting_offsets: str) -> DataFrame:
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
    )


def with_event_time(df: DataFrame, ts_col: str = "event_ts") -> DataFrame:
    return df.withColumn("event_ts_parsed", F.to_timestamp(F.col(ts_col))).withColumn(
        "event_date", F.to_date(F.col("event_ts_parsed"))
    )


def prepare_traffic(df: DataFrame) -> DataFrame:
    out = _parse_kafka_df(df, traffic_schema())
    out = with_event_time(out, "event_ts")
    out = (
        out.withColumn("departure_ts_parsed", F.to_timestamp(F.col("departure_ts")))
        .withColumn("observed_duration_min", F.col("observed_duration_min").cast("double"))
        .withColumn("base_duration_min", F.col("base_duration_min").cast("double"))
        .withColumn("distance_km", F.col("distance_km").cast("double"))
        .withColumn("precipitation_mm", F.coalesce(F.col("precipitation_mm").cast("double"), F.lit(0.0)))
        .withColumn("rain_mm", F.coalesce(F.col("rain_mm").cast("double"), F.lit(0.0)))
        .withColumn("weathercode", F.coalesce(F.col("weathercode").cast("double"), F.lit(0.0)))
        .withColumn("windspeed_10m", F.coalesce(F.col("windspeed_10m").cast("double"), F.lit(0.0)))
        .withColumn("source", F.coalesce(F.col("source"), F.lit("unknown")))
        .filter(F.col("route_id").isNotNull() & F.col("event_ts_parsed").isNotNull())
        .dropDuplicates(["event_id"])
    )
    return out


def prepare_weather(df: DataFrame) -> DataFrame:
    out = _parse_kafka_df(df, weather_schema())
    out = with_event_time(out, "event_ts")
    out = (
        out.withColumn("timestamp_local_ts", F.to_timestamp(F.col("timestamp_local")))
        .withColumn("precipitation_mm", F.coalesce(F.col("precipitation_mm").cast("double"), F.lit(0.0)))
        .withColumn("rain_mm", F.coalesce(F.col("rain_mm").cast("double"), F.lit(0.0)))
        .withColumn("weathercode", F.coalesce(F.col("weathercode").cast("double"), F.lit(0.0)))
        .withColumn("windspeed_10m", F.coalesce(F.col("windspeed_10m").cast("double"), F.lit(0.0)))
        .withColumn("weather_hour_ts", F.date_trunc("hour", F.col("event_ts_parsed")))
        .filter(F.col("route_id").isNotNull() & F.col("event_ts_parsed").isNotNull())
        .dropDuplicates(["event_id"])
    )
    return out


def write_bronze_stream(df: DataFrame, output_path: str, checkpoint_path: str, query_name: str):
    return (
        df.writeStream.format("parquet")
        .outputMode("append")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .partitionBy("route_id", "event_date")
        .queryName(query_name)
        .trigger(processingTime="30 seconds")
        .start()
    )


def write_weather_rolling_stream(weather_df: DataFrame, output_path: str, checkpoint_path: str):
    """Compute sliding 3-hour rainfall aggregates from weather stream and persist to Silver."""
    rolling = (
        weather_df.withWatermark("event_ts_parsed", "2 hours")
        .groupBy(
            F.col("route_id"),
            F.window(F.col("event_ts_parsed"), "3 hours", "1 hour").alias("rain_window"),
        )
        .agg(
            F.sum("precipitation_mm").alias("rolling_precip_3h_mm"),
            F.sum("rain_mm").alias("rolling_rain_3h_mm"),
            F.max("event_ts_parsed").alias("last_event_ts"),
        )
        .select(
            "route_id",
            F.col("rain_window.start").alias("window_start"),
            F.col("rain_window.end").alias("window_end"),
            "rolling_precip_3h_mm",
            "rolling_rain_3h_mm",
            "last_event_ts",
            F.to_date(F.col("rain_window.start")).alias("event_date"),
        )
    )

    return (
        rolling.writeStream.format("parquet")
        .outputMode("append")
        .option("path", output_path)
        .option("checkpointLocation", checkpoint_path)
        .partitionBy("route_id", "event_date")
        .queryName("weather_rolling_silver")
        .trigger(processingTime="60 seconds")
        .start()
    )


def _latest_weather_lookup(spark: SparkSession, weather_bronze_path: str) -> DataFrame:
    """Batch lookup of latest weather per route/hour from bronze parquet for micro-batch enrichment."""
    try:
        w = spark.read.parquet(weather_bronze_path)
    except Exception:
        return spark.createDataFrame(
            [],
            T.StructType(
                [
                    T.StructField("route_id", T.StringType(), True),
                    T.StructField("weather_hour_ts", T.TimestampType(), True),
                    T.StructField("precipitation_mm", T.DoubleType(), True),
                    T.StructField("rain_mm", T.DoubleType(), True),
                    T.StructField("weathercode", T.DoubleType(), True),
                    T.StructField("windspeed_10m", T.DoubleType(), True),
                ]
            ),
        )
    cols = w.columns
    if "weather_hour_ts" not in cols and "event_ts_parsed" in cols:
        w = w.withColumn("weather_hour_ts", F.date_trunc("hour", F.col("event_ts_parsed")))
    needed = ["route_id", "weather_hour_ts", "precipitation_mm", "rain_mm", "weathercode", "windspeed_10m", "event_ts_parsed"]
    for c in needed:
        if c not in w.columns:
            w = w.withColumn(c, F.lit(None))
    latest = (
        w.select(*needed)
        .withColumn("precipitation_mm", F.coalesce(F.col("precipitation_mm").cast("double"), F.lit(0.0)))
        .withColumn("rain_mm", F.coalesce(F.col("rain_mm").cast("double"), F.lit(0.0)))
        .withColumn("weathercode", F.coalesce(F.col("weathercode").cast("double"), F.lit(0.0)))
        .withColumn("windspeed_10m", F.coalesce(F.col("windspeed_10m").cast("double"), F.lit(0.0)))
        .dropna(subset=["route_id", "weather_hour_ts"])
        .orderBy(F.col("event_ts_parsed").desc())
        .dropDuplicates(["route_id", "weather_hour_ts"])
    )
    return latest


def _detect_alerts(enriched_batch: DataFrame) -> DataFrame:
    base = F.coalesce(F.col("base_duration_min"), F.lit(35.0))
    ratio = F.when(base > 0, F.col("observed_duration_min") / base).otherwise(F.lit(1.0))
    delay_cliff = (F.col("observed_duration_min") - base >= F.lit(12.0)) | (ratio >= F.lit(1.15))
    heavy_rain = F.coalesce(F.col("precipitation_mm"), F.lit(0.0)) >= F.lit(5.0)

    alerts = (
        enriched_batch.withColumn("is_delay_cliff", delay_cliff)
        .withColumn("is_heavy_rain", heavy_rain)
        .filter(F.col("is_delay_cliff") | F.col("is_heavy_rain"))
        .withColumn(
            "alert_type",
            F.when(F.col("is_delay_cliff") & F.col("is_heavy_rain"), F.lit("delay_cliff_heavy_rain"))
            .when(F.col("is_delay_cliff"), F.lit("delay_cliff"))
            .otherwise(F.lit("heavy_rain_surge")),
        )
        .withColumn(
            "severity",
            F.when(F.col("alert_type") == F.lit("delay_cliff_heavy_rain"), F.lit("critical"))
            .when(F.col("alert_type") == F.lit("delay_cliff"), F.lit("high"))
            .otherwise(F.lit("medium")),
        )
        .withColumn(
            "message",
            F.concat_ws(
                " ",
                F.lit("Route"),
                F.col("route_id"),
                F.lit("alert:"),
                F.col("alert_type"),
                F.lit("duration_min="),
                F.round(F.col("observed_duration_min"), 1),
                F.lit("base_min="),
                F.round(base, 1),
                F.lit("rain_mm="),
                F.round(F.coalesce(F.col("precipitation_mm"), F.lit(0.0)), 2),
            ),
        )
        .select(
            F.expr("uuid()").alias("alert_id"),
            F.current_timestamp().alias("alert_ts"),
            "route_id",
            "alert_type",
            "severity",
            "message",
            F.col("observed_duration_min").cast("double").alias("observed_duration_min"),
            base.cast("double").alias("baseline_duration_min"),
            F.coalesce(F.col("precipitation_mm"), F.lit(0.0)).cast("double").alias("precipitation_mm"),
            F.coalesce(F.col("rain_mm"), F.lit(0.0)).cast("double").alias("rain_mm"),
            F.col("event_ts_parsed").alias("event_ts"),
        )
    )
    return alerts


def write_alerts_to_kafka(alerts_df: DataFrame, bootstrap: str, topic: str) -> None:
    if alerts_df.rdd.isEmpty():
        return
    payload = alerts_df.select(
        F.col("route_id").cast("string").alias("key"),
        F.to_json(F.struct(*[F.col(c) for c in alerts_df.columns])).alias("value"),
    )
    (
        payload.write.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("topic", topic)
        .save()
    )


def start_enriched_silver_and_alerts(
    spark: SparkSession,
    traffic_stream: DataFrame,
    *,
    weather_bronze_path: str,
    silver_output_path: str,
    silver_checkpoint: str,
    alerts_topic: str,
    kafka_bootstrap: str,
):
    """Enrich traffic with latest hourly weather and emit Silver + alerts.

    Uses foreachBatch so we can do a lightweight batch lookup against weather bronze
    (and optionally future static dims) while keeping the traffic source as a stream.
    """

    def _foreach_batch(batch_df: DataFrame, batch_id: int) -> None:
        if batch_df.rdd.isEmpty():
            return
        batch = batch_df.withColumn("weather_hour_ts", F.date_trunc("hour", F.col("event_ts_parsed")))
        weather_latest = _latest_weather_lookup(spark, weather_bronze_path)
        if weather_latest.rdd.isEmpty():
            enriched = (
                batch.withColumn("precipitation_mm", F.coalesce(F.col("precipitation_mm"), F.lit(0.0)))
                .withColumn("rain_mm", F.coalesce(F.col("rain_mm"), F.lit(0.0)))
                .withColumn("weathercode", F.coalesce(F.col("weathercode"), F.lit(0.0)))
                .withColumn("windspeed_10m", F.coalesce(F.col("windspeed_10m"), F.lit(0.0)))
            )
        else:
            w = weather_latest.alias("w")
            t = batch.alias("t")
            enriched = (
                t.join(w, on=[F.col("t.route_id") == F.col("w.route_id"), F.col("t.weather_hour_ts") == F.col("w.weather_hour_ts")], how="left")
                .select(
                    F.col("t.*"),
                    F.coalesce(F.col("w.precipitation_mm"), F.col("t.precipitation_mm"), F.lit(0.0)).alias("precipitation_mm_enriched"),
                    F.coalesce(F.col("w.rain_mm"), F.col("t.rain_mm"), F.lit(0.0)).alias("rain_mm_enriched"),
                    F.coalesce(F.col("w.weathercode"), F.col("t.weathercode"), F.lit(0.0)).alias("weathercode_enriched"),
                    F.coalesce(F.col("w.windspeed_10m"), F.col("t.windspeed_10m"), F.lit(0.0)).alias("windspeed_10m_enriched"),
                )
                .drop("precipitation_mm", "rain_mm", "weathercode", "windspeed_10m")
                .withColumnRenamed("precipitation_mm_enriched", "precipitation_mm")
                .withColumnRenamed("rain_mm_enriched", "rain_mm")
                .withColumnRenamed("weathercode_enriched", "weathercode")
                .withColumnRenamed("windspeed_10m_enriched", "windspeed_10m")
            )

        enriched = (
            enriched.withColumn("processing_ts", F.current_timestamp())
            .withColumn("event_date", F.to_date(F.col("event_ts_parsed")))
            .dropDuplicates(["event_id"])
        )

        (
            enriched.write.mode("append")
            .format("parquet")
            .partitionBy("route_id", "event_date")
            .save(silver_output_path)
        )

        alerts = _detect_alerts(enriched)
        write_alerts_to_kafka(alerts, bootstrap=kafka_bootstrap, topic=alerts_topic)
        LOGGER.info("foreachBatch batch_id=%s rows=%s alerts=%s", batch_id, enriched.count(), alerts.count())

    return (
        traffic_stream.withWatermark("event_ts_parsed", "10 minutes")
        .writeStream.foreachBatch(_foreach_batch)
        .option("checkpointLocation", silver_checkpoint)
        .queryName("silver_enriched_and_alerts")
        .trigger(processingTime="30 seconds")
        .start()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spark Structured Streaming processor for commute lakehouse")
    parser.add_argument("--kafka-bootstrap", default="localhost:29092")
    parser.add_argument("--traffic-topic", default=TRAFFIC_TOPIC)
    parser.add_argument("--weather-topic", default=WEATHER_TOPIC)
    parser.add_argument("--alerts-topic", default=ALERTS_TOPIC)
    parser.add_argument("--hdfs-base", default="hdfs://namenode:9000/datalake", help="Base path for lakehouse outputs")
    parser.add_argument(
        "--checkpoint-base",
        default=None,
        help="Checkpoint base path; defaults to <hdfs-base>/checkpoints/stream",
    )
    parser.add_argument("--starting-offsets", default="latest", choices=["latest", "earliest"])
    parser.add_argument("--local-mode", action="store_true", help="Run Spark in local[*] mode")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def ensure_local_dirs(base: str) -> None:
    if "://" in base and not base.startswith("file:"):
        return
    # local path
    path = base.replace("file:", "")
    for sub in [
        "bronze/traffic",
        "bronze/weather",
        "silver/commute_enriched",
        "silver/weather_rolling_3h",
        "checkpoints/stream",
    ]:
        Path(path, sub).mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    hdfs_base = args.hdfs_base.rstrip("/")
    checkpoint_base = (args.checkpoint_base or f"{hdfs_base}/checkpoints/stream").rstrip("/")
    ensure_local_dirs(hdfs_base)

    spark = build_spark("commute-stream-processor", local_mode=args.local_mode)
    LOGGER.info("Starting stream processor. kafka=%s base=%s local=%s", args.kafka_bootstrap, hdfs_base, args.local_mode)

    traffic_raw = read_kafka_stream(spark, args.kafka_bootstrap, args.traffic_topic, args.starting_offsets)
    weather_raw = read_kafka_stream(spark, args.kafka_bootstrap, args.weather_topic, args.starting_offsets)

    traffic = prepare_traffic(traffic_raw)
    weather = prepare_weather(weather_raw)

    queries = []
    queries.append(
        write_bronze_stream(
            traffic,
            output_path=f"{hdfs_base}/bronze/traffic",
            checkpoint_path=f"{checkpoint_base}/traffic_bronze",
            query_name="traffic_bronze",
        )
    )
    queries.append(
        write_bronze_stream(
            weather,
            output_path=f"{hdfs_base}/bronze/weather",
            checkpoint_path=f"{checkpoint_base}/weather_bronze",
            query_name="weather_bronze",
        )
    )
    queries.append(
        write_weather_rolling_stream(
            weather,
            output_path=f"{hdfs_base}/silver/weather_rolling_3h",
            checkpoint_path=f"{checkpoint_base}/weather_rolling",
        )
    )
    queries.append(
        start_enriched_silver_and_alerts(
            spark,
            traffic,
            weather_bronze_path=f"{hdfs_base}/bronze/weather",
            silver_output_path=f"{hdfs_base}/silver/commute_enriched",
            silver_checkpoint=f"{checkpoint_base}/silver_enriched_alerts",
            alerts_topic=args.alerts_topic,
            kafka_bootstrap=args.kafka_bootstrap,
        )
    )

    LOGGER.info("Started %s streaming queries", len(queries))
    for q in queries:
        LOGGER.info("Query: name=%s id=%s active=%s", q.name, q.id, q.isActive)

    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        LOGGER.info("Stopping queries...")
        for q in spark.streams.active:
            q.stop()
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
