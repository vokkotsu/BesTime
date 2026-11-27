#!/usr/bin/env bash
set -euo pipefail

# Helper script for the Commute Optimizer Big Data Lakehouse stack.
# Works with Docker Compose v2 (`docker compose`) or v1 (`docker-compose`).

COMPOSE_CMD="${COMPOSE_CMD:-}"
if [ -z "$COMPOSE_CMD" ]; then
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
  elif docker-compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
  else
    echo "Docker Compose is required." >&2
    exit 1
  fi
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

KAFKA_CONTAINER="${KAFKA_CONTAINER:-commute-kafka}"
NAMENODE_CONTAINER="${NAMENODE_CONTAINER:-commute-namenode}"
SPARK_MASTER_CONTAINER="${SPARK_MASTER_CONTAINER:-commute-spark-master}"
KAFKA_BOOTSTRAP="${KAFKA_BOOTSTRAP:-localhost:9092}"
SPARK_MASTER_URL="${SPARK_MASTER_URL:-spark://spark-master:7077}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_bigdata_pipeline.sh <command>

Commands:
  up                Start Kafka, HDFS, and Spark containers
  down              Stop containers (add --volumes to remove volumes)
  init-hdfs         Create HDFS medallion/checkpoint directories
  create-topics     Create Kafka topics (idempotent)
  submit-stream     Submit Spark Structured Streaming job
  submit-batch      Submit Spark Gold ETL batch job
  submit-all        Submit stream job in background + run batch job once
  status            Show container status and key URLs

Environment overrides:
  COMPOSE_CMD, KAFKA_CONTAINER, NAMENODE_CONTAINER, SPARK_MASTER_CONTAINER,
  KAFKA_BOOTSTRAP, SPARK_MASTER_URL
USAGE
}

wait_for_kafka() {
  echo "[wait] Kafka broker..."
  for i in $(seq 1 40); do
    if docker exec "$KAFKA_CONTAINER" /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list >/dev/null 2>&1; then
      echo "[ok] Kafka is ready"
      return 0
    fi
    sleep 3
  done
  echo "[error] Kafka did not become ready" >&2
  return 1
}

wait_for_hdfs() {
  echo "[wait] HDFS NameNode..."
  for i in $(seq 1 40); do
    if docker exec "$NAMENODE_CONTAINER" hdfs dfs -ls / >/dev/null 2>&1; then
      echo "[ok] HDFS is ready"
      return 0
    fi
    sleep 3
  done
  echo "[error] HDFS did not become ready" >&2
  return 1
}

cmd_up() {
  echo "[up] Starting big data stack..."
  $COMPOSE_CMD up -d
  wait_for_kafka
  wait_for_hdfs
  cmd_init_hdfs
  cmd_create_topics
  cmd_status
}

cmd_down() {
  if [ "${1:-}" = "--volumes" ]; then
    echo "[down] Stopping stack and removing volumes..."
    $COMPOSE_CMD down -v
  else
    echo "[down] Stopping stack..."
    $COMPOSE_CMD down
  fi
}

cmd_init_hdfs() {
  wait_for_hdfs
  echo "[hdfs] Creating medallion directories..."
  docker exec "$NAMENODE_CONTAINER" hdfs dfs -mkdir -p \
    /datalake/bronze/traffic \
    /datalake/bronze/weather \
    /datalake/silver/commute_enriched \
    /datalake/silver/weather_rolling_3h \
    /datalake/gold/feature_store \
    /datalake/checkpoints/stream/traffic_bronze \
    /datalake/checkpoints/stream/weather_bronze \
    /datalake/checkpoints/stream/weather_rolling \
    /datalake/checkpoints/stream/silver_enriched_alerts
  docker exec "$NAMENODE_CONTAINER" hdfs dfs -chmod -R 777 /datalake || true
  echo "[hdfs] Current /datalake layout:"
  docker exec "$NAMENODE_CONTAINER" hdfs dfs -ls -R /datalake | sed -n '1,120p'
}

cmd_create_topics() {
  wait_for_kafka
  echo "[kafka] Creating topics..."
  for topic in commute.traffic.raw commute.weather.raw commute.alerts.realtime; do
    docker exec "$KAFKA_CONTAINER" /opt/kafka/bin/kafka-topics.sh \
      --bootstrap-server localhost:9092 \
      --create --if-not-exists \
      --topic "$topic" \
      --partitions 3 \
      --replication-factor 1
  done
  echo "[kafka] Topics:"
  docker exec "$KAFKA_CONTAINER" /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
}

spark_packages() {
  # Kafka connector for Structured Streaming + Spark SQL Kafka sink.
  echo "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5"
}

cmd_submit_stream() {
  echo "[spark] Submitting Structured Streaming job..."
  docker exec "$SPARK_MASTER_CONTAINER" /opt/spark/bin/spark-submit \
    --master "$SPARK_MASTER_URL" \
    --deploy-mode client \
    --packages "$(spark_packages)" \
    --conf spark.executor.memory=1g \
    --conf spark.driver.memory=1g \
    --conf spark.sql.shuffle.partitions=4 \
    --conf spark.sql.streaming.schemaInference=false \
    --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
    /opt/workspace/src/streaming/spark_stream_processor.py \
    --kafka-bootstrap kafka:9092 \
    --hdfs-base hdfs://namenode:9000/datalake \
    --checkpoint-base hdfs://namenode:9000/datalake/checkpoints/stream
}

cmd_submit_batch() {
  echo "[spark] Submitting Gold ETL batch job..."
  docker exec "$SPARK_MASTER_CONTAINER" /opt/spark/bin/spark-submit \
    --master "$SPARK_MASTER_URL" \
    --deploy-mode client \
    --packages "$(spark_packages)" \
    --conf spark.executor.memory=1g \
    --conf spark.driver.memory=1g \
    --conf spark.sql.shuffle.partitions=4 \
    --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 \
    /opt/workspace/src/batch/spark_gold_etl.py \
    --hdfs-base hdfs://namenode:9000/datalake \
    --duckdb-path /opt/workspace/data/commute.duckdb
}

cmd_submit_all() {
  echo "[spark] Starting stream job in background (logs: ./.spark-stream.log)..."
  mkdir -p "$PROJECT_ROOT/.logs"
  nohup "$0" submit-stream > "$PROJECT_ROOT/.logs/spark-stream.log" 2>&1 &
  echo $! > "$PROJECT_ROOT/.logs/spark-stream.pid"
  echo "[spark] Stream PID: $(cat "$PROJECT_ROOT/.logs/spark-stream.pid")"
  sleep 5
  cmd_submit_batch
}

cmd_status() {
  $COMPOSE_CMD ps
  cat <<'STATUS'

Key endpoints:
  Kafka bootstrap (host):    localhost:29092  (container: kafka:9092)
  HDFS NameNode UI:          http://localhost:9870
  HDFS RPC:                  hdfs://localhost:9000
  Spark Master UI:           http://localhost:8080
  Spark Worker UI:           http://localhost:8081

Kafka topics (if Kafka is up):
  commute.traffic.raw
  commute.weather.raw
  commute.alerts.realtime
STATUS
}

main() {
  local cmd="${1:-}"
  shift || true
  case "$cmd" in
    up) cmd_up "$@" ;;
    down) cmd_down "$@" ;;
    init-hdfs) cmd_init_hdfs "$@" ;;
    create-topics) cmd_create_topics "$@" ;;
    submit-stream) cmd_submit_stream "$@" ;;
    submit-batch) cmd_submit_batch "$@" ;;
    submit-all) cmd_submit_all "$@" ;;
    status) cmd_status "$@" ;;
    ""|-h|--help|help) usage ;;
    *) echo "Unknown command: $cmd" >&2; usage; exit 1 ;;
  esac
}

main "$@"
