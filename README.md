# Commute Departure & Delay Risk Optimizer

MVP for an **Anti-Traffic & Rain Commute Calculator** that models commute travel time as a departure-time curve (P50/P90), instead of a single static ETA. It flags:

- **Diminishing Return Zones**: leaving much earlier saves little/no time because rush-hour saturation is already high.
- **Delay Cliff Zones**: leaving slightly later causes a non-linear jump in travel time (rush wave + rain interaction / tail risk).

The project now includes an upgrade path to a **distributed Big Data & Real-Time Streaming Lakehouse** using **Apache Kafka (KRaft)**, **Apache Spark 3.5 Structured Streaming / PySpark**, and **Apache Hadoop HDFS**, while keeping the existing **LightGBM quantile models (`P50`/`P90`)** and **Streamlit dashboard** compatible.

## Architecture

### Local MVP (existing, still supported)
- **Python 3.11+**
- **DuckDB** (`data/commute.duckdb`) for local analytical storage
- **Open-Meteo** hourly weather (no API key)
- **TomTom / OSRM / synthetic simulator** for travel duration observations
- **pandas/numpy/scikit-learn/lightgbm** for feature engineering and quantile regression (`P50`, `P90`)
- **Streamlit** dashboard for scenario exploration

### Big Data Lakehouse (new)
1. **Event Streaming (Apache Kafka, KRaft, no Zookeeper)**
   - `commute.traffic.raw`: real-time travel duration telemetry (TomTom poller, replay, synthetic simulator)
   - `commute.weather.raw`: hourly/live weather telemetry
   - `commute.alerts.realtime`: real-time `delay_cliff` / heavy-rain surge alerts emitted by Spark
2. **Distributed Storage (Hadoop HDFS, Parquet, Medallion)**
   - Bronze: `/datalake/bronze/traffic/`, `/datalake/bronze/weather/` (raw append-only)
   - Silver: `/datalake/silver/commute_enriched/`, `/datalake/silver/weather_rolling_3h/`
   - Gold: `/datalake/gold/feature_store/` (feature store for training)
   - Checkpoints: `/datalake/checkpoints/stream/...`
3. **Processing (Apache Spark 3.5+)**
   - `src/streaming/spark_stream_processor.py`: Structured Streaming Kafka -> Bronze/Silver/Alerts
   - `src/batch/spark_gold_etl.py`: Silver -> Gold feature store (PySpark window features) -> DuckDB sync for existing training/dashboard

## Project Structure

```text
commute-optimizer/  (workspace root)
├── docker-compose.yml                  # Kafka (KRaft), Hadoop HDFS, Spark Master/Worker
├── hadoop.env                          # HDFS/Hadoop config
├── scripts/
│   └── run_bigdata_pipeline.sh         # HDFS dirs, Kafka topics, Spark submit helpers
├── data/                               # DuckDB database & generated artifacts/models
├── src/
│   ├── config.py                       # Routes, time grid, API endpoints, model config
│   ├── ingestion/
│   │   ├── weather.py                  # Open-Meteo hourly forecast & historical fetch
│   │   ├── routing.py                  # OSRM/TomTom baseline + synthetic simulator
│   │   ├── pipeline.py                 # Idempotent ETL orchestrator into DuckDB
│   │   └── collector.py                # Continuous live collector
│   ├── streaming/
│   │   ├── __init__.py
│   │   ├── kafka_producer.py           # Publishes live/replay/synthetic events to Kafka
│   │   └── spark_stream_processor.py   # Structured Streaming: Kafka -> HDFS + alerts
│   ├── batch/
│   │   ├── __init__.py
│   │   └── spark_gold_etl.py           # Silver -> Gold Feature Store -> DuckDB sync
│   ├── features/
│   │   └── build_features.py           # Pandas feature pipeline (MVP/training parity target)
│   ├── models/
│   │   ├── train.py                    # LightGBM quantile regressors (P50/P90)
│   │   └── predict.py                  # 15-min departure curve + zone detection
│   └── app/
│       └── dashboard.py                # Streamlit UI + Big Data status/alerts panel
├── tests/
│   ├── test_features_and_cliff.py
│   └── test_spark_transformations.py   # PySpark vs pandas feature parity tests
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Optional `.env`:
```bash
copy .env.example .env
# set TOMTOM_API_KEY=... (optional; simulator fallback works without it)
```

## Run the Local MVP (unchanged)

### 1. Bootstrap data
```bash
python -m src.ingestion.pipeline
```

### 2. Train quantile models (P50/P90)
```bash
python -m src.models.train --route-id bsd_jakarta
```

### 3. Launch dashboard
```bash
streamlit run src/app/dashboard.py
```

## Run the Big Data Lakehouse (Kafka + HDFS + Spark)

### Prerequisites
- Docker Desktop / Docker Engine with Docker Compose v2
- 8GB-16GB RAM recommended (containers have JVM/memory caps)
- Python deps installed locally (`pip install -r requirements.txt`) for producer/dashboard
- On Windows, run `scripts/run_bigdata_pipeline.sh` from **Git Bash** or **WSL** (PowerShell commands below are also provided)

### 1. Start infrastructure
**Git Bash / WSL**
```bash
bash scripts/run_bigdata_pipeline.sh up
```

**PowerShell (equivalent)**
```powershell
docker compose up -d
docker exec commute-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
docker exec commute-namenode hdfs dfs -mkdir -p `
  /datalake/bronze/traffic /datalake/bronze/weather `
  /datalake/silver/commute_enriched /datalake/silver/weather_rolling_3h `
  /datalake/gold/feature_store `
  /datalake/checkpoints/stream/traffic_bronze `
  /datalake/checkpoints/stream/weather_bronze `
  /datalake/checkpoints/stream/weather_rolling `
  /datalake/checkpoints/stream/silver_enriched_alerts
docker exec commute-namenode hdfs dfs -chmod -R 777 /datalake

docker exec commute-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic commute.traffic.raw --partitions 3 --replication-factor 1
docker exec commute-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic commute.weather.raw --partitions 3 --replication-factor 1
docker exec commute-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --create --if-not-exists --topic commute.alerts.realtime --partitions 3 --replication-factor 1
docker compose ps
```

### 2. Verify services
```powershell
docker compose ps
docker exec commute-kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list
docker exec commute-namenode hdfs dfs -ls -R /datalake
```

Endpoints:
- Kafka bootstrap (host): `localhost:29092` (container internal: `kafka:9092`)
- HDFS NameNode UI: `http://localhost:9870`
- HDFS RPC: `hdfs://localhost:9000` (host) / `hdfs://namenode:9000` (containers)
- Spark Master UI: `http://localhost:8080`
- Spark Worker UI: `http://localhost:8081`

### 3. Start Spark Structured Streaming (Kafka -> Bronze/Silver/Alerts)
**Git Bash / WSL**
```bash
bash scripts/run_bigdata_pipeline.sh submit-stream
```

**PowerShell**
```powershell
docker exec commute-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 `
  --conf spark.executor.memory=1g `
  --conf spark.driver.memory=1g `
  --conf spark.sql.shuffle.partitions=4 `
  --conf spark.sql.streaming.schemaInference=false `
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 `
  /opt/workspace/src/streaming/spark_stream_processor.py `
  --kafka-bootstrap kafka:9092 `
  --hdfs-base hdfs://namenode:9000/datalake `
  --checkpoint-base hdfs://namenode:9000/datalake/checkpoints/stream
```

> Package/version note: `scripts/run_bigdata_pipeline.sh` uses `org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5`, which matches the Spark image tag `apache/spark:3.5.5` in `docker-compose.yml`.

### 4. Publish telemetry events to Kafka
The producer runs on the host and publishes to `localhost:29092`.

```powershell
# Ensure local DuckDB has data for replay mode
python -m src.ingestion.pipeline

# Replay historical observations to Kafka
python -m src.streaming.kafka_producer --bootstrap-servers localhost:29092 --mode replay --route-id bsd_jakarta --limit 200

# Synthetic stream (no external APIs)
python -m src.streaming.kafka_producer --bootstrap-servers localhost:29092 --mode synthetic --route-id bsd_jakarta --limit 200

# Live polling (TomTom/Open-Meteo with fallback)
python -m src.streaming.kafka_producer --bootstrap-servers localhost:29092 --mode live --route-id bsd_jakarta --interval-seconds 60
```

### 5. Run Spark Gold Batch ETL (Silver -> Gold -> DuckDB sync)
**Git Bash / WSL**
```bash
bash scripts/run_bigdata_pipeline.sh submit-batch
```

**PowerShell**
```powershell
docker exec commute-spark-master /opt/spark/bin/spark-submit `
  --master spark://spark-master:7077 `
  --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 `
  --conf spark.executor.memory=1g `
  --conf spark.driver.memory=1g `
  --conf spark.sql.shuffle.partitions=4 `
  --conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000 `
  /opt/workspace/src/batch/spark_gold_etl.py `
  --hdfs-base hdfs://namenode:9000/datalake `
  --duckdb-path /opt/workspace/data/commute.duckdb
```

What this does:
- Reads Silver Parquet (`/datalake/silver/commute_enriched`, and optionally `/datalake/silver/weather_rolling_3h`)
- Computes leakage-safe PySpark window features and time/rush/rain interaction features aligned to `MODEL_FEATURES`
- Writes Gold Feature Store to `/datalake/gold/feature_store` partitioned by `route_id,event_date`
- Upserts snapshot tables into local DuckDB (`gold_features`, `trip_observations`, `weather_hourly`, route baseline updates) so existing `src/models/train.py` and `src/app/dashboard.py` keep working

### 6. Train models from Lakehouse-synced DuckDB
```powershell
python -m src.models.train --route-id bsd_jakarta
```

### 7. Launch dashboard (with Big Data status + Kafka alerts)
```powershell
streamlit run src/app/dashboard.py
```

In the sidebar, open **"Big Data Infrastructure Status"** to see:
- Kafka broker reachability (`KAFKA_BOOTSTRAP_SERVERS`, default `localhost:29092`)
- HDFS NameNode reachability (`HDFS_NAMENODE_UI`, default `http://localhost:9870`)
- Recent real-time alerts consumed from `commute.alerts.realtime`

## Local-Mode Spark Testing (no Docker required)

`src/batch/spark_gold_etl.py` supports `--local-mode` and local filesystem paths (e.g., `./data/lakehouse`). This is useful for testing Spark transforms against local Parquet without HDFS containers.

```powershell
# Example: run Gold ETL locally against local lakehouse Parquet
python -m src.batch.spark_gold_etl --local-mode --hdfs-base ./data/lakehouse --duckdb-path ./data/commute.duckdb

# or spark-submit local mode
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 `
  src/batch/spark_gold_etl.py --local-mode --hdfs-base ./data/lakehouse --duckdb-path ./data/commute.duckdb
```

For the streaming processor, `--local-mode` is also supported:
```powershell
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5 `
  src/streaming/spark_stream_processor.py --local-mode --kafka-bootstrap localhost:9092 --hdfs-base ./data/lakehouse
```

## Feature Parity / Modeling Notes

- `src/features/build_features.py` remains the pandas reference feature pipeline for the MVP.
- `src/batch/spark_gold_etl.py` computes the same model-facing columns (`MODEL_FEATURES`) in PySpark, including:
  - cyclical time: `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
  - rush flags/intensity: `is_morning_rush`, `is_evening_rush`, `rush_intensity`
  - leakage-safe rolling rain: `precip_rolling_3h`, `rain_rolling_3h` using prior-row window (`rowsBetween(-12,-1)`) equivalent to pandas `shift(1).rolling(window=12, min_periods=1).mean()`
  - interaction: `rain_x_rush`
- `tests/test_spark_transformations.py` validates Spark/pandas parity for key transformations.

## Testing

```powershell
# MVP tests
pytest -q

# Spark parity tests (requires pyspark; skipped automatically if not installed)
pytest -q tests/test_spark_transformations.py
```

## Operational Notes / Guardrails

- **Schema enforcement**: Spark streaming uses explicit `StructType` schemas for Kafka JSON payloads; `spark.sql.streaming.schemaInference=false`.
- **Fault tolerance**: structured streaming checkpoints are stored under `/datalake/checkpoints/stream/...` for restart recovery.
- **Resource caps**: Docker services are configured with local-friendly JVM/memory limits (Kafka 512m-1g heap, Hadoop 1g, Spark master 1g daemon, worker 2G).
- **Compatibility**: existing DuckDB + LightGBM + Streamlit flow remains usable. Gold ETL syncs Lakehouse outputs into DuckDB for seamless training/dashboard use.
- **Security/local demo**: HDFS permissions are relaxed and Kafka is plaintext for local development only. Harden before any shared deployment.

## Troubleshooting

- **Spark package mismatch**: ensure the Kafka connector package matches Spark version (`3.5.5`).
- **Kafka connection from host**: use `localhost:29092`; from containers use `kafka:9092`.
- **No alerts in dashboard**: the sidebar reads from `latest` offsets and only shows newly consumed messages; produce/stream fresh alerts, then click "Refresh alerts".
- **Empty Gold ETL**: run the stream processor and producer first so Silver Parquet has data.
- **Windows shell script**: run `scripts/run_bigdata_pipeline.sh` via Git Bash/WSL, or use the PowerShell commands in this README.