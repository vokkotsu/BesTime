# Commute Departure & Delay Risk Optimizer

MVP for an **Anti-Traffic & Rain Commute Calculator** that models commute travel time as a departure-time curve (P50/P90), instead of a single static ETA. It flags:

- **Diminishing Return Zones**: leaving much earlier saves little/no time because rush-hour saturation is already high.
- **Delay Cliff Zones**: leaving slightly later causes a non-linear jump in travel time (rush wave + rain interaction / tail risk).

## Architecture

Lightweight, local-first, modular Python stack:

- **Python 3.11+**
- **DuckDB** (`data/commute.duckdb`) for local analytical storage
- **Parquet** exports possible via DuckDB/pandas (raw/processed snapshots)
- **Open-Meteo** for hourly weather forecast/archive (`precipitation`, `rain`, `weathercode`, `windspeed_10m`) - no API key
- **OSRM public demo API** for baseline route distance/duration, with a realistic synthetic rush-hour + rain penalty fallback simulator when unavailable
- **User Trip Logger** stored in DuckDB (`trip_observations`) for continuous improvement
- **pandas/numpy/scikit-learn/lightgbm** for feature engineering and quantile regression (`P50`, `P90`)
- **Streamlit** dashboard for interactive scenario exploration

## Project Structure

```text
commute-optimizer/  (this workspace root contains the MVP files)
├── data/                       # DuckDB database & generated artifacts/models
├── src/
│   ├── config.py               # Routes, time grid, API endpoints, model config
│   ├── ingestion/
│   │   ├── weather.py          # Open-Meteo hourly forecast & historical fetch
│   │   ├── routing.py          # OSRM baseline + synthetic rush-hour/rain simulator
│   │   └── pipeline.py         # Idempotent ETL orchestrator into DuckDB
│   ├── features/
│   │   └── build_features.py   # Cyclical time, rush flags, rolling rain, interactions
│   ├── models/
│   │   ├── train.py            # LightGBM quantile regressors (P50/P90), temporal split
│   │   └── predict.py          # 15-min departure curve + zone detection
│   └── app/
│       └── dashboard.py        # Streamlit UI
├── tests/
│   └── test_features_and_cliff.py
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

## Run the MVP

### 1. Bootstrap data (ingestion + synthetic trips if no logs)

```bash
python -m src.ingestion.pipeline
```

This creates/updates `data/commute.duckdb` with:
- `routes`
- `weather_hourly`
- `trip_observations` (bootstrapped synthetic observations when empty)

### 2. Train quantile models (P50/P90)

```bash
python -m src.models.train --route-id bsd_jakarta
```

Model artifacts and metrics are saved under `data/models/`.

### 3. Launch dashboard

```bash
streamlit run src/app/dashboard.py
```

The dashboard lets you:
- select a sample route and date
- choose a weather scenario (Forecast/Dry/Light rain/Heavy rain)
- view P50/P90 departure-vs-duration curve
- see flagged diminishing-return and delay-cliff zones
- log actual commute trips into DuckDB

## How It Works

### Data ingestion (`src/ingestion/`)
- **Weather**: pulls hourly Open-Meteo forecast + recent archive data.
- **Routing baseline**: asks OSRM for route distance/duration; falls back to haversine + synthetic baseline.
- **Synthetic bootstrap**: generates realistic non-linear trip durations using rush-hour intensity and rain penalties (including rare incident tail risk) when user logs are sparse.
- **Idempotent ETL**: upserts by keys (`route_id,timestamp_local`, `trip_id`) so reruns are safe.

### Feature engineering (`src/features/build_features.py`)
- Cyclical time encoding: hour/day-of-week sin/cos
- Rush-hour flags and smooth rush intensity
- Rolling rain features (`3h`) with leakage protection (`shift`)
- Interaction term: `rain_x_rush`
- Weather features: precipitation, rain, weathercode, windspeed

### Modeling (`src/models/`)
- **LightGBM Quantile Regressors** with `objective="quantile"`
  - `alpha=0.5` for P50 (median)
  - `alpha=0.9` for P90 (tail risk)
- **Temporal train/test split** (last N days as test; ratio fallback for small data)
- Metrics: MAE, pinball loss, P90 coverage, baseline hour-median MAE

### Prediction & zone detection (`src/models/predict.py`)
- Builds 15-minute departure grid for a selected day
- Predicts P50/P90 duration and arrival times
- Detects:
  - **Delay cliff**: sharp increase vs previous slot (absolute or percentage threshold)
  - **Diminishing return**: leaving earlier by 45 min saves <= configured threshold
  - **High tail risk**: elevated P90 spread

## Testing

```bash
pytest -q
```

Included tests cover:
- delay-cliff detection
- diminishing-return detection
- feature inference columns
- pipeline + feature smoke test on a temporary DuckDB

## Notes / Production Next Steps

- Replace synthetic bootstrap with real user trip logs over time (the logger is already wired).
- Add route-specific calibration and holiday/event features.
- Use scheduled ingestion (cron/Airflow/Dagster) and model retraining.
- Add map visualization and confidence intervals beyond P50/P90.
- Respect OSRM demo usage policies; self-host OSRM or use a production routing provider for scale.
