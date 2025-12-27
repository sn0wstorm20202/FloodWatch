# FloodWatch (Kolkata) — Backend

FastAPI backend for flood risk scoring + flood-aware routing + crowd reports + ML training persistence + training reports/plots.

## Quick start (Windows)

### 1) Create venv (recommended location)

Run these commands from the project root:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2) Run the API server

```powershell
cd backend
.\venv\Scripts\uvicorn.exe app:app --reload --host 127.0.0.1 --port 8000
```

Open:
- `http://127.0.0.1:8000/docs` (Swagger UI)
- `http://127.0.0.1:8000/openapi.json` (OpenAPI spec for frontend)

## API endpoints (for frontend)

Base URL (local): `http://127.0.0.1:8000`

### Core endpoints

#### 1) GET `/risk`

Query params:
- `lat` (float)
- `lon` (float)

Response:
- `risk_score` (0..1)
- `risk_level` (string)
- `ml_details.cluster` (int)
- `ml_details.anomaly` (0..1)

Example:
- `/risk?lat=22.57&lon=88.36`

#### 2) GET `/route`

Query params:
- `start_lat`, `start_lon`, `end_lat`, `end_lon`

Response:
- `distance_km`
- `route_risk` (0..1)
- `geometry` (GeoJSON LineString)

Example:
- `/route?start_lat=22.57&start_lon=88.36&end_lat=22.656&end_lon=88.438`

#### 3) POST `/report`

Body (JSON):
```json
{ "lat": 22.57, "lon": 88.36, "severity": 4 }
```

Response:
- `{ "status": "report received" }`

#### 4) GET `/alerts`

Response:
- `{ "alerts": [...], "messages": [...] }`

`alerts` is marker-friendly (each entry is a dict):
- `lat`, `lon`
- `risk_score`
- `count`
- `severity_max`, `severity_avg`
- `message`

`messages` is a legacy list of strings for simple UIs.

### ML / training report endpoints

#### 5) GET `/ml/stats`

Response:
- `{ "trained": true/false, "training_report": { ... } }`

This includes (examples):
- `samples`, `pseudo_label_rate`, `cluster_counts`, `silhouette_score`
- `holdout_metrics` (accuracy/precision/recall/f1/roc_auc/etc)
- `classifier_coefficients`
- `plots` (mapping of plot keys -> PNG filenames)

#### 6) GET `/ml/plots`

Response:
- `{ "plots": { "plot_key": "filename.png", ... } }`

Plot keys currently generated:
- `cluster_counts`
- `anomaly_hist`
- `risk_hist`
- `feature_hists`
- `pca_clusters`
- `risk_heatmap`

#### 7) GET `/ml/plot/{plot_key}`

Serves a PNG.

Examples:
- `/ml/plot/cluster_counts`
- `/ml/plot/risk_heatmap`

#### 7b) GET `/ml/plot_meta/{plot_key}`

Returns metadata needed to correctly place the plot on a map.

Example:
- `/ml/plot_meta/risk_heatmap`

Response shape:
```json
{
  "plot": "risk_heatmap",
  "meta": {
    "bounds": {"west": 88.30, "east": 88.47, "south": 22.47, "north": 22.65},
    "bins": 80
  }
}
```

#### 8) POST `/ml/retrain`

Forces retraining immediately.

Response:
- `{ "status": "retrained", "trained": true/false, "training_report": { ... } }`

## ML model persistence (important)

### Where artifacts are saved

After training, artifacts are stored here:
- `backend/ml_artifacts/ml_model.joblib` (persisted model)
- `backend/ml_artifacts/ml_training_report.json` (training report)
- `backend/ml_artifacts/plots/*.png` (training plots)

### When the server retrains vs loads

On startup the backend calls `load_or_train(force_retrain=False)`:
- If a persisted model exists AND the dataset fingerprint matches current dataset files, it **loads** the model (fast).
- If the fingerprint changed (you changed/replaced a dataset file), it **re-trains** and saves a new model.

### How to retrain when you change datasets

Option A (recommended): restart the server after changing dataset files.
- The backend will detect changes (mtime/size fingerprint) and retrain automatically.

Option B: call the retrain API.

PowerShell:
```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/ml/retrain"
```

cURL:
```bash
curl -X POST http://127.0.0.1:8000/ml/retrain
```

## Notes

- Plot generation uses matplotlib with a headless backend (`Agg`).
- If you ever want to disable plot generation: set env `FLOODWATCH_DISABLE_PLOTS=1` before starting the server.

## React Native map layer (Waterlogging Layer)

Recommended flow for your `MapLayersSheet` waterlogging toggle:

### Toggle ON

1) Fetch the PNG heatmap:
- `GET /ml/plot/risk_heatmap`

2) Fetch overlay bounds:
- `GET /ml/plot_meta/risk_heatmap`

Use the returned `bounds` to place the PNG overlay correctly on the map.

3) Fetch alert markers:
- `GET /alerts`

Render one marker per `alerts[]` item using `lat`/`lon` and show `message`/`severity_max`.

### Toggle OFF

Remove overlay + clear markers.
