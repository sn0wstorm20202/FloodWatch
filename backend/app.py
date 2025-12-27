from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import config
from alerts import AlertsEngine
from data_loader import DataLoader
from ml_engine import MLEngine
from routing_engine import RoutingEngine
from store import ReportStore
from utils import haversine_m, risk_level

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger("floodwatch")

loader: DataLoader | None = None
store: ReportStore | None = None
ml_engine: MLEngine | None = None
routing_engine: RoutingEngine | None = None
alerts_engine: AlertsEngine | None = None

DEFAULT_PLOTS = {
    "risk_heatmap": "risk_heatmap.png",
}

DEFAULT_PLOT_META = {
    "risk_heatmap": {
        "bounds": {
            "west": 88.30827491320474,
            "east": 88.47776904101242,
            "south": 22.478332926893593,
            "north": 22.65688806153375,
        },
        "bins": 80,
    }
}


def _ensure_training_report_loaded() -> dict | None:
    global ml_engine
    if ml_engine is None:
        return None
    if isinstance(getattr(ml_engine, "training_report", None), dict):
        return ml_engine.training_report
    try:
        if config.ML_REPORT_PATH.exists():
            with config.ML_REPORT_PATH.open("r", encoding="utf-8") as f:
                ml_engine.training_report = json.load(f)
    except Exception:
        logger.exception("Failed to load ML training report from disk")
        return None
    return ml_engine.training_report if isinstance(getattr(ml_engine, "training_report", None), dict) else None


class ReportIn(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    severity: int = Field(..., ge=config.CROWD_SEVERITY_MIN, le=config.CROWD_SEVERITY_MAX)


def _require(obj, name: str):
    if obj is None:
        raise HTTPException(status_code=503, detail=f"Service not ready: {name}")
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loader, store, ml_engine, routing_engine, alerts_engine

    logger.info("Starting FloodWatch backend (DEMO_MODE=%s)", config.DEMO_MODE)
    logger.info(
        "Startup flags: ML_AUTO_TRAIN_ON_STARTUP=%s ML_ALLOW_API_RETRAIN=%s LOAD_ROADS_GRAPH=%s ML_PERSIST_MODELS=%s",
        config.ML_AUTO_TRAIN_ON_STARTUP,
        config.ML_ALLOW_API_RETRAIN,
        config.LOAD_ROADS_GRAPH,
        config.ML_PERSIST_MODELS,
    )

    loader = DataLoader()
    try:
        loader.load_all()
    except Exception:
        # Demo-safe: never crash on startup.
        logger.exception("Data loading failed; continuing with partial/empty data")

    store = ReportStore()
    ml_engine = MLEngine(loader)
    try:
        loaded = False
        try:
            if config.ML_PERSIST_MODELS:
                loaded = bool(ml_engine._try_load_persisted_model())
        except Exception:
            loaded = False

        if not loaded and config.ML_AUTO_TRAIN_ON_STARTUP:
            ml_engine.load_or_train(force_retrain=True)
    except Exception:
        logger.exception("ML training failed; continuing in demo-safe fallback mode")

    try:
        if ml_engine is not None and ml_engine.training_report is None and config.ML_REPORT_PATH.exists():
            with config.ML_REPORT_PATH.open("r", encoding="utf-8") as f:
                ml_engine.training_report = json.load(f)
    except Exception:
        logger.exception("Failed to load ML training report from disk")

    routing_engine = RoutingEngine(loader, ml_engine)
    alerts_engine = AlertsEngine(store, ml_engine)

    yield

    # Best-effort cleanup
    try:
        if loader and loader.elevation_ds:
            loader.elevation_ds.close()
        if loader and loader.landcover_ds:
            loader.landcover_ds.close()
    except Exception:
        logger.exception("Dataset cleanup failed")


app = FastAPI(title="FloodWatch Kolkata – Backend Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/ready")
def readiness_check():
    return {
        "ready": all(x is not None for x in [loader, store, ml_engine, routing_engine, alerts_engine]),
        "components": {
            "loader": loader is not None,
            "store": store is not None,
            "ml_engine": ml_engine is not None,
            "routing_engine": routing_engine is not None,
            "alerts_engine": alerts_engine is not None,
        },
        "ml_trained": bool(getattr(ml_engine, "models", None) is not None) if ml_engine is not None else False,
    }


@app.get("/risk")
def get_risk(
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
):
    try:
        me = _require(ml_engine, "ml_engine")
        out = me.predict_risk(float(lat), float(lon))
        score = float(out["ml_risk"])
        return {
            "risk_score": score,
            "risk_level": risk_level(score),
            "ml_details": {
                "cluster": int(out["cluster_id"]),
                "anomaly": float(out["anomaly_score"]),
            },
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("/risk failed")
        # Demo-safe: return a meaningful fallback instead of failing.
        return {
            "risk_score": 0.75 if config.DEMO_MODE else 0.40,
            "risk_level": "Flooded" if config.DEMO_MODE else "Risky",
            "ml_details": {"cluster": 0, "anomaly": 0.5},
        }


@app.get("/route")
def get_route(
    start_lat: float = Query(..., ge=-90.0, le=90.0),
    start_lon: float = Query(..., ge=-180.0, le=180.0),
    end_lat: float = Query(..., ge=-90.0, le=90.0),
    end_lon: float = Query(..., ge=-180.0, le=180.0),
):
    rt = _require(routing_engine, "routing_engine")
    try:
        dist_km, route_risk, geom = rt.route(float(start_lat), float(start_lon), float(end_lat), float(end_lon))
        return {
            "distance_km": float(round(dist_km, 3)),
            "route_risk": float(round(route_risk, 3)),
            "geometry": geom,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("/route failed")
        # Demo-safe: never crash live. Return a straight-line fallback.
        me = _require(ml_engine, "ml_engine")
        dist_m = haversine_m(float(start_lat), float(start_lon), float(end_lat), float(end_lon))
        route_risk = (
            float(me.predict_risk(float(start_lat), float(start_lon))["ml_risk"]) +
            float(me.predict_risk(float(end_lat), float(end_lon))["ml_risk"])
        ) / 2.0
        return {
            "distance_km": float(round(dist_m / 1000.0, 3)),
            "route_risk": float(round(route_risk, 3)),
            "geometry": {
                "type": "LineString",
                "coordinates": [[float(start_lon), float(start_lat)], [float(end_lon), float(end_lat)]],
            },
        }


@app.post("/report")
def post_report(payload: ReportIn = Body(...)):
    st = _require(store, "store")
    try:
        st.add_report(payload.lat, payload.lon, payload.severity)
        return {"status": "report received"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("/report failed")
        # Demo-safe
        return {"status": "report received"}


@app.get("/alerts")
def get_alerts():
    ae = _require(alerts_engine, "alerts_engine")
    try:
        alerts = ae.get_alerts()
        legacy = []
        try:
            legacy = [a.get("message") for a in alerts if isinstance(a, dict) and isinstance(a.get("message"), str)]
        except Exception:
            legacy = []
        return {"alerts": alerts, "messages": legacy}
    except HTTPException:
        raise
    except Exception:
        logger.exception("/alerts failed")
        return {"alerts": [], "messages": []}


@app.get("/ml/stats")
def get_ml_stats():
    me = _require(ml_engine, "ml_engine")
    return {"trained": me.models is not None, "training_report": me.training_report}


@app.get("/ml/plots")
def get_ml_plots():
    me = _require(ml_engine, "ml_engine")
    report = _ensure_training_report_loaded()
    plots: dict = {}
    if isinstance(report, dict):
        plots = report.get("plots") or {}

    try:
        if (not plots) and config.ML_PLOTS_DIR.exists():
            for p in config.ML_PLOTS_DIR.glob("*.png"):
                plots[p.stem] = p.name
    except Exception:
        pass

    if not plots:
        plots = dict(DEFAULT_PLOTS)
    return {"plots": plots}


@app.get("/ml/plot/{plot_name}")
def get_ml_plot(plot_name: str):
    me = _require(ml_engine, "ml_engine")
    report = _ensure_training_report_loaded()

    filename = None
    if isinstance(report, dict):
        plots = report.get("plots") or {}
        filename = plots.get(plot_name)

    if not isinstance(filename, str) or not filename:
        filename = DEFAULT_PLOTS.get(plot_name)

    if not isinstance(filename, str) or not filename:
        candidate = config.ML_PLOTS_DIR / f"{plot_name}.png"
        try:
            if candidate.exists():
                filename = candidate.name
        except Exception:
            filename = None

    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=404, detail="Plot not found")

    p = config.ML_PLOTS_DIR / filename
    try:
        if not p.exists():
            raise HTTPException(status_code=404, detail="Plot file missing")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Plot file missing")
    return FileResponse(path=str(p), media_type="image/png", filename=filename)


@app.get("/ml/plot_meta/{plot_name}")
def get_ml_plot_meta(plot_name: str):
    me = _require(ml_engine, "ml_engine")
    report = _ensure_training_report_loaded()

    meta = {}
    if isinstance(report, dict):
        meta = report.get("plot_meta") or {}
        if not isinstance(meta, dict):
            meta = {}

    out = meta.get(plot_name) if isinstance(meta, dict) else None
    if out is None:
        out = DEFAULT_PLOT_META.get(plot_name)
    if out is None:
        raise HTTPException(status_code=404, detail="Plot metadata not found")
    return {"plot": plot_name, "meta": out}


@app.post("/ml/retrain")
def retrain_ml():
    me = _require(ml_engine, "ml_engine")
    if not config.ML_ALLOW_API_RETRAIN:
        raise HTTPException(status_code=403, detail="API retrain disabled")
    try:
        me.load_or_train(force_retrain=True)
    except Exception:
        logger.exception("/ml/retrain failed")
        raise HTTPException(status_code=500, detail="Retrain failed")
    return {"status": "retrained", "trained": me.models is not None, "training_report": me.training_report}
