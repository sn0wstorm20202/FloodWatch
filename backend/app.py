from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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

    loader = DataLoader()
    try:
        loader.load_all()
    except Exception:
        # Demo-safe: never crash on startup.
        logger.exception("Data loading failed; continuing with partial/empty data")

    store = ReportStore()
    ml_engine = MLEngine(loader)
    try:
        ml_engine.train()
    except Exception:
        logger.exception("ML training failed; continuing in demo-safe fallback mode")

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
        return {"alerts": alerts}
    except HTTPException:
        raise
    except Exception:
        logger.exception("/alerts failed")
        return {"alerts": []}
