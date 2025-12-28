from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

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

DEFAULT_PLOTS = {
    "risk_heatmap": "risk_heatmap.png",
    "waterlogging_hotspots_28y": "waterlogging_hotspots_28y.png",
    "waterlogging_combined": "waterlogging_combined.png",
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
    },
    "waterlogging_hotspots_28y": {
        "bounds": {
            "west": 88.30827491320474,
            "east": 88.47776904101242,
            "south": 22.478332926893593,
            "north": 22.65688806153375,
        },
        "bins": 256,
    },
}


_waterlogging_hotspots_meta: dict | None = None


_waterlogging_combined_meta: dict | None = None
_waterlogging_combined_last_write_ts: float = 0.0
_waterlogging_combined_last_range_sig: tuple[float, float] | None = None


_risk_heatmap_last_write_ts: float = 0.0
_risk_heatmap_last_at_sig: float | None = None


def _ensure_risk_heatmap_plot(at_ts: float | None = None) -> Path:
    global _risk_heatmap_last_write_ts, _risk_heatmap_last_at_sig

    meta = DEFAULT_PLOT_META.get("risk_heatmap") if isinstance(DEFAULT_PLOT_META, dict) else None
    if not isinstance(meta, dict):
        raise HTTPException(status_code=404, detail="Plot not found")

    b = meta.get("bounds") if isinstance(meta.get("bounds"), dict) else {}
    west = float(b.get("west"))
    east = float(b.get("east"))
    south = float(b.get("south"))
    north = float(b.get("north"))
    bins = int(meta.get("bins") or 80)

    sig = float(at_ts) if at_ts is not None else None
    out_name = "risk_heatmap.png" if sig is None else f"risk_heatmap_{int(sig)}.png"
    out_path = config.ML_PLOTS_DIR / out_name
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    now = time.time()
    # Prefer reusing already-generated timestamped heatmaps even across restarts.
    try:
        if sig is not None and out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        pass

    # For live (no at=), allow brief caching so we don't regenerate on every request.
    try:
        if (
            sig is None
            and out_path.exists()
            and out_path.stat().st_size > 0
            and (now - float(out_path.stat().st_mtime)) < 60.0
        ):
            return out_path
    except Exception:
        pass

    ld = _require(loader, "loader")
    me = _require(ml_engine, "ml_engine")

    rainfall_raw = None
    try:
        rainfall_raw = ld.rainfall_value_at(float(sig)) if sig is not None else ld.latest_rainfall_value()
    except Exception:
        rainfall_raw = None

    # Performance note:
    # Full-resolution (bins x bins) inference can be slow because it calls into raster sampling
    # and ML inference thousands of times. For interactive map overlays we compute a smaller grid
    # and upsample it back to (bins x bins).
    compute_bins = int(min(int(bins), 12))
    lats_small = np.linspace(float(south), float(north), int(compute_bins), dtype=float)
    lons_small = np.linspace(float(west), float(east), int(compute_bins), dtype=float)
    grid_small = np.zeros((int(compute_bins), int(compute_bins)), dtype=float)

    for i, lat in enumerate(lats_small):
        for j, lon in enumerate(lons_small):
            try:
                if sig is None:
                    grid_small[i, j] = float(me.predict_risk(float(lat), float(lon))["ml_risk"])
                else:
                    grid_small[i, j] = float(me.predict_risk_at(float(lat), float(lon), float(sig))["ml_risk"])
            except Exception:
                grid_small[i, j] = 0.0

    # Upsample to bins using separable 1D interpolation.
    x_small = np.linspace(0.0, 1.0, int(compute_bins), dtype=float)
    x_full = np.linspace(0.0, 1.0, int(bins), dtype=float)

    tmp = np.zeros((int(compute_bins), int(bins)), dtype=float)
    for i in range(int(compute_bins)):
        tmp[i, :] = np.interp(x_full, x_small, np.asarray(grid_small[i, :], dtype=float))

    grid = np.zeros((int(bins), int(bins)), dtype=float)
    for j in range(int(bins)):
        grid[:, j] = np.interp(x_full, x_small, np.asarray(tmp[:, j], dtype=float))

    grid = np.clip(grid, 0.0, 1.0)
    grid = np.flipud(grid)

    v = np.asarray(grid, dtype=float)
    t = np.clip(v, 0.0, 1.0)

    # Simple heatmap gradient: blue -> cyan -> yellow -> red
    c0 = np.array([0.00, 0.10, 0.60], dtype=float)
    c1 = np.array([0.00, 0.85, 0.95], dtype=float)
    c2 = np.array([1.00, 0.95, 0.00], dtype=float)
    c3 = np.array([1.00, 0.00, 0.00], dtype=float)

    rgb = np.zeros((t.shape[0], t.shape[1], 3), dtype=float)
    m1 = t < (1.0 / 3.0)
    m2 = (t >= (1.0 / 3.0)) & (t < (2.0 / 3.0))
    m3 = t >= (2.0 / 3.0)

    a1 = np.zeros_like(t)
    a2 = np.zeros_like(t)
    a3 = np.zeros_like(t)
    a1[m1] = t[m1] * 3.0
    a2[m2] = (t[m2] - (1.0 / 3.0)) * 3.0
    a3[m3] = (t[m3] - (2.0 / 3.0)) * 3.0

    for k in range(3):
        rgb[..., k] = 0.0
        rgb[..., k] += m1.astype(float) * (c0[k] + (c1[k] - c0[k]) * a1)
        rgb[..., k] += m2.astype(float) * (c1[k] + (c2[k] - c1[k]) * a2)
        rgb[..., k] += m3.astype(float) * (c2[k] + (c3[k] - c2[k]) * a3)

    alpha = np.clip(t ** 0.75, 0.0, 1.0) * 0.85
    alpha = np.where(t < 0.02, 0.0, alpha)

    rgba = np.zeros((t.shape[0], t.shape[1], 4), dtype=float)
    rgba[..., 0:3] = np.clip(rgb, 0.0, 1.0)
    rgba[..., 3] = np.clip(alpha, 0.0, 1.0)

    img = (rgba * 255.0).astype("uint8")
    try:
        Image.fromarray(img, mode="RGBA").save(str(out_path), format="PNG")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save heatmap")

    _risk_heatmap_last_write_ts = float(now)
    _risk_heatmap_last_at_sig = sig
    _log_event(
        "plot_generated",
        {
            "plot": "risk_heatmap",
            "bins": int(bins),
            "compute_bins": int(compute_bins),
            "at_ts": float(sig) if sig is not None else None,
            "rainfall_raw": float(rainfall_raw) if rainfall_raw is not None else None,
        },
    )
    try:
        logger.info(
            "Risk heatmap generated: at_ts=%s rainfall_raw=%s bins=%d",
            str(int(sig)) if sig is not None else "live",
            _fmt_num(rainfall_raw, 3),
            int(bins),
        )
    except Exception:
        pass
    return out_path


def _compute_waterlogging_hotspots_meta(bins: int = 256) -> dict | None:
    global loader, _waterlogging_hotspots_meta
    if isinstance(_waterlogging_hotspots_meta, dict):
        return _waterlogging_hotspots_meta

    ld = loader
    if ld is None or ld.surface_water_df is None or ld._water_points_lonlat is None:
        return None

    pts = ld._water_points_lonlat
    try:
        lon = pts[:, 0].astype(float)
        lat = pts[:, 1].astype(float)
    except Exception:
        return None

    if lon.size == 0 or lat.size == 0:
        return None

    try:
        west = float(np.nanpercentile(lon, 0.5))
        east = float(np.nanpercentile(lon, 99.5))
        south = float(np.nanpercentile(lat, 0.5))
        north = float(np.nanpercentile(lat, 99.5))
    except Exception:
        try:
            west = float(lon.min())
            east = float(lon.max())
            south = float(lat.min())
            north = float(lat.max())
        except Exception:
            return None

    try:
        pad_lon = max(0.001, (east - west) * 0.02)
        pad_lat = max(0.001, (north - south) * 0.02)
        west -= pad_lon
        east += pad_lon
        south -= pad_lat
        north += pad_lat
    except Exception:
        pass

    out = {
        "bounds": {
            "west": west,
            "east": east,
            "south": south,
            "north": north,
        },
        "bins": int(bins),
        "metric": "water_occurrence",
        "metric_units": "percent_of_time",
    }
    _waterlogging_hotspots_meta = out
    return out


def _compute_waterlogging_combined_meta(bins: int = 256, hours: int = 6) -> dict | None:
    global _waterlogging_combined_meta
    if isinstance(_waterlogging_combined_meta, dict):
        cached = _waterlogging_combined_meta
        if int(cached.get("hours", -1)) == int(hours) and int(cached.get("bins", -1)) == int(bins):
            return cached

    base = _compute_waterlogging_hotspots_meta(bins=bins)
    if not isinstance(base, dict):
        return None
    out = dict(base)
    out["sources"] = ["surface_water_occurrence", "crowd_reports"]
    out["hours"] = int(hours)
    _waterlogging_combined_meta = out
    return out


def _ensure_waterlogging_combined_plot(hours: int = 6, since_ts: float | None = None, until_ts: float | None = None) -> Path:
    global _waterlogging_combined_last_write_ts, _waterlogging_combined_last_range_sig
    meta = _compute_waterlogging_combined_meta(bins=256, hours=int(hours))
    if not isinstance(meta, dict):
        raise HTTPException(status_code=503, detail="Surface water dataset not available")

    out_path = config.ML_PLOTS_DIR / "waterlogging_combined.png"
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    now = time.time()
    if since_ts is not None or until_ts is not None:
        if since_ts is None or until_ts is None:
            raise HTTPException(status_code=400, detail="Both 'from' and 'to' are required")
        try:
            since_ts = float(since_ts)
            until_ts = float(until_ts)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid 'from'/'to' timestamp")
        if until_ts <= since_ts:
            raise HTTPException(status_code=400, detail="Invalid time window: to must be after from")
        range_sig: tuple[float, float] | None = (float(since_ts), float(until_ts))
    else:
        since_ts = now - float(int(hours)) * 3600.0
        until_ts = now
        range_sig = None

    try:
        if (
            out_path.exists()
            and out_path.stat().st_size > 0
            and (now - float(_waterlogging_combined_last_write_ts)) < 15.0
            and (
                (range_sig is None)
                or (
                    isinstance(_waterlogging_combined_last_range_sig, tuple)
                    and float(_waterlogging_combined_last_range_sig[0]) == float(range_sig[0])
                    and float(_waterlogging_combined_last_range_sig[1]) == float(range_sig[1])
                )
            )
        ):
            return out_path
    except Exception:
        pass

    ld = _require(loader, "loader")
    st = _require(store, "store")
    if ld._water_points_lonlat is None:
        raise HTTPException(status_code=503, detail="Surface water dataset not available")

    try:
        import matplotlib.cm as cm
    except Exception:
        raise HTTPException(status_code=500, detail="Plot dependencies missing")

    b = meta.get("bounds") if isinstance(meta.get("bounds"), dict) else {}
    west = float(b.get("west"))
    east = float(b.get("east"))
    south = float(b.get("south"))
    north = float(b.get("north"))
    bins = int(meta.get("bins") or 256)

    pts = ld._water_points_lonlat
    lon = pts[:, 0].astype(float)
    lat = pts[:, 1].astype(float)

    base_weights = None
    try:
        w = getattr(ld, "_water_occurrence", None)
        if w is not None:
            w = np.asarray(w, dtype=float)
            if w.shape[0] == lat.shape[0]:
                occ = np.clip(w, 0.0, 1.0)
                base_weights = np.sqrt(np.clip(occ * (1.0 - occ), 0.0, 1.0))
                base_weights = np.where(occ >= 0.70, 0.0, base_weights)
    except Exception:
        base_weights = None

    try:
        hist_base, _xedges, _yedges = np.histogram2d(
            lat,
            lon,
            bins=(bins, bins),
            range=[[south, north], [west, east]],
            weights=base_weights,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate base heatmap")

    reports = []
    try:
        reports = st.all_reports()
    except Exception:
        reports = []

    rlat: list[float] = []
    rlon: list[float] = []
    rwt: list[float] = []
    for r in reports:
        try:
            rts = float(r.ts)
            if rts < float(since_ts) or rts > float(until_ts):
                continue
            sev = float(getattr(r, "severity", 1))
            sev_norm = max(0.0, min(1.0, (sev - float(config.CROWD_SEVERITY_MIN)) / float(config.CROWD_SEVERITY_MAX - config.CROWD_SEVERITY_MIN)))
            rlat.append(float(getattr(r, "lat")))
            rlon.append(float(getattr(r, "lon")))
            rwt.append(float(sev_norm) ** 1.25)
        except Exception:
            continue

    hist_live = np.zeros_like(hist_base, dtype=float)
    if rlat:
        try:
            hist_live, _x2, _y2 = np.histogram2d(
                np.asarray(rlat, dtype=float),
                np.asarray(rlon, dtype=float),
                bins=(bins, bins),
                range=[[south, north], [west, east]],
                weights=np.asarray(rwt, dtype=float),
            )
        except Exception:
            hist_live = np.zeros_like(hist_base, dtype=float)

    combined = np.asarray(hist_base, dtype=float) + (2.5 * np.asarray(hist_live, dtype=float))
    combined = np.flipud(combined)
    combined = np.nan_to_num(combined, nan=0.0)
    combined = np.log1p(combined)

    mx = float(combined.max()) if combined.size else 0.0
    if mx <= 0.0:
        raise HTTPException(status_code=500, detail="Heatmap has no data")

    norm = np.clip(combined / mx, 0.0, 1.0)
    cmap = cm.get_cmap("inferno")
    rgba = cmap(norm)

    alpha = np.clip(norm ** 0.65, 0.0, 1.0) * 0.85
    alpha = np.where(norm < 0.02, 0.0, alpha)
    rgba[..., 3] = alpha

    img = (rgba * 255.0).astype("uint8")
    try:
        Image.fromarray(img, mode="RGBA").save(str(out_path), format="PNG")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save heatmap")

    _waterlogging_combined_last_write_ts = float(now)
    _waterlogging_combined_last_range_sig = range_sig
    _log_event(
        "plot_generated",
        {
            "plot": "waterlogging_combined",
            "hours": int(hours),
            "report_points": int(len(rlat)),
            "from_ts": float(since_ts),
            "to_ts": float(until_ts),
        },
    )
    try:
        logger.info(
            "Waterlogging plot generated: from_ts=%.0f to_ts=%.0f hours=%d report_points=%d",
            float(since_ts),
            float(until_ts),
            int(hours),
            int(len(rlat)),
        )
    except Exception:
        pass
    return out_path


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


class ReportAtIn(BaseModel):
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    severity: int = Field(..., ge=config.CROWD_SEVERITY_MIN, le=config.CROWD_SEVERITY_MAX)
    ts: str = Field(...)


def _require(obj, name: str):
    if obj is None:
        raise HTTPException(status_code=503, detail=f"Service not ready: {name}")
    return obj


@asynccontextmanager
async def lifespan(app: FastAPI):
    global loader, store, ml_engine, routing_engine, alerts_engine, _startup_task

    logger.info("Starting FloodWatch backend (DEMO_MODE=%s)", config.DEMO_MODE)
    logger.info(
        "Startup flags: ML_AUTO_TRAIN_ON_STARTUP=%s ML_ALLOW_API_RETRAIN=%s LOAD_ROADS_GRAPH=%s ML_PERSIST_MODELS=%s",
        config.ML_AUTO_TRAIN_ON_STARTUP,
        config.ML_ALLOW_API_RETRAIN,
        config.LOAD_ROADS_GRAPH,
        config.ML_PERSIST_MODELS,
    )

    loader = DataLoader()
    store = ReportStore()
    ml_engine = MLEngine(loader)
    routing_engine = RoutingEngine(loader, ml_engine, store)
    alerts_engine = AlertsEngine(store, ml_engine)
    _log_event("startup", {"demo_mode": bool(config.DEMO_MODE)})

    async def _background_init() -> None:
        try:
            try:
                await asyncio.to_thread(loader.load_all)
                try:
                    rt = routing_engine
                    if rt is not None:
                        rt.graph = rt._prepare_graph(loader.graph)
                except Exception:
                    pass
            except Exception:
                # Demo-safe: never crash on startup.
                logger.exception("Data loading failed; continuing with partial/empty data")

            try:
                loaded = False
                try:
                    if config.ML_PERSIST_MODELS:
                        loaded = bool(ml_engine._try_load_persisted_model())
                except Exception:
                    loaded = False

                if loaded:
                    _log_event("ml_model_loaded", {})

                if not loaded and config.ML_AUTO_TRAIN_ON_STARTUP:
                    _log_event("ml_startup_train_start", {})
                    await asyncio.to_thread(ml_engine.load_or_train, True)
                    _log_event(
                        "ml_startup_train_done",
                        {"trained": bool(getattr(ml_engine, "models", None) is not None)},
                    )
            except Exception:
                logger.exception("ML training failed; continuing in demo-safe fallback mode")
                _log_event("ml_startup_train_failed", {})

            try:
                rep = getattr(ml_engine, "training_report", None)

                if not (isinstance(rep, dict) and rep):
                    try:
                        if config.ML_REPORT_PATH.exists():
                            with config.ML_REPORT_PATH.open("r", encoding="utf-8") as f:
                                rep = json.load(f)
                                ml_engine.training_report = rep
                    except Exception:
                        rep = None

                if isinstance(rep, dict) and rep:
                    logger.info("\n%s", _format_training_report(rep))

            except Exception:
                logger.info("ML training report available but could not be serialized")
        except Exception:
            logger.exception("Background startup failed")

    try:
        _startup_task = asyncio.create_task(_background_init())
    except Exception:
        _startup_task = None

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


@app.get("/")
def root():
    return {
        "service": "FloodWatch Kolkata – Backend Service",
        "status": "ok",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


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


@app.get("/ml/rainfall_times")
def get_rainfall_times(limit: int = Query(30, ge=1, le=500)):
    ld = loader
    if ld is None:
        ld = DataLoader()
    if getattr(ld, "_rain_ts_epoch", None) is None:
        try:
            ld._load_rainfall()
        except Exception:
            pass
    arr = getattr(ld, "_rain_ts_epoch", None)
    if arr is None:
        return {"times": []}
    try:
        ts_arr = list(arr)
    except Exception:
        ts_arr = []
    if not ts_arr:
        return {"times": []}

    ts_sel = ts_arr[-int(limit) :]
    out = []
    for ts in reversed(ts_sel):
        try:
            t = float(ts)
            out.append({"ts_epoch": t, "ts_utc": datetime.fromtimestamp(t, tz=timezone.utc).isoformat()})
        except Exception:
            continue
    return {"times": out}


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
    at: str | None = Query(None),
):
    try:
        me = _require(ml_engine, "ml_engine")
        ts_epoch = None
        if at is not None:
            ts_epoch = _parse_ts(str(at))
            if ts_epoch is None:
                raise HTTPException(status_code=400, detail="Invalid 'at' format")

            try:
                ld = _require(loader, "loader")
                rain_exact = None
                if getattr(ld, "_rain_ts_epoch", None) is not None:
                    rain_exact = ld.rainfall_value_at_exact(float(ts_epoch), tolerance_s=1.0)
                if getattr(ld, "_rain_ts_epoch", None) is not None and rain_exact is None:
                    raise HTTPException(status_code=404, detail="No data found for requested timestamp")
            except HTTPException:
                raise
            except Exception:
                pass

            out = me.predict_risk_at(float(lat), float(lon), float(ts_epoch))
        else:
            out = me.predict_risk(float(lat), float(lon))
        score = float(out["ml_risk"])
        resp = {
            "risk_score": score,
            "risk_level": risk_level(score),
            "ml_details": {
                "cluster": int(out["cluster_id"]),
                "anomaly": float(out["anomaly_score"]),
            },
        }

        if ts_epoch is not None:
            rainfall_raw = None
            try:
                ld = _require(loader, "loader")
                rainfall_raw = ld.rainfall_value_at_exact(float(ts_epoch), tolerance_s=1.0)
            except Exception:
                rainfall_raw = None
            resp["as_of"] = {
                "ts_epoch": float(ts_epoch),
                "ts_utc": datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).isoformat(),
                "rainfall_raw": float(rainfall_raw) if rainfall_raw is not None else None,
            }
            _log_event(
                "risk_query",
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "at_ts": float(ts_epoch),
                    "rainfall_raw": float(rainfall_raw) if rainfall_raw is not None else None,
                    "score": float(score),
                },
            )
            try:
                logger.info(
                    "Risk query: lat=%.6f lon=%.6f at_ts=%.0f rainfall_raw=%s score=%.3f",
                    float(lat),
                    float(lon),
                    float(ts_epoch),
                    _fmt_num(rainfall_raw, 3),
                    float(score),
                )
            except Exception:
                pass

        return resp
    except HTTPException:
        raise
    except Exception:
        logger.exception("/risk failed")
        # Demo-safe
        try:
            me = ml_engine
            fb = None
            if me is not None:
                fb_fn = getattr(me, "_predict_risk_fallback", None)
                if callable(fb_fn):
                    ts_epoch = _parse_ts(str(at)) if at is not None else None
                    rainfall_raw = None
                    try:
                        if ts_epoch is not None and loader is not None:
                            rainfall_raw = loader.rainfall_value_at(float(ts_epoch))
                    except Exception:
                        rainfall_raw = None
                    try:
                        fb = fb_fn(float(lat), float(lon), rainfall_raw_override=rainfall_raw)
                    except TypeError:
                        fb = fb_fn(float(lat), float(lon))

            if isinstance(fb, dict):
                score = float(fb.get("ml_risk", 0.40))
                return {
                    "risk_score": score,
                    "risk_level": risk_level(score),
                    "ml_details": {
                        "cluster": int(fb.get("cluster_id", 0)),
                        "anomaly": float(fb.get("anomaly_score", 0.5)),
                    },
                }
        except Exception:
            pass

        score = 0.75 if config.DEMO_MODE else 0.40
        return {
            "risk_score": score,
            "risk_level": risk_level(score),
            "ml_details": {"cluster": 0, "anomaly": 0.5},
        }


@app.get("/route")
def get_route(
    start_lat: float = Query(..., ge=-90.0, le=90.0),
    start_lon: float = Query(..., ge=-180.0, le=180.0),
    end_lat: float = Query(..., ge=-90.0, le=90.0),
    end_lon: float = Query(..., ge=-180.0, le=180.0),
    waterlogging: bool = Query(False),
):
    rt = _require(routing_engine, "routing_engine")
    try:
        dist_km, route_risk, geom = rt.route(
            float(start_lat),
            float(start_lon),
            float(end_lat),
            float(end_lon),
            use_crowd=bool(waterlogging),
        )

        is_fallback = False
        try:
            if isinstance(geom, dict) and geom.get("type") == "LineString":
                coords = geom.get("coordinates")
                if isinstance(coords, list) and len(coords) == 2:
                    a, b = coords[0], coords[1]
                    if (
                        isinstance(a, (list, tuple))
                        and isinstance(b, (list, tuple))
                        and len(a) >= 2
                        and len(b) >= 2
                    ):
                        eps = 1e-6
                        start_match = abs(float(a[0]) - float(start_lon)) < eps and abs(float(a[1]) - float(start_lat)) < eps
                        end_match = abs(float(b[0]) - float(end_lon)) < eps and abs(float(b[1]) - float(end_lat)) < eps
                        is_fallback = bool(start_match and end_match)
        except Exception:
            is_fallback = False

        return {
            "distance_km": float(round(dist_km, 3)),
            "route_risk": float(round(route_risk, 3)),
            "geometry": geom,
            "is_fallback": bool(is_fallback),
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
            "is_fallback": True,
        }


@app.post("/report")
def post_report(payload: ReportIn = Body(...)):
    st = _require(store, "store")
    try:
        st.add_report(payload.lat, payload.lon, payload.severity)
        _log_event("report_added", {"lat": float(payload.lat), "lon": float(payload.lon), "severity": int(payload.severity)})
        return {"status": "report received"}
    except HTTPException:
        raise
    except Exception:
        logger.exception("/report failed")
        return {"status": "report received"}


@app.post("/debug/report_at")
def debug_report_at(payload: ReportAtIn = Body(...)):
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail="Not found")
    st = _require(store, "store")
    ts = _parse_ts(str(payload.ts))
    if ts is None:
        raise HTTPException(status_code=400, detail="Invalid ts format")
    try:
        st.add_report_at(payload.lat, payload.lon, payload.severity, float(ts))
        _log_event(
            "report_added_at",
            {"lat": float(payload.lat), "lon": float(payload.lon), "severity": int(payload.severity), "ts": float(ts)},
        )
        return {"status": "report inserted", "ts": float(ts)}
    except HTTPException:
        raise
    except Exception:
        logger.exception("/debug/report_at failed")
        raise HTTPException(status_code=500, detail="Failed to insert report")


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
    report = _ensure_training_report_loaded()
    out = {"trained": me.models is not None, "training_report": report}
    try:
        out["model_path"] = str(config.ML_MODEL_PATH)
        out["report_path"] = str(config.ML_REPORT_PATH)
        out["model_exists"] = bool(config.ML_MODEL_PATH.exists())
        out["report_exists"] = bool(config.ML_REPORT_PATH.exists())
    except Exception:
        pass
    try:
        if config.ML_MODEL_PATH.exists():
            out["model_mtime"] = float(config.ML_MODEL_PATH.stat().st_mtime)
    except Exception:
        pass
    try:
        if config.ML_REPORT_PATH.exists():
            out["report_mtime"] = float(config.ML_REPORT_PATH.stat().st_mtime)
    except Exception:
        pass
    return out


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
def get_ml_plot(
    plot_name: str,
    hours: int = Query(6, ge=1, le=168),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
    at: str | None = Query(None),
):
    me = _require(ml_engine, "ml_engine")
    report = _ensure_training_report_loaded()

    base_name = str(plot_name)
    if base_name.lower().endswith(".png"):
        base_name = base_name[:-4]

    if base_name == "waterlogging_hotspots_28y":
        p = _ensure_waterlogging_hotspots_plot()
        return FileResponse(path=str(p), media_type="image/png", filename=p.name)

    if base_name == "risk_heatmap":
        ts_epoch = _parse_ts(str(at)) if at is not None else None
        if at is not None and ts_epoch is None:
            raise HTTPException(status_code=400, detail="Invalid 'at' format")

        if ts_epoch is not None:
            try:
                ld = _require(loader, "loader")
                if getattr(ld, "_rain_ts_epoch", None) is not None:
                    rain_exact = ld.rainfall_value_at_exact(float(ts_epoch), tolerance_s=1.0)
                    if rain_exact is None:
                        raise HTTPException(status_code=404, detail="No data found for requested timestamp")
            except HTTPException:
                raise
            except Exception:
                pass
        p = _ensure_risk_heatmap_plot(at_ts=float(ts_epoch) if ts_epoch is not None else None)
        return FileResponse(path=str(p), media_type="image/png", filename=p.name)

    if base_name == "waterlogging_combined":
        if from_ is not None or to is not None:
            if from_ is None or to is None:
                raise HTTPException(status_code=400, detail="Both 'from' and 'to' are required")
            s = _parse_ts(from_)
            u = _parse_ts(to)
            if s is None or u is None:
                raise HTTPException(status_code=400, detail="Invalid 'from'/'to' format")
            p = _ensure_waterlogging_combined_plot(hours=int(hours), since_ts=float(s), until_ts=float(u))
        else:
            p = _ensure_waterlogging_combined_plot(hours=int(hours))
        return FileResponse(path=str(p), media_type="image/png", filename=p.name)

    filename = None
    if isinstance(report, dict):
        plots = report.get("plots") or {}
        filename = plots.get(base_name)

    if not isinstance(filename, str) or not filename:
        filename = DEFAULT_PLOTS.get(base_name)

    if not isinstance(filename, str) or not filename:
        candidate = config.ML_PLOTS_DIR / f"{base_name}.png"
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
def get_ml_plot_meta(
    plot_name: str,
    hours: int = Query(6, ge=1, le=168),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
    at: str | None = Query(None),
):
    me = _require(ml_engine, "ml_engine")
    report = _ensure_training_report_loaded()

    base_name = str(plot_name)
    if base_name.lower().endswith(".png"):
        base_name = base_name[:-4]

    if base_name == "risk_heatmap":
        ts_epoch = _parse_ts(str(at)) if at is not None else None
        if at is not None and ts_epoch is None:
            raise HTTPException(status_code=400, detail="Invalid 'at' format")
        rainfall_raw = None
        try:
            ld = _require(loader, "loader")
            if ts_epoch is not None:
                if getattr(ld, "_rain_ts_epoch", None) is not None:
                    rainfall_raw = ld.rainfall_value_at_exact(float(ts_epoch), tolerance_s=1.0)
                    if rainfall_raw is None:
                        raise HTTPException(status_code=404, detail="No data found for requested timestamp")
                else:
                    rainfall_raw = ld.latest_rainfall_value()
            else:
                rainfall_raw = ld.latest_rainfall_value()
        except HTTPException:
            raise
        except Exception:
            rainfall_raw = None

        out = DEFAULT_PLOT_META.get("risk_heatmap")
        as_of = None
        if ts_epoch is not None:
            as_of = {
                "ts_epoch": float(ts_epoch),
                "ts_utc": datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc).isoformat(),
                "rainfall_raw": float(rainfall_raw) if rainfall_raw is not None else None,
            }
        return {"plot": base_name, "meta": out, "as_of": as_of}

    if base_name == "waterlogging_hotspots_28y":
        meta = _compute_waterlogging_hotspots_meta(bins=256)
        if meta is not None:
            return {"plot": base_name, "meta": meta}
        raise HTTPException(status_code=503, detail="Surface water dataset not available")

    if base_name == "waterlogging_combined":
        s = None
        u = None
        if from_ is not None or to is not None:
            if from_ is None or to is None:
                raise HTTPException(status_code=400, detail="Both 'from' and 'to' are required")
            s = _parse_ts(from_)
            u = _parse_ts(to)
            if s is None or u is None:
                raise HTTPException(status_code=400, detail="Invalid 'from'/'to' format")
            if float(u) <= float(s):
                raise HTTPException(status_code=400, detail="Invalid time window: to must be after from")

        meta = _compute_waterlogging_combined_meta(bins=256, hours=int(hours))
        if meta is not None:
            window = None
            if s is not None and u is not None:
                window = {"from_ts": float(s), "to_ts": float(u)}
            return {"plot": base_name, "meta": meta, "window": window}
        raise HTTPException(status_code=503, detail="Surface water dataset not available")

    meta = {}
    if isinstance(report, dict):
        meta = report.get("plot_meta") or {}
        if not isinstance(meta, dict):
            meta = {}

    out = meta.get(base_name) if isinstance(meta, dict) else None
    if out is None:
        out = DEFAULT_PLOT_META.get(base_name)
    if out is None:
        raise HTTPException(status_code=404, detail="Plot metadata not found")
    return {"plot": base_name, "meta": out}


@app.post("/ml/retrain")
def retrain_ml():
    me = _require(ml_engine, "ml_engine")
    if not config.ML_ALLOW_API_RETRAIN:
        raise HTTPException(status_code=403, detail="API retrain disabled")
    try:
        _log_event("ml_retrain_start", {})
        me.load_or_train(force_retrain=True)
        _log_event("ml_retrain_done", {"trained": bool(me.models is not None)})
    except Exception:
        logger.exception("/ml/retrain failed")
        raise HTTPException(status_code=500, detail="Retrain failed")

    try:
        rep = getattr(me, "training_report", None)
        if isinstance(rep, dict) and rep:
            logger.info("\n%s", _format_training_report(rep))
    except Exception:
        logger.info("ML training report available after retrain but could not be serialized")

    return {"status": "retrained", "trained": me.models is not None, "training_report": me.training_report}


def _parse_ts(value: str) -> float | None:
    try:
        if value is None:
            return None
        v = str(value).strip()
        if not v:
            return None
        try:
            return float(v)
        except Exception:
            pass
        try:
            v2 = v.replace("Z", "+00:00")
            dt = datetime.fromisoformat(v2)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return float(dt.astimezone(timezone.utc).timestamp())
        except Exception:
            pass
        try:
            dt = datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
            return float(dt.astimezone(timezone.utc).timestamp())
        except Exception:
            return None
    except Exception:
        return None


@app.get("/debug/events")
def debug_events(since: str | None = None, until: str | None = None, limit: int = Query(200, ge=1, le=500)):
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail="Not found")
    s = _parse_ts(since) if since is not None else None
    u = _parse_ts(until) if until is not None else None
    out = []
    for e in list(_event_log):
        try:
            ts = float(e.get("ts_epoch", 0.0))
            if s is not None and ts < float(s):
                continue
            if u is not None and ts > float(u):
                continue
            out.append(e)
        except Exception:
            continue
    return {"events": out[-int(limit):]}


@app.post("/debug/smoke_test")
def debug_smoke_test(
    hours: int = Query(6, ge=1, le=168),
    from_: str | None = Query(None, alias="from"),
    to: str | None = Query(None, alias="to"),
    at: str | None = Query(None),
):
    if not DEBUG_MODE:
        raise HTTPException(status_code=404, detail="Not found")
    _log_event("smoke_test_start", {"hours": int(hours)})
    me = _require(ml_engine, "ml_engine")
    ok = True
    results: dict = {}
    try:
        if at is not None:
            ts_epoch = _parse_ts(str(at))
            if ts_epoch is None:
                raise HTTPException(status_code=400, detail="Invalid 'at' format")
            results["risk_a"] = me.predict_risk_at(22.5171688, 88.4187615, float(ts_epoch))
            results["risk_b"] = me.predict_risk_at(22.57, 88.36, float(ts_epoch))
            _ensure_risk_heatmap_plot(at_ts=float(ts_epoch))
        else:
            results["risk_a"] = me.predict_risk(22.5171688, 88.4187615)
            results["risk_b"] = me.predict_risk(22.57, 88.36)
            _ensure_risk_heatmap_plot(at_ts=None)
    except Exception:
        ok = False
        results["risk_error"] = True
    try:
        if from_ is not None or to is not None:
            if from_ is None or to is None:
                raise HTTPException(status_code=400, detail="Both 'from' and 'to' are required")
            s = _parse_ts(from_)
            u = _parse_ts(to)
            if s is None or u is None:
                raise HTTPException(status_code=400, detail="Invalid 'from'/'to' format")
            _ensure_waterlogging_combined_plot(hours=int(hours), since_ts=float(s), until_ts=float(u))
        else:
            _ensure_waterlogging_combined_plot(hours=int(hours))
        results["plot"] = "ok"
    except Exception:
        ok = False
        results["plot"] = "fail"
    _log_event("smoke_test_done", {"ok": bool(ok)})
    return {"ok": bool(ok), "results": results}
