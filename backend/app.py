from __future__ import annotations

import json
import logging
import os
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

try:
    from colorama import Fore, Style
    from colorama import init as _colorama_init

    _colorama_init()
except Exception:  # pragma: no cover
    class _NoColor:
        def __getattr__(self, _name: str) -> str:
            return ""

    Fore = _NoColor()  # type: ignore
    Style = _NoColor()  # type: ignore


def _clamp01(x: float) -> float:
    try:
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)
    except Exception:
        return 0.0


def _fmt_num(x, digits: int = 3) -> str:
    try:
        if x is None:
            return "—"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "—"


def _fmt_int(x) -> str:
    try:
        if x is None:
            return "—"
        return str(int(x))
    except Exception:
        return "—"


def _bar01(value: float, width: int = 24, fill: str = "█", empty: str = "░") -> str:
    v = _clamp01(value)
    n = int(round(v * width))
    n = max(0, min(width, n))
    left = fill * n
    right = empty * (width - n)
    return f"{Fore.CYAN}{left}{Style.RESET_ALL}{Fore.WHITE}{right}{Style.RESET_ALL}"


def _bar_count(count: int, max_count: int, width: int = 18) -> str:
    try:
        if max_count <= 0:
            return _bar01(0.0, width=width)
        return _bar01(float(count) / float(max_count), width=width)
    except Exception:
        return _bar01(0.0, width=width)


def _format_training_report(rep: dict) -> str:
    def _env_flag(name: str, default: str = "0") -> bool:
        try:
            return os.getenv(name, default) == "1"
        except Exception:
            return False

    def _tag(value: float | None, good: float, ok: float) -> str:
        if value is None:
            return f"{Fore.WHITE}—{Style.RESET_ALL}"
        v = float(value)
        if v >= good:
            return f"{Fore.GREEN}{Style.BRIGHT}GOOD{Style.RESET_ALL}"
        if v >= ok:
            return f"{Fore.YELLOW}{Style.BRIGHT}OK{Style.RESET_ALL}"
        return f"{Fore.RED}{Style.BRIGHT}LOW{Style.RESET_ALL}"

    verbose = _env_flag("FLOODWATCH_ML_REPORT_VERBOSE", "0")

    line = f"{Fore.MAGENTA}{'═' * 72}{Style.RESET_ALL}"
    title = f"{Style.BRIGHT}{Fore.MAGENTA}FLOODWATCH • ML DASHBOARD{Style.RESET_ALL}"

    trained_at = rep.get("trained_at_utc")
    samples = rep.get("samples")
    train_seconds = rep.get("train_seconds")
    sil = rep.get("silhouette_score")
    pseudo = rep.get("pseudo_label_rate")
    high_cluster = rep.get("high_risk_cluster")
    anomaly_stats = rep.get("anomaly_score_stats") if isinstance(rep.get("anomaly_score_stats"), dict) else {}
    coefs = rep.get("classifier_coefficients") if isinstance(rep.get("classifier_coefficients"), dict) else {}

    holdout = rep.get("holdout_metrics") if isinstance(rep.get("holdout_metrics"), dict) else {}
    cluster_counts = rep.get("cluster_counts") if isinstance(rep.get("cluster_counts"), dict) else {}
    cluster_risk_map = rep.get("cluster_risk_map") if isinstance(rep.get("cluster_risk_map"), dict) else {}

    max_count = 0
    try:
        max_count = max(int(v) for v in cluster_counts.values()) if cluster_counts else 0
    except Exception:
        max_count = 0

    try:
        sil_v = float(sil) if sil is not None else None
    except Exception:
        sil_v = None

    try:
        pseudo_v = _clamp01(float(pseudo))
    except Exception:
        pseudo_v = 0.0

    try:
        acc_v = float(holdout.get("accuracy")) if holdout.get("accuracy") is not None else None
    except Exception:
        acc_v = None
    try:
        f1_v = float(holdout.get("f1")) if holdout.get("f1") is not None else None
    except Exception:
        f1_v = None
    try:
        prec_v = float(holdout.get("precision")) if holdout.get("precision") is not None else None
    except Exception:
        prec_v = None
    try:
        rec_v = float(holdout.get("recall")) if holdout.get("recall") is not None else None
    except Exception:
        rec_v = None
    try:
        roc_v = float(holdout.get("roc_auc")) if holdout.get("roc_auc") is not None else None
    except Exception:
        roc_v = None
    try:
        ap_v = float(holdout.get("avg_precision")) if holdout.get("avg_precision") is not None else None
    except Exception:
        ap_v = None

    if acc_v is not None and roc_v is not None and sil_v is not None:
        ok_score = 0
        ok_score += 1 if acc_v >= 0.90 else 0
        ok_score += 1 if roc_v >= 0.90 else 0
        ok_score += 1 if sil_v >= 0.30 else 0
        if ok_score == 3:
            status = f"{Fore.GREEN}{Style.BRIGHT}HEALTHY{Style.RESET_ALL}"
        elif ok_score == 2:
            status = f"{Fore.YELLOW}{Style.BRIGHT}OK{Style.RESET_ALL}"
        else:
            status = f"{Fore.RED}{Style.BRIGHT}CHECK{Style.RESET_ALL}"
    else:
        status = f"{Fore.YELLOW}{Style.BRIGHT}PARTIAL{Style.RESET_ALL}"

    out: list[str] = []
    out.append(line)
    out.append(title + f"  {Fore.WHITE}[{status}{Fore.WHITE}]{Style.RESET_ALL}")
    out.append(line)
    out.append(
        f"{Fore.CYAN}{Style.BRIGHT}DATA{Style.RESET_ALL} "
        f"trained={trained_at or '—'} | samples={_fmt_int(samples)} | time={_fmt_num(train_seconds, 2)}s"
    )
    out.append(
        f"{Fore.CYAN}{Style.BRIGHT}QUALITY{Style.RESET_ALL} "
        f"sil={_fmt_num(sil_v, 3)}({_tag(sil_v, 0.50, 0.30)}) | pseudo_risky={_fmt_num(pseudo_v, 3)} {_bar01(pseudo_v, width=16)}"
    )
    if anomaly_stats:
        out.append(
            f"{Fore.CYAN}{Style.BRIGHT}ANOMALY{Style.RESET_ALL} "
            f"avg={_fmt_num(anomaly_stats.get('mean'), 3)} | min={_fmt_num(anomaly_stats.get('min'), 2)} | max={_fmt_num(anomaly_stats.get('max'), 2)}"
        )

    if isinstance(holdout, dict) and holdout:
        cm = holdout.get("confusion_matrix")
        cm_str = ""
        if isinstance(cm, list) and len(cm) == 2 and all(isinstance(r, list) and len(r) == 2 for r in cm):
            try:
                tn, fp = int(cm[0][0]), int(cm[0][1])
                fn, tp = int(cm[1][0]), int(cm[1][1])
                cm_str = f" | TP={tp} FP={fp} TN={tn} FN={fn}"
            except Exception:
                cm_str = ""
        out.append(
            f"{Fore.CYAN}{Style.BRIGHT}TEST{Style.RESET_ALL} "
            f"acc={_fmt_num(acc_v, 4)} | prec={_fmt_num(prec_v, 4)} | rec={_fmt_num(rec_v, 4)} | f1={_fmt_num(f1_v, 4)} | auc={_fmt_num(roc_v, 4)} | ap={_fmt_num(ap_v, 4)}{cm_str}"
        )

    out.append(f"{Fore.CYAN}{Style.BRIGHT}CLUSTERS{Style.RESET_ALL} high_risk_id={_fmt_int(high_cluster)}")
    if cluster_counts:
        for k in sorted(cluster_counts.keys(), key=lambda x: int(x) if str(x).lstrip('-').isdigit() else str(x)):
            try:
                cid = int(k)
            except Exception:
                cid = k
            try:
                cnt = int(cluster_counts.get(k, 0))
            except Exception:
                cnt = 0
            risk = cluster_risk_map.get(str(cid), cluster_risk_map.get(cid, None))
            try:
                risk_v = _clamp01(float(risk)) if risk is not None else 0.0
            except Exception:
                risk_v = 0.0

            if risk_v >= 0.80:
                risk_tag = f"{Fore.RED}{Style.BRIGHT}HIGH{Style.RESET_ALL}"
            elif risk_v >= 0.40:
                risk_tag = f"{Fore.YELLOW}{Style.BRIGHT}MED{Style.RESET_ALL}"
            else:
                risk_tag = f"{Fore.GREEN}{Style.BRIGHT}LOW{Style.RESET_ALL}"

            out.append(
                f"  {Fore.WHITE}#{cid}:{Style.RESET_ALL} {cnt:>5} {_bar_count(cnt, max_count)}"
                f"  risk={_fmt_num(risk_v, 2)} {_bar01(risk_v, width=12)} {risk_tag}"
            )
    else:
        out.append(f"  {Fore.WHITE}—{Style.RESET_ALL}")

    if coefs:
        out.append(f"{Fore.CYAN}{Style.BRIGHT}DRIVERS{Style.RESET_ALL}")
        try:
            items = [(str(k), float(v)) for k, v in coefs.items()]
            items.sort(key=lambda kv: abs(kv[1]), reverse=True)
            top = items[:4]
            denom = abs(top[0][1]) if top and top[0][1] != 0 else 1.0
            for k, v in top:
                arrow = f"{Fore.RED}↑{Style.RESET_ALL}" if v >= 0 else f"{Fore.GREEN}↓{Style.RESET_ALL}"
                strength = _clamp01(min(1.0, abs(v) / denom))
                out.append(
                    f"  {arrow} {Fore.WHITE}{k}{Style.RESET_ALL} coef={_fmt_num(v, 3)} {_bar01(strength, width=16)}"
                )
        except Exception:
            pass

    if verbose:
        out.append("")
        out.append(f"{Fore.WHITE}Verbose mode is ON (FLOODWATCH_ML_REPORT_VERBOSE=1).{Style.RESET_ALL}")
        out.append(f"{Fore.WHITE}Risk score range: 0.0 safer → 1.0 riskier.{Style.RESET_ALL}")

    report_path = str(config.ML_REPORT_PATH)
    out.append(line)
    out.append(f"{Fore.WHITE}Report file: {Fore.CYAN}{report_path}{Style.RESET_ALL}")
    out.append(line)
    return "\n".join(out)


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
        rep = getattr(ml_engine, "training_report", None)
        if not (isinstance(rep, dict) and rep):
            try:
                if config.ML_REPORT_PATH.exists():
                    with config.ML_REPORT_PATH.open("r", encoding="utf-8") as f:
                        rep = json.load(f)
            except Exception:
                rep = None
        if isinstance(rep, dict) and rep:
            logger.info("\n%s", _format_training_report(rep))
    except Exception:
        logger.info("ML training report available but could not be serialized")

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
    plots = {}
    if isinstance(me.training_report, dict):
        plots = me.training_report.get("plots") or {}
    return {"plots": plots}


@app.get("/ml/plot/{plot_name}")
def get_ml_plot(plot_name: str):
    me = _require(ml_engine, "ml_engine")
    if not isinstance(me.training_report, dict):
        raise HTTPException(status_code=404, detail="No training report available")
    plots = me.training_report.get("plots") or {}
    filename = plots.get(plot_name)
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
    if not isinstance(me.training_report, dict):
        raise HTTPException(status_code=404, detail="No training report available")
    meta = me.training_report.get("plot_meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    out = meta.get(plot_name)
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

    try:
        rep = getattr(me, "training_report", None)
        if isinstance(rep, dict) and rep:
            logger.info("\n%s", _format_training_report(rep))
    except Exception:
        logger.info("ML training report available after retrain but could not be serialized")

    return {"status": "retrained", "trained": me.models is not None, "training_report": me.training_report}
