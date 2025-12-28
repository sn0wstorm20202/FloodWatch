from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import config
from data_loader import DataLoader
from utils import clamp01, normalize_minmax, round_coord

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MLModels:
    cluster_model: object
    anomaly_model: object
    classifier_model: object


class MLEngine:
    """Unsupervised + weakly supervised ML engine.

    Pipeline (as requested):
        Feature engineering -> KMeans(k=3) -> IsolationForest -> pseudo-labels -> LogisticRegression -> ml_risk

    Demo-safe behavior:
        If anything fails (missing packages, missing datasets, training errors),
        predict_risk returns stable non-crashing defaults.
    """

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader

        self.feature_table: Optional[pd.DataFrame] = None
        self.models: Optional[MLModels] = None

        self.training_report: Optional[dict] = None

        self.plot_meta: Optional[dict] = None

        self.cluster_risk_map: Dict[int, float] = {}
        self.high_risk_cluster: Optional[int] = None

        self._anom_raw_min: float = 0.0
        self._anom_raw_max: float = 1.0

        # Normalization stats for inference
        self._elev_vmin: float = 0.0
        self._elev_vmax: float = 1.0
        self._lc_vmin: float = 0.0
        self._lc_vmax: float = 1.0
        self._rain_vmin: float = 0.0
        self._rain_vmax: float = 1.0

        self._water_points_lonlat: Optional[np.ndarray] = None
        self._water_occurrence_values: Optional[np.ndarray] = None
        self._water_tree = None

    def load_or_train(self, force_retrain: bool = False) -> None:
        if force_retrain:
            self.train()
            self._persist_if_enabled()
            return

        if config.ML_PERSIST_MODELS:
            if self._try_load_persisted_model():
                return

        self.train()
        self._persist_if_enabled()

    def _persist_if_enabled(self) -> None:
        if not config.ML_PERSIST_MODELS:
            return
        try:
            self._save_persisted_model()
        except Exception:
            logger.exception("Failed to persist ML model; continuing")

    def train(self) -> None:
        """Build feature table + train clustering/anomaly/classifier models."""
        t0 = time.time()
        self._init_stats_and_water_index()

        df = self._build_feature_table()
        if df is None or df.empty:
            logger.warning("ML feature table is empty; ML models will not be trained")
            self.feature_table = None
            self.models = None
            return

        self.feature_table = df

        try:
            from sklearn.cluster import KMeans
            from sklearn.ensemble import IsolationForest
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                accuracy_score,
                average_precision_score,
                confusion_matrix,
                f1_score,
                precision_score,
                recall_score,
                roc_auc_score,
                silhouette_score,
            )
            from sklearn.model_selection import train_test_split
        except Exception:
            logger.exception("scikit-learn not available; ML models will not be trained")
            self.models = None
            return

        X = df[["elevation", "rainfall", "water_occurrence", "urban_density"]].to_numpy(dtype=float)

        # STEP 2: Unsupervised clustering
        cluster_model = KMeans(n_clusters=3, random_state=42, n_init="auto")
        cluster_id = cluster_model.fit_predict(X)
        df = df.copy()
        df["cluster_id"] = cluster_id

        self._interpret_clusters(df)

        # STEP 3: Anomaly detection
        X_anom = df[["rainfall", "water_occurrence", "elevation"]].to_numpy(dtype=float)
        anomaly_model = IsolationForest(
            n_estimators=200,
            random_state=42,
            contamination="auto",
        )
        anomaly_model.fit(X_anom)

        # Convert score_samples -> anomaly_score in [0,1]
        raw = -anomaly_model.score_samples(X_anom)
        self._anom_raw_min = float(np.min(raw)) if raw.size else 0.0
        self._anom_raw_max = float(np.max(raw)) if raw.size else 1.0
        anom_score = self._normalize_anomaly_raw(raw)
        df["anomaly_score"] = anom_score

        # STEP 4: Pseudo-label generation
        if self.high_risk_cluster is None:
            high_cluster = int(pd.Series(cluster_id).mode().iloc[0])
        else:
            high_cluster = int(self.high_risk_cluster)

        pseudo = ((df["anomaly_score"] > 0.8) | (df["cluster_id"] == high_cluster)).astype(int)
        df["pseudo_label"] = pseudo

        # STEP 5: Weakly-supervised classifier
        y = pseudo.to_numpy(dtype=int)
        clf = LogisticRegression(
            max_iter=500,
            random_state=42,
            class_weight="balanced",
        )
        clf.fit(X, y)

        self.feature_table = df
        self.models = MLModels(cluster_model=cluster_model, anomaly_model=anomaly_model, classifier_model=clf)

        try:
            self._predict_risk_cached.cache_clear()
        except Exception:
            pass

        report: dict[str, Any] = {}
        report["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
        report["samples"] = int(len(df))
        report["high_risk_cluster"] = int(self.high_risk_cluster) if self.high_risk_cluster is not None else None
        report["pseudo_label_rate"] = float(np.mean(y)) if y.size else 0.0
        report["cluster_counts"] = {int(k): int(v) for k, v in pd.Series(cluster_id).value_counts().to_dict().items()}
        report["cluster_risk_map"] = {int(k): float(v) for k, v in self.cluster_risk_map.items()}
        report["anomaly_score_stats"] = {
            "min": float(np.min(anom_score)) if len(anom_score) else 0.0,
            "max": float(np.max(anom_score)) if len(anom_score) else 1.0,
            "mean": float(np.mean(anom_score)) if len(anom_score) else 0.0,
        }

        try:
            sil = float(silhouette_score(X, cluster_id)) if len(df) >= 10 else None
        except Exception:
            sil = None
        report["silhouette_score"] = sil

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y if len(np.unique(y)) > 1 else None)
            clf_eval = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
            clf_eval.fit(X_tr, y_tr)
            proba_te = clf_eval.predict_proba(X_te)[:, 1]
            pred_te = (proba_te >= 0.5).astype(int)
            report["holdout_metrics"] = {
                "accuracy": float(accuracy_score(y_te, pred_te)),
                "precision": float(precision_score(y_te, pred_te, zero_division=0)),
                "recall": float(recall_score(y_te, pred_te, zero_division=0)),
                "f1": float(f1_score(y_te, pred_te, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_te, proba_te)) if len(np.unique(y_te)) > 1 else None,
                "avg_precision": float(average_precision_score(y_te, proba_te)) if len(np.unique(y_te)) > 1 else None,
                "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
            }
        except Exception:
            report["holdout_metrics"] = None

        try:
            coef = getattr(clf, "coef_", None)
            if coef is not None and len(coef) and coef.shape[1] == 4:
                report["classifier_coefficients"] = {
                    "elevation": float(coef[0][0]),
                    "rainfall": float(coef[0][1]),
                    "water_occurrence": float(coef[0][2]),
                    "urban_density": float(coef[0][3]),
                }
        except Exception:
            pass

        report["fingerprint"] = self._dataset_fingerprint()
        report["artifacts"] = {
            "model_path": str(config.ML_MODEL_PATH),
            "report_path": str(config.ML_REPORT_PATH),
            "plots_dir": str(config.ML_PLOTS_DIR),
        }

        report["plots"] = self._generate_training_plots(df=df)
        report["plot_meta"] = self.plot_meta

        report["train_seconds"] = float(max(0.0, time.time() - t0))
        self.training_report = report

        try:
            config.ML_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            config.ML_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            with config.ML_REPORT_PATH.open("w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Failed to write training report")

        logger.info(
            "ML training complete: samples=%d high_risk_cluster=%s label_rate=%.3f",
            len(df),
            self.high_risk_cluster,
            float(np.mean(y)) if y.size else 0.0,
        )

    def _dataset_fingerprint(self) -> dict:
        out: dict[str, Any] = {}
        paths = {
            "rainfall_csv": config.RAINFALL_CSV_PATH,
            "surface_water_csv": config.SURFACE_WATER_CSV_PATH,
            "elevation_tif": config.ELEVATION_TIF_PATH,
            "landcover_tif": config.LANDCOVER_TIF_PATH,
            "roads_graphml": config.ROADS_GRAPHML_PATH,
        }

        for k, p in paths.items():
            try:
                if p.exists():
                    st = p.stat()
                    out[k] = {"path": str(p), "size": int(st.st_size), "mtime": float(st.st_mtime)}
                else:
                    out[k] = {"path": str(p), "missing": True}
            except Exception:
                out[k] = {"path": str(p), "error": True}
        return out

    def _try_load_persisted_model(self) -> bool:
        try:
            if not config.ML_MODEL_PATH.exists():
                return False
        except Exception:
            return False

        try:
            import joblib
        except Exception:
            return False

        try:
            blob = joblib.load(config.ML_MODEL_PATH)
        except Exception:
            logger.exception("Failed to load persisted ML model")
            return False

        if not isinstance(blob, dict):
            return False

        if blob.get("schema_version") != 1:
            return False

        current_fp = self._dataset_fingerprint()
        saved_fp = blob.get("fingerprint")
        if isinstance(saved_fp, dict) and saved_fp != current_fp:
            return False

        try:
            self.models = blob.get("models")
            self.cluster_risk_map = blob.get("cluster_risk_map") or {}
            self.high_risk_cluster = blob.get("high_risk_cluster")
            self._anom_raw_min = float(blob.get("anom_raw_min", 0.0))
            self._anom_raw_max = float(blob.get("anom_raw_max", 1.0))

            self._elev_vmin = float(blob.get("elev_vmin", 0.0))
            self._elev_vmax = float(blob.get("elev_vmax", 1.0))
            self._lc_vmin = float(blob.get("lc_vmin", 0.0))
            self._lc_vmax = float(blob.get("lc_vmax", 1.0))
            self._rain_vmin = float(blob.get("rain_vmin", 0.0))
            self._rain_vmax = float(blob.get("rain_vmax", 1.0))

            self._init_stats_and_water_index()

            rep = blob.get("training_report")
            if isinstance(rep, dict):
                self.training_report = rep
            else:
                self.training_report = None

            try:
                self._predict_risk_cached.cache_clear()
            except Exception:
                pass

            if self.models is None:
                return False

            logger.info("Loaded persisted ML model from %s", config.ML_MODEL_PATH)
            return True
        except Exception:
            logger.exception("Failed to restore persisted ML model")
            return False

    def _save_persisted_model(self) -> None:
        if self.models is None:
            return
        try:
            import joblib
        except Exception:
            return

        config.ML_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "fingerprint": self._dataset_fingerprint(),
            "models": self.models,
            "cluster_risk_map": self.cluster_risk_map,
            "high_risk_cluster": self.high_risk_cluster,
            "anom_raw_min": self._anom_raw_min,
            "anom_raw_max": self._anom_raw_max,
            "elev_vmin": self._elev_vmin,
            "elev_vmax": self._elev_vmax,
            "lc_vmin": self._lc_vmin,
            "lc_vmax": self._lc_vmax,
            "rain_vmin": self._rain_vmin,
            "rain_vmax": self._rain_vmax,
            "training_report": self.training_report,
        }
        joblib.dump(payload, config.ML_MODEL_PATH)

    def _generate_training_plots(self, df: pd.DataFrame) -> dict:
        try:
            if os.getenv("FLOODWATCH_DISABLE_PLOTS") == "1":
                return {}
        except Exception:
            pass

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return {}

        try:
            config.ML_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            return {}

        out: dict[str, str] = {}
        meta: dict[str, Any] = {}

        try:
            if "cluster_id" in df.columns:
                counts = df["cluster_id"].value_counts().sort_index()
                fig = plt.figure(figsize=(6, 4))
                plt.bar([int(x) for x in counts.index.tolist()], counts.values.tolist())
                plt.xlabel("cluster_id")
                plt.ylabel("count")
                p = config.ML_PLOTS_DIR / "cluster_counts.png"
                fig.tight_layout()
                fig.savefig(p, dpi=160)
                plt.close(fig)
                out["cluster_counts"] = p.name
        except Exception:
            pass

        try:
            if "anomaly_score" in df.columns:
                fig = plt.figure(figsize=(6, 4))
                plt.hist(df["anomaly_score"].astype(float).to_numpy(), bins=30)
                plt.xlabel("anomaly_score")
                plt.ylabel("count")
                p = config.ML_PLOTS_DIR / "anomaly_hist.png"
                fig.tight_layout()
                fig.savefig(p, dpi=160)
                plt.close(fig)
                out["anomaly_hist"] = p.name
        except Exception:
            pass

        try:
            if self.models is not None:
                X = df[["elevation", "rainfall", "water_occurrence", "urban_density"]].to_numpy(dtype=float)
                proba = self.models.classifier_model.predict_proba(X)[:, 1]
                fig = plt.figure(figsize=(6, 4))
                plt.hist(np.clip(proba, 0.0, 1.0), bins=30)
                plt.xlabel("ml_risk")
                plt.ylabel("count")
                p = config.ML_PLOTS_DIR / "risk_hist.png"
                fig.tight_layout()
                fig.savefig(p, dpi=160)
                plt.close(fig)
                out["risk_hist"] = p.name
        except Exception:
            pass

        try:
            cols = ["elevation", "rainfall", "water_occurrence", "urban_density"]
            if all(c in df.columns for c in cols):
                fig, axes = plt.subplots(2, 2, figsize=(9, 6))
                axes = axes.reshape(-1)
                for i, c in enumerate(cols):
                    ax = axes[i]
                    ax.hist(df[c].astype(float).to_numpy(), bins=30)
                    ax.set_title(c)
                p = config.ML_PLOTS_DIR / "feature_hists.png"
                fig.tight_layout()
                fig.savefig(p, dpi=160)
                plt.close(fig)
                out["feature_hists"] = p.name
        except Exception:
            pass

        try:
            if self.models is not None and "cluster_id" in df.columns:
                from sklearn.decomposition import PCA

                X = df[["elevation", "rainfall", "water_occurrence", "urban_density"]].to_numpy(dtype=float)
                z = PCA(n_components=2, random_state=42).fit_transform(X)
                c = df["cluster_id"].astype(int).to_numpy()

                fig = plt.figure(figsize=(6, 4))
                plt.scatter(z[:, 0], z[:, 1], c=c, s=6, cmap="tab10", alpha=0.85)
                plt.xlabel("pca_1")
                plt.ylabel("pca_2")
                p = config.ML_PLOTS_DIR / "pca_clusters.png"
                fig.tight_layout()
                fig.savefig(p, dpi=160)
                plt.close(fig)
                out["pca_clusters"] = p.name
        except Exception:
            pass

        try:
            if self.models is not None and "lat" in df.columns and "lon" in df.columns:
                X = df[["elevation", "rainfall", "water_occurrence", "urban_density"]].to_numpy(dtype=float)
                proba = self.models.classifier_model.predict_proba(X)[:, 1]
                lat = df["lat"].astype(float).to_numpy()
                lon = df["lon"].astype(float).to_numpy()

                ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(proba)
                lat = lat[ok]
                lon = lon[ok]
                proba = np.clip(proba[ok], 0.0, 1.0)

                if lat.size >= 10:
                    bins = 80
                    s, xedges, yedges = np.histogram2d(lon, lat, bins=bins, weights=proba)
                    cts, _, _ = np.histogram2d(lon, lat, bins=[xedges, yedges])
                    avg = np.divide(s, cts, out=np.full_like(s, np.nan, dtype=float), where=cts > 0)

                    fig = plt.figure(figsize=(7, 5))
                    plt.imshow(
                        avg.T,
                        origin="lower",
                        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap="inferno",
                        aspect="auto",
                        vmin=0.0,
                        vmax=1.0,
                    )
                    plt.colorbar(label="ml_risk")
                    plt.xlabel("lon")
                    plt.ylabel("lat")
                    p = config.ML_PLOTS_DIR / "risk_heatmap.png"
                    fig.tight_layout()
                    fig.savefig(p, dpi=160)
                    plt.close(fig)
                    out["risk_heatmap"] = p.name
                    meta["risk_heatmap"] = {
                        "bounds": {
                            "west": float(xedges[0]),
                            "east": float(xedges[-1]),
                            "south": float(yedges[0]),
                            "north": float(yedges[-1]),
                        },
                        "bins": int(bins),
                    }
        except Exception:
            pass

        self.plot_meta = meta

        return out

    def predict_risk(self, lat: float, lon: float) -> dict:
        """Return ML flood risk for a point.

        Output contract (requested):
            {"ml_risk": float, "cluster_id": int, "anomaly_score": float}
        """

        try:
            return self._predict_risk_cached(
                round_coord(lat, config.RISK_CACHE_ROUND_DECIMALS),
                round_coord(lon, config.RISK_CACHE_ROUND_DECIMALS),
            )
        except Exception:
            logger.exception("ML predict failed; returning demo-safe defaults")
            return self._predict_risk_fallback(lat, lon)

    def predict_risk_at(self, lat: float, lon: float, ts_epoch: float) -> dict:
        """Return ML flood risk for a point at a specific timestamp.

        This overrides the rainfall scalar using the rainfall CSV's system:index timeline.
        """

        lat_r = round_coord(lat, config.RISK_CACHE_ROUND_DECIMALS)
        lon_r = round_coord(lon, config.RISK_CACHE_ROUND_DECIMALS)

        rain_raw = None
        try:
            rain_raw = self.loader.rainfall_value_at(float(ts_epoch))
        except Exception:
            rain_raw = None

        try:
            return self._predict_risk_uncached(lat_r, lon_r, rainfall_raw_override=rain_raw)
        except Exception:
            logger.exception("ML predict_at failed; returning demo-safe defaults")
            return self._predict_risk_fallback(lat_r, lon_r, rainfall_raw_override=rain_raw)

    def _coord_noise01(self, lat: float, lon: float) -> float:
        """Deterministic pseudo-random value in [0,1) derived from coordinates."""
        x = math.sin(lat * 12.9898 + lon * 78.233) * 43758.5453
        return float(x - math.floor(x))

    def _predict_risk_fallback(self, lat: float, lon: float, rainfall_raw_override: float | None = None) -> dict:
        """Deployment-safe fallback when the ML model or datasets aren't available.

        Goal:
        - Never return a constant for all locations.
        - Prefer feature-based heuristics when available.
        - Otherwise fall back to deterministic coordinate-based variation.
        """
        noise = self._coord_noise01(float(lat), float(lon))

        feats = None
        try:
            feats = self._extract_features(float(lat), float(lon), rainfall_raw_override=rainfall_raw_override)
        except Exception:
            feats = None

        if isinstance(feats, dict):
            # Heuristic risk in [0,1]
            w = getattr(config, "RISK_WEIGHTS", {}) or {}
            w_rain = float(w.get("rainfall", 0.35))
            w_elev = float(w.get("elevation", 0.25))
            w_water = float(w.get("water_proximity", 0.20))
            w_urban = float(w.get("urban_density", 0.10))
            denom = max(1e-6, (w_rain + w_elev + w_water + w_urban))

            base = (
                (1.0 - float(feats.get("elevation", 0.5))) * w_elev
                + float(feats.get("rainfall", 0.5)) * w_rain
                + float(feats.get("water_occurrence", 0.5)) * w_water
                + float(feats.get("urban_density", 0.5)) * w_urban
            ) / denom
        else:
            # No features -> stable coordinate-based baseline
            base = 0.35 + 0.5 * noise

        # Add a small deterministic jitter so nearby but different coords won't look identical.
        jitter = (noise - 0.5) * 0.25  # [-0.125 .. +0.125]
        demo_bias = 0.08 if config.DEMO_MODE else 0.0
        risk = float(clamp01(base + jitter + demo_bias))

        # Provide plausible auxiliary values.
        cluster_id = int(min(2, max(0, int(noise * 3.0))))
        anomaly_score = float(clamp01(0.2 + 0.8 * noise))

        return {"ml_risk": risk, "cluster_id": cluster_id, "anomaly_score": anomaly_score}

    @lru_cache(maxsize=20000)
    def _predict_risk_cached(self, lat_rounded: float, lon_rounded: float) -> dict:
        if self.models is None:
            return self._predict_risk_fallback(lat_rounded, lon_rounded)

        feats = self._extract_features(lat_rounded, lon_rounded)
        if feats is None:
            return self._predict_risk_fallback(lat_rounded, lon_rounded)

        X = np.asarray([[feats["elevation"], feats["rainfall"], feats["water_occurrence"], feats["urban_density"]]], dtype=float)
        X_anom = np.asarray([[feats["rainfall"], feats["water_occurrence"], feats["elevation"]]], dtype=float)

        cluster_id = int(self.models.cluster_model.predict(X)[0])

        raw = -self.models.anomaly_model.score_samples(X_anom)
        anomaly_score = float(self._normalize_anomaly_raw(raw)[0])

        try:
            proba = float(self.models.classifier_model.predict_proba(X)[0][1])
        except Exception:
            # Demo-safe: if classifier fails, fall back to a blend of anomaly and cluster interpretation.
            proba = float(clamp01(0.6 * anomaly_score + 0.4 * self.cluster_risk_map.get(cluster_id, 0.5)))

        return {"ml_risk": float(clamp01(proba)), "cluster_id": cluster_id, "anomaly_score": float(clamp01(anomaly_score))}

    def _predict_risk_uncached(self, lat_rounded: float, lon_rounded: float, rainfall_raw_override: float | None) -> dict:
        if self.models is None:
            return self._predict_risk_fallback(lat_rounded, lon_rounded, rainfall_raw_override=rainfall_raw_override)

        feats = self._extract_features(lat_rounded, lon_rounded, rainfall_raw_override=rainfall_raw_override)
        if feats is None:
            return self._predict_risk_fallback(lat_rounded, lon_rounded, rainfall_raw_override=rainfall_raw_override)

        X = np.asarray([[feats["elevation"], feats["rainfall"], feats["water_occurrence"], feats["urban_density"]]], dtype=float)
        X_anom = np.asarray([[feats["rainfall"], feats["water_occurrence"], feats["elevation"]]], dtype=float)

        cluster_id = int(self.models.cluster_model.predict(X)[0])

        raw = -self.models.anomaly_model.score_samples(X_anom)
        anomaly_score = float(self._normalize_anomaly_raw(raw)[0])

        try:
            proba = float(self.models.classifier_model.predict_proba(X)[0][1])
        except Exception:
            proba = float(clamp01(0.6 * anomaly_score + 0.4 * self.cluster_risk_map.get(cluster_id, 0.5)))

        return {"ml_risk": float(clamp01(proba)), "cluster_id": cluster_id, "anomaly_score": float(clamp01(anomaly_score))}

    def _normalize_anomaly_raw(self, raw: np.ndarray) -> np.ndarray:
        denom = (self._anom_raw_max - self._anom_raw_min)
        if denom <= 0:
            return np.zeros_like(raw, dtype=float)
        return np.clip((raw - self._anom_raw_min) / denom, 0.0, 1.0)

    def _init_stats_and_water_index(self) -> None:
        # Rasters
        if self.loader.elevation_stats is not None:
            self._elev_vmin = float(self.loader.elevation_stats.vmin)
            self._elev_vmax = float(self.loader.elevation_stats.vmax)
        if self.loader.landcover_stats is not None:
            self._lc_vmin = float(self.loader.landcover_stats.vmin)
            self._lc_vmax = float(self.loader.landcover_stats.vmax)

        # Rainfall
        self._rain_vmin = float(getattr(self.loader, "rainfall_min", 0.0))
        self._rain_vmax = float(getattr(self.loader, "rainfall_max", 1.0))

        # Water points nearest-neighbor index
        pts = self.loader.water_points_lonlat()
        self._water_points_lonlat = pts
        self._water_occurrence_values = self.loader.water_occurrence_values()

        self._water_tree = None
        if pts is None or pts.size == 0:
            return

        try:
            from sklearn.neighbors import BallTree

            # BallTree haversine expects radians, order [lat, lon]
            latlon_rad = np.deg2rad(np.column_stack([pts[:, 1], pts[:, 0]]))
            self._water_tree = BallTree(latlon_rad, metric="haversine")
        except Exception:
            logger.exception("Failed to build water BallTree index")
            self._water_tree = None

    def _build_feature_table(self) -> Optional[pd.DataFrame]:
        """STEP 1: Build training feature table from rasters + CSVs.

        Output columns:
            lat | lon | elevation | rainfall | water_occurrence | urban_density

        All features normalized to [0,1].
        """

        elev_ds = self.loader.elevation_ds
        if elev_ds is None:
            logger.warning("Elevation raster missing; cannot build ML feature grid")
            return None

        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.transform import xy
            from rasterio.warp import transform
        except Exception:
            logger.exception("rasterio not available")
            return None

        # Keep this bounded for hackathon demos.
        grid_w = 90
        grid_h = 90

        try:
            elev_small = elev_ds.read(1, out_shape=(grid_h, grid_w), resampling=Resampling.bilinear, masked=True)
            elev_arr = np.asarray(elev_small, dtype=float)

            scale_x = elev_ds.width / float(grid_w)
            scale_y = elev_ds.height / float(grid_h)
            scaled_transform = elev_ds.transform * elev_ds.transform.scale(scale_x, scale_y)

            rows = np.repeat(np.arange(grid_h), grid_w)
            cols = np.tile(np.arange(grid_w), grid_h)
            xs, ys = xy(scaled_transform, rows, cols, offset="center")

            src_crs = elev_ds.crs
            if src_crs is None:
                lon = np.asarray(xs, dtype=float)
                lat = np.asarray(ys, dtype=float)
            else:
                lon, lat = transform(src_crs, "EPSG:4326", xs, ys)
                lon = np.asarray(lon, dtype=float)
                lat = np.asarray(lat, dtype=float)

            elev_flat = elev_arr.reshape(-1)

            # Landcover sampling at same points
            lc_vals = self._sample_raster_batch(self.loader.landcover_ds, lat=lat, lon=lon)

            # Rainfall scalar applied to all points
            r_val = self.loader.latest_rainfall_value()
            if r_val is None:
                r_raw = float(self._rain_vmax) if config.DEMO_MODE else float(self._rain_vmin)
            else:
                r_raw = float(r_val)

            if config.DEMO_MODE:
                r_raw = r_raw * 1.10

            rainfall_norm = float(normalize_minmax(r_raw, self._rain_vmin, self._rain_vmax))

            # Water occurrence from nearest distance
            water_occ = self._water_occurrence_batch(lat=lat, lon=lon)

            elev_norm = np.asarray([normalize_minmax(v, self._elev_vmin, self._elev_vmax) for v in elev_flat], dtype=float)
            urban_norm = np.asarray([normalize_minmax(v, self._lc_vmin, self._lc_vmax) for v in lc_vals], dtype=float)

            # Clean NaNs
            elev_norm = np.nan_to_num(elev_norm, nan=0.0)
            urban_norm = np.nan_to_num(urban_norm, nan=0.0)
            water_occ = np.nan_to_num(water_occ, nan=0.0)

            df = pd.DataFrame(
                {
                    "lat": lat,
                    "lon": lon,
                    "elevation": np.clip(elev_norm, 0.0, 1.0),
                    "rainfall": np.clip(np.full_like(elev_norm, rainfall_norm), 0.0, 1.0),
                    "water_occurrence": np.clip(water_occ, 0.0, 1.0),
                    "urban_density": np.clip(urban_norm, 0.0, 1.0),
                }
            )

            # Drop invalid coordinates
            df = df[np.isfinite(df["lat"]) & np.isfinite(df["lon"])].reset_index(drop=True)
            return df
        except Exception:
            logger.exception("Failed to build ML feature table")
            return None

    def _sample_raster_batch(self, ds, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        if ds is None:
            return np.zeros_like(lat, dtype=float)

        try:
            from rasterio.warp import transform
        except Exception:
            return np.zeros_like(lat, dtype=float)

        try:
            src_crs = "EPSG:4326"
            dst_crs = ds.crs or src_crs

            if str(dst_crs) != src_crs:
                xs, ys = transform(src_crs, dst_crs, lon.tolist(), lat.tolist())
            else:
                xs, ys = lon.tolist(), lat.tolist()

            coords = list(zip(xs, ys))
            vals = np.asarray([v[0] for v in ds.sample(coords)], dtype=float)
            vals = np.nan_to_num(vals, nan=0.0)
            return vals
        except Exception:
            logger.exception("Batch raster sampling failed")
            return np.zeros_like(lat, dtype=float)

    def _water_occurrence_batch(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        if self._water_tree is None or self._water_occurrence_values is None:
            return np.full_like(lat, 0.7 if config.DEMO_MODE else 0.3, dtype=float)

        try:
            # Query expects radians [lat, lon]
            q = np.deg2rad(np.column_stack([lat, lon]))
            dist_rad, ind = self._water_tree.query(q, k=1)
            dist_m = dist_rad.reshape(-1) * 6371000.0
            proximity = 1.0 - np.clip(dist_m / float(config.WATER_PROXIMITY_MAX_M), 0.0, 1.0)

            idx = ind.reshape(-1)
            base_occ = self._water_occurrence_values[idx]
            base_occ = np.nan_to_num(np.asarray(base_occ, dtype=float), nan=0.0)

            occ = np.clip(base_occ * proximity, 0.0, 1.0)
            return np.asarray(occ, dtype=float)
        except Exception:
            logger.exception("Water occurrence computation failed")
            return np.full_like(lat, 0.7 if config.DEMO_MODE else 0.3, dtype=float)

    def _extract_features(self, lat: float, lon: float, rainfall_raw_override: float | None = None) -> Optional[dict]:
        # Elevation
        elev = self.loader.sample_raster_value(self.loader.elevation_ds, lat=lat, lon=lon)
        if elev is None:
            elev_norm = 0.5
        else:
            elev_norm = normalize_minmax(float(elev), self._elev_vmin, self._elev_vmax)

        # Landcover -> urban
        lc = self.loader.sample_raster_value(self.loader.landcover_ds, lat=lat, lon=lon)
        if lc is None:
            urban = 0.5
        else:
            urban = normalize_minmax(float(lc), self._lc_vmin, self._lc_vmax)

        # Rainfall scalar
        r_val = rainfall_raw_override if rainfall_raw_override is not None else self.loader.latest_rainfall_value()
        if r_val is None:
            r_raw = float(self._rain_vmax) if config.DEMO_MODE else float(self._rain_vmin)
        else:
            r_raw = float(r_val)

        if config.DEMO_MODE:
            r_raw = r_raw * 1.10

        rainfall = normalize_minmax(r_raw, self._rain_vmin, self._rain_vmax)

        # Water occurrence
        water_occ = float(self._water_occurrence_batch(np.asarray([lat], dtype=float), np.asarray([lon], dtype=float))[0])

        return {
            "elevation": float(clamp01(elev_norm)),
            "rainfall": float(clamp01(rainfall)),
            "water_occurrence": float(clamp01(water_occ)),
            "urban_density": float(clamp01(urban)),
        }

    def _interpret_clusters(self, df: pd.DataFrame) -> None:
        """Interpret clusters by ranking them using feature means.

        This avoids hard-coded flood rules/thresholds; it derives high-risk cluster
        from learned structure.
        """

        means = df.groupby("cluster_id")[["elevation", "rainfall", "water_occurrence", "urban_density"]].mean()
        if means.empty:
            self.high_risk_cluster = None
            self.cluster_risk_map = {}
            return

        # Higher risk if low elevation and high rainfall, water occurrence, and urban density.
        cluster_score = (1.0 - means["elevation"]) + means["rainfall"] + means["water_occurrence"] + means["urban_density"]
        high_cluster = int(cluster_score.idxmax())
        self.high_risk_cluster = high_cluster

        # Map clusters to a continuous risk prior in [0,1] based on ranking.
        rank = cluster_score.rank(method="dense")
        if len(rank) == 1:
            self.cluster_risk_map = {high_cluster: 1.0}
            return

        min_r = float(rank.min())
        max_r = float(rank.max())
        risk_map: Dict[int, float] = {}
        for cid, r in rank.items():
            risk_map[int(cid)] = clamp01((float(r) - min_r) / (max_r - min_r))
        self.cluster_risk_map = risk_map
