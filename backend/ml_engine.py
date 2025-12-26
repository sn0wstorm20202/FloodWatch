from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

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
        self._water_tree = None

    def train(self) -> None:
        """Build feature table + train clustering/anomaly/classifier models."""
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

        logger.info(
            "ML training complete: samples=%d high_risk_cluster=%s label_rate=%.3f",
            len(df),
            self.high_risk_cluster,
            float(np.mean(y)) if y.size else 0.0,
        )

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
            return {"ml_risk": 0.80 if config.DEMO_MODE else 0.40, "cluster_id": 0, "anomaly_score": 0.50}

    @lru_cache(maxsize=20000)
    def _predict_risk_cached(self, lat_rounded: float, lon_rounded: float) -> dict:
        if self.models is None:
            return {"ml_risk": 0.80 if config.DEMO_MODE else 0.40, "cluster_id": 0, "anomaly_score": 0.50}

        feats = self._extract_features(lat_rounded, lon_rounded)
        if feats is None:
            return {"ml_risk": 0.80 if config.DEMO_MODE else 0.40, "cluster_id": 0, "anomaly_score": 0.50}

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
        if self._water_tree is None:
            return np.full_like(lat, 0.7 if config.DEMO_MODE else 0.3, dtype=float)

        try:
            # Query expects radians [lat, lon]
            q = np.deg2rad(np.column_stack([lat, lon]))
            dist_rad, _ind = self._water_tree.query(q, k=1)
            dist_m = dist_rad.reshape(-1) * 6371000.0
            occ = 1.0 - np.clip(dist_m / float(config.WATER_PROXIMITY_MAX_M), 0.0, 1.0)
            return np.asarray(occ, dtype=float)
        except Exception:
            logger.exception("Water occurrence computation failed")
            return np.full_like(lat, 0.7 if config.DEMO_MODE else 0.3, dtype=float)

    def _extract_features(self, lat: float, lon: float) -> Optional[dict]:
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
        r_val = self.loader.latest_rainfall_value()
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
