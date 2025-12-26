from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

import config
from data_loader import DataLoader
from store import ReportStore
from utils import clamp01, mean, normalize_minmax, risk_level, round_coord

logger = logging.getLogger(__name__)


class RiskEngine:
    """Explainable weighted flood risk model.

    Mandatory formula:
        risk_score = 0.35*rainfall + 0.25*low_elevation + 0.20*water_proximity + 0.10*urban_density + 0.10*crowd_reports

    Every component is normalized to [0,1] and returned in the explanation.
    """

    def __init__(self, loader: DataLoader, store: ReportStore) -> None:
        self.loader = loader
        self.store = store

    def invalidate_cache(self) -> None:
        try:
            self.cached_risk_score.cache_clear()
        except Exception:
            logger.exception("Failed to clear risk cache")

    def compute(self, lat: float, lon: float) -> Tuple[float, str, Dict[str, float]]:
        score, expl = self._compute_components(lat, lon)
        return score, risk_level(score), expl

    def _compute_components(self, lat: float, lon: float) -> Tuple[float, Dict[str, float]]:
        rainfall = self._rainfall_component()
        elevation = self._low_elevation_component(lat, lon)
        water = self._water_proximity_component(lat, lon)
        urban = self._urban_density_component(lat, lon)
        crowd = self._crowd_component(lat, lon)

        # Demo amplifications
        if config.DEMO_MODE:
            rainfall = clamp01(rainfall * float(config.DEMO_RAINFALL_MULT))
            crowd = clamp01(crowd * float(config.DEMO_CROWD_MULT))

        rainfall = clamp01(rainfall)
        elevation = clamp01(elevation)
        water = clamp01(water)
        urban = clamp01(urban)
        crowd = clamp01(crowd)

        score = (
            float(config.RISK_WEIGHTS["rainfall"]) * rainfall
            + float(config.RISK_WEIGHTS["elevation"]) * elevation
            + float(config.RISK_WEIGHTS["water_proximity"]) * water
            + float(config.RISK_WEIGHTS["urban_density"]) * urban
            + float(config.RISK_WEIGHTS["crowd_reports"]) * crowd
        )
        score = clamp01(score)

        expl = {
            "rainfall": rainfall,
            "elevation": elevation,
            "water_proximity": water,
            "urban_density": urban,
            "crowd_reports": crowd,
        }
        return score, expl

    def _rainfall_component(self) -> float:
        v = self.loader.latest_rainfall_value()
        if v is None:
            return 0.85 if config.DEMO_MODE else 0.30
        try:
            return normalize_minmax(float(v), float(self.loader.rainfall_min), float(self.loader.rainfall_max))
        except Exception:
            logger.exception("Rainfall normalization failed")
            return 0.85 if config.DEMO_MODE else 0.30

    def _low_elevation_component(self, lat: float, lon: float) -> float:
        ds = self.loader.elevation_ds
        stats = self.loader.elevation_stats
        if ds is None or stats is None:
            return 0.80 if config.DEMO_MODE else 0.35

        v = self.loader.sample_raster_value(ds, lat=lat, lon=lon)
        if v is None:
            return 0.80 if config.DEMO_MODE else 0.35

        # Normalize elevation, then invert: lower elevation => higher flood risk.
        elev_norm = normalize_minmax(float(v), float(stats.vmin), float(stats.vmax))
        return clamp01(1.0 - elev_norm)

    def _urban_density_component(self, lat: float, lon: float) -> float:
        ds = self.loader.landcover_ds
        stats = self.loader.landcover_stats
        if ds is None or stats is None:
            return 0.65 if config.DEMO_MODE else 0.40

        v = self.loader.sample_raster_value(ds, lat=lat, lon=lon)
        if v is None:
            return 0.65 if config.DEMO_MODE else 0.40

        # Landcover class/value is dataset-dependent.
        # For demo-safe explainability we treat higher normalized landcover values as higher urban density.
        return normalize_minmax(float(v), float(stats.vmin), float(stats.vmax))

    def _water_proximity_component(self, lat: float, lon: float) -> float:
        pts = self.loader.water_points_lonlat()
        if pts is None or pts.size == 0:
            return 0.75 if config.DEMO_MODE else 0.30

        # Fast equirectangular approximation for city-scale distances.
        lat0 = float(lat)
        lon0 = float(lon)
        rad = np.pi / 180.0
        cos_lat = np.cos(lat0 * rad)

        dlon = (pts[:, 0] - lon0) * 111320.0 * cos_lat
        dlat = (pts[:, 1] - lat0) * 110540.0
        dist_m = np.sqrt(dlon * dlon + dlat * dlat)
        min_m = float(np.min(dist_m))

        # Closer to water => higher risk (inverse normalized)
        return clamp01(1.0 - clamp01(min_m / float(config.WATER_PROXIMITY_MAX_M)))

    def _crowd_component(self, lat: float, lon: float) -> float:
        count, sev_sum = self.store.stats_near(lat, lon, radius_m=float(config.CROWD_INFLUENCE_RADIUS_M))
        # Blend count and severity. We normalize by a maximum expected local report density.
        density_score = clamp01(float(count) / float(config.CROWD_MAX_REPORTS_FOR_FULL_SCORE))
        severity_score = clamp01(float(sev_sum) / float(config.CROWD_MAX_REPORTS_FOR_FULL_SCORE))
        return clamp01(0.6 * density_score + 0.4 * severity_score)

    @lru_cache(maxsize=10000)
    def cached_risk_score(self, lat_rounded: float, lon_rounded: float) -> float:
        score, _lvl, _expl = self.compute(lat_rounded, lon_rounded)
        return float(score)

    def risk_score_cached(self, lat: float, lon: float) -> float:
        lat_r = round_coord(lat, config.RISK_CACHE_ROUND_DECIMALS)
        lon_r = round_coord(lon, config.RISK_CACHE_ROUND_DECIMALS)
        return self.cached_risk_score(lat_r, lon_r)

    def route_risk_from_points(self, points_latlon: list[tuple[float, float]]) -> float:
        if not points_latlon:
            return 0.0
        vals = [self.risk_score_cached(lat, lon) for (lat, lon) in points_latlon]
        return clamp01(mean(vals))
