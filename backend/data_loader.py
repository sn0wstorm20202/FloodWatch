from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd
import rasterio
try:
    import geopandas as gpd
    from shapely.geometry import Point
except Exception:  # pragma: no cover
    gpd = None
    Point = None
from rasterio.enums import Resampling
from rasterio.warp import transform

import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RasterStats:
    vmin: float
    vmax: float


class DataLoader:
    """Loads and holds all datasets in memory / open handles.

    Design goals:
    - Load once at startup
    - Never crash the service if a dataset is missing/corrupt
    - Provide cheap sampling helpers for per-request computations
    """

    def __init__(self) -> None:
        self.rainfall_df: Optional[pd.DataFrame] = None
        self.rainfall_col: Optional[str] = None
        self.rainfall_min: float = 0.0
        self.rainfall_max: float = 1.0

        self.surface_water_df: Optional[pd.DataFrame] = None
        self.surface_water_gdf = None
        self.water_lat_col: Optional[str] = None
        self.water_lon_col: Optional[str] = None
        self._water_points_lonlat: Optional[np.ndarray] = None
        self._water_occurrence: Optional[np.ndarray] = None

        self.elevation_ds: Optional[rasterio.io.DatasetReader] = None
        self.elevation_stats: Optional[RasterStats] = None

        self.landcover_ds: Optional[rasterio.io.DatasetReader] = None
        self.landcover_stats: Optional[RasterStats] = None

        self.graph: Optional[nx.Graph] = None
        self._graph_nodes_lonlat: Optional[np.ndarray] = None
        self._graph_node_ids: Optional[list] = None

    def load_all(self) -> None:
        self._load_rainfall()
        self._load_surface_water()
        self._load_elevation_raster()
        self._load_landcover_raster()
        self._load_roads_graph()

    def _candidate_paths(self, primary: Path) -> list[Path]:
        candidates: list[Path] = [primary]

        # If the project data/ directory isn't populated yet, fall back to common local locations.
        home = Path.home()
        candidates.append(config.PROJECT_DIR / primary.name)
        candidates.append(home / "Downloads" / primary.name)

        # Also allow explicit override via environment variable for container deployments.
        env_key = f"FLOODWATCH_{primary.stem.upper()}_PATH"
        if os.getenv(env_key):
            candidates.insert(0, Path(os.environ[env_key]))

        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[Path] = []
        for p in candidates:
            ps = str(p)
            if ps in seen:
                continue
            seen.add(ps)
            deduped.append(p)
        return deduped

    def _resolve_existing_path(self, primary: Path) -> Optional[Path]:
        for p in self._candidate_paths(primary):
            try:
                if p.exists():
                    return p
            except OSError:
                continue
        return None

    def _load_rainfall(self) -> None:
        path = self._resolve_existing_path(config.RAINFALL_CSV_PATH)
        if not path:
            logger.warning("Rainfall CSV not found at %s (or fallbacks). Using demo-safe defaults.", config.RAINFALL_CSV_PATH)
            self.rainfall_df = None
            self.rainfall_col = None
            self.rainfall_min = 0.0
            self.rainfall_max = 1.0
            return

        try:
            df = pd.read_csv(path)
        except Exception:
            logger.exception("Failed to load rainfall CSV from %s. Using demo-safe defaults.", path)
            self.rainfall_df = None
            self.rainfall_col = None
            self.rainfall_min = 0.0
            self.rainfall_max = 1.0
            return

        rainfall_col = self._pick_numeric_column(df, preferred_substrings=["rain", "mm"])  # heuristic
        if not rainfall_col:
            logger.warning("Rainfall CSV loaded but no numeric column found. Using demo-safe defaults.")
            self.rainfall_df = df
            self.rainfall_col = None
            self.rainfall_min = 0.0
            self.rainfall_max = 1.0
            return

        series = pd.to_numeric(df[rainfall_col], errors="coerce").dropna()
        if series.empty:
            self.rainfall_df = df
            self.rainfall_col = rainfall_col
            self.rainfall_min = 0.0
            self.rainfall_max = 1.0
            return

        self.rainfall_df = df
        self.rainfall_col = rainfall_col

        # Use robust min/max to reduce outlier sensitivity.
        self.rainfall_min = float(series.quantile(0.05))
        self.rainfall_max = float(series.quantile(0.95))
        if self.rainfall_max <= self.rainfall_min:
            self.rainfall_min = float(series.min())
            self.rainfall_max = float(series.max())
        if self.rainfall_max <= self.rainfall_min:
            self.rainfall_min = 0.0
            self.rainfall_max = max(1.0, float(series.max()))

        logger.info(
            "Loaded rainfall CSV: path=%s col=%s vmin=%.3f vmax=%.3f", path, rainfall_col, self.rainfall_min, self.rainfall_max
        )

    def _load_surface_water(self) -> None:
        path = self._resolve_existing_path(config.SURFACE_WATER_CSV_PATH)
        if not path:
            logger.warning("Surface water CSV not found at %s (or fallbacks).", config.SURFACE_WATER_CSV_PATH)
            self.surface_water_df = None
            return

        try:
            df = pd.read_csv(path)
        except Exception:
            logger.exception("Failed to load surface water CSV from %s.", path)
            self.surface_water_df = None
            return

        lat_col, lon_col = self._pick_lat_lon_columns(df)
        if not lat_col or not lon_col:
            geo_col = ".geo" if ".geo" in df.columns else None
            if geo_col is not None:
                lats: list[float] = []
                lons: list[float] = []

                for v in df[geo_col].tolist():
                    try:
                        if not isinstance(v, str) or not v:
                            lats.append(np.nan)
                            lons.append(np.nan)
                            continue
                        gj = json.loads(v)
                        coords = gj.get("coordinates")
                        if not coords or len(coords) < 2:
                            lats.append(np.nan)
                            lons.append(np.nan)
                            continue
                        lon, lat = float(coords[0]), float(coords[1])
                        lats.append(lat)
                        lons.append(lon)
                    except Exception:
                        lats.append(np.nan)
                        lons.append(np.nan)

                df = df.copy()
                df["lat"] = lats
                df["lon"] = lons
                lat_col, lon_col = "lat", "lon"

        if not lat_col or not lon_col:
            logger.warning("Surface water CSV loaded but lat/lon columns not detected.")
            self.surface_water_df = df
            return

        occ_col = "occurrence" if "occurrence" in df.columns else None
        if occ_col is None:
            occ_col = self._pick_numeric_column(df, preferred_substrings=["occur", "occ"])

        lat = pd.to_numeric(df[lat_col], errors="coerce")
        lon = pd.to_numeric(df[lon_col], errors="coerce")

        mask = lat.notna() & lon.notna()
        df = df.loc[mask].copy()

        self.surface_water_df = df
        if gpd is not None and Point is not None:
            try:
                self.surface_water_gdf = gpd.GeoDataFrame(
                    df,
                    geometry=[Point(xy) for xy in zip(df[lon_col].astype(float), df[lat_col].astype(float))],
                    crs="EPSG:4326",
                )
            except Exception:
                logger.exception("Failed to build GeoDataFrame for surface water; continuing with DataFrame")
                self.surface_water_gdf = None
        self.water_lat_col = lat_col
        self.water_lon_col = lon_col
        self._water_points_lonlat = np.column_stack(
            [df[lon_col].astype(float).to_numpy(), df[lat_col].astype(float).to_numpy()]
        )

        try:
            occ_clean = pd.to_numeric(df[occ_col], errors="coerce") if occ_col else pd.Series(np.full(len(df), np.nan))
            occ_clean = occ_clean.fillna(0.0).astype(float)
            self._water_occurrence = np.clip(occ_clean.to_numpy(dtype=float) / 100.0, 0.0, 1.0)
        except Exception:
            logger.exception("Failed to normalize surface water occurrence; defaulting to zeros")
            self._water_occurrence = np.zeros(len(df), dtype=float)

        logger.info("Loaded surface water CSV: path=%s rows=%d lat_col=%s lon_col=%s", path, len(df), lat_col, lon_col)

    def _load_elevation_raster(self) -> None:
        path = self._resolve_existing_path(config.ELEVATION_TIF_PATH)
        if not path:
            logger.warning("Elevation TIFF not found at %s (or fallbacks).", config.ELEVATION_TIF_PATH)
            self.elevation_ds = None
            self.elevation_stats = None
            return

        try:
            ds = rasterio.open(path)
        except Exception:
            logger.exception("Failed to open elevation TIFF: %s", path)
            self.elevation_ds = None
            self.elevation_stats = None
            return

        stats = self._estimate_raster_stats(ds)
        self.elevation_ds = ds
        self.elevation_stats = stats
        logger.info("Opened elevation raster: path=%s vmin=%.3f vmax=%.3f crs=%s", path, stats.vmin, stats.vmax, ds.crs)

    def _load_landcover_raster(self) -> None:
        path = self._resolve_existing_path(config.LANDCOVER_TIF_PATH)
        if not path:
            logger.warning("Landcover TIFF not found at %s (or fallbacks).", config.LANDCOVER_TIF_PATH)
            self.landcover_ds = None
            self.landcover_stats = None
            return

        try:
            ds = rasterio.open(path)
        except Exception:
            logger.exception("Failed to open landcover TIFF: %s", path)
            self.landcover_ds = None
            self.landcover_stats = None
            return

        stats = self._estimate_raster_stats(ds)
        self.landcover_ds = ds
        self.landcover_stats = stats
        logger.info("Opened landcover raster: path=%s vmin=%.3f vmax=%.3f crs=%s", path, stats.vmin, stats.vmax, ds.crs)

    def _load_roads_graph(self) -> None:
        path = self._resolve_existing_path(config.ROADS_GRAPHML_PATH)
        if not path:
            logger.warning("Road graph GraphML not found at %s (or fallbacks).", config.ROADS_GRAPHML_PATH)
            self.graph = None
            self._graph_nodes_lonlat = None
            self._graph_node_ids = None
            return

        try:
            G = nx.read_graphml(path)
        except Exception:
            logger.exception("Failed to read roads graph from %s", path)
            self.graph = None
            self._graph_nodes_lonlat = None
            self._graph_node_ids = None
            return

        # graphml reads node ids as strings; that's fine.
        node_ids: list = []
        coords: list[list[float]] = []
        for n, data in G.nodes(data=True):
            lon, lat = self._node_lonlat(data)
            if lon is None or lat is None:
                continue
            node_ids.append(n)
            coords.append([lon, lat])

        self.graph = G
        self._graph_node_ids = node_ids
        self._graph_nodes_lonlat = np.asarray(coords, dtype=float) if coords else None

        logger.info("Loaded roads graph: path=%s nodes=%d edges=%d", path, G.number_of_nodes(), G.number_of_edges())

    def _pick_numeric_column(self, df: pd.DataFrame, preferred_substrings: list[str]) -> Optional[str]:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            # Try to coerce columns with preferred substrings
            for c in df.columns:
                if any(s in str(c).lower() for s in preferred_substrings):
                    series = pd.to_numeric(df[c], errors="coerce")
                    if series.notna().any():
                        return str(c)
            # Fallback: first coercible column
            for c in df.columns:
                series = pd.to_numeric(df[c], errors="coerce")
                if series.notna().any():
                    return str(c)
            return None

        for c in numeric_cols:
            if any(s in str(c).lower() for s in preferred_substrings):
                return str(c)
        return str(numeric_cols[0])

    def _pick_lat_lon_columns(self, df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
        cols = [str(c) for c in df.columns]
        lower = {c: c.lower() for c in cols}

        lat_candidates = [c for c in cols if lower[c] in {"lat", "latitude"} or "lat" == lower[c][-3:]]
        lon_candidates = [c for c in cols if lower[c] in {"lon", "lng", "longitude"} or "lon" == lower[c][-3:] or "lng" == lower[c][-3:]]

        if not lat_candidates:
            lat_candidates = [c for c in cols if "lat" in lower[c]]
        if not lon_candidates:
            lon_candidates = [c for c in cols if "lon" in lower[c] or "lng" in lower[c]]

        lat_col = lat_candidates[0] if lat_candidates else None
        lon_col = lon_candidates[0] if lon_candidates else None
        return lat_col, lon_col

    def _estimate_raster_stats(self, ds: rasterio.io.DatasetReader) -> RasterStats:
        try:
            band = ds.read(
                1,
                out_shape=(1, 256, 256),
                resampling=Resampling.bilinear,
                masked=True,
            )
            arr = np.asarray(band, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                return RasterStats(vmin=0.0, vmax=1.0)
            vmin = float(np.nanpercentile(arr, 5))
            vmax = float(np.nanpercentile(arr, 95))
            if vmax <= vmin:
                vmin = float(np.nanmin(arr))
                vmax = float(np.nanmax(arr))
            if vmax <= vmin:
                vmin, vmax = 0.0, 1.0
            return RasterStats(vmin=vmin, vmax=vmax)
        except Exception:
            logger.exception("Failed to estimate raster stats; using defaults")
            return RasterStats(vmin=0.0, vmax=1.0)

    def sample_raster_value(self, ds: Optional[rasterio.io.DatasetReader], lat: float, lon: float) -> Optional[float]:
        if ds is None:
            return None

        try:
            src_crs = "EPSG:4326"
            dst_crs = ds.crs or src_crs

            if str(dst_crs) != src_crs:
                xs, ys = transform(src_crs, dst_crs, [lon], [lat])
                x, y = xs[0], ys[0]
            else:
                x, y = lon, lat

            vals = list(ds.sample([(x, y)]))
            if not vals:
                return None
            v = float(vals[0][0])
            if not np.isfinite(v):
                return None
            return v
        except Exception:
            logger.exception("Raster sample failed")
            return None

    def latest_rainfall_value(self) -> Optional[float]:
        if self.rainfall_df is None or not self.rainfall_col:
            return None
        try:
            series = pd.to_numeric(self.rainfall_df[self.rainfall_col], errors="coerce").dropna()
            if series.empty:
                return None
            return float(series.iloc[-1])
        except Exception:
            logger.exception("Failed to compute latest rainfall")
            return None

    def water_points_lonlat(self) -> Optional[np.ndarray]:
        return self._water_points_lonlat

    def water_occurrence_values(self) -> Optional[np.ndarray]:
        return self._water_occurrence

    def nearest_graph_node(self, lat: float, lon: float) -> Optional[str]:
        if self._graph_nodes_lonlat is None or not self._graph_node_ids:
            return None
        coords = self._graph_nodes_lonlat
        # Approximate nearest neighbor in degrees (good enough for a city-scale graph).
        d2 = (coords[:, 0] - lon) ** 2 + (coords[:, 1] - lat) ** 2
        idx = int(np.argmin(d2))
        return str(self._graph_node_ids[idx])

    def node_lonlat_by_id(self, node_id: str) -> tuple[Optional[float], Optional[float]]:
        if not self.graph:
            return None, None
        data = self.graph.nodes.get(node_id)
        if not data:
            return None, None
        lon, lat = self._node_lonlat(data)
        return lon, lat

    def _node_lonlat(self, data: dict) -> tuple[Optional[float], Optional[float]]:
        # OSM-style graphs commonly store x=lon, y=lat.
        lon = data.get("x", data.get("lon", data.get("lng")))
        lat = data.get("y", data.get("lat"))
        try:
            if lon is None or lat is None:
                return None, None
            return float(lon), float(lat)
        except Exception:
            return None, None
