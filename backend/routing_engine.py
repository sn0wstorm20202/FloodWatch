from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import networkx as nx

import config
from data_loader import DataLoader
from ml_engine import MLEngine
from store import ReportStore
from utils import clamp01, haversine_m, km_from_m, mean, to_linestring_geojson_lonlat

logger = logging.getLogger(__name__)


class RoutingEngine:
    """Flood-aware routing on the Kolkata roads graph.

    Edge weight (MANDATORY):
        edge_weight = length * (1 + ml_risk)
    """

    def __init__(self, loader: DataLoader, ml_engine: MLEngine, store: ReportStore | None = None) -> None:
        self.loader = loader
        self.ml_engine = ml_engine
        self.store = store
        self._use_crowd_for_weight: bool = False
        self.graph = self._prepare_graph(loader.graph)

    def _prepare_graph(self, G: Optional[nx.Graph]) -> Optional[nx.Graph]:
        if G is None:
            return None

        # NetworkX GraphML loaders may produce MultiDiGraph. We simplify to DiGraph
        # by keeping the smallest-length edge between node pairs.
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            H = nx.DiGraph() if G.is_directed() else nx.Graph()
            H.add_nodes_from(G.nodes(data=True))
            best: dict[tuple[str, str], tuple[float, dict]] = {}
            for u, v, _k, data in G.edges(keys=True, data=True):
                u2, v2 = str(u), str(v)
                length = self._edge_length_m(u2, v2, data)
                key = (u2, v2)
                if key not in best or length < best[key][0]:
                    best[key] = (length, dict(data))
            for (u, v), (_len, data) in best.items():
                H.add_edge(u, v, **data)
            return H

        return G

    def route(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        use_crowd: bool = False,
    ) -> Tuple[float, float, dict]:
        """Compute safest route.

        Returns:
            distance_km, route_risk, geometry_geojson

        Demo-safe behavior: if routing fails or graph is unavailable,
        returns a straight-line LineString.
        """

        if self.graph is None:
            return self._fallback_route(start_lat, start_lon, end_lat, end_lon)

        start_node = self.loader.nearest_graph_node(start_lat, start_lon)
        end_node = self.loader.nearest_graph_node(end_lat, end_lon)
        if start_node is None or end_node is None:
            return self._fallback_route(start_lat, start_lon, end_lat, end_lon)

        prev = bool(self._use_crowd_for_weight)
        self._use_crowd_for_weight = bool(use_crowd)
        try:
            path = nx.shortest_path(self.graph, source=start_node, target=end_node, weight=self._weight)
        except Exception:
            logger.exception("Routing failed; using fallback route")
            self._use_crowd_for_weight = prev
            return self._fallback_route(start_lat, start_lon, end_lat, end_lon)
        finally:
            self._use_crowd_for_weight = prev

        coords_lonlat: list[list[float]] = []
        points_latlon: list[tuple[float, float]] = []
        for n in path:
            lon, lat = self.loader.node_lonlat_by_id(str(n))
            if lon is None or lat is None:
                continue
            coords_lonlat.append([float(lon), float(lat)])
            points_latlon.append((float(lat), float(lon)))

        if len(coords_lonlat) < 2:
            return self._fallback_route(start_lat, start_lon, end_lat, end_lon)

        distance_m = self._path_length_m(path)
        route_risk = self._route_risk_from_points(points_latlon, use_crowd=bool(use_crowd))
        return km_from_m(distance_m), float(route_risk), to_linestring_geojson_lonlat(coords_lonlat)

    def _route_risk_from_points(self, points_latlon: list[tuple[float, float]], use_crowd: bool) -> float:
        if not points_latlon:
            return 0.0
        vals = [float(self._risk_at_point(lat, lon, use_crowd=bool(use_crowd))) for (lat, lon) in points_latlon]
        return float(mean(vals))

    def _crowd_risk_at(self, lat: float, lon: float) -> float:
        st = self.store
        if st is None:
            return 0.0
        try:
            count, sev_sum = st.stats_near(lat, lon, radius_m=float(config.CROWD_INFLUENCE_RADIUS_M))
            density_score = clamp01(float(count) / float(config.CROWD_MAX_REPORTS_FOR_FULL_SCORE))
            severity_score = clamp01(float(sev_sum) / float(config.CROWD_MAX_REPORTS_FOR_FULL_SCORE))
            return clamp01(0.6 * density_score + 0.4 * severity_score)
        except Exception:
            return 0.0

    def _risk_at_point(self, lat: float, lon: float, use_crowd: bool) -> float:
        try:
            ml = float(self.ml_engine.predict_risk(float(lat), float(lon))["ml_risk"])
        except Exception:
            ml = 0.7 if config.DEMO_MODE else 0.3

        if not use_crowd:
            return clamp01(ml)

        crowd = self._crowd_risk_at(float(lat), float(lon))
        return clamp01(max(float(ml), float(crowd)))

    def _path_length_m(self, path: list[str]) -> float:
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            data = self._get_edge_data_best(u, v)
            lon_u, lat_u = self.loader.node_lonlat_by_id(str(u))
            lon_v, lat_v = self.loader.node_lonlat_by_id(str(v))
            total += self._edge_length_m(str(u), str(v), data, fallback_lonlat=(lon_u, lat_u, lon_v, lat_v))
        return float(total)

    def _get_edge_data_best(self, u: str, v: str) -> Dict[str, Any]:
        try:
            data = self.graph.get_edge_data(u, v)
            if data is None:
                return {}
            # For MultiGraphs this could be a dict-of-dicts; but we already simplified.
            if isinstance(data, dict) and any(isinstance(val, dict) for val in data.values()):
                # choose arbitrary
                first = next(iter(data.values()))
                return dict(first)
            return dict(data)
        except Exception:
            return {}

    def _edge_length_m(
        self,
        u: str,
        v: str,
        data: Dict[str, Any],
        fallback_lonlat: Optional[tuple[Optional[float], Optional[float], Optional[float], Optional[float]]] = None,
    ) -> float:
        for key in ("length", "Length", "len", "distance", "dist"):
            if key in data:
                try:
                    return float(data[key])
                except Exception:
                    pass

        if fallback_lonlat is not None:
            lon_u, lat_u, lon_v, lat_v = fallback_lonlat
            if None not in (lon_u, lat_u, lon_v, lat_v):
                return float(haversine_m(float(lat_u), float(lon_u), float(lat_v), float(lon_v)))

        lon_u, lat_u = self.loader.node_lonlat_by_id(str(u))
        lon_v, lat_v = self.loader.node_lonlat_by_id(str(v))
        if None in (lon_u, lat_u, lon_v, lat_v):
            return 1.0
        return float(haversine_m(float(lat_u), float(lon_u), float(lat_v), float(lon_v)))

    def _edge_midpoint_latlon(self, u: str, v: str) -> Optional[tuple[float, float]]:
        lon_u, lat_u = self.loader.node_lonlat_by_id(str(u))
        lon_v, lat_v = self.loader.node_lonlat_by_id(str(v))
        if None in (lon_u, lat_u, lon_v, lat_v):
            return None
        return (float(lat_u + lat_v) / 2.0, float(lon_u + lon_v) / 2.0)

    def _weight(self, u: str, v: str, data: Dict[str, Any]) -> float:
        length_m = self._edge_length_m(str(u), str(v), data)
        mid = self._edge_midpoint_latlon(str(u), str(v))
        if mid is None:
            risk = 0.7 if config.DEMO_MODE else 0.3
        else:
            risk = float(self._risk_at_point(mid[0], mid[1], use_crowd=bool(self._use_crowd_for_weight)))

        # Mandatory weighting
        return float(length_m) * (1.0 + float(risk))

    def _fallback_route(self, start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> Tuple[float, float, dict]:
        dist_m = haversine_m(start_lat, start_lon, end_lat, end_lon)
        route_risk = mean(
            [
                float(self._risk_at_point(start_lat, start_lon, use_crowd=bool(self._use_crowd_for_weight))),
                float(self._risk_at_point(end_lat, end_lon, use_crowd=bool(self._use_crowd_for_weight))),
            ]
        )
        geom = to_linestring_geojson_lonlat([[float(start_lon), float(start_lat)], [float(end_lon), float(end_lat)]])
        return km_from_m(dist_m), float(route_risk), geom
