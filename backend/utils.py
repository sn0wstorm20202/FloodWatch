from __future__ import annotations

import math
from typing import Iterable


def clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))


def normalize_minmax(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.0
    return clamp01((float(value) - float(vmin)) / (float(vmax) - float(vmin)))


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def round_coord(value: float, decimals: int) -> float:
    return round(float(value), int(decimals))


def to_linestring_geojson_lonlat(coords_lonlat: list[list[float]]) -> dict:
    return {"type": "LineString", "coordinates": coords_lonlat}


def km_from_m(meters: float) -> float:
    return float(meters) / 1000.0


def risk_level(score: float) -> str:
    score = clamp01(score)
    if score <= 0.30:
        return "Safe"
    if score <= 0.60:
        return "Risky"
    return "Flooded"
