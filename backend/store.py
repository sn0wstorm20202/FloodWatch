from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import config
from utils import clamp01, haversine_m


@dataclass(frozen=True)
class CrowdReport:
    lat: float
    lon: float
    severity: int
    ts: float


class ReportStore:
    """In-memory store for crowd flood reports.

    This store is intentionally simple (demo-safe). In production, replace with Redis/PostGIS.
    """

    def __init__(self) -> None:
        self._reports: List[CrowdReport] = []

    def _now(self) -> float:
        return time.time()

    def prune(self) -> None:
        cutoff = self._now() - float(config.CROWD_REPORT_TTL_SECONDS)
        self._reports = [r for r in self._reports if r.ts >= cutoff]

    def add_report(self, lat: float, lon: float, severity: int) -> None:
        self.prune()
        severity = int(severity)
        severity = max(config.CROWD_SEVERITY_MIN, min(config.CROWD_SEVERITY_MAX, severity))
        self._reports.append(CrowdReport(lat=float(lat), lon=float(lon), severity=severity, ts=self._now()))

    def add_report_at(self, lat: float, lon: float, severity: int, ts: float) -> None:
        self.prune()
        severity = int(severity)
        severity = max(config.CROWD_SEVERITY_MIN, min(config.CROWD_SEVERITY_MAX, severity))
        self._reports.append(CrowdReport(lat=float(lat), lon=float(lon), severity=severity, ts=float(ts)))

    def all_reports(self) -> List[CrowdReport]:
        self.prune()
        return list(self._reports)

    def stats_near(self, lat: float, lon: float, radius_m: float = config.CROWD_INFLUENCE_RADIUS_M) -> Tuple[int, float]:
        """Return (count, severity_sum_norm) within radius.

        severity_sum_norm sums each report severity scaled to [0,1].
        """
        self.prune()
        count = 0
        sev_sum = 0.0
        for r in self._reports:
            if haversine_m(lat, lon, r.lat, r.lon) <= float(radius_m):
                count += 1
                sev_sum += clamp01((r.severity - config.CROWD_SEVERITY_MIN) / (config.CROWD_SEVERITY_MAX - config.CROWD_SEVERITY_MIN))
        return count, sev_sum

    def hotspots(self, grid_decimals: int = 2) -> List[Tuple[float, float, int]]:
        """Return hotspots as (lat, lon, report_count) grouped by rounded lat/lon."""
        self.prune()
        buckets = {}
        for r in self._reports:
            key = (round(float(r.lat), grid_decimals), round(float(r.lon), grid_decimals))
            buckets[key] = buckets.get(key, 0) + 1
        out = [(lat, lon, cnt) for (lat, lon), cnt in buckets.items()]
        out.sort(key=lambda x: x[2], reverse=True)
        return out

    def hotspots_detailed(self, grid_decimals: int = 2) -> List[dict]:
        self.prune()
        buckets: dict[tuple[float, float], dict[str, float]] = {}
        for r in self._reports:
            key = (round(float(r.lat), grid_decimals), round(float(r.lon), grid_decimals))
            b = buckets.get(key)
            if b is None:
                buckets[key] = {"count": 0.0, "sev_sum": 0.0, "sev_max": float(r.severity)}
                b = buckets[key]
            b["count"] = float(b.get("count", 0.0)) + 1.0
            b["sev_sum"] = float(b.get("sev_sum", 0.0)) + float(r.severity)
            b["sev_max"] = float(max(float(b.get("sev_max", 0.0)), float(r.severity)))

        out: List[dict] = []
        for (lat, lon), b in buckets.items():
            cnt = int(b.get("count", 0.0))
            if cnt <= 0:
                continue
            sev_avg = float(b.get("sev_sum", 0.0)) / float(cnt)
            out.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "count": int(cnt),
                    "severity_max": int(b.get("sev_max", 0.0)),
                    "severity_avg": float(sev_avg),
                }
            )
        out.sort(key=lambda x: int(x.get("count", 0)), reverse=True)
        return out
