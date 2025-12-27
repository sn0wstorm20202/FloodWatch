from __future__ import annotations

import logging
from typing import List

import config
from ml_engine import MLEngine
from store import ReportStore

logger = logging.getLogger(__name__)


class AlertsEngine:
    """Mock authority alert logic.

    Trigger alert if:
        risk_score > 0.7 AND reports >= 3

    Implementation:
    - We scan report hotspots (rounded grid) and evaluate risk there.
    - If any hotspot meets threshold, we emit an alert string.
    """

    def __init__(self, store: ReportStore, ml_engine: MLEngine) -> None:
        self.store = store
        self.ml_engine = ml_engine

    def get_alerts(self) -> List[dict]:
        alerts: List[dict] = []

        hotspots = self.store.hotspots_detailed(grid_decimals=2)
        if not hotspots:
            return []

        for h in hotspots:
            lat = float(h.get("lat", 0.0))
            lon = float(h.get("lon", 0.0))
            count = int(h.get("count", 0))
            sev_max = int(h.get("severity_max", 0))
            sev_avg = float(h.get("severity_avg", 0.0))

            try:
                score = float(self.ml_engine.predict_risk(float(lat), float(lon))["ml_risk"])
            except Exception:
                logger.exception("Failed to compute risk for alerts")
                continue

            if float(score) > 0.70 and int(count) >= 3:
                msg = f"High flood risk near {float(lat):.2f}, {float(lon):.2f} – dispatch recommended"
                alerts.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "risk_score": float(score),
                        "count": int(count),
                        "severity_max": int(sev_max),
                        "severity_avg": float(sev_avg),
                        "message": msg,
                    }
                )

        if alerts:
            for a in alerts:
                try:
                    logger.warning("ALERT (mock): %s", a.get("message"))
                except Exception:
                    pass
        return alerts
