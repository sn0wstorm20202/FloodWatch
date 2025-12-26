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

    def get_alerts(self) -> List[str]:
        alerts: List[str] = []

        hotspots = self.store.hotspots(grid_decimals=2)
        if not hotspots:
            # Demo-safe: if no crowd reports but demo mode is on, still return empty list (contract).
            return []

        for lat, lon, count in hotspots:
            try:
                score = float(self.ml_engine.predict_risk(float(lat), float(lon))["ml_risk"])
            except Exception:
                logger.exception("Failed to compute risk for alerts")
                continue

            if float(score) > 0.70 and int(count) >= 3:
                alerts.append(
                    f"High flood risk near {float(lat):.2f}, {float(lon):.2f} – dispatch recommended"
                )

        if alerts:
            for a in alerts:
                logger.warning("ALERT (mock): %s", a)
        return alerts
