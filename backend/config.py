from __future__ import annotations

from pathlib import Path

# Demo-safe mode: intentionally biases the system toward higher visible risk.
DEMO_MODE = True

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
_DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DATA_DIR = _DEFAULT_DATA_DIR if _DEFAULT_DATA_DIR.exists() else PROJECT_DIR

RAINFALL_CSV_PATH = DATA_DIR / "Kolkata_Rainfall.csv"
SURFACE_WATER_CSV_PATH = DATA_DIR / "Kolkata_Surface_Water_CSV.csv"
ELEVATION_TIF_PATH = DATA_DIR / "Kolkata_Elevation.tif"
LANDCOVER_TIF_PATH = DATA_DIR / "Kolkata_Landcover.tif"
ROADS_GRAPHML_PATH = DATA_DIR / "kolkata_roads.graphml"

# Flood risk model weights (MANDATORY)
RISK_WEIGHTS = {
    "rainfall": 0.35,
    "elevation": 0.25,
    "water_proximity": 0.20,
    "urban_density": 0.10,
    "crowd_reports": 0.10,
}

# Risk level thresholds
SAFE_MAX = 0.30
RISKY_MAX = 0.60

# Demo mode amplification (applied to component values, then clamped to [0,1])
DEMO_RAINFALL_MULT = 1.35
DEMO_CROWD_MULT = 1.50

# Spatial / feature normalization parameters
WATER_PROXIMITY_MAX_M = 2000.0

CROWD_INFLUENCE_RADIUS_M = 500.0
CROWD_MAX_REPORTS_FOR_FULL_SCORE = 6
CROWD_SEVERITY_MIN = 1
CROWD_SEVERITY_MAX = 5
CROWD_REPORT_TTL_SECONDS = 6 * 60 * 60

# Routing
FLOODED_EDGE_THRESHOLD = 0.60
FLOODED_EDGE_PENALTY = 50.0

# Caching
RISK_CACHE_ROUND_DECIMALS = 3

# Logging
LOG_LEVEL = "INFO"
