from __future__ import annotations

import argparse
import logging

import config
from data_loader import DataLoader
from ml_engine import MLEngine

logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, logging.INFO))
logger = logging.getLogger("floodwatch.train_ml")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/retrain FloodWatch ML model and write artifacts.")
    parser.add_argument("--force", action="store_true", help="Force retrain even if a persisted model exists")
    args = parser.parse_args()

    loader = DataLoader()

    # Roads graph is not needed for ML training; keep memory lower by disabling it here.
    try:
        config.LOAD_ROADS_GRAPH = False
    except Exception:
        pass

    loader.load_all()

    me = MLEngine(loader)
    me.load_or_train(force_retrain=bool(args.force))

    if me.models is None:
        logger.error("Training did not produce a model. Check datasets and logs.")
        return 2

    logger.info("Training complete. Artifacts written to %s", str(config.ML_ARTIFACT_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
