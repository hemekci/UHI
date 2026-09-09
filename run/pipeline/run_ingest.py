"""Stage-3 step 1: per-city end-to-end ingestion to parquet (real data).

Usage:
    python run/pipeline/run_ingest.py cities=demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.ingest import City, GEEClient, OpenBuildingsClient, run_city  # noqa: E402
from src.utils import get_logger, set_seed  # noqa: E402

logger = get_logger(__name__, level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    logger.info("Ingest run — %d cities (%s)", len(cfg.cities.cities), cfg.cities.name)

    out_dir = Path(cfg.data_root) / "patches"

    gee = GEEClient(cfg.ingest)
    gee.authenticate()
    ob = OpenBuildingsClient(cfg.ingest)

    skipped = []
    done = []
    failed = []
    for entry in cfg.cities.cities:
        city = City.from_config(dict(entry))
        existing = out_dir / f"{city.id}.parquet"
        if existing.exists() and existing.stat().st_size > 0:
            logger.info("City %s: cached parquet exists, skipping", city.id)
            skipped.append(city.id)
            continue
        try:
            run_city(city, gee, ob, cfg.ingest, out_dir)
            done.append(city.id)
        except Exception:
            logger.exception("City %s failed; continuing", city.id)
            failed.append(city.id)

    logger.info("Ingestion complete. new=%d skipped=%d failed=%d", len(done), len(skipped), len(failed))
    if failed:
        logger.warning("Failed cities: %s", failed)


if __name__ == "__main__":
    main()
