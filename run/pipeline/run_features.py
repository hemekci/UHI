"""Stage-3 step 2: combine per-city patch parquets into a harmonized feature table.

Reads:  data/patches/<city_id>.parquet for each city in the selected cities config.
Writes: data/features/<cities_name>.parquet

Also derives:
- impervious_frac = wc_built_frac + wc_bare_frac
- vegetation_frac = wc_tree_frac + wc_shrub_frac + wc_grass_frac + wc_cropland_frac

Usage:
    python run/pipeline/run_features.py cities=demo
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.utils import get_logger, write_patches  # noqa: E402

logger = get_logger(__name__, level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    patch_dir = Path(cfg.data_root) / "patches"
    out_dir = Path(cfg.data_root) / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.cities.name}.parquet"

    frames = []
    missing = []
    for entry in cfg.cities.cities:
        pid = entry["id"]
        pp = patch_dir / f"{pid}.parquet"
        if not pp.exists():
            missing.append(pid)
            continue
        frames.append(pd.read_parquet(pp))
    if not frames:
        logger.error("No patch parquets found in %s. Run run_ingest.py first.", patch_dir)
        return
    if missing:
        logger.warning("Missing patch parquets for: %s", missing)

    df = pd.concat(frames, ignore_index=True)
    logger.info("Combined %d cities -> %d patches", len(frames), len(df))

    # Derive convenience features.
    if {"wc_built_frac", "wc_bare_frac"}.issubset(df.columns):
        df["impervious_frac"] = df["wc_built_frac"] + df["wc_bare_frac"]
    for col in ("wc_tree_frac", "wc_shrub_frac", "wc_grass_frac", "wc_cropland_frac"):
        if col not in df.columns:
            df[col] = 0.0
    df["vegetation_frac"] = (
        df["wc_tree_frac"] + df["wc_shrub_frac"] + df["wc_grass_frac"] + df["wc_cropland_frac"]
    )

    # Drop rows without a valid UHI target.
    n_before = len(df)
    df = df.dropna(subset=["uhi_anomaly_c"])
    logger.info("Dropped %d rows missing uhi_anomaly_c", n_before - len(df))

    write_patches(df, out_path)
    logger.info("Wrote %d rows, %d cols -> %s", len(df), df.shape[1], out_path)

    # Quick per-zone summary for sanity.
    logger.info("Per-zone counts:")
    for zone, sub in df.groupby("koppen_zone"):
        logger.info(
            "  %s: n=%d, UHI mean=%.2f std=%.2f, density mean=%.3f",
            zone, len(sub), sub.uhi_anomaly_c.mean(), sub.uhi_anomaly_c.std(),
            sub.building_density.mean(),
        )


if __name__ == "__main__":
    main()
