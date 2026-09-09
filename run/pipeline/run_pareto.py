"""Stage-3 step 4: extract per-Koppen-zone empirical Pareto fronts.

Reads:  data/features/<cities_name>.parquet
Writes: <hydra_run_dir>/pareto_summary.json and data/features/<cities_name>.pareto.parquet
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.features import quality_screen  # noqa: E402
from src.pareto import extract_pareto  # noqa: E402
from src.utils import get_logger, read_patches, set_seed  # noqa: E402

logger = get_logger(__name__, level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    data_path = Path(cfg.data_root) / "features" / f"{cfg.cities.name}.parquet"
    if not data_path.exists():
        logger.error("Feature table not found at %s.", data_path)
        return
    df = read_patches(data_path)
    df = quality_screen(df, abs_max_c=getattr(cfg, "uhi_abs_max_c", 40.0))

    results = extract_pareto(
        df,
        objectives=[dict(o) for o in cfg.pareto.objectives],
        group_by=cfg.pareto.group_by,
        min_points_per_zone=cfg.pareto.min_points_per_zone,
    )

    # Collect Pareto-efficient patches across zones.
    pareto_frames = []
    for r in results:
        sub = df.loc[r.indices].copy()
        sub["_pareto_zone"] = r.zone
        pareto_frames.append(sub)
    pareto_df = pd.concat(pareto_frames, ignore_index=True) if pareto_frames else df.iloc[0:0].copy()

    pareto_path = data_path.with_suffix(".pareto.parquet")
    pareto_df.to_parquet(pareto_path, index=False)

    summary = {
        "n_zones": len(results),
        "n_pareto_total": len(pareto_df),
        "per_zone": [
            {
                "zone": r.zone,
                "n_total": r.n_total,
                "n_pareto": r.n_pareto,
                "pareto_fraction": r.n_pareto / r.n_total if r.n_total else 0.0,
            }
            for r in results
        ],
        "pareto_parquet": str(pareto_path),
    }
    out_dir = Path.cwd()
    (out_dir / "pareto_summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Pareto: %s zones, %s patches total", len(results), len(pareto_df))
    for r in results:
        logger.info("  %s: %d/%d (%.1f%%)", r.zone, r.n_pareto, r.n_total, 100 * r.n_pareto / r.n_total)


if __name__ == "__main__":
    main()
