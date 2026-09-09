"""Stage-3 step 5: cluster Pareto-efficient patches into climate-smart archetypes.

Reads:  data/features/<cities_name>.pareto.parquet
Writes: <hydra_run_dir>/typology.json, typology_labels.parquet, archetype_cards.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.typology import cluster_pareto_front  # noqa: E402
from src.utils import get_logger, read_patches, set_seed  # noqa: E402

logger = get_logger(__name__, level=logging.INFO)


_ARCHETYPE_FEATURES = [
    "building_density",
    "height_mean",
    "height_cv",
    "ndvi_p80",
    "wc_tree_frac",
    "wc_built_frac",
    "wc_bare_frac",
    "wc_water_frac",
    "impervious_frac",
    "vegetation_frac",
]

_OBJECTIVE_COLS = ["uhi_anomaly_c", "building_density"]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    pareto_path = Path(cfg.data_root) / "features" / f"{cfg.cities.name}.pareto.parquet"
    if not pareto_path.exists():
        logger.error("Pareto parquet not found at %s. Run run_pareto.py first.", pareto_path)
        return
    df = read_patches(pareto_path)
    logger.info("Clustering %d Pareto patches over zones: %s",
                len(df), sorted(df._pareto_zone.unique().tolist()))

    feats = [c for c in _ARCHETYPE_FEATURES if c in df.columns]
    X = df[feats].fillna(0.0).to_numpy(dtype=float)
    F = df[_OBJECTIVE_COLS].fillna(0.0).to_numpy(dtype=float)

    k_candidates = tuple(cfg.typology.k_candidates)
    if len(X) < max(k_candidates) + 1:
        logger.error("Only %d Pareto points — cannot cluster into up to k=%d. Aborting.",
                     len(X), max(k_candidates))
        return

    tc = cluster_pareto_front(
        X_pareto=X,
        F_pareto=F,
        k_candidates=k_candidates,
        random_seed=cfg.typology.random_seed,
    )
    logger.info("Selected k=%d (silhouette=%.3f)", tc.k, tc.silhouette)

    out_dir = Path.cwd()
    df_out = df.copy()
    df_out["archetype"] = tc.labels.astype(int)
    df_out.to_parquet(out_dir / "typology_labels.parquet", index=False)

    # Build archetype cards: per-cluster medians + IQRs + top contributing cities.
    cards = []
    for k in range(tc.k):
        idx = tc.labels == k
        if not idx.any():
            continue
        sub = df.loc[idx]
        card = {
            "archetype": k,
            "n_patches": int(idx.sum()),
            "median_features": {c: float(np.nanmedian(sub[c])) for c in feats},
            "iqr_features": {
                c: [float(np.nanpercentile(sub[c], 25)), float(np.nanpercentile(sub[c], 75))]
                for c in feats
            },
            "uhi_anomaly_c": {
                "median": float(np.nanmedian(sub["uhi_anomaly_c"])),
                "iqr": [
                    float(np.nanpercentile(sub["uhi_anomaly_c"], 25)),
                    float(np.nanpercentile(sub["uhi_anomaly_c"], 75)),
                ],
            },
            "building_density": {
                "median": float(np.nanmedian(sub["building_density"])),
                "iqr": [
                    float(np.nanpercentile(sub["building_density"], 25)),
                    float(np.nanpercentile(sub["building_density"], 75)),
                ],
            },
            "top_contributing_cities": sub["city_id"].value_counts().head(3).to_dict(),
            "koppen_zones_represented": sorted(sub._pareto_zone.unique().tolist()),
        }
        cards.append(card)

    (out_dir / "archetype_cards.json").write_text(json.dumps(cards, indent=2))
    summary = {
        "k": int(tc.k),
        "silhouette": float(tc.silhouette),
        "n_pareto_patches": len(X),
        "features_used": feats,
    }
    (out_dir / "typology.json").write_text(json.dumps(summary, indent=2))
    logger.info("Wrote typology_labels.parquet, archetype_cards.json, typology.json")


if __name__ == "__main__":
    main()
