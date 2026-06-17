# UHI — Continuous Morphology Archetypes for Heat Mitigation

[![CI](https://github.com/hemekci/UHI/actions/workflows/ci.yml/badge.svg)](https://github.com/hemekci/UHI/actions/workflows/ci.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19897974.svg)](https://doi.org/10.5281/zenodo.19897974)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Companion repository for the manuscript:

> **Beyond Local Climate Zones: Continuous Morphology Archetypes for Heat
> Mitigation Across 66 Cities**
> Seyda Emekci, Hakan Emekci.
> Submitted to *Sustainable Cities and Society* (Elsevier).

This repository will host **the complete code, data-acquisition recipes, and
reproducibility notebooks** behind every figure and table in the paper. It is
designed so that an independent reader can re-run the entire ingest →
features → models → archetypes pipeline on the public satellite and
building-footprint products, and recover the published results bit-for-bit
(modulo non-deterministic XGBoost threading).

---

## Status

The manuscript is **under major revision** at *Sustainable Cities and Society*.
The skeleton below is in place; the runnable pipeline, harmonised patch-level
table, and figure-generating notebooks will land here on or before the
published-paper milestone, under the open MIT licence.

Headline result: across 66 cities and 14 Köppen zones, tree-canopy fraction is
the only universal cooling lever (negative in all 14 zones, significant in 13);
the global within-city mixed-effects model reaches marginal R² = 0.25 and
conditional R² = 0.39, and three exploratory morphology archetypes summarise the
empirical (UHI, density) Pareto front.

If you are a reviewer or editor and need pre-publication access to the code
or data, please contact the corresponding author (see [Contact](#contact)).

---

## What this repository will contain

Everything needed to reproduce the paper end-to-end:

- **Pipeline code** (`pipeline/`) — modular Python package covering
  - `ingest/` — deterministic recipes pulling Landsat 8/9 LST, MODIS LST,
    Microsoft / Google / VIDA building footprints with heights, 3D-GloBFP,
    GlobalBuildingAtlas, GLAMOUR, ESA WorldCover, ERA5-Land, and Köppen
    classifications;
  - `features/` — per-1 km-patch morphology features. Eight are retained for
    inference (building density, mean height, height CV, tree-canopy, built-up,
    bare-soil and water fractions, and 80th-percentile NDVI); canyon aspect
    ratio and street-orientation entropy are computed but excluded for
    instability and uneven Global-South coverage. A physical-plausibility
    quality screen (`|UHI anomaly| <= 40 C`) removes residual cloud / fill
    artifacts before modelling;
  - `models/` — within-city-centred linear mixed-effects model with city
    random intercept, complementary XGBoost regressor with TreeExplainer SHAP,
    and per-Köppen-zone refits;
  - `archetypes/` — empirical Pareto extraction in (UHI anomaly, density)
    space and morphology archetype clustering with archetype-card export.
- **Hydra configs** (`configs/`) — every experiment in the paper expressed
  as a named, version-pinned config composition.
- **Notebooks** (`notebooks/`) — one notebook per figure/table, runnable
  from the harmonised intermediate artefacts.
- **Scripts** (`scripts/`) — thin CLI wrappers for end-to-end runs.
- **Tests** (`tests/`) — pytest suite for every module; CI gate.
- **Data manifest** (`data/README.md`) — exact source URLs, version tags,
  and on-disk layout. The harmonised patch-level table will be deposited on
  Zenodo with a DOI cited from the paper.

---

## Repository layout

```
UHI/
├── pipeline/           Python package (ingest → features → models → archetypes)
│   ├── ingest/
│   ├── features/
│   ├── models/
│   └── archetypes/
├── configs/            Hydra + OmegaConf configuration tree
├── notebooks/          one notebook per figure / table
├── scripts/            runnable entry points (run_ingest, run_features, ...)
├── tests/              pytest suite
├── data/               local-only; raw / interim / processed (gitignored)
├── .github/workflows/  CI pipeline (lint, type-check, test)
├── pyproject.toml      uv / pip-installable package definition
├── LICENSE             MIT
└── README.md
```

---

## Reproducibility quick-start

> The package is currently a skeleton; the commands below describe the
> intended end-to-end workflow once the pipeline lands.

```bash
# 1. Clone and enter
git clone https://github.com/hemekci/UHI.git
cd UHI

# 2. Create a reproducible environment with uv
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# 3. Pull the data manifest (URLs, hashes, version tags)
cat data/README.md

# 4. Run the full pipeline for one city to smoke-test the install
uv run python scripts/run_ingest.py     city=istanbul
uv run python scripts/run_features.py   city=istanbul
uv run python scripts/run_models.py     experiment=baseline
uv run python scripts/run_archetypes.py experiment=baseline

# 5. Reproduce all figures and tables
uv run jupyter nbconvert --execute --to notebook notebooks/*.ipynb
```

Reproducibility guarantees:

- Random seeds set globally (`set_seed(42)` in `pipeline.utils`);
- Every Hydra run snapshots its full resolved config and `pip freeze` next
  to its outputs;
- Dataset and feature-table SHA-256 hashes recorded in run logs;
- CI re-runs the test suite on every push to guard against regressions.

---

## Data

All inputs are public, with stable identifiers. See
[`data/README.md`](data/README.md) for the full list of products, version
tags, and access instructions. Highlights:

- **Land-surface temperature** — Landsat 8/9 Collection 2 Level-2 ST and
  MODIS MOD11A1 / MYD11A1.
- **Building footprints with heights** — Microsoft Global ML Building
  Footprints (Jan 2026 update), Google Open Buildings v3, the
  VIDA-distributed combined dataset, 3D-GloBFP (Che et al. 2024),
  GlobalBuildingAtlas (Zhu et al. 2025), GLAMOUR (Li et al. 2024).
- **Land cover & vegetation** — ESA WorldCover 10 m (2021), MODIS NDVI.
- **Reanalysis & climate** — ERA5-Land hourly, Köppen–Geiger 2007–2023
  classification.

The harmonised per-patch table for all 66 cities will be deposited on
**Zenodo** with a citable DOI; the DOI will be linked here on release.

---

## Citation

While the manuscript is under review, please cite the software via Zenodo:

> Emekci, S., & Emekci, H. (2026). *UHI: Continuous Morphology Archetypes
> for Heat Mitigation Across Cities* (Version v0.1.0) [Software]. Zenodo.
> https://doi.org/10.5281/zenodo.19897974

The DOI above is the **concept DOI** — it always resolves to the latest
release. The article DOI will be added here once the paper is accepted.

```bibtex
@software{emekci2026uhi,
  author    = {Emekci, Seyda and Emekci, Hakan},
  title     = {{UHI: Continuous Morphology Archetypes for Heat Mitigation Across Cities}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19897974},
  url       = {https://doi.org/10.5281/zenodo.19897974}
}
```

---

## License

[MIT](LICENSE) for code. The harmonised patch-level table on Zenodo will be
released under **CC-BY-4.0**. Upstream data products retain their own
licences — see [`data/README.md`](data/README.md).

---

## Contact

- **Seyda Emekci** — Department of Architecture, Ankara Yıldırım Beyazıt
  University · `semekci@aybu.edu.tr`
- **Hakan Emekci** — Graduate School of Sciences, TED University ·
  `hakan.emekci@tedu.edu.tr`

Issues and discussion: please use the
[issue tracker](https://github.com/hemekci/UHI/issues) once the pipeline is
released.
