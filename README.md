<h1 align="center">Beyond Local Climate Zones</h1>
<p align="center"><b>Continuous Morphology Archetypes for Heat Mitigation Across 66 Cities</b></p>

<p align="center">
  <a href="https://doi.org/10.1016/j.scs.2026.107814"><img alt="Article" src="https://img.shields.io/badge/Article-10.1016%2Fj.scs.2026.107814-B31B1B.svg"></a>
  <a href="https://doi.org/10.5281/zenodo.19897974"><img alt="Zenodo" src="https://zenodo.org/badge/DOI/10.5281/zenodo.19897974.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="Code: MIT" src="https://img.shields.io/badge/Code-MIT-yellow.svg"></a>
  <a href="https://creativecommons.org/licenses/by/4.0/"><img alt="Data: CC BY 4.0" src="https://img.shields.io/badge/Data-CC%20BY%204.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
</p>

How does the shape of a city block relate to how hot it gets? This repository holds the
code and the open dataset behind a study that measures urban morphology **continuously** —
rather than through categorical Local Climate Zone classes — and relates it to summer
surface urban heat island (SUHI) intensity across **66 cities in 15 Köppen climate zones**.

> **Emekci, S., & Emekci, H. (2026).** Beyond local climate zones: Continuous morphology
> archetypes for heat mitigation across 66 cities. *Sustainable Cities and Society*, **150**, 107814.
> [doi.org/10.1016/j.scs.2026.107814](https://doi.org/10.1016/j.scs.2026.107814)

![The 66-city sample across Köppen climate zones](docs/figures/fig02_city_map.png)

## Key findings

**Tree-canopy fraction is the only universal cooling lever.** Its coefficient is negative in
all 14 modelled climate zones and significant in 13 (standardised β = −0.64, *p* < 10⁻¹²²).
Every other lever changes rank, or even sign, between climate regimes.

| Lever | Standardised β | Direction |
|---|---:|---|
| Tree-canopy fraction | −0.64 | cooling |
| Water fraction | −0.58 | cooling |
| 80th-percentile NDVI | −0.31 | cooling |
| Building-height CV | −0.18 | cooling |
| Mean building height | −0.09 | cooling |
| Building density | +0.06 | warming |
| Built-up fraction | +0.31 | warming |
| Bare-soil fraction | +0.42 | warming |

Morphology explains a limited but consistent share of within-city SUHI variance
(marginal *R*² = 0.25, conditional *R*² = 0.39); background climate dominates, as the
literature would predict. Per-zone models raise the pooled marginal *R*² to 0.38, reaching
0.55 in Dwa. From the 144 Pareto-efficient patches, three designer-facing archetypes emerge:

| | *n* | UHI anomaly | Density | Tree canopy | Built-up | Mean height |
|---|---:|---:|---:|---:|---:|---:|
| **A1** dense paved | 81 | −0.26 °C | 0.54 | 5 % | 90 % | 16 m |
| **A2** open vegetated | 52 | −2.95 °C | 0.26 | 24 % | 61 % | 11 m |
| **A3** tall sparse | 11 | −4.41 °C | 0.25 | 8 % | 62 % | 27 m |

## The dataset

`data/published/` holds the harmonised patch-level table — the component most likely to be
useful independently of this study's own analysis. Each row is a **1 km² built-up patch**
carrying a summer surface-UHI anomaly and eight continuous morphology descriptors,
comparable across all 66 cities.

| File | Rows | |
|---|---:|---|
| `uhi_morphology_patches_v1.parquet` | 33,715 | full harmonised sample |
| `uhi_morphology_patches_screened_v1.parquet` | 33,498 | modelling sample, after the quality screen |
| `diagnostics.json` · `diagnostics_perzone.json` | — | model coefficients and fit statistics |
| `pareto_summary.json` · `archetype_cards.json` | — | Pareto fronts and the three archetypes |

Every column is documented in [`data/published/README.md`](data/published/README.md),
including known missing values. Released under CC BY 4.0.

```python
import pandas as pd
df = pd.read_parquet("data/published/uhi_morphology_patches_screened_v1.parquet")
df.shape   # (33498, 28)
```

## Reproducing the results

```bash
git clone https://github.com/hemekci/UHI.git && cd UHI
uv venv && uv pip install -e .

mkdir -p data/features
cp data/published/uhi_morphology_patches_v1.parquet data/features/full.parquet

python run/pipeline/run_diagnostics.py          cities=full   # global mixed-effects model
python run/pipeline/run_perzone_diagnostics.py  cities=full   # per-zone models
python run/pipeline/run_pareto.py               cities=full   # Pareto fronts
python run/pipeline/run_typology.py             cities=full   # archetype clustering
python run/pipeline/make_figures.py                           # regenerate every figure
```

This reproduces the published coefficients exactly. Re-running the *ingest* stage
additionally needs a Google Earth Engine account, since the Landsat, WorldCover and
ERA5-Land layers are pulled from GEE; everything downstream runs from the published tables.

## Repository layout

```
src/              analysis package — ingest, features, morphology, model,
                  pareto, typology, metrics
run/conf/         Hydra configuration
run/pipeline/     stage entry points, incl. make_figures.py
data/published/   harmonised tables and model outputs (CC BY 4.0)
docs/figures/     all 15 published figures as PNG
tests/            18 unit and smoke tests
```

## How to cite

Please cite the article. It is the citable record for this work — the code and data are
released as its supplement rather than as separately citable outputs.

```bibtex
@article{emekci2026morphology,
  author  = {Emekci, Seyda and Emekci, Hakan},
  title   = {Beyond local climate zones: Continuous morphology archetypes for heat mitigation across 66 cities},
  journal = {Sustainable Cities and Society},
  year    = {2026},
  volume  = {150},
  pages   = {107814},
  doi     = {10.1016/j.scs.2026.107814}
}
```

GitHub's *Cite this repository* button returns the same reference from
[`CITATION.cff`](CITATION.cff). Machine-readable metadata is in
[`codemeta.json`](codemeta.json) and [`.zenodo.json`](.zenodo.json), and a condensed
summary for automated readers is in [`llms.txt`](llms.txt).

This release is archived at Zenodo under
[10.5281/zenodo.19897974](https://doi.org/10.5281/zenodo.19897974) for permanence and
version pinning; quote that DOI if you need to identify the exact version you ran.

## Scope and limitations

The dependent variable is *surface* UHI from Landsat land-surface temperature. It is not a
measure of canopy-layer air temperature or of human thermal comfort, and the design
implications are framed accordingly. Building-height coverage is uneven and absent for
about half the patches, sparsest in the Global South; a height-exclusion sensitivity
analysis in the article shows the conclusions do not rest on those two features.
Coefficients are specific to the 1 km aggregation scale. The analysis is observational, so
coefficients quantify association rather than identified causal effects.

## Data sources

All inputs are public and open: Landsat 8/9 Collection 2 Level 2 (USGS), Microsoft Global
ML Building Footprints, Google Open Buildings v3, the VIDA Google–Microsoft combined
distribution, GlobalBuildingAtlas, GLAMOUR, ESA WorldCover, ERA5-Land (ECMWF Copernicus),
WUDAPT LCZ and SRTM. Provenance and licences for each layer are in
[`data/README.md`](data/README.md).

## Licence and contact

Code under the [MIT licence](LICENSE); data and paper text under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**Seyda Emekci** (corresponding) — Department of Architecture, Ankara Yıldırım Beyazıt
University, Ankara, Türkiye — semekci@aybu.edu.tr
**Hakan Emekci** — Institute of Informatics, Hacettepe University, Ankara, Türkiye
