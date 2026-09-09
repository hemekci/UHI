# Beyond Local Climate Zones — Continuous Morphology Archetypes for Heat Mitigation

[![Article](https://img.shields.io/badge/Article-10.1016%2Fj.scs.2026.107814-B31B1B.svg)](https://doi.org/10.1016/j.scs.2026.107814)
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.19897974.svg)](https://doi.org/10.5281/zenodo.19897974)
[![License: MIT](https://img.shields.io/badge/Code-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Companion code and data for the published article:

> **Emekci, S., & Emekci, H. (2026).** Beyond local climate zones: Continuous morphology
> archetypes for heat mitigation across 66 cities. *Sustainable Cities and Society*, **150**, 107814.
> https://doi.org/10.1016/j.scs.2026.107814

**Article:** [doi.org/10.1016/j.scs.2026.107814](https://doi.org/10.1016/j.scs.2026.107814) ·
**Archived release:** [doi.org/10.5281/zenodo.19897974](https://doi.org/10.5281/zenodo.19897974) ·
**Repository:** [github.com/hemekci/UHI](https://github.com/hemekci/UHI)

---

## How to cite

Please cite **the article** for the findings, and **the archived release** if you use the code
or the harmonised data table.

### Article

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

### Software and data

```bibtex
@software{emekci2026uhi,
  author    = {Emekci, Seyda and Emekci, Hakan},
  title     = {{UHI}: Continuous Morphology Archetypes for Heat Mitigation Across Cities},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19897974},
  url       = {https://github.com/hemekci/UHI}
}
```

The Zenodo DOI above is a *concept* DOI: it always resolves to the most recent release.
To cite a specific version, use the version DOI shown on that release's Zenodo page.

GitHub's **Cite this repository** button (top right) produces the same entries from
[`CITATION.cff`](CITATION.cff). Machine-readable metadata is also provided in
[`codemeta.json`](codemeta.json) and [`.zenodo.json`](.zenodo.json), and a condensed
plain-text summary of the study, its numbers and its file layout is in
[`llms.txt`](llms.txt) for automated readers.

---

## What this study does

Urban block morphology is fixed at masterplan stage and locks in thermal performance for
50–100 years, yet design-stage guidance has been derived city by city. This study measures
morphology **continuously** — rather than through categorical Local Climate Zone classes —
across a harmonised multi-city sample, and attributes surface urban heat island (SUHI)
variation to it within climate zones.

| | |
|---|---|
| **Cities** | 66, across 15 Köppen climate zones |
| **Sample** | 33,715 harmonised 1 km² patches (33,498 after a physical-plausibility screen) |
| **Target** | 3-summer rolling-mean Landsat 8/9 surface-temperature anomaly, within-city centred |
| **Morphology** | 8 continuous features from open global building-footprint-with-height products |
| **Models** | Linear mixed effects with city random intercept; XGBoost + SHAP for corroboration |
| **Outputs** | Per-zone coefficients, empirical Pareto fronts, three designer-facing archetypes |

### Headline results

- **Tree-canopy fraction is the only universal cooling lever** — negative in all 14 modelled
  zones and significant in 13 (β = −0.64, *p* < 10⁻¹²²).
- Morphology explains a **limited but consistent** share of within-city SUHI variance
  (marginal *R*² = 0.25, conditional *R*² = 0.39); background climate dominates, as expected.
- Per-zone models lift the pooled marginal *R*² to **0.38**, reaching 0.55 in Dwa.
- **144 Pareto-efficient patches** cluster into **three climate-smart archetypes**.

### The three archetypes

Extracted from the 144 Pareto-efficient patches; medians per archetype.

| | *n* | UHI anomaly | Building density | Tree canopy | Built-up | Mean height |
|---|---:|---:|---:|---:|---:|---:|
| **A1** dense paved | 81 | −0.26 °C | 0.54 | 5 % | 90 % | 16 m |
| **A2** open vegetated | 52 | −2.95 °C | 0.26 | 24 % | 61 % | 11 m |
| **A3** tall sparse | 11 | −4.41 °C | 0.25 | 8 % | 62 % | 27 m |

Full profiles, including the Köppen zones each archetype appears in, are in
[`data/published/archetype_cards.json`](data/published/archetype_cards.json).

---

## Figures

Effect-size ranking of the morphology fixed effects:

![Fixed-effect coefficients](docs/figures/fig05_shap_bar.png)

Where the 66 cities sit across Köppen climate zones:

![City sample](docs/figures/fig02_city_map.png)

Which morphology features are significant in which zone:

![Per-zone significance](docs/figures/fig06_perzone_significance.png)

The three climate-smart morphology archetypes:

![Archetypes](docs/figures/fig09_archetype_radars.png)

All 15 figures are in [`docs/figures/`](docs/figures/).

---

## Repository layout

```
src/              analysis package
  ingest/         Earth Engine ingest, city rosters
  features/       patch features, quality screening
  morphology/     building-footprint morphology extraction
  model/          mixed-effects and XGBoost + SHAP
  pareto/         empirical Pareto extraction
  typology/       archetype clustering
  metrics/        R², ICC, hierarchical diagnostics
run/
  conf/           Hydra configuration (cities, features, models, Pareto, typology)
  pipeline/       stage entry points, incl. make_figures.py
data/published/   harmonised patch tables + model outputs (CC BY 4.0)
docs/figures/     all 15 published figures as PNG
tests/            unit and smoke tests
```

## Reproducing the results

```bash
git clone https://github.com/hemekci/UHI.git && cd UHI
uv venv && uv pip install -e .

# place the published table where the pipeline expects it
mkdir -p data/features
cp data/published/uhi_morphology_patches_v1.parquet data/features/full.parquet

# every modelling stage then runs from the repository alone
python run/pipeline/run_diagnostics.py          cities=full   # global mixed-effects model
python run/pipeline/run_perzone_diagnostics.py  cities=full   # per-zone models
python run/pipeline/run_pareto.py               cities=full   # Pareto fronts
python run/pipeline/run_typology.py             cities=full   # archetype clustering
python run/pipeline/make_figures.py                           # regenerate every figure
```

Re-running the *ingest* stage additionally requires a Google Earth Engine account,
since the Landsat, WorldCover and ERA5-Land layers are pulled from GEE. Everything
downstream of ingest runs from the published tables alone.

## Reusing the dataset

The harmonised table is the component most likely to be useful independently of the
article's own analysis. Each row is a 1 km² built-up patch with a summer surface-UHI
anomaly and eight continuous morphology descriptors, comparable across 66 cities and
15 Köppen zones — a sample design that is expensive to assemble and is released here so
that it need not be rebuilt.

It supports, among other uses:

- benchmarking alternative UHI models on a fixed, documented multi-city sample;
- testing whether a morphology–temperature relationship established in one city
  generalises across climate regimes;
- extending the analysis with additional predictors, cities or time periods;
- comparing continuous morphology descriptors against categorical Local Climate Zone
  classes on identical patches, since `lcz_class` is included;
- teaching multi-city urban-climate analysis with a ready, reproducible dataset.

### Scope and limitations

The dependent variable is *surface* UHI from Landsat land-surface temperature. It is not
a measure of canopy-layer air temperature or of human thermal comfort, and the article's
design implications are framed accordingly. Building-height coverage is uneven and absent
for about half the patches, sparsest in the Global South; the article reports a
height-exclusion sensitivity analysis showing the conclusions do not rest on those two
features. Coefficients are specific to the 1 km aggregation scale. The analysis is
observational, so the coefficients quantify association rather than identified causal
effect.

## Data sources

All inputs are public and open. The harmonised patch-level table is derived from
Landsat 8/9 Collection 2 Level 2 (USGS), Microsoft Global ML Building Footprints,
Google Open Buildings v3, the VIDA Google–Microsoft combined distribution,
GlobalBuildingAtlas, GLAMOUR, ESA WorldCover, ERA5-Land (ECMWF Copernicus),
WUDAPT LCZ and SRTM. Provenance, licences and access instructions for each layer are
in [`data/README.md`](data/README.md).

## Licence

Code is released under the [MIT licence](LICENSE). The harmonised data table and the
paper text are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

**Seyda Emekci** (corresponding author) — Department of Architecture,
Ankara Yıldırım Beyazıt University, Ankara, Türkiye — semekci@aybu.edu.tr

**Hakan Emekci** — Institute of Informatics, Hacettepe University, Ankara, Türkiye

---

**Keywords:** urban heat island · urban morphology · surface urban heat island · SUHI ·
land surface temperature · Landsat · building footprints · building height · Local Climate Zones ·
LCZ · Köppen climate zones · mixed-effects models · XGBoost · SHAP · explainable machine learning ·
Pareto frontier · urban climate · climate-smart design · heat mitigation · multi-city study
