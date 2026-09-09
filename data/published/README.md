# Published data tables

Released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) alongside
[Emekci & Emekci (2026)](https://doi.org/10.1016/j.scs.2026.107814).

| File | Rows | Description |
|---|---:|---|
| `uhi_morphology_patches_v1.parquet` | 33,715 | Harmonised patch-level table, full sample |
| `uhi_morphology_patches_screened_v1.parquet` | 33,498 | After the physical-plausibility screen (`\|UHI anomaly\| ≤ 40 °C`) used for all modelling |
| `diagnostics.json` | — | Global mixed-effects fixed effects, CIs, R², ICC |
| `diagnostics_perzone.json` | — | Per-zone mixed-effects coefficients and fit statistics |
| `pareto_summary.json` | — | Per-zone empirical Pareto counts and fractions |
| `archetype_cards.json` | — | The three climate-smart morphology archetypes |

The screened table is the one behind every figure and table in the article. The 217
removed patches are residual-cloud artefacts, all in the tropical-rainforest zone Af.

## Columns

One row per 1 km² built-up patch.

**Identifiers** — `patch_id`, `city_id`, `country`, `koppen_zone`

**Target** — `uhi_anomaly_c` (summer surface-UHI anomaly, °C, within-city centred),
`lst_summer_mean_c` (3-summer rolling-mean Landsat 8/9 LST),
`rural_reference_lst_c` (median LST of the 10–20 km rural annulus)

**Morphology (the 8 modelled features)** — `building_density`, `height_mean`, `height_cv`,
`ndvi_p80`, `wc_tree_frac`, `wc_built_frac`, `wc_bare_frac`, `wc_water_frac`

**Additional morphology** — `total_footprint_m2`, `n_buildings_patch`, `height_std`,
`height_coverage_patch`, `built_up_fraction`, `impervious_frac`, `vegetation_frac`

**Land cover (ESA WorldCover fractions)** — `wc_shrub_frac`, `wc_grass_frac`, `wc_cropland_frac`

**Context** — `era5_summer_t2m_c` (ERA5-Land summer air temperature), `elevation_m` (SRTM),
`lcz_class` (WUDAPT LCZ)

## Known missing values

`height_mean` and `height_std` are null for 16,931 patches (50.2%) where the open
building-height products have no coverage — sparsest in the Global South. The article
reports a height-exclusion sensitivity analysis showing the conclusions do not depend on
these two features (marginal R² 0.252 → 0.244; every lever keeps its sign, significance and
rank). `era5_summer_t2m_c` is null for 1,077 patches outside the ERA5-Land land mask.

## Reading the tables

```python
import pandas as pd
df = pd.read_parquet("data/published/uhi_morphology_patches_screened_v1.parquet")
print(df.shape)          # (33498, 28)
```
