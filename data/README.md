# data/

Local-only data directory. Contents under `data/raw/`, `data/interim/`, and
`data/processed/` are gitignored — never commit raw or processed data here.

External data sources used by the pipeline (links to be finalised on release):

- **Landsat 8/9 LST** — USGS Earth Explorer / Google Earth Engine
- **MODIS LST (MOD11A1 / MYD11A1)** — NASA LP DAAC
- **Microsoft Global ML Building Footprints** — github.com/microsoft/GlobalMLBuildingFootprints
- **Google Open Buildings v3** — sites.research.google/open-buildings
- **VIDA Google–Microsoft combined** — source.coop/vida/google-microsoft-open-buildings
- **3D-GloBFP** — Earth System Science Data 16 (Che et al. 2024)
- **GlobalBuildingAtlas** — Earth System Science Data 17 (Zhu et al. 2025)
- **GLAMOUR** — Scientific Data 11 (Li et al. 2024)
- **ESA WorldCover** — esa-worldcover.org
- **ERA5-Land** — Copernicus Climate Data Store
- **Köppen–Geiger 2007–2023** — Beck et al. (2023)

The harmonised per-patch table covers **66 cities / 14 Köppen zones / 33,715
one-km² built-up patches**. A physical-plausibility quality screen
(`|UHI anomaly| <= 40 °C`) is applied at analysis time, leaving **33,498**
patches for modelling (the 217 removed patches are residual-cloud artifacts,
all in the tropical-rainforest zone Af). The raw table preserves the full
sample; screening happens in-pipeline.

Suggested local layout (not enforced):

```
data/
├── raw/          downloaded layers, untouched
├── interim/      per-city tiles, masked, reprojected
└── processed/    final per-patch morphology + LST table
```
