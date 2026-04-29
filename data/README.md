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

Suggested local layout (not enforced):

```
data/
├── raw/          downloaded layers, untouched
├── interim/      per-city tiles, masked, reprojected
└── processed/    final per-patch morphology + LST table
```
