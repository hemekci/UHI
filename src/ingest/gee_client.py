"""Google Earth Engine client — fetches LST, NDVI, WorldCover, LCZ, ERA5-Land, SRTM.

All fetchers return ``ee.Image`` or ``ee.Number`` objects. Heavy reduceRegions
batching over the patch grid happens in `run/pipeline/run_ingest.py` so that
network calls are amortized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .city import City


@dataclass(frozen=True)
class PatchFeatures:
    """Raw GEE outputs for one patch, pre-feature-engineering."""

    patch_id: str
    city_id: str
    lst_summer_mean_c: float | None
    ndvi_p80: float | None
    worldcover_shares: dict[str, float]
    lcz_class: int | None
    era5_summer_t2m_c: float | None
    elevation_m: float | None


# Landsat 8/9 Collection 2 Level 2 ST_B10 scale/offset (Celsius).
# LST_C = DN * 0.00341802 + 149.0 - 273.15.
_LANDSAT_ST_SCALE = 0.00341802
_LANDSAT_ST_OFFSET = 149.0
_KELVIN_ZERO = 273.15


class GEEClient:
    """Thin wrapper that initializes Earth Engine and exposes image fetchers.

    ``cfg`` is a Hydra config node matching run/conf/ingest/default.yaml.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        """Initialize earthengine-api with the configured Cloud project."""
        import ee  # type: ignore

        project_id = self.cfg.gee.project
        if not project_id:
            raise RuntimeError(
                "ingest.gee.project is required. Set it in run/conf/ingest/default.yaml."
            )
        ee.Initialize(project=project_id)
        self._initialized = True

    # ------------------------------------------------------------------
    # City geometry helpers
    # ------------------------------------------------------------------
    # Extra buffer beyond city radius so the rural reference ring (10-20 km)
    # falls inside the fetched image extent even for small city radii.
    _EXTRA_RADIUS_KM_FOR_RURAL = 25.0

    @staticmethod
    def _city_bbox(city: City, extra_km: float | None = None) -> Any:
        import ee  # type: ignore

        extra = GEEClient._EXTRA_RADIUS_KM_FOR_RURAL if extra_km is None else extra_km
        return (
            ee.Geometry.Point([city.lon, city.lat])
            .buffer(int((city.radius_km + extra) * 1000))
            .bounds()
        )

    @staticmethod
    def _summer_filter(city: City) -> Any:
        """Return an ee.Filter that keeps summer months per hemisphere."""
        import ee  # type: ignore

        if city.lat >= 0:
            return ee.Filter.calendarRange(6, 8, "month")  # JJA
        return ee.Filter.calendarRange(12, 2, "month")     # DJF

    # ------------------------------------------------------------------
    # Landsat 8/9 LST
    # ------------------------------------------------------------------
    def fetch_landsat_lst(self, city: City) -> Any:
        """Summer-mean Landsat 8+9 LST in Celsius across the configured year range.

        Applies CFMask via QA_PIXEL and per-hemisphere summer filtering.
        Returns an ``ee.Image`` with single band ``lst_c``.
        """
        import ee  # type: ignore

        region = self._city_bbox(city)
        y_first = int(self.cfg.landsat.years[0])
        y_last = int(self.cfg.landsat.years[-1])
        start = f"{y_first}-01-01"
        end = f"{y_last}-12-31"

        def _prep(img):
            # Cloud + shadow mask from QA_PIXEL (bits 3=cloud, 4=cloud shadow).
            qa = img.select("QA_PIXEL")
            cloud = qa.bitwiseAnd(1 << 3).neq(0)
            shadow = qa.bitwiseAnd(1 << 4).neq(0)
            mask = cloud.Or(shadow).Not()
            lst = (
                img.select("ST_B10")
                .multiply(_LANDSAT_ST_SCALE)
                .add(_LANDSAT_ST_OFFSET)
                .subtract(_KELVIN_ZERO)
                .rename("lst_c")
            )
            return lst.updateMask(mask)

        def _collection(col_id: str):
            return (
                ee.ImageCollection(col_id)
                .filterBounds(region)
                .filterDate(start, end)
                .filter(self._summer_filter(city))
                .filter(ee.Filter.lt("CLOUD_COVER", self.cfg.landsat.max_cloud_cover_pct))
                .map(_prep)
            )

        l8 = _collection("LANDSAT/LC08/C02/T1_L2")
        l9 = _collection("LANDSAT/LC09/C02/T1_L2")
        merged = l8.merge(l9)
        return merged.mean().clip(region)

    # ------------------------------------------------------------------
    # NDVI from same Landsat scenes
    # ------------------------------------------------------------------
    def fetch_ndvi(self, city: City) -> Any:
        """Summer-median NDVI from Landsat 8+9 SR bands over the configured window."""
        import ee  # type: ignore

        region = self._city_bbox(city)
        y_first = int(self.cfg.landsat.years[0])
        y_last = int(self.cfg.landsat.years[-1])
        start = f"{y_first}-01-01"
        end = f"{y_last}-12-31"

        def _ndvi(img):
            qa = img.select("QA_PIXEL")
            mask = qa.bitwiseAnd(1 << 3).neq(0).Or(qa.bitwiseAnd(1 << 4).neq(0)).Not()
            nir = img.select("SR_B5").multiply(0.0000275).add(-0.2)
            red = img.select("SR_B4").multiply(0.0000275).add(-0.2)
            ndvi = nir.subtract(red).divide(nir.add(red)).rename("ndvi")
            return ndvi.updateMask(mask)

        def _collection(col_id: str):
            return (
                ee.ImageCollection(col_id)
                .filterBounds(region)
                .filterDate(start, end)
                .filter(self._summer_filter(city))
                .filter(ee.Filter.lt("CLOUD_COVER", self.cfg.landsat.max_cloud_cover_pct))
                .map(_ndvi)
            )

        l8 = _collection("LANDSAT/LC08/C02/T1_L2")
        l9 = _collection("LANDSAT/LC09/C02/T1_L2")
        return l8.merge(l9).median().clip(region)

    # ------------------------------------------------------------------
    # ESA WorldCover land-cover
    # ------------------------------------------------------------------
    def fetch_worldcover(self, city: City) -> Any:
        """ESA WorldCover v200 class mosaic clipped to the city bbox.

        WorldCover is an ImageCollection (one image per version) — we mosaic to a single Image.
        """
        import ee  # type: ignore

        region = self._city_bbox(city)
        return ee.ImageCollection(self.cfg.worldcover.image).mosaic().clip(region)

    # ------------------------------------------------------------------
    # LCZ
    # ------------------------------------------------------------------
    def fetch_lcz(self, city: City) -> Any:
        """Demuzere 2022 global LCZ map (mosaic) clipped to city bbox.

        Categorical 1..17 in the ``LCZ_Filter`` band.
        """
        import ee  # type: ignore

        region = self._city_bbox(city)
        col = ee.ImageCollection("RUB/RUBCLIM/LCZ/global_lcz_map/latest")
        img = col.mosaic().select("LCZ_Filter")
        return img.clip(region)

    # ------------------------------------------------------------------
    # ERA5-Land summer air T
    # ------------------------------------------------------------------
    def fetch_era5_summer_t2m(self, city: City) -> Any:
        """Summer-mean ERA5-Land 2 m air temperature in Celsius."""
        import ee  # type: ignore

        region = self._city_bbox(city)
        y_first = int(self.cfg.landsat.years[0])
        y_last = int(self.cfg.landsat.years[-1])
        start = f"{y_first}-01-01"
        end = f"{y_last}-12-31"

        col = (
            ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
            .filterDate(start, end)
            .filter(self._summer_filter(city))
            .select("temperature_2m")
        )
        return col.mean().subtract(_KELVIN_ZERO).rename("era5_summer_t2m_c").clip(region)

    # ------------------------------------------------------------------
    # SRTM elevation
    # ------------------------------------------------------------------
    def fetch_srtm(self, city: City) -> Any:
        """SRTM 30 m elevation clipped to city bbox."""
        import ee  # type: ignore

        region = self._city_bbox(city)
        return ee.Image("USGS/SRTMGL1_003").rename("elevation_m").clip(region)
