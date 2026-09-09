"""Open-buildings client — queries Overture Maps buildings via DuckDB httpfs.

Overture Maps Foundation integrates Google Open Buildings, Microsoft Global ML
Footprints, and OpenStreetMap into a harmonized global dataset, released monthly
as GeoParquet on a public S3 bucket. We query it with DuckDB's httpfs + spatial
extensions, filter by city bbox, and return a GeoDataFrame.

Building-height coverage varies by region (best in EU/N.America/E.Asia; sparse
parts of sub-Saharan Africa). We surface a per-city height-confidence flag for
use by the G7/C7 uncertainty-aware analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .city import City

if TYPE_CHECKING:
    import geopandas as gpd


# Pinned Overture release. Update monthly as needed.
# Format: s3://overturemaps-us-west-2/release/YYYY-MM-DD.X/theme=buildings/type=building/
OVERTURE_RELEASE = "2026-04-15.0"
OVERTURE_BUCKET = "overturemaps-us-west-2"


@dataclass(frozen=True)
class BuildingsBundle:
    """Per-city cached building layer."""

    city_id: str
    parquet_path: Path
    n_buildings: int
    has_heights_fraction: float


class OpenBuildingsClient:
    """Queries Overture Maps buildings and caches per-city GeoParquet locally."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.cache_dir = Path(cfg.open_buildings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # DuckDB session
    # ------------------------------------------------------------------
    def _duckdb(self):
        import duckdb

        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")
        con.execute("INSTALL spatial; LOAD spatial;")
        con.execute("SET s3_region='us-west-2';")
        con.execute("SET s3_use_ssl=true;")
        # Public bucket, no credentials needed
        con.execute("SET s3_url_style='path';")
        return con

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------
    def fetch_city(self, city: City) -> BuildingsBundle:
        """Return a cached BuildingsBundle for the city; fetch from Overture if missing."""
        cache_path = self.cache_dir / f"{city.id}_buildings.parquet"
        if cache_path.exists():
            return self._summarize_cache(city, cache_path)

        # Local azimuthal equal-area CRS -> lat/lon bbox for the query.
        # Approximate: radius_km converted to degrees with a simple factor.
        bbox = self._latlon_bbox(city)
        xmin, ymin, xmax, ymax = bbox

        s3_glob = (
            f"s3://{OVERTURE_BUCKET}/release/{OVERTURE_RELEASE}"
            f"/theme=buildings/type=building/*"
        )
        query = f"""
        COPY (
          SELECT
            id,
            ST_AsWKB(geometry) AS geom_wkb,
            height,
            num_floors,
            class,
            sources
          FROM read_parquet('{s3_glob}', hive_partitioning=1)
          WHERE bbox.xmin >= {xmin}
            AND bbox.xmax <= {xmax}
            AND bbox.ymin >= {ymin}
            AND bbox.ymax <= {ymax}
        ) TO '{cache_path}' (FORMAT 'parquet', COMPRESSION 'snappy');
        """
        con = self._duckdb()
        con.execute(query)
        con.close()
        return self._summarize_cache(city, cache_path)

    def _latlon_bbox(self, city: City) -> tuple[float, float, float, float]:
        """Approximate lat/lon bbox around a city for the given radius in km."""
        import math

        lat_deg_per_km = 1.0 / 111.0
        lon_deg_per_km = 1.0 / (111.0 * max(0.1, math.cos(math.radians(city.lat))))
        dlat = city.radius_km * lat_deg_per_km
        dlon = city.radius_km * lon_deg_per_km
        return (
            city.lon - dlon,
            city.lat - dlat,
            city.lon + dlon,
            city.lat + dlat,
        )

    def _summarize_cache(self, city: City, cache_path: Path) -> BuildingsBundle:
        import pyarrow.parquet as pq

        tbl = pq.read_table(cache_path, columns=["height"])
        heights = tbl.column("height").to_pandas()
        n = len(heights)
        has_h = float(heights.notna().mean()) if n else 0.0
        return BuildingsBundle(
            city_id=city.id,
            parquet_path=cache_path,
            n_buildings=n,
            has_heights_fraction=has_h,
        )

    # ------------------------------------------------------------------
    # Morphology aggregation per patch
    # ------------------------------------------------------------------
    def buildings_to_geodataframe(self, bundle: BuildingsBundle) -> gpd.GeoDataFrame:
        """Load the cached per-city buildings parquet into a GeoDataFrame in EPSG:4326."""
        import geopandas as gpd
        import pyarrow.parquet as pq
        from shapely import wkb

        tbl = pq.read_table(bundle.parquet_path).to_pandas()
        tbl["geometry"] = tbl["geom_wkb"].map(wkb.loads)
        gdf = gpd.GeoDataFrame(tbl.drop(columns=["geom_wkb"]), geometry="geometry", crs="EPSG:4326")
        return gdf
