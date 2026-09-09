"""Patch-grid generation over a city bounding box.

We produce a regular 1 km x 1 km grid in a locally appropriate equal-area projection
around the city centroid, then transform back to WGS84 for downstream use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .city import City

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


@dataclass(frozen=True)
class PatchGrid:
    """A collection of patch polygons tied to a city."""

    city_id: str
    patches: GeoDataFrame
    crs_local: str
    resolution_m: int


def build_patch_grid(
    city: City,
    resolution_m: int = 1000,
) -> PatchGrid:
    """Generate a square patch grid around a city within radius_km, in a local equal-area CRS.

    The local CRS is a custom azimuthal equal-area projection centred on the city.
    This keeps patch areas approximately 1 km² everywhere.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError as exc:
        raise ImportError("geopandas and shapely are required for patching") from exc

    # Local azimuthal equal-area projection centred at city
    crs_local = (
        f"+proj=laea +lat_0={city.lat} +lon_0={city.lon} +x_0=0 +y_0=0 "
        "+datum=WGS84 +units=m +no_defs"
    )
    r_m = int(city.radius_km * 1000)

    polygons = []
    ids = []
    idx = 0
    x = -r_m
    while x < r_m:
        y = -r_m
        while y < r_m:
            if math.hypot(x + resolution_m / 2, y + resolution_m / 2) <= r_m:
                polygons.append(box(x, y, x + resolution_m, y + resolution_m))
                ids.append(f"{city.id}_{idx:05d}")
                idx += 1
            y += resolution_m
        x += resolution_m

    gdf = gpd.GeoDataFrame(
        {"patch_id": ids, "city_id": city.id},
        geometry=polygons,
        crs=crs_local,
    )
    return PatchGrid(
        city_id=city.id,
        patches=gdf,
        crs_local=crs_local,
        resolution_m=resolution_m,
    )
