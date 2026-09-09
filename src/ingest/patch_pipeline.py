"""End-to-end per-city patch feature extraction.

For each patch we compute:
- LST (summer-mean), NDVI P80, WorldCover class fractions, LCZ mode, ERA5 T2m, elevation (GEE)
- building_density, height_mean, height_cv, canyon_aspect_proxy, orientation_entropy (local)
- UHI anomaly = patch LST - rural reference P50 (per city)

Results are written as one parquet per city into ``data/patches/<city_id>.parquet``.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .city import City
from .gee_client import GEEClient
from .open_buildings import OpenBuildingsClient
from .patching import build_patch_grid

logger = logging.getLogger(__name__)

# WorldCover v200 class codes we care about
_WC_CLASSES = {
    "tree": 10,
    "shrub": 20,
    "grass": 30,
    "cropland": 40,
    "built": 50,
    "bare": 60,
    "water": 80,
}

_PATCH_CHUNK = 400  # reduceRegions batch size


def run_city(
    city: City,
    gee: GEEClient,
    ob: OpenBuildingsClient,
    cfg_ingest: Any,
    out_dir: Path,
) -> Path:
    """Ingest one city end-to-end; return the path to the written parquet."""
    import ee

    t_start = time.time()
    logger.info("City %s (%s, %s): starting pipeline", city.id, city.name, city.koppen)

    # 1) Patch grid in local equal-area.
    grid = build_patch_grid(city, resolution_m=cfg_ingest.patch.resolution_m)
    logger.info("  patches=%d radius_km=%s", len(grid.patches), city.radius_km)

    # Project patches to EPSG:4326 for GEE + geopandas ops.
    patches_wgs = grid.patches.to_crs("EPSG:4326")

    # 2) GEE feature images.
    lst_img = gee.fetch_landsat_lst(city)
    ndvi_img = gee.fetch_ndvi(city)
    wc_img = gee.fetch_worldcover(city)
    lcz_img = gee.fetch_lcz(city)
    era5_img = gee.fetch_era5_summer_t2m(city)
    srtm_img = gee.fetch_srtm(city)

    # 3) Build ee.FeatureCollection of patches.
    def _patch_features_for_gee():
        feats = []
        for _, row in patches_wgs.iterrows():
            poly = row.geometry
            coords = [list(poly.exterior.coords)]
            feats.append(ee.Feature(ee.Geometry.Polygon(coords), {"patch_id": row.patch_id}))
        return feats

    gee_feats = _patch_features_for_gee()
    logger.info("  running GEE reduceRegions in chunks of %d", _PATCH_CHUNK)

    gee_rows = []
    for i in range(0, len(gee_feats), _PATCH_CHUNK):
        chunk = ee.FeatureCollection(gee_feats[i : i + _PATCH_CHUNK])
        t0 = time.time()

        # LST mean
        lst_fc = lst_img.reduceRegions(chunk, ee.Reducer.mean(), 30).getInfo()
        lst_map = {f["properties"]["patch_id"]: f["properties"].get("mean") for f in lst_fc["features"]}

        # NDVI P80
        ndvi_fc = ndvi_img.reduceRegions(chunk, ee.Reducer.percentile([80]), 30).getInfo()
        ndvi_map = {f["properties"]["patch_id"]: f["properties"].get("p80") for f in ndvi_fc["features"]}

        # WorldCover class fractions (reducer = frequencyHistogram at 10 m)
        wc_fc = wc_img.reduceRegions(chunk, ee.Reducer.frequencyHistogram(), 10).getInfo()
        wc_map = {f["properties"]["patch_id"]: f["properties"].get("histogram") or {} for f in wc_fc["features"]}

        # LCZ mode at 100 m
        lcz_fc = lcz_img.reduceRegions(chunk, ee.Reducer.mode(), 100).getInfo()
        lcz_map = {f["properties"]["patch_id"]: f["properties"].get("mode") for f in lcz_fc["features"]}

        # ERA5 T2m (summer mean, single scalar per patch)
        era5_fc = era5_img.reduceRegions(chunk, ee.Reducer.mean(), 1000).getInfo()
        era5_map = {f["properties"]["patch_id"]: f["properties"].get("mean") for f in era5_fc["features"]}

        # SRTM elevation
        srtm_fc = srtm_img.reduceRegions(chunk, ee.Reducer.mean(), 30).getInfo()
        srtm_map = {f["properties"]["patch_id"]: f["properties"].get("mean") for f in srtm_fc["features"]}

        for f in gee_feats[i : i + _PATCH_CHUNK]:
            pid = f.getInfo()["properties"]["patch_id"]  # lightweight
            wc_hist = wc_map.get(pid, {}) or {}
            wc_total = sum(wc_hist.values()) or 1.0
            row = {
                "patch_id": pid,
                "city_id": city.id,
                "koppen_zone": city.koppen,
                "country": city.country,
                "lst_summer_mean_c": lst_map.get(pid),
                "ndvi_p80": ndvi_map.get(pid),
                "lcz_class": lcz_map.get(pid),
                "era5_summer_t2m_c": era5_map.get(pid),
                "elevation_m": srtm_map.get(pid),
                **{
                    f"wc_{cname}_frac": (wc_hist.get(str(code), 0) or 0) / wc_total
                    for cname, code in _WC_CLASSES.items()
                },
            }
            gee_rows.append(row)
        logger.info("  gee chunk %d/%d done (%.1fs)", i // _PATCH_CHUNK + 1, math.ceil(len(gee_feats) / _PATCH_CHUNK), time.time() - t0)

    gee_df = pd.DataFrame(gee_rows)

    # 4) Open Buildings fetch + per-patch spatial join.
    t0 = time.time()
    bundle = ob.fetch_city(city)
    logger.info("  buildings: n=%d, height_coverage=%.1f%% (%.1fs)", bundle.n_buildings, bundle.has_heights_fraction * 100, time.time() - t0)

    gdf_b = ob.buildings_to_geodataframe(bundle)
    if len(gdf_b):
        # Project buildings to local LAEA for area math.
        gdf_b_local = gdf_b.to_crs(grid.crs_local)
        gdf_b_local["footprint_m2"] = gdf_b_local.geometry.area
        # Join patches (local) <- buildings (local) via intersects.
        import geopandas as gpd

        joined = gpd.sjoin(gdf_b_local, grid.patches[["patch_id", "geometry"]], predicate="intersects", how="inner")
        # Aggregate per patch.
        agg = (
            joined.groupby("patch_id")
            .apply(
                lambda g: pd.Series(
                    {
                        "total_footprint_m2": float(g["footprint_m2"].sum()),
                        "height_mean": float(np.nanmean(g["height"])) if g["height"].notna().any() else np.nan,
                        "height_std": float(np.nanstd(g["height"])) if g["height"].notna().any() else np.nan,
                        "n_buildings_patch": len(g),
                        "height_coverage_patch": float(g["height"].notna().mean()),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
    else:
        agg = pd.DataFrame(columns=["patch_id", "total_footprint_m2", "height_mean", "height_std", "n_buildings_patch", "height_coverage_patch"])

    patch_area_m2 = cfg_ingest.patch.resolution_m ** 2
    agg["building_density"] = agg["total_footprint_m2"].fillna(0) / patch_area_m2
    agg["height_cv"] = agg["height_std"] / agg["height_mean"].replace({0: np.nan})
    agg["height_cv"] = agg["height_cv"].fillna(0.0)

    # 5) Merge GEE + buildings.
    df = gee_df.merge(agg, on="patch_id", how="left")
    # Fill buildings NaNs with zeros for density etc.
    for col in ["total_footprint_m2", "building_density", "n_buildings_patch"]:
        df[col] = df[col].fillna(0.0)

    # 6) Rural reference LST — mean over rural ring.
    rural_start_km, rural_end_km = cfg_ingest.patch.rural_reference_ring_km
    rural_ring = ee.Geometry.Point([city.lon, city.lat]).buffer(rural_end_km * 1000).difference(
        ee.Geometry.Point([city.lon, city.lat]).buffer(rural_start_km * 1000)
    )
    # Mask to non-built WorldCover (tree, shrub, grass, cropland, bare).
    non_built_mask = wc_img.neq(_WC_CLASSES["built"])
    rural_lst = lst_img.updateMask(non_built_mask).reduceRegion(
        reducer=ee.Reducer.percentile([50]),
        geometry=rural_ring,
        scale=30,
        maxPixels=1e10,
    ).get("lst_c").getInfo()
    logger.info("  rural reference P50 LST (%d-%d km ring, non-built): %s", rural_start_km, rural_end_km, rural_lst)

    if rural_lst is not None:
        df["uhi_anomaly_c"] = df["lst_summer_mean_c"] - float(rural_lst)
    else:
        df["uhi_anomaly_c"] = np.nan
    df["rural_reference_lst_c"] = rural_lst

    # 7) Built-up filter.
    df["built_up_fraction"] = df["wc_built_frac"]
    df = df[df["built_up_fraction"] >= cfg_ingest.patch.min_built_up_fraction].copy()

    # 8) Write.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city.id}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("  wrote %d built-up patches -> %s (total %.1fs)", len(df), out_path, time.time() - t_start)
    return out_path
