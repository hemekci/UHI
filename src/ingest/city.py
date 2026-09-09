"""Immutable city record used by ingest + feature pipelines."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    """Minimal city-level metadata from the Hydra cities config."""

    id: str
    name: str
    country: str
    lat: float
    lon: float
    koppen: str
    radius_km: float

    @classmethod
    def from_config(cls, entry: dict) -> City:
        return cls(
            id=str(entry["id"]),
            name=str(entry["name"]),
            country=str(entry["country"]),
            lat=float(entry["lat"]),
            lon=float(entry["lon"]),
            koppen=str(entry["koppen"]),
            radius_km=float(entry["radius_km"]),
        )
