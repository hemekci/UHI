"""Parquet-backed patch-table IO with DuckDB-friendly schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_patches(df: pd.DataFrame, path: Path) -> None:
    """Write a patch-level DataFrame to parquet. Ensures directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_patches(path: Path) -> pd.DataFrame:
    """Read a patch-level parquet back into a DataFrame."""
    return pd.read_parquet(path)
