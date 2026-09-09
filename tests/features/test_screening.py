"""Tests for the physical-plausibility quality screen."""

import pandas as pd

from src.features import DEFAULT_UHI_ABS_MAX_C, quality_screen


def test_default_bound_is_40c():
    assert DEFAULT_UHI_ABS_MAX_C == 40.0


def test_drops_only_out_of_bound_patches():
    df = pd.DataFrame({"uhi_anomaly_c": [5.0, -95.0, 40.0, 41.0, -40.0, -41.0]})
    out = quality_screen(df, abs_max_c=40.0)
    # 5, 40, -40 are within |x| <= 40; -95, 41, -41 are dropped.
    assert sorted(out["uhi_anomaly_c"].tolist()) == [-40.0, 5.0, 40.0]


def test_custom_bound():
    df = pd.DataFrame({"uhi_anomaly_c": [10.0, 25.0, 35.0]})
    assert len(quality_screen(df, abs_max_c=30.0)) == 2


def test_missing_column_is_noop():
    df = pd.DataFrame({"other": [1, 2, 3]})
    assert len(quality_screen(df)) == 3


def test_returns_copy_not_view():
    df = pd.DataFrame({"uhi_anomaly_c": [1.0, 2.0]})
    out = quality_screen(df)
    out.loc[out.index[0], "uhi_anomaly_c"] = 999.0
    assert df["uhi_anomaly_c"].iloc[0] == 1.0
