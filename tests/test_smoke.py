"""Smoke tests verifying the package skeleton imports cleanly.

These are placeholders that will be replaced by real tests once the
pipeline modules are implemented.
"""

import importlib


def test_package_imports() -> None:
    pkg = importlib.import_module("pipeline")
    assert hasattr(pkg, "__version__")


def test_subpackages_import() -> None:
    for name in ("ingest", "features", "models", "archetypes"):
        importlib.import_module(f"pipeline.{name}")
