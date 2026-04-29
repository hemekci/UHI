# scripts/

Runnable entry points wrapping `pipeline/` modules for end-to-end execution.
Driven by Hydra configs under `configs/`.

Planned:

- `run_ingest.py` — pull and harmonise raw layers for one or more cities
- `run_features.py` — build the per-patch morphology table
- `run_models.py` — fit mixed-effects + XGBoost + SHAP, produce attribution figures
- `run_archetypes.py` — Pareto extraction + clustering + archetype cards

To be added after manuscript acceptance.
