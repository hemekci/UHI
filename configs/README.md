# configs/

Hydra + OmegaConf configuration tree.

Planned layout:

```
configs/
├── config.yaml           # top-level defaults
├── data/                 # data source configs (LST product, footprint vendor, ...)
├── features/             # feature-extraction parameters
├── model/                # mixed-effects, XGBoost, archetype clustering
└── experiment/           # named experiments composing the above
```

To be populated alongside `pipeline/` after manuscript acceptance.
