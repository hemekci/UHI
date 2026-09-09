"""UHI model registry + factory."""

from __future__ import annotations

from collections.abc import Callable

from .base import BaseUHIModel

MODEL_REGISTRY: dict[str, type[BaseUHIModel]] = {}


def register_model(name: str) -> Callable[[type[BaseUHIModel]], type[BaseUHIModel]]:
    def decorator(cls: type[BaseUHIModel]) -> type[BaseUHIModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} already registered")
        MODEL_REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def ModelFactory(cfg) -> BaseUHIModel:
    key = cfg.name
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {key!r}. Known: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key](cfg)
