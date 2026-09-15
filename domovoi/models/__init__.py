"""Model registry.

`build_model(model_cfg)` turns a config's `model:` section into a model
instance.  New architectures register themselves here so config files stay
identical in shape across experiments — only the `architecture` string
changes.
"""

from __future__ import annotations

from .baseline_transformer import (BaselineTransformer,
                                   BaselineTransformerConfig, ModelOutput,
                                   build_baseline_transformer)

ARCHITECTURES = {
    "baseline_transformer": build_baseline_transformer,
}


def build_model(model_cfg):
    """Build the model named by model_cfg.architecture."""
    builder = ARCHITECTURES.get(model_cfg.architecture)
    if builder is None:
        raise ValueError(
            f"Unknown architecture {model_cfg.architecture!r}. "
            f"Registered: {sorted(ARCHITECTURES)}")
    return builder(model_cfg)


__all__ = [
    "BaselineTransformer", "BaselineTransformerConfig", "ModelOutput",
    "build_baseline_transformer", "build_model", "ARCHITECTURES",
]