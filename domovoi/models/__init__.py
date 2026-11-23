"""Model registry.

`build_model(model_cfg)` turns a config's `model:` section into a model
instance.  New architectures register themselves here so config files stay
identical in shape across experiments — only the `architecture` string
changes.  domovoi.config validates `architecture` against this dict, so
registering here is the only step needed to make a name selectable.
"""

from __future__ import annotations

from .baseline_transformer import (BaselineTransformer,
                                   BaselineTransformerConfig, ModelOutput,
                                   build_baseline_transformer)
from .domovoi_recurrent import (RECURRENCE_ORDERS, DomovoiRecurrentTransformer,
                                RecurrentTransformerConfig,
                                build_domovoi_recurrent)
from .kv_cache import ApplicationCache, CacheValidationError, validate_caches
from .signature import config_signature, model_signature, signature_differences

ARCHITECTURES = {
    "baseline_transformer": build_baseline_transformer,
    "domovoi_recurrent_v1": build_domovoi_recurrent,
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
    "DomovoiRecurrentTransformer", "RecurrentTransformerConfig",
    "build_domovoi_recurrent", "RECURRENCE_ORDERS",
    "ApplicationCache", "CacheValidationError", "validate_caches",
    "config_signature", "model_signature", "signature_differences",
]
