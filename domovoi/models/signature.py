"""Architecture signatures: the computation graph a set of weights belongs to.

Two models whose parameter tensors have identical names and shapes can still
compute different functions: a depth-2 "cycle" recurrent model and a depth-2
"sequence" one, or a 2-block baseline and a 2-block recurrent model, all
share one state_dict layout.  Loading weights by shape alone therefore
cannot tell them apart (Codex audit DML-A01).  A signature records every
model setting that changes how stored weights are wired into the forward
pass, including the explicit block-application schedule, so checkpoint
restore, training resume and profiling can refuse a mismatch.

dropout is deliberately excluded: it changes training noise, not which
function the weights compute.  Training resume checks it separately.
"""

from __future__ import annotations

# fields copied verbatim from the model config; order is the report order
_FIELDS = ("architecture", "vocab_size", "context_length", "d_model",
           "n_layers", "n_heads", "ffn_dim", "tie_embeddings",
           "recurrence_depth", "recurrence_order")


def model_signature(model) -> dict:
    """Signature of an instantiated model (what it will actually compute)."""
    cfg = model.cfg
    sig = {
        "architecture": model.architecture_id,
        "vocab_size": cfg.vocab_size,
        "context_length": cfg.context_length,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "ffn_dim": cfg.ffn_dim,
        "tie_embeddings": cfg.tie_embeddings,
        "recurrence_depth": getattr(cfg, "recurrence_depth", 1),
        "recurrence_order": getattr(cfg, "recurrence_order", "cycle"),
    }
    schedule = [int(i) for i in model.application_schedule()]
    sig["application_schedule"] = schedule
    sig["effective_depth"] = len(schedule)
    return sig


def config_signature(model_section) -> dict:
    """Signature implied by a `model:` config (a ModelSection or the dict
    stored in checkpoint.json).  Dicts are normalised through ModelSection
    first, so legacy checkpoints written before the recurrence keys existed
    get the defaults (depth 1, cycle) exactly as config loading does."""
    from ..config import ModelSection
    from .domovoi_recurrent import recurrence_schedule

    m = ModelSection(**model_section) if isinstance(model_section, dict) else model_section
    sig = {f: getattr(m, f) for f in _FIELDS}
    schedule = recurrence_schedule(m.n_layers, m.recurrence_depth, m.recurrence_order)
    sig["application_schedule"] = schedule
    sig["effective_depth"] = len(schedule)
    return sig


def signature_differences(expected: dict, actual: dict,
                          expected_label: str = "checkpoint",
                          actual_label: str = "model") -> list[str]:
    """Human-readable '<field>: <expected_label>=X, <actual_label>=Y' lines,
    one per differing field; empty when the signatures are equal."""
    keys = list(expected) + [k for k in actual if k not in expected]
    return [f"{k}: {expected_label}={expected.get(k)!r}, {actual_label}={actual.get(k)!r}"
            for k in keys if expected.get(k) != actual.get(k)]
