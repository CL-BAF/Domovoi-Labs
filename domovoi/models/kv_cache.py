"""Strict validation of per-application KV caches (Codex audit DML-A02).

A cache is a list with one (k, v) pair per block *application*, in schedule
order; k and v are (batch, n_heads, prefix_len, head_dim).  Before this
module the forward pass read the prefix length from slot 0 only, so a cache
whose other slots had a different length, a None hole, or the wrong batch,
head layout, device or dtype either crashed deep inside attention or, worse,
silently produced different logits.  validate_caches() checks everything
that can be checked before any block runs and raises CacheValidationError
(a ValueError) with the failing slot named.

Provenance.  Tensors carry no record of which application produced them, so
two same-shaped caches from different schedules (depth-2 cycle vs depth-2
sequence) or different models are indistinguishable by shape.  Caches
returned by forward() are therefore an ApplicationCache: a plain list that
also remembers the producing model and its schedule.  Such a cache is
rejected by any other model or schedule.  A cache rebuilt as a plain list
(e.g. list(cache) or hand-assembled) has no provenance and gets the
structural checks only; see POST_AUDIT_FIXES.txt for that remaining limit.
"""

from __future__ import annotations

import weakref

import torch


class CacheValidationError(ValueError):
    """A KV cache that does not fit this model / input.  Subclasses ValueError
    so existing callers that catch ValueError keep working."""


class ApplicationCache(list):
    """list of per-application (k, v) pairs plus provenance.  Behaves exactly
    like the list forward() used to return."""

    def __init__(self, items=(), model=None, schedule=()):
        super().__init__(items)
        self.owner = weakref.ref(model) if model is not None else None
        self.schedule = tuple(schedule)


def expected_kv_dtype(model, device: torch.device) -> torch.dtype:
    """dtype the attention projections will produce for new keys/values:
    the autocast dtype when autocast is active, else the parameter dtype."""
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    return model.wte.weight.dtype


def validate_caches(model, idx: torch.Tensor, caches) -> int:
    """Check `caches` against `model` and the new tokens `idx`.  Returns the
    cached prefix length (0 when caches is None or all slots are None).
    Raises CacheValidationError on any inconsistency."""
    if idx.dim() != 2 or idx.size(1) < 1:
        raise CacheValidationError(
            f"idx must be (batch, T) with T >= 1; got shape {tuple(idx.shape)}.")
    B, T = idx.shape
    ctx = model.cfg.context_length
    if caches is None:
        prefix = 0
    else:
        prefix = _validate_entries(model, idx, caches)
    if prefix + T > ctx:
        raise CacheValidationError(
            f"Sequence length {prefix + T} exceeds context_length {ctx}. "
            f"(prefix {prefix} + new {T})")
    return prefix


def _validate_entries(model, idx: torch.Tensor, caches) -> int:
    schedule = tuple(model.application_schedule())
    if not isinstance(caches, (list, tuple)):
        raise CacheValidationError(
            f"caches must be a list of per-application (k, v) pairs; got {type(caches).__name__}.")
    if len(caches) != len(schedule):
        # checked before any slot is indexed (an empty list used to IndexError)
        raise CacheValidationError(
            f"Got {len(caches)} cache entries; this model applies "
            f"{len(schedule)} blocks per forward (schedule {list(schedule)}).")
    if isinstance(caches, ApplicationCache):
        if caches.owner is not None and caches.owner() is not model:
            raise CacheValidationError(
                "This cache was produced by a different model instance; K/V "
                "from other weights would silently change the logits.")
        if caches.schedule != schedule:
            raise CacheValidationError(
                f"This cache was produced under application schedule "
                f"{list(caches.schedule)}, but the model now runs "
                f"{list(schedule)}; slot j must hold K/V of application j.")

    n_none = sum(entry is None for entry in caches)
    if n_none == len(caches):
        return 0                      # explicit "no prefix yet", as before
    if n_none:
        holes = [j for j, entry in enumerate(caches) if entry is None]
        raise CacheValidationError(
            f"Cache slots {holes} are None while others hold K/V; every "
            f"application needs the same cached prefix.")

    B = idx.size(0)
    n_heads = model.cfg.n_heads
    head_dim = model.cfg.d_model // n_heads
    device = model.wte.weight.device
    want_dtype = expected_kv_dtype(model, device)
    prefix = None
    for j, entry in enumerate(caches):
        where = f"cache slot {j} (application of block {schedule[j]})"
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise CacheValidationError(f"{where} must be a (k, v) pair.")
        k, v = entry
        if not (isinstance(k, torch.Tensor) and isinstance(v, torch.Tensor)):
            raise CacheValidationError(f"{where}: k and v must be tensors.")
        if k.dim() != 4 or k.shape != v.shape:
            raise CacheValidationError(
                f"{where}: k and v must both be (batch, heads, length, head_dim); "
                f"got k {tuple(k.shape)}, v {tuple(v.shape)}.")
        kb, kh, kl, kd = k.shape
        if kb != B:
            raise CacheValidationError(
                f"{where}: cache batch {kb} != input batch {B}.")
        if kh != n_heads or kd != head_dim:
            raise CacheValidationError(
                f"{where}: expected {n_heads} heads x {head_dim} dims, got {kh} x {kd}.")
        if prefix is None:
            prefix = kl
        elif kl != prefix:
            raise CacheValidationError(
                f"{where}: cached length {kl} != slot 0 length {prefix}; "
                f"all applications must cache the same prefix.")
        if k.device != device or v.device != device:
            raise CacheValidationError(
                f"{where}: cache on {k.device}/{v.device}, model on {device}.")
        if k.dtype != want_dtype or v.dtype != want_dtype:
            raise CacheValidationError(
                f"{where}: cache dtype {k.dtype}/{v.dtype}, but this forward "
                f"produces {want_dtype} keys/values (mixing would silently "
                f"type-promote).")
    return prefix
