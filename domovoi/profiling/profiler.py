"""Profiling utilities.

Records the efficiency side of EVALS.md's fast-research layer:

- total (unique) / trainable / active / applied parameter counts
- checkpoint size on disk
- peak GPU VRAM allocated and reserved (REAL torch.cuda statistics —
  never estimated from parameter counts)
- prefill tok/s, decode tok/s, time to first token
- training tok/s and training peak VRAM (filled in by the Trainer)
- VRAM vs context length sweep

On CPU, VRAM fields are explicitly null with "cuda": false — the report
never fabricates memory numbers it could not measure.

Labelling: keys under "analytic_estimates" are closed-form estimates
(accounting.py, computed in the model's real dtype).  "measured_parameter_bytes"
and "kv_cache_*_bytes" are exact counts from the tensors.  VRAM peaks are
torch.cuda measurements and each result carries a "measurement_basis".
"""

from __future__ import annotations

import pathlib
import time
from typing import Optional

import torch

from ..inference.generate import generate
from ..models.kv_cache import expected_kv_dtype
from ..models.signature import config_signature, model_signature, signature_differences
from .accounting import architecture_accounting


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Module.parameters() deduplicates shared (tied or reused) tensors, so
    # `total` is the number of UNIQUE stored parameters — honest even with
    # tied embeddings and with blocks applied several times.
    schedule = model.application_schedule()
    stored_blocks = sum(p.numel() for p in model.blocks.parameters())
    used_blocks = sum(sum(p.numel() for p in model.blocks[i].parameters())
                      for i in schedule)
    return {
        "total": total,
        "trainable": trainable,
        # dense models: every stored parameter is active for every token.
        # Sparse / MoE / conditional architectures override this.
        "active_per_token": total,
        # parameter USES per token: a block applied k times counts k times.
        # Equals total for the baseline; larger for weight-shared recurrence.
        "applied_per_token": total - stored_blocks + used_blocks,
        "effective_depth": len(schedule),
    }


def checkpoint_size_mb(ckpt_dir: str | pathlib.Path) -> float:
    return round(pathlib.Path(ckpt_dir).joinpath("model.safetensors").stat().st_size / (1024 * 1024), 3)


def vram_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"peak_allocated_mb": None, "peak_reserved_mb": None, "cuda": False}
    return {
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2),
        "cuda": True,
    }


def reset_vram_tracking(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _median(values):
    values = sorted(v for v in values if v is not None)
    return values[len(values) // 2] if values else None


def device_wide_used_mb(device: torch.device) -> Optional[float]:
    """Memory in use on the whole GPU right now, by ALL processes (driver
    view, torch.cuda.mem_get_info).  A snapshot, not a peak; None on CPU."""
    if device.type != "cuda":
        return None
    free, total = torch.cuda.mem_get_info(device)
    return round((total - free) / (1024 * 1024), 2)


def _vram_basis(device: torch.device) -> str:
    if device.type != "cuda":
        return "not measured: no CUDA device (VRAM fields are null, never estimated)"
    return ("measured: torch.cuda.max_memory_allocated / max_memory_reserved of "
            "this process since the reset before the timed runs (peak over all "
            "of them)")


def profile_inference(model, tokenizer, device, prompt: str = "Once upon a time",
                      max_new_tokens: int = 64, n_repeats: int = 3,
                      prompt_ids: Optional[list[int]] = None) -> dict:
    """Measure generation speed.  Runs one warmup then `n_repeats` timed
    generations and reports medians.  The VRAM peak covers ALL timed runs
    (generate() is told not to reset it per run; previously only the last
    run's peak was reported)."""
    if prompt_ids is not None:
        prompt = None          # exact ids win; the text is decoded from them
    # warmup (JIT/cudnn autotune, cuda context init)
    generate(model, prompt, tokenizer, max_new_tokens=min(8, max_new_tokens),
             temperature=0.0, prompt_ids=prompt_ids)
    reset_vram_tracking(device)

    runs = [generate(model, prompt, tokenizer, max_new_tokens=max_new_tokens,
                     temperature=0.0, prompt_ids=prompt_ids, reset_peak=False)
            for _ in range(n_repeats)]
    stats = vram_stats(device)
    # decode_tok_per_s is None when the decode loop produced no tokens
    # (max_new_tokens <= 1); never fabricate a number for those runs
    ttft = _median(r["ttft_s"] for r in runs)
    last = runs[-1]
    return {
        "prompt": runs[0]["prompt"],
        "n_prompt_tokens": runs[0]["n_prompt_tokens"],
        "n_new_tokens": last["n_new_tokens"],
        "final_sequence_length": last["final_sequence_length"],
        "max_new_tokens": max_new_tokens,
        "n_repeats": n_repeats,
        "prefill_tok_per_s_median": _median(r["prefill_tok_per_s"] for r in runs),
        "decode_tok_per_s_median": _median(r["decode_tok_per_s"] for r in runs),
        "ttft_s_median": round(ttft, 6) if ttft is not None else None,
        "peak_vram_allocated_mb": stats["peak_allocated_mb"],
        "peak_vram_reserved_mb": stats["peak_reserved_mb"],
        "device_wide_used_mb_after": device_wide_used_mb(device),
        # exact tensor byte counts of the final cache (measured, CPU or GPU)
        "kv_cache_logical_bytes": last["kv_cache_logical_bytes"],
        "kv_cache_storage_bytes": last["kv_cache_storage_bytes"],
        "effective_depth": last["effective_depth"],
        "weight_dtype": last["weight_dtype"],
        "kv_dtype": last["kv_dtype"],
        "batch_size": 1,
        "device": str(device),
        "measurement_basis": {
            "timing": "wall clock; CUDA-synchronized before every clock read",
            "vram": _vram_basis(device),
            "device_wide_used_mb_after": "snapshot of all processes on the GPU after the runs; not a peak",
            "kv_cache_bytes": "counted from the final cache tensors (exact, not estimated)",
        },
    }


def profile_context_sweep(model, tokenizer, device,
                          lengths: Optional[list[int]] = None,
                          max_new_tokens: int = 8) -> list[dict]:
    """Peak VRAM vs context length (EVALS.md 'context efficiency').

    Prompts are EXACT token-id sequences (filler ids repeated and cut to the
    target), so the measured length is the requested length; text-level
    padding used to undershoot (16 and 32 both became 9 tokens with a word
    tokenizer).  Each target is capped at context_length - max_new_tokens so
    every row still decodes its max_new_tokens; a capped row says so."""
    lengths = lengths or [64, 128, 256, 512]
    ctx_cap = model.cfg.context_length
    max_prompt = ctx_cap - max_new_tokens
    if max_prompt < 1:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} leaves no room for a prompt in "
            f"context_length={ctx_cap}.")
    filler_ids = tokenizer.encode("the quick brown fox jumps over the lazy dog ")
    if not filler_ids:
        raise ValueError("Filler text tokenized to zero tokens.")
    results = []
    for requested in lengths:
        target = min(requested, max_prompt)
        ids = (filler_ids * (target // len(filler_ids) + 1))[:target]
        reset_vram_tracking(device)
        r = generate(model, None, tokenizer, max_new_tokens=max_new_tokens,
                     temperature=0.0, prompt_ids=ids, reset_peak=False)
        stats = vram_stats(device)
        results.append({
            "requested_tokens": requested,
            "target_tokens": target,
            "capped_for_decode_headroom": target < requested,
            "actual_prompt_tokens": r["n_prompt_tokens"],
            "max_new_tokens": max_new_tokens,
            "n_new_tokens": r["n_new_tokens"],
            "final_sequence_length": r["final_sequence_length"],
            "context_length": ctx_cap,
            "effective_depth": r["effective_depth"],
            "kv_cache_logical_bytes": r["kv_cache_logical_bytes"],
            "kv_cache_storage_bytes": r["kv_cache_storage_bytes"],
            "peak_vram_allocated_mb": stats["peak_allocated_mb"],
            "peak_vram_reserved_mb": stats["peak_reserved_mb"],
            "ttft_s": r["ttft_s"],
            "decode_tok_per_s": r["decode_tok_per_s"],
        })
    return results


def profile_model(model, cfg, tokenizer, device: torch.device,
                  ckpt_dir: Optional[str | pathlib.Path] = None,
                  max_new_tokens: int = 64, context_sweep: bool = True,
                  context_lengths: Optional[list[int]] = None) -> dict:
    """Full profile of an already-built, already-loaded model.  `ckpt_dir`
    is optional and only used to report the checkpoint path/size; callers
    that already have the model in memory pass it to avoid building and
    reading the checkpoint a second time.

    Raises ValueError when `cfg` does not describe `model` (architecture,
    dimensions or recurrence schedule): the analytic estimates come from
    cfg and the measurements from model, so a mismatch would silently pair
    one architecture's estimates with another's measurements."""
    diffs = signature_differences(config_signature(cfg.model), model_signature(model),
                                  "config", "model")
    if diffs:
        raise ValueError("profile_model: config does not describe this model:\n  "
                         + "\n  ".join(diffs))
    model.eval()
    sig = model_signature(model)
    # analytic byte estimates use the dtypes ACTUALLY in effect: the weights'
    # dtype for parameters, and for K/V the dtype this forward will produce
    # (the autocast dtype under autocast, else the weight dtype) -- the same
    # rule the cache validator enforces (Codex re-review: FP32 weights + bf16
    # autocast were estimated at 4 KV bytes/element but held 2)
    elem = next(model.parameters()).element_size()
    kv_dtype = expected_kv_dtype(model, next(model.parameters()).device)
    kv_elem = torch.empty((), dtype=kv_dtype).element_size()
    # parameters() yields each shared (tied / reused) tensor once
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    profile = {
        "checkpoint": str(ckpt_dir) if ckpt_dir is not None else None,
        "architecture": cfg.model.architecture,
        "context_length": cfg.model.context_length,
        "recurrence": {k: sig[k] for k in ("recurrence_depth", "recurrence_order",
                                            "application_schedule", "effective_depth")},
        "weight_dtype": str(next(model.parameters()).dtype),
        "kv_dtype": str(kv_dtype),
        "autocast": torch.is_autocast_enabled(next(model.parameters()).device.type),
        "device": str(device),
        "parameters": count_parameters(model),
        # exact: bytes of the parameter tensors as loaded (not a device peak)
        "measured_parameter_bytes": param_bytes,
        # closed-form, config-derived (NOT measured): FLOPs, KV bytes, etc.
        "analytic_estimates": architecture_accounting(
            cfg.model, bytes_per_param=elem, bytes_per_kv=kv_elem),
        "checkpoint_size_mb": checkpoint_size_mb(ckpt_dir)
            if ckpt_dir is not None else None,
        "inference": profile_inference(model, tokenizer, device,
                                       max_new_tokens=max_new_tokens),
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if context_sweep:
        profile["context_sweep"] = profile_context_sweep(
            model, tokenizer, device, lengths=context_lengths)
    return profile


def profile_checkpoint(ckpt_dir, tokenizer, device: Optional[torch.device] = None,
                       max_new_tokens: int = 64, context_sweep: bool = True,
                       context_lengths: Optional[list[int]] = None) -> dict:
    """Full profile of one checkpoint (builds the model from disk).  Returns
    the profile.json payload.  Callers that already built the model should
    call profile_model directly."""
    from ..training.checkpoint import build_model_from_checkpoint
    from ..utils import detect_device

    device = device if device is not None else detect_device()
    model, cfg, meta = build_model_from_checkpoint(ckpt_dir)
    model = model.to(device)
    model.eval()
    return profile_model(model, cfg, tokenizer, device, ckpt_dir=ckpt_dir,
                         max_new_tokens=max_new_tokens,
                         context_sweep=context_sweep,
                         context_lengths=context_lengths)