"""Profiling utilities.

Records the efficiency side of EVALS.md's fast-research layer:

- total / trainable / active parameter counts
- checkpoint size on disk
- peak GPU VRAM allocated and reserved (REAL torch.cuda statistics —
  never estimated from parameter counts)
- prefill tok/s, decode tok/s, time to first token
- training tok/s and training peak VRAM (filled in by the Trainer)
- VRAM vs context length sweep

On CPU, VRAM fields are explicitly null with "cuda": false — the report
never fabricates memory numbers it could not measure.
"""

from __future__ import annotations

import pathlib
import time
from typing import Optional

import torch

from ..inference.generate import generate


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Module.parameters() deduplicates shared (tied) tensors, so `total` is
    # honest even with tied embeddings.
    return {
        "total": total,
        "trainable": trainable,
        # dense baseline: every parameter is active for every token.  Sparse /
        # MoE / conditional architectures override this with their real value.
        "active_per_token": total,
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


def profile_inference(model, tokenizer, device, prompt: str = "Once upon a time",
                      max_new_tokens: int = 64, n_repeats: int = 3) -> dict:
    """Measure generation speed.  Runs one warmup then `n_repeats` timed
    generations and reports the median decode tok/s."""
    reset_vram_tracking(device)
    # warmup (JIT/cudnn autotune, cuda context init)
    generate(model, prompt, tokenizer, max_new_tokens=min(8, max_new_tokens),
             temperature=0.0)
    reset_vram_tracking(device)

    runs = [generate(model, prompt, tokenizer, max_new_tokens=max_new_tokens,
                     temperature=0.0) for _ in range(n_repeats)]
    decode = sorted(r["decode_tok_per_s"] for r in runs)
    prefill = sorted(r["prefill_tok_per_s"] for r in runs)
    ttft = sorted(r["ttft_s"] for r in runs)
    stats = vram_stats(device)
    return {
        "prompt": prompt,
        "n_prompt_tokens": runs[0]["n_prompt_tokens"],
        "max_new_tokens": max_new_tokens,
        "n_repeats": n_repeats,
        "prefill_tok_per_s_median": prefill[len(prefill) // 2],
        "decode_tok_per_s_median": decode[len(decode) // 2],
        "ttft_s_median": round(ttft[len(ttft) // 2], 6),
        "peak_vram_allocated_mb": stats["peak_allocated_mb"],
        "peak_vram_reserved_mb": stats["peak_reserved_mb"],
        "batch_size": 1,
        "device": str(device),
    }


def profile_context_sweep(model, tokenizer, device,
                          lengths: Optional[list[int]] = None,
                          max_new_tokens: int = 8) -> list[dict]:
    """Peak VRAM vs context length (EVALS.md 'context efficiency').  Uses
    prompts padded with repeated filler to the requested token length,
    capped by the model's context window."""
    lengths = lengths or [64, 128, 256, 512]
    ctx_cap = model.cfg.context_length
    filler = "the quick brown fox jumps over the lazy dog "
    results = []
    for target in lengths:
        target = min(target, ctx_cap)
        prompt = (filler * (target // len(filler) + 1))[:target * 6]
        ids = tokenizer.encode(prompt)
        if len(ids) > target:
            prompt = tokenizer.decode(ids[:target])
        reset_vram_tracking(device)
        r = generate(model, prompt, tokenizer, max_new_tokens=max_new_tokens,
                     temperature=0.0)
        stats = vram_stats(device)
        results.append({
            "target_tokens": target,
            "actual_prompt_tokens": r["n_prompt_tokens"],
            "max_new_tokens": max_new_tokens,
            "peak_vram_allocated_mb": stats["peak_allocated_mb"],
            "peak_vram_reserved_mb": stats["peak_reserved_mb"],
            "decode_tok_per_s": r["decode_tok_per_s"],
        })
    return results


def profile_checkpoint(ckpt_dir, tokenizer, device: Optional[torch.device] = None,
                       max_new_tokens: int = 64, context_sweep: bool = True,
                       context_lengths: Optional[list[int]] = None) -> dict:
    """Full profile of one checkpoint.  Returns the profile.json payload."""
    from ..training.checkpoint import build_model_from_checkpoint
    from ..utils import detect_device

    device = device if device is not None else detect_device()
    model, cfg, meta = build_model_from_checkpoint(ckpt_dir)
    model = model.to(device)
    model.eval()

    profile = {
        "checkpoint": str(ckpt_dir),
        "architecture": cfg.model.architecture,
        "context_length": cfg.model.context_length,
        "parameters": count_parameters(model),
        "checkpoint_size_mb": checkpoint_size_mb(ckpt_dir),
        "inference": profile_inference(model, tokenizer, device,
                                       max_new_tokens=max_new_tokens),
        "profiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if context_sweep:
        profile["context_sweep"] = profile_context_sweep(
            model, tokenizer, device, lengths=context_lengths)
    return profile