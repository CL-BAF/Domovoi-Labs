"""Text generation from a checkpoint.

Independent of training: `load_model_for_inference` rebuilds the model from
a checkpoint directory (config travels inside checkpoint.json), and
`generate` runs an incremental KV-cache decode loop with temperature,
top-k and top-p sampling.

Speed measurement: prefill (first forward over the prompt) and decode (the
per-token loop) are timed separately, so time-to-first-token, prefill tok/s
and decode tok/s are all real measurements, never estimates.
"""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn.functional as F

from ..training.checkpoint import build_model_from_checkpoint
from ..utils.seeding import set_seed


def load_model_for_inference(ckpt_dir, device: Optional[torch.device] = None):
    """Build model + load weights from a checkpoint dir.  Returns
    (model, config, checkpoint_meta, device).  Uses safetensors load and
    puts the model in eval mode on the requested (or best available) device."""
    from ..utils import detect_device
    device = device if device is not None else detect_device()
    model, cfg, meta = build_model_from_checkpoint(ckpt_dir)
    model = model.to(device)
    model.eval()
    return model, cfg, meta, device


@torch.no_grad()
def _sample_next(logits: torch.Tensor, temperature: float, top_k: int,
                 top_p: float, generator: Optional[torch.Generator]) -> int:
    """Sample one token id from last-position logits (shape (B, vocab) or
    (vocab,)); B is assumed 1 for simplicity (batch-1 generation)."""
    logits = logits.reshape(-1)
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / temperature
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        v, _ = torch.topk(logits, k)
        logits[logits < v[-1]] = float("-inf")
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # keep tokens whose cumulative probability mass stays <= top_p, plus
        # the first token that crosses it (so there is always at least one)
        keep = cum - probs <= top_p
        keep[0] = True
        sorted_logits[~keep] = float("-inf")
        logits = torch.full_like(logits, float("-inf"))
        logits[sorted_idx] = sorted_logits
    probs = F.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator).item())


@torch.no_grad()
def generate(
    model,
    prompt: str,
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Generate continuation of `prompt`.  Returns a result dict:

        text           - prompt + generated text
        new_text       - only the generated part
        n_prompt_tokens, n_new_tokens
        ttft_s         - time to first token (prefill forward)
        prefill_tok_per_s, decode_tok_per_s
        total_tok_per_s
    """
    if seed is not None:
        set_seed(seed)
    device = device or next(model.parameters()).device

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to zero tokens; provide a non-empty prompt.")
    ctx_len = model.cfg.context_length
    if len(prompt_ids) > ctx_len:
        raise ValueError(
            f"Prompt has {len(prompt_ids)} tokens; model context is {ctx_len}.")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # reset CUDA peak-memory tracking so the profile of *this* generation is
    # measurable (real stats, not guesses)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # sampling generator must live on the same device as the logits
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # ---- prefill -----------------------------------------------------
    t0 = time.perf_counter()
    out = model(x, use_cache=True)
    caches = out.caches
    ttft = time.perf_counter() - t0

    # a token sampled after a forward that leaves cache length L occupies
    # sequence position L, so it is inside the context window only if
    # L < ctx_len.  Every emitted token must live inside the window, hence:
    #   prefill sample (position n_prompt) needs n_prompt < ctx_len, and
    #   each decode sample needs cache_len + 2 <= ctx_len.
    new_ids = []
    if len(prompt_ids) < ctx_len:
        next_tok = _sample_next(out.logits[0, -1], temperature, top_k, top_p, gen)
        new_ids.append(next_tok)

    t1 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        # stop before sampling a token that would live outside the window
        if caches[0][0].size(2) + 2 > ctx_len:
            break
        x = torch.tensor([[next_tok]], dtype=torch.long, device=device)
        out = model(x, use_cache=True, caches=caches)
        caches = out.caches
        next_tok = _sample_next(out.logits[0, -1], temperature, top_k, top_p, gen)
        new_ids.append(next_tok)
    decode_s = time.perf_counter() - t1

    n_new = len(new_ids)
    result = {
        "prompt": prompt,
        "text": prompt + tokenizer.decode(new_ids),
        "new_text": tokenizer.decode(new_ids),
        "n_prompt_tokens": len(prompt_ids),
        "n_new_tokens": n_new,
        "ttft_s": round(ttft, 6),
        "prefill_tok_per_s": round(len(prompt_ids) / max(1e-9, ttft), 1),
        "decode_tok_per_s": round(n_new / max(1e-9, decode_s), 1) if n_new else None,
        "decode_s": round(decode_s, 6),
        "settings": {
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "max_new_tokens": max_new_tokens, "seed": seed,
        },
    }
    if device.type == "cuda":
        result["peak_vram_allocated_mb"] = round(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
        result["peak_vram_reserved_mb"] = round(
            torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2)
    return result