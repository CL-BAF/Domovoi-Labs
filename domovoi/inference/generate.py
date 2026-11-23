"""Text generation from a checkpoint.

Independent of training: `load_model_for_inference` rebuilds the model from
a checkpoint directory (config travels inside checkpoint.json), and
`generate` runs an incremental KV-cache decode loop with temperature,
top-k and top-p sampling.

Speed measurement: prefill (first forward over the prompt) and decode (the
per-token loop) are timed separately, so time-to-first-token, prefill tok/s
and decode tok/s are all real measurements, never estimates.  On CUDA the
clock is read only after torch.cuda.synchronize, and TTFT runs until the
first token id is on the host (prefill + first sampling).  The first new
token is sampled from the prefill logits, so it is NOT counted in
decode_tok_per_s — that number covers exactly the forwards that were timed
(and is null when the decode loop never ran, rather than fabricating one).
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


def _sync(device: torch.device) -> None:
    """CUDA launches are asynchronous: without a sync, perf_counter measures
    the launch, not the work (audit DML-A03).  No-op on CPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cache_bytes(caches) -> dict:
    """Exact byte counts of a KV cache (counted from the tensors, not
    estimated).  logical = bytes of the K/V views themselves; storage =
    bytes of the distinct underlying buffers they keep alive.  After
    prefill K/V are views into the fused QKV output, so storage > logical."""
    if not caches:
        return {"kv_cache_logical_bytes": 0, "kv_cache_storage_bytes": 0}
    tensors = [t for pair in caches for t in pair]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
    return {"kv_cache_logical_bytes": sum(t.numel() * t.element_size() for t in tensors),
            "kv_cache_storage_bytes": sum(storages.values())}


@torch.no_grad()
def generate(
    model,
    prompt: Optional[str],
    tokenizer,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    prompt_ids: Optional[list[int]] = None,
    reset_peak: bool = True,
) -> dict:
    """Generate continuation of `prompt`.  Returns a result dict:

        text           - prompt + generated text
        new_text       - only the generated part
        n_prompt_tokens, n_new_tokens
        prefill_s      - prefill forward, CUDA-synchronized
        ttft_s         - time to first token: prefill forward + sampling the
                         first token (None if no token could be sampled)
        prefill_tok_per_s, decode_tok_per_s  (decode_tok_per_s is None when
        the decode loop produced no tokens, e.g. max_new_tokens <= 1)
        decode_s       - seconds spent in the decode loop
        final_sequence_length, kv_cache_*_bytes, effective_depth,
        weight_dtype, kv_dtype

    prompt_ids: exact token ids to use instead of encoding `prompt` (the
    profiler needs exact lengths; text re-encoding can change them).
    reset_peak: reset CUDA peak-memory stats first (the profiler turns this
    off to take one peak across several runs).
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0; got {max_new_tokens}")
    if seed is not None:
        set_seed(seed)
    device = device or next(model.parameters()).device

    if prompt_ids is None:
        prompt_ids = tokenizer.encode(prompt)
    else:
        prompt_ids = [int(i) for i in prompt_ids]
        if prompt is None:
            prompt = tokenizer.decode(prompt_ids)
    if not prompt_ids:
        raise ValueError("Prompt tokenized to zero tokens; provide a non-empty prompt.")
    ctx_len = model.cfg.context_length
    if len(prompt_ids) > ctx_len:
        raise ValueError(
            f"Prompt has {len(prompt_ids)} tokens; model context is {ctx_len}.")

    x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # reset CUDA peak-memory tracking so the profile of *this* generation is
    # measurable (real stats, not guesses)
    if device.type == "cuda" and reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    # sampling generator must live on the same device as the logits
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)

    # ---- prefill -----------------------------------------------------
    _sync(device)
    t0 = time.perf_counter()
    out = model(x, use_cache=True)
    caches = out.caches
    _sync(device)
    prefill_s = time.perf_counter() - t0
    ttft = None

    # a token sampled after a forward that leaves cache length L occupies
    # sequence position L, so it is inside the context window only if
    # L < ctx_len.  Every emitted token must live inside the window, hence:
    #   prefill sample (position n_prompt) needs n_prompt < ctx_len, and
    #   each decode sample needs cache_len + 2 <= ctx_len.
    new_ids = []
    decode_ids = []   # tokens sampled INSIDE the timed decode loop
    if max_new_tokens >= 1 and len(prompt_ids) < ctx_len:
        next_tok = _sample_next(out.logits[0, -1], temperature, top_k, top_p, gen)
        new_ids.append(next_tok)
        # .item() inside sampling already waits for the device; the first
        # token id is now on the host
        ttft = time.perf_counter() - t0

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
        decode_ids.append(next_tok)
    _sync(device)
    decode_s = time.perf_counter() - t1

    n_new = len(new_ids)
    n_decode = len(decode_ids)
    result = {
        "prompt": prompt,
        "text": prompt + tokenizer.decode(new_ids),
        "new_text": tokenizer.decode(new_ids),
        "n_prompt_tokens": len(prompt_ids),
        "n_new_tokens": n_new,
        "prefill_s": round(prefill_s, 6),
        "ttft_s": round(ttft, 6) if ttft is not None else None,
        "prefill_tok_per_s": round(len(prompt_ids) / max(1e-9, prefill_s), 1),
        # decode throughput covers exactly the timed decode-loop tokens;
        # the prefill-sampled first token belongs to the prefill measurement
        "decode_tok_per_s": round(n_decode / max(1e-9, decode_s), 1)
            if n_decode else None,
        "decode_s": round(decode_s, 6),
        # tokens held in the KV cache after the last forward (the sampled
        # final token is never fed back, so it is not cached)
        "final_sequence_length": caches[0][0].size(2),
        **cache_bytes(caches),
        "effective_depth": len(caches),
        # weights and K/V can differ (FP32 weights, bf16 K/V under autocast)
        "weight_dtype": str(next(model.parameters()).dtype),
        "kv_dtype": str(caches[0][0].dtype),
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