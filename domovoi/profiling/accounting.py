"""Analytic (closed-form) cost accounting from a `model:` config section.

Everything here is DERIVED FROM THE CONFIG, never measured.  It exists to
design parameter-matched and compute-matched comparisons before anything is
trained, and to sanity-check measured profiles.  Measured numbers (VRAM,
tok/s) come only from profiler.py.  Every key that is an estimate says so.

Conventions (d = d_model, f = ffn_dim, V = vocab_size, A = block applications
per forward, i.e. n_layers * recurrence_depth):
- one multiply-add = 2 FLOPs;
- per application: matmuls 2*(4d^2 + 2df) FLOPs per token, attention scores +
  weighted sum 4*d*L per token where L is the number of keys attended;
- output head 2*V*d per token (counted even when tied: tying saves memory,
  not compute);
- biases, LayerNorm, softmax, GELU and embedding lookups are ignored (small);
- training FLOPs ~= 3x forward (backward ~= 2x forward), the usual estimate.
"""

from __future__ import annotations


def block_params(d_model: int, ffn_dim: int) -> int:
    """Parameters of one pre-LN block with biases (matches baseline Block)."""
    d, f = d_model, ffn_dim
    ln = 2 * d                       # weight + bias
    attn = (3 * d * d + 3 * d) + (d * d + d)
    mlp = (d * f + f) + (f * d + d)
    return 2 * ln + attn + mlp       # = 4d^2 + 2df + 9d + f


def architecture_accounting(m, attended_tokens: int | None = None,
                            bytes_per_param: int = 2, bytes_per_kv: int = 2) -> dict:
    """Closed-form costs for a ModelSection-like object `m`.

    attended_tokens: keys each query attends to (default: m.context_length,
    i.e. a decode step at a full window; use (T+1)/2 for the average over a
    causal training sequence of length T).
    bytes_per_param / bytes_per_kv: 2 = bf16/fp16 storage."""
    d, V = m.d_model, m.vocab_size
    depth = getattr(m, "recurrence_depth", 1)
    apps = m.n_layers * depth
    L = m.context_length if attended_tokens is None else attended_tokens
    per_block = block_params(d, m.ffn_dim)
    embed = V * d + m.context_length * d            # token + learned position table
    head = 0 if m.tie_embeddings else V * d
    unique = embed + m.n_layers * per_block + 2 * d + head   # + final LayerNorm
    matmul_per_app = 2 * (4 * d * d + 2 * d * m.ffn_dim)
    fwd = apps * (matmul_per_app + 4 * d * L) + 2 * V * d
    return {
        # every number below is a closed-form estimate under these assumptions
        "basis": "analytic estimate from the config; not a measurement",
        "bytes_per_param": bytes_per_param,
        "bytes_per_kv": bytes_per_kv,
        "unique_params": unique,
        "block_params": per_block,
        "stored_blocks": m.n_layers,
        "effective_depth": apps,
        "applied_params": unique + (apps - m.n_layers) * per_block,
        "weight_bytes": unique * bytes_per_param,
        "kv_cache_bytes_per_token": 2 * apps * d * bytes_per_kv,
        "kv_cache_bytes_full_context": 2 * apps * d * bytes_per_kv * m.context_length,
        "attended_tokens": L,
        "est_forward_flops_per_token": fwd,
        "est_train_flops_per_token": 3 * fwd,
    }
