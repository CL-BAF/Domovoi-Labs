"""Baseline decoder-only Transformer (GPT-2 style, pre-LayerNorm).

This is intentionally a boring, conventional, well-understood architecture.
It is the reference point ("the laboratory control group") against which all
future Domovoi architectures are compared.  It is written in plain PyTorch
with no Hugging Face abstractions so it stays easy to read, modify and
replace.

Key implementation points:

- Pre-LayerNorm residual blocks: x = x + attn(ln1(x)); x = x + mlp(ln2(x)).
- Causal masking via F.scaled_dot_product_attention (flash/mem-efficient
  kernels when available, exact math regardless).
- Optional per-layer key/value cache for incremental decoding.  The cache
  is a list of (k, v) tensor tuples, one entry per layer.  Passing
  ``use_cache=True`` with no previous cache performs prefill and returns the
  initial cache; passing ``caches=...`` extends it.  This keeps the
  generation loop independent of the model internals as much as possible
  while staying in plain PyTorch.
- Optional tied input/output embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineTransformerConfig:
    """All knobs of the baseline architecture.  Mirrors the `model:` config
    section (domovoi.config.ModelSection.to_model_config)."""
    vocab_size: int = 4096
    context_length: int = 512
    d_model: int = 640
    n_layers: int = 10
    n_heads: int = 10
    ffn_dim: int = 1728
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Non-config internals, fixed for clarity:
    bias: bool = True                 # use bias in linear projections
    initializer_range: float = 0.02


class ModelOutput(NamedTuple):
    logits: torch.Tensor              # (B, T, vocab)
    loss: Optional[torch.Tensor]      # scalar cross-entropy, if targets given
    caches: Optional[list]            # per-layer (k, v) tuples, if use_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: BaselineTransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0, \
            f"d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.dropout = cfg.dropout
        # fused qkv projection (common, readable, efficient)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor, cache: Optional[tuple] = None):
        """x: (B, T, C).  cache: optional (k, v) each (B, n_heads, T_prev, head_dim).

        Returns (attn_output, new_cache_or_None).
        """
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        kv_len = k.size(2)

        dropout_p = self.dropout if self.training else 0.0

        if cache is None:
            # Full self-attention over a fresh sequence: the built-in causal
            # flag is the clearest and fastest formulation.
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p, is_causal=True)
        elif T == 1:
            # Single new token attends to everything in the cache: no mask.
            y = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p)
        else:
            # Prefill/extend with an existing cache: query i (of T) may attend
            # keys up to (kv_len - T) + i.  Boolean mask: True = allowed.
            mask = torch.ones(T, kv_len, dtype=torch.bool, device=x.device) \
                .tril(diagonal=kv_len - T)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=dropout_p)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: BaselineTransformerConfig):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.ffn_dim, bias=cfg.bias)
        self.proj = nn.Linear(cfg.ffn_dim, cfg.d_model, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, cfg: BaselineTransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, cache: Optional[tuple] = None):
        attn_out, new_cache = self.attn(self.ln1(x), cache=cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class BaselineTransformer(nn.Module):
    def __init__(self, cfg: BaselineTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            # Tied weights: output projection reuses the input embedding.
            # This is a *real* weight share, not a copy.
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    # GPT-2 style: scale down residual-path output projections by sqrt(depth)
    def _rescale_residual_projections(self):
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0,
                            std=self.cfg.initializer_range / math.sqrt(2 * self.cfg.n_layers))
            nn.init.normal_(block.mlp.proj.weight, mean=0.0,
                            std=self.cfg.initializer_range / math.sqrt(2 * self.cfg.n_layers))

    # ------------------------------------------------------------------
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        caches: Optional[list] = None,
    ) -> ModelOutput:
        """idx: (B, T) token ids.  targets: (B, T) next-token ids.

        caches: optional per-layer (k, v) tuples from a previous call.  When
        given, idx contains only the *new* tokens; positions resume after the
        cached prefix.
        """
        B, T = idx.shape
        max_len = self.cfg.context_length
        if caches is not None:
            prefix_len = caches[0][0].size(2) if caches[0] is not None else 0
        else:
            prefix_len = 0
        total_len = prefix_len + T
        if total_len > max_len:
            raise ValueError(
                f"Sequence length {total_len} exceeds context_length {max_len}. "
                f"(prefix {prefix_len} + new {T})")

        # positions for the new tokens continue after the cached prefix
        pos = torch.arange(prefix_len, prefix_len + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        if caches is None:
            caches = [None] * len(self.blocks)
        new_caches = [] if use_cache else None
        for block, cache in zip(self.blocks, caches):
            x, new_cache = block(x, cache=cache)
            if use_cache:
                new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1))

        return ModelOutput(logits=logits, loss=loss, caches=new_caches)

    # ------------------------------------------------------------------
    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        # wpe counts as embedding; tied lm_head shares storage with wte, and
        # Module.parameters() already deduplicates shared tensors.
        if non_embedding:
            n -= self.wpe.weight.numel()
        return n

    def param_breakdown(self) -> dict:
        """Detailed parameter accounting for research comparisons."""
        emb = self.wte.weight.numel()
        pos = self.wpe.weight.numel()
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        ln_f = sum(p.numel() for p in self.ln_f.parameters())
        return {
            "token_embeddings": emb,
            "position_embeddings": pos,
            "per_block": per_block,
            "blocks": per_block * self.cfg.n_layers,
            "final_layernorm": ln_f,
            # lm_head is tied (0 extra) or vocab*d_model
            "lm_head": 0 if self.cfg.tie_embeddings else self.lm_head.weight.numel(),
            "total": self.num_params(non_embedding=False),
            "total_non_embedding": self.num_params(non_embedding=True),
        }


def build_baseline_transformer(cfg) -> BaselineTransformer:
    """Build a BaselineTransformer from a domovoi.config.ModelSection (or
    anything with the same attribute names)."""
    model_cfg = BaselineTransformerConfig(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        tie_embeddings=cfg.tie_embeddings,
    )
    model = BaselineTransformer(model_cfg)
    model._rescale_residual_projections()
    return model