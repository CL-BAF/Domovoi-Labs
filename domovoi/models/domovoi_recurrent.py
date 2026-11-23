"""Domovoi Shared-Core Recurrent Transformer (architecture id: domovoi_recurrent_v1).

EXPERIMENTAL.  Hypothesis under test: re-applying a small set of shared
Transformer blocks several times buys more capability per GiB of stored
weights than giving every depth position its own block.

What changes relative to BaselineTransformer, and nothing else:

- ``n_layers`` stored blocks are applied ``recurrence_depth`` times each, so
  the residual stream passes through n_layers * recurrence_depth block
  applications ("effective depth").  Reuse is real: every application calls
  the *same* nn.Module, so there is one copy of each weight tensor and its
  gradient is the sum of the gradients from every application.
- ``recurrence_order`` picks how applications are arranged (names follow
  Takase & Kiyono, "Lessons on Parameter Sharing across Layers in
  Transformers"):
      cycle     0,1,2, 0,1,2, ...   (whole stack looped; Universal-Transformer-like)
      sequence  0,0, 1,1, 2,2, ...  (each block repeated back to back)
- Residual-projection init is scaled by the effective depth, not n_layers.

Embeddings, attention, MLP, LayerNorms, positional encoding, weight tying and
the output head are the baseline's own classes, so an A/B difference can only
come from sharing.  With recurrence_depth == 1 the model is exactly the
baseline (same parameter names, same init RNG stream, identical logits).

The recurrent working state is the residual stream itself (hidden-state
recurrence across *depth*).  There is no extra memory across sequence
positions beyond ordinary causal attention, and no persistent memory across
generated tokens beyond the KV cache.  Each application produces its own
keys/values, so the KV cache grows with effective depth even though weights
do not: see ARCHITECTURE_IMPLEMENTATION_EXPLAINED.txt.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .baseline_transformer import BaselineTransformer, BaselineTransformerConfig

RECURRENCE_ORDERS = ("cycle", "sequence")


@dataclass
class RecurrentTransformerConfig(BaselineTransformerConfig):
    """Baseline knobs plus the two recurrence knobs.  n_layers here means the
    number of *distinct* (independently parameterised) blocks."""
    recurrence_depth: int = 1
    recurrence_order: str = "cycle"


def recurrence_schedule(n_blocks: int, depth: int, order: str) -> list[int]:
    """Stored-block index for every application, in execution order."""
    # bool is an int subclass: True would silently mean depth 1 (audit DML-A08)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (n_blocks, depth)):
        raise ValueError(f"n_blocks and depth must be integers; got {n_blocks!r}, {depth!r}")
    if n_blocks < 1 or depth < 1:
        raise ValueError(f"need n_blocks >= 1 and depth >= 1; got {n_blocks}, {depth}")
    if order == "cycle":
        return [b for _ in range(depth) for b in range(n_blocks)]
    if order == "sequence":
        return [b for b in range(n_blocks) for _ in range(depth)]
    raise ValueError(f"recurrence_order must be one of {RECURRENCE_ORDERS}; got {order!r}")


class DomovoiRecurrentTransformer(BaselineTransformer):
    architecture_id = "domovoi_recurrent_v1"   # registry name; used in signatures

    def __init__(self, cfg: RecurrentTransformerConfig):
        # validate (raises) before allocating any weights
        schedule = recurrence_schedule(cfg.n_layers, cfg.recurrence_depth,
                                       cfg.recurrence_order)
        super().__init__(cfg)          # builds n_layers blocks, nothing more
        self._schedule = tuple(schedule)

    def application_schedule(self) -> list[int]:
        return list(self._schedule)

    @property
    def effective_depth(self) -> int:
        return len(self._schedule)

    def application_hidden_states(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """Residual stream after every block application (uncached, no head).

        Research diagnostic: lets experiments read out (or early-exit) the
        state after each recurrence iteration.  Element -1 fed through ln_f and
        lm_head reproduces forward(idx).logits; a test enforces that."""
        T = idx.size(1)
        if T > self.cfg.context_length:
            raise ValueError(f"Sequence length {T} exceeds context_length "
                             f"{self.cfg.context_length}.")
        x = self.wte(idx) + self.wpe(torch.arange(T, device=idx.device))
        states = []
        for block_idx in self._schedule:
            x, _ = self.blocks[block_idx](x)
            states.append(x)
        return states

    def param_breakdown(self) -> dict:
        b = super().param_breakdown()
        b.update({
            "unique_blocks": self.cfg.n_layers,
            "recurrence_depth": self.cfg.recurrence_depth,
            "recurrence_order": self.cfg.recurrence_order,
            "effective_depth": self.effective_depth,
            # parameter *uses* per token: a block applied k times counts k times
            "applied_blocks": b["per_block"] * self.effective_depth,
        })
        return b


def build_domovoi_recurrent(cfg) -> DomovoiRecurrentTransformer:
    """Build from a domovoi.config.ModelSection (or anything with the same
    attribute names).  Mirrors build_baseline_transformer, including the
    residual rescale that the baseline also applies only in its builder."""
    model_cfg = RecurrentTransformerConfig(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        dropout=cfg.dropout,
        tie_embeddings=cfg.tie_embeddings,
        recurrence_depth=cfg.recurrence_depth,
        recurrence_order=cfg.recurrence_order,
    )
    model = DomovoiRecurrentTransformer(model_cfg)
    model._rescale_residual_projections()
    return model
