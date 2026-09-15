"""Optimizer and LR schedule construction.

Standard AdamW with weight decay applied to matrices only (biases and
LayerNorm parameters are excluded, as is conventional).  Cosine LR decay
with linear warmup from 0 to learning_rate over warmup_steps, then cosine
decay to learning_rate * min_lr_ratio at training.steps.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def build_optimizer(model: nn.Module, lr: float, weight_decay: float,
                    betas: tuple) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1-d parameters (biases, layernorm gains) are not decayed.
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas))


def build_scheduler(optimizer, warmup_steps: int, total_steps: int,
                    min_lr_ratio: float) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine to lr * min_lr_ratio."""
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (1.0 - min_lr_ratio) * cos + min_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0