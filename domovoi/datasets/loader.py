"""Batch loading over packed token files.

Training batches are drawn at random offsets (GPT-2 style) using an explicit
`torch.Generator` owned by the trainer, so the stream is reproducible from
the seed and — crucially — **restorable across checkpoints** (the generator
state is saved in the checkpoint).

Validation batches use a fixed, constant seed so validation loss is
comparable across steps, checkpoints and runs.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch


class PackedTokenData:
    """Memory-mapped uint16 packed token file."""

    def __init__(self, bin_path: str | pathlib.Path):
        self.path = str(bin_path)
        self.data = np.memmap(self.path, dtype=np.uint16, mode="r")
        self.n_tokens = int(self.data.size)

    def __len__(self) -> int:
        return self.n_tokens

    def get_batch(
        self,
        batch_size: int,
        block_size: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random-offset batch of (x, y) next-token pairs."""
        if self.n_tokens < block_size + 1:
            raise RuntimeError(
                f"Packed token file {self.path} has only {self.n_tokens} tokens; "
                f"need at least block_size+1 = {block_size + 1}.")
        ix = torch.randint(self.n_tokens - block_size - 1, (batch_size,),
                           generator=generator)
        x = torch.stack([
            torch.from_numpy(self.data[i:i + block_size].astype(np.int64))
            for i in ix.tolist()
        ])
        y = torch.stack([
            torch.from_numpy(self.data[i + 1:i + 1 + block_size].astype(np.int64))
            for i in ix.tolist()
        ])
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), \
                   y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def fixed_val_generator(device: torch.device) -> torch.Generator:
    """Constant-seed generator: validation batches are the same on every
    call, machine and run.  (Note: CUDA RNG and CPU RNG produce different
    streams for the same seed; validation loss is still deterministic per
    device, which is what comparability across steps requires.)"""
    g = torch.Generator()
    g.manual_seed(0)
    return g


def estimate_tokens(trained_steps: int, batch_size: int, block_size: int,
                    grad_accum: int) -> int:
    """Tokens of training signal consumed: each optimizer step processes
    batch_size * block_size * grad_accum forward passes worth of tokens."""
    return trained_steps * batch_size * block_size * grad_accum