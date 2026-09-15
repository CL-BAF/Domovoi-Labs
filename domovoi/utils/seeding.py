"""Deterministic seeding.

`set_seed` seeds every RNG that the pipeline touches.  `capture_rng_state`
/ `restore_rng_state` allow checkpoints to record RNG state so a resumed
run continues the exact same sampling stream instead of silently diverging.
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, torch (CPU and all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    """Snapshot every RNG state we can restore later."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [
            torch.cuda.get_rng_state(i) for i in range(torch.cuda.device_count())
        ]
    return state


def restore_rng_state(state: dict) -> None:
    """Restore a state captured by capture_rng_state().  Missing keys (e.g.
    CUDA state captured on a CUDA machine then restored on CPU) are skipped
    with a warning-free best effort: the run continues deterministically on
    the devices that exist."""
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu() if isinstance(state["torch_cpu"], torch.Tensor) else state["torch_cpu"])
    cuda_states = state.get("torch_cuda")
    if cuda_states and torch.cuda.is_available():
        for i, s in enumerate(cuda_states):
            if i < torch.cuda.device_count():
                torch.cuda.set_rng_state(s if isinstance(s, torch.ByteTensor) else torch.tensor(s, dtype=torch.uint8), i)
            else:
                break