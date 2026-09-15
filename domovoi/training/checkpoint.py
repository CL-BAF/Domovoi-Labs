"""Checkpoint save / load.

A checkpoint directory contains:

    model.safetensors  - model weights (safetensors: fast, safe, portable)
    optimizer.pt       - optimizer state
    scheduler.pt       - LR scheduler state (last step)
    rng.pt             - RNG states (python/numpy/torch/cuda + trainer extras)
    checkpoint.json    - step, tokens_seen, config, timestamps, git commit

Save is atomic-ish: we write to a `*.tmp` sibling then os.replace it into
place, so a crash mid-save can never leave a half-written checkpoint that
looks valid.
"""

from __future__ import annotations

import json
import os
import pathlib
import time

import torch
from safetensors.torch import save_file

from ..utils.seeding import capture_rng_state, restore_rng_state


class CheckpointError(RuntimeError):
    pass


def save_checkpoint(
    ckpt_dir: str | pathlib.Path,
    model,
    optimizer,
    scheduler,
    step: int,
    tokens_seen: int,
    config,
    extra: dict | None = None,
    rng_extra: dict | None = None,
) -> dict:
    """Save a checkpoint.  `rng_extra` can carry additional RNG states (e.g.
    the data-batch generator); they are stored alongside the standard RNG
    capture inside rng.pt so resume is bit-exact."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = ckpt_dir.parent / (ckpt_dir.name + ".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # safetensors refuses tensors that share storage (e.g. tied
    # lm_head.weight == wte.weight).  Keep the first key of each storage
    # alias; load_state_dict(strict=False) restores the tie via the shared
    # Parameter object.
    state, seen_ptrs = {}, set()
    for k, v in model.state_dict().items():
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs.add(ptr)
        state[k] = v.detach().cpu().contiguous()
    save_file(state, str(tmp_dir / "model.safetensors"))
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp_dir / "scheduler.pt")
    torch.save({"standard": capture_rng_state(), "extra": rng_extra or {}},
               tmp_dir / "rng.pt")

    meta = {
        "step": step,
        "tokens_seen": tokens_seen,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_architecture": config.model.architecture,
        "config": config.to_dict(),
        **(extra or {}),
    }
    (tmp_dir / "checkpoint.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # atomic directory swap: target removed first so os.replace is a plain
    # rename, which works on both POSIX and Windows
    if ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir)
    os.replace(tmp_dir, ckpt_dir)
    return meta


def load_checkpoint(
    ckpt_dir: str | pathlib.Path,
    model,
    optimizer=None,
    scheduler=None,
    restore_rng: bool = True,
) -> tuple[dict, dict]:
    """Load a checkpoint into `model` (required) and, if given, optimizer and
    scheduler.  Returns (checkpoint.json contents, full rng.pt contents).
    The standard RNG state is restored here; rng["extra"] is left for the
    caller (the trainer restores its data-batch generator from it)."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    if not (ckpt_dir / "checkpoint.json").is_file():
        raise CheckpointError(f"No checkpoint found at {ckpt_dir}")

    from safetensors.torch import load_file
    state = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # tied lm_head.weight does not appear in the file (shared storage);
    # everything else must match exactly.
    if unexpected:
        raise CheckpointError(f"Checkpoint has unexpected keys: {unexpected}")
    real_missing = [k for k in missing if k != "lm_head.weight"]
    if real_missing:
        raise CheckpointError(f"Checkpoint is missing keys: {real_missing}")

    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(ckpt_dir / "optimizer.pt", map_location="cpu", weights_only=False))
        # move optimizer state tensors to the model's device
        device = next(model.parameters()).device
        for st in optimizer.state.values():
            for k, v in st.items():
                if isinstance(v, torch.Tensor):
                    st[k] = v.to(device)
    if scheduler is not None:
        scheduler.load_state_dict(
            torch.load(ckpt_dir / "scheduler.pt", map_location="cpu", weights_only=False))
    rng = {"standard": None, "extra": {}}
    if restore_rng:
        rng = torch.load(ckpt_dir / "rng.pt", map_location="cpu", weights_only=False)
        if rng.get("standard"):
            restore_rng_state(rng["standard"])

    meta = json.loads((ckpt_dir / "checkpoint.json").read_text(encoding="utf-8"))
    return meta, rng


def build_model_from_checkpoint(ckpt_dir: str | pathlib.Path):
    """Rebuild the model exactly as the checkpoint was trained: read the
    config out of checkpoint.json, construct the architecture, load the
    weights.  Used by inference/eval/profiling so they never depend on a
    config file being present at a particular path."""
    from ..config import config_from_dict
    from ..models import build_model

    ckpt_dir = pathlib.Path(ckpt_dir)
    meta = json.loads((ckpt_dir / "checkpoint.json").read_text(encoding="utf-8"))
    cfg = config_from_dict(meta["config"])
    model = build_model(cfg.model)
    from safetensors.torch import load_file
    state = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or [k for k in missing if k != "lm_head.weight"]:
        raise CheckpointError(
            f"Checkpoint at {ckpt_dir} does not match the architecture it "
            f"was trained with (missing={missing}, unexpected={unexpected}).")
    return model, cfg, meta


class CheckpointManager:
    """Tracks checkpoints under an experiment directory: numbered
    `step_XXXXXXXX` dirs plus `latest.json` pointing at the newest one."""

    def __init__(self, root: str | pathlib.Path):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, step: int, **kwargs) -> pathlib.Path:
        name = f"step_{step:08d}"
        path = self.root / name
        save_checkpoint(path, **kwargs)
        (self.root / "latest.json").write_text(
            json.dumps({"step": step, "path": name}, indent=2), encoding="utf-8")
        return path

    def latest(self) -> pathlib.Path | None:
        latest = self.root / "latest.json"
        if not latest.is_file():
            return None
        info = json.loads(latest.read_text(encoding="utf-8"))
        path = self.root / info["path"]
        return path if (path / "checkpoint.json").is_file() else None

    def all(self) -> list[pathlib.Path]:
        return sorted(p for p in self.root.glob("step_*")
                      if (p / "checkpoint.json").is_file())