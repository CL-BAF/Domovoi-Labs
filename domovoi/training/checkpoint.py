"""Checkpoint save / load.

A checkpoint directory contains:

    model.safetensors  - model weights (safetensors: fast, safe, portable)
    optimizer.pt       - optimizer state
    scheduler.pt       - LR scheduler state (last step)
    scaler.pt          - torch.amp.GradScaler state (present when fp16 AMP is
                         used; restored so fp16 resume stays bit-exact)
    rng.pt             - RNG states (python/numpy/torch/cuda + trainer extras)
    checkpoint.json    - step, tokens_seen, config, timestamps, git commit

Save is crash-safe: we write to a `*.tmp` sibling, rename the previous
version aside to `*.bak`, then os.replace the new one into place.  A crash
mid-save can never leave a half-written checkpoint that looks valid; the
worst case is a `*.bak` sibling that CheckpointManager.latest() recovers from.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
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
    scaler=None,
) -> dict:
    """Save a checkpoint.  `rng_extra` can carry additional RNG states (e.g.
    the data-batch generator); they are stored alongside the standard RNG
    capture inside rng.pt so resume is bit-exact.  `scaler` (a
    torch.amp.GradScaler) is persisted as scaler.pt so fp16 resume restores
    the loss scale — a restarted scale changes which steps overflow and
    would diverge the trajectory."""
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
    if scaler is not None:
        torch.save(scaler.state_dict(), tmp_dir / "scaler.pt")
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

    # crash-safe directory swap: rename the previous version aside (a plain
    # rename, reversible), move the new one into place, then delete the old.
    # A crash between the two renames leaves a `.bak` sibling instead of
    # destroying the previous checkpoint; latest() recovers from that.
    bak_dir = ckpt_dir.parent / (ckpt_dir.name + ".bak")
    if bak_dir.exists():
        shutil.rmtree(bak_dir)
    if ckpt_dir.exists():
        os.replace(ckpt_dir, bak_dir)
    os.replace(tmp_dir, ckpt_dir)
    if bak_dir.exists():
        shutil.rmtree(bak_dir)
    return meta


def load_checkpoint(
    ckpt_dir: str | pathlib.Path,
    model,
    optimizer=None,
    scheduler=None,
    restore_rng: bool = True,
    scaler=None,
) -> tuple[dict, dict]:
    """Load a checkpoint into `model` (required) and, if given, optimizer,
    scheduler and GradScaler.  Returns (checkpoint.json contents, full
    rng.pt contents).  The rng.pt contents are ALWAYS returned so the caller
    can restore extra RNG states (the data-batch generator); `restore_rng`
    only controls whether the standard RNG state is applied here.  With
    restore_rng=False the caller is responsible for restoring
    rng["standard"] itself.  scaler.pt (written by newer checkpoints) is
    loaded when present; checkpoints from before scaler persistence simply
    leave the scaler at its fresh defaults."""
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
    if scaler is not None and (ckpt_dir / "scaler.pt").is_file():
        scaler_state = torch.load(ckpt_dir / "scaler.pt", map_location="cpu",
                                  weights_only=False)
        # disabled scalers persist an empty dict (and loading one is a no-op);
        # an enabled scaler with no stored state (fp32 checkpoint resumed as
        # fp16) legitimately has nothing to restore
        if scaler_state:
            scaler.load_state_dict(scaler_state)
    rng_pt = ckpt_dir / "rng.pt"
    rng = torch.load(rng_pt, map_location="cpu", weights_only=False) \
        if rng_pt.is_file() else {"standard": None, "extra": {}}
    if restore_rng and rng.get("standard"):
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
        save_checkpoint(path, step=step, **kwargs)
        (self.root / "latest.json").write_text(
            json.dumps({"step": step, "path": name}, indent=2), encoding="utf-8")
        return path

    def latest(self) -> pathlib.Path | None:
        latest = self.root / "latest.json"
        if not latest.is_file():
            return None
        info = json.loads(latest.read_text(encoding="utf-8"))
        path = self.root / info["path"]
        if not (path / "checkpoint.json").is_file():
            # a crash between the two renames of a save left the previous
            # version aside as `*.bak` — recover it
            bak = path.parent / (path.name + ".bak")
            if bak.is_dir() and (bak / "checkpoint.json").is_file():
                os.replace(bak, path)
            else:
                return None
        return path

    def all(self) -> list[pathlib.Path]:
        return sorted(p for p in self.root.glob("step_*")
                      if (p / "checkpoint.json").is_file())