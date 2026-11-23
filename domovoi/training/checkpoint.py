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
import re
import shutil
import time
from typing import NamedTuple, Optional

import torch
from safetensors.torch import save_file

from ..models.signature import config_signature, model_signature, signature_differences
from ..utils.seeding import capture_rng_state, restore_rng_state


class CheckpointError(RuntimeError):
    pass


def read_checkpoint_meta(ckpt_dir: str | pathlib.Path) -> dict:
    """checkpoint.json of a checkpoint dir, without touching any tensors."""
    ckpt_dir = pathlib.Path(ckpt_dir)
    if not (ckpt_dir / "checkpoint.json").is_file():
        raise CheckpointError(f"No checkpoint found at {ckpt_dir}")
    return json.loads((ckpt_dir / "checkpoint.json").read_text(encoding="utf-8"))


def saved_signature(meta: dict, ckpt_dir) -> dict:
    """The architecture signature a checkpoint was saved under.  Checkpoints
    written before signatures existed (e.g. the DML-0001 era) are read from
    their stored model config, with absent recurrence keys defaulting to the
    conventional depth-1 schedule.  When an explicit signature is present it
    must agree with the stored config: a disagreement means the meaning of a
    recurrence setting changed in code since the save."""
    model_cfg = (meta.get("config") or {}).get("model")
    if model_cfg is None:
        raise CheckpointError(
            f"Checkpoint at {ckpt_dir} records no model config, so the "
            f"computation graph of its weights cannot be verified.")
    from_config = config_signature(model_cfg)
    stored = meta.get("model_signature")
    if stored is not None and stored != from_config:
        diffs = signature_differences(stored, from_config, "saved signature",
                                      "signature re-derived from saved config")
        raise CheckpointError(
            f"Checkpoint at {ckpt_dir} is internally inconsistent; the stored "
            f"schedule no longer matches what its config means in this code "
            f"version:\n  " + "\n  ".join(diffs))
    return stored if stored is not None else from_config


def verify_checkpoint_architecture(meta: dict, model, ckpt_dir) -> None:
    """Refuse to put checkpoint weights into a model with a different
    computation graph, even when every tensor shape matches (DML-A01)."""
    diffs = signature_differences(saved_signature(meta, ckpt_dir), model_signature(model))
    if diffs:
        raise CheckpointError(
            f"Checkpoint at {ckpt_dir} was saved by a different architecture "
            f"than the model it is being loaded into; its weights would be "
            f"interpreted under a different computation graph:\n  "
            + "\n  ".join(diffs)
            + "\nTo resume, use the checkpoint's own model config (stored in "
              "checkpoint.json under config.model). To train a different "
              "architecture or schedule, start a new experiment. For "
              "inference, use build_model_from_checkpoint(), which rebuilds "
              "the saved architecture.")


def _check_loaded_keys(missing, unexpected, model, ckpt_dir) -> None:
    """Only a TIED lm_head.weight may be absent from the file (it shares
    storage with wte.weight).  For an untied model a missing head would
    survive as random init (audit DML-A05)."""
    if unexpected:
        raise CheckpointError(f"Checkpoint at {ckpt_dir} has unexpected keys: {unexpected}")
    waivable = {"lm_head.weight"} if model.cfg.tie_embeddings else set()
    real_missing = [k for k in missing if k not in waivable]
    if real_missing:
        raise CheckpointError(f"Checkpoint at {ckpt_dir} is missing keys: {real_missing}")


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
        # the computation graph these weights belong to, including the
        # explicit block-application schedule; verified on every load
        "model_signature": model_signature(model),
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
    # architecture/schedule check BEFORE any tensor reaches the model
    meta = read_checkpoint_meta(ckpt_dir)
    verify_checkpoint_architecture(meta, model, ckpt_dir)

    from safetensors.torch import load_file
    state = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    _check_loaded_keys(missing, unexpected, model, ckpt_dir)

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

    return meta, rng


def build_model_from_checkpoint(ckpt_dir: str | pathlib.Path):
    """Rebuild the model exactly as the checkpoint was trained: read the
    config out of checkpoint.json, construct the architecture, load the
    weights.  Used by inference/eval/profiling so they never depend on a
    config file being present at a particular path."""
    from ..config import config_from_dict
    from ..models import build_model

    ckpt_dir = pathlib.Path(ckpt_dir)
    meta = read_checkpoint_meta(ckpt_dir)
    cfg = config_from_dict(meta["config"])
    model = build_model(cfg.model)
    # the rebuilt model must run the schedule recorded at save time (guards
    # against a code change in what a recurrence setting means)
    verify_checkpoint_architecture(meta, model, ckpt_dir)
    from safetensors.torch import load_file
    state = load_file(str(ckpt_dir / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    _check_loaded_keys(missing, unexpected, model, ckpt_dir)
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
        """Newest checkpoint per latest.json, RECOVERING a crashed swap (a
        `*.bak` sibling is renamed back into place).  This writes; callers
        that must not modify anything before validating (Trainer resume) use
        scan_checkpoints() + recover_checkpoint() instead."""
        found = scan_checkpoints(self.root)["latest"]
        if found is None:
            return None
        if found.recover_from is not None:
            recover_checkpoint(found)
        return found.path

    def all(self) -> list[pathlib.Path]:
        return sorted(p for p in self.root.glob("step_*")
                      if (p / "checkpoint.json").is_file())


class LatestCheckpoint(NamedTuple):
    path: pathlib.Path                    # where the checkpoint belongs (step_XXXXXXXX)
    read_from: pathlib.Path               # where its files are right now
    recover_from: Optional[pathlib.Path]  # *.bak to rename into `path`, if any
    step: int


def _step_of(name: str) -> Optional[int]:
    m = re.fullmatch(r"step_(\d+)(?:\.bak|\.tmp)?", name)
    return int(m.group(1)) if m else None


def scan_checkpoints(root: Optional[str | pathlib.Path]) -> dict:
    """READ-ONLY inventory of a checkpoints directory (never creates, moves
    or deletes anything; Codex re-review).  Returns
      latest    LatestCheckpoint per latest.json, or None
      complete  step dirs holding a checkpoint.json
      bak/tmp   leftover *.bak / *.tmp step dirs of an interrupted save
      other     step_* entries without a checkpoint.json
      index     True if latest.json exists
      any       True if ANY trace of a previous run is present
    Step directories are listed independently of latest.json, so a missing
    or stale index cannot hide existing checkpoints."""
    inv = {"latest": None, "complete": [], "bak": [], "tmp": [], "other": [],
           "index": False, "any": False}
    if root is None or not pathlib.Path(root).is_dir():
        return inv
    root = pathlib.Path(root)
    for p in sorted(root.iterdir()):
        if _step_of(p.name) is None:
            continue
        has_meta = (p / "checkpoint.json").is_file()
        if p.name.endswith(".bak"):
            inv["bak"].append(p)
        elif p.name.endswith(".tmp"):
            inv["tmp"].append(p)
        elif has_meta:
            inv["complete"].append(p)
        else:
            inv["other"].append(p)
    index = root / "latest.json"
    inv["index"] = index.is_file()
    if inv["index"]:
        target = root / json.loads(index.read_text(encoding="utf-8"))["path"]
        bak = target.parent / (target.name + ".bak")
        step = _step_of(target.name)
        if (target / "checkpoint.json").is_file():
            inv["latest"] = LatestCheckpoint(target, target, None, step)
        elif (bak / "checkpoint.json").is_file():
            # a crash between the two renames of a save left it aside
            inv["latest"] = LatestCheckpoint(target, bak, bak, step)
    inv["any"] = bool(inv["index"] or inv["complete"] or inv["bak"]
                      or inv["tmp"] or inv["other"])
    return inv


def recover_checkpoint(found: LatestCheckpoint) -> pathlib.Path:
    """Rename a crashed swap's *.bak back into place.  Call only AFTER the
    checkpoint read from `found.read_from` has been validated."""
    if found.recover_from is not None:
        os.replace(found.recover_from, found.path)
    return found.path