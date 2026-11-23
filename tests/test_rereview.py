"""Regression tests for the two findings of Codex's RE-review of the
post-audit fixes (POST_AUDIT_FIXES.txt, "Re-review round").

  RR1 (High)   checkpoint preflight must be read-only: a refused resume
               must not recover a *.bak checkpoint; a fresh run must see
               step directories even when latest.json is missing; data
               provenance is checked on resume; one writer at a time.
  RR2 (Medium) profiling must report weight and KV dtypes separately and
               estimate KV bytes in the KV dtype (FP32 weights + bf16
               autocast).

The .bak recovery, missing-index fresh run and autocast KV-estimate cases
were reproduced before the fix (post_audit/repro_rereview_before.txt); the
stale-index, data-provenance and writer-lock tests cover protections that
did not exist before.  CPU only, synthetic data.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib

import pytest
import torch

from domovoi.config import clone_config, config_from_dict
from domovoi.models import build_model
from domovoi.profiling.profiler import profile_model
from domovoi.training import CheckpointError, CheckpointManager, Trainer
from domovoi.training.checkpoint import scan_checkpoints


def _tree(root: pathlib.Path) -> dict:
    """path -> sha256 for files, 'dir' for directories (renames count)."""
    return {str(p.relative_to(root)): ("dir" if p.is_dir() else
                                       hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*"))}


def _cfg(smoke_dataset, **model):
    d = json.loads(json.dumps(smoke_dataset["config_dict"]))
    d["model"].update({"architecture": "domovoi_recurrent_v1", "recurrence_depth": 2, **model})
    d["training"].update(steps=2, checkpoint_interval=2, eval_interval=2)
    return d


def _run(smoke_dataset, name="exp"):
    d = _cfg(smoke_dataset)
    exp = smoke_dataset["tmp_path"] / name
    Trainer(config_from_dict(d), experiment_dir=exp).train()
    return d, exp


def _crash_swap(exp):
    """Simulate a crash between the two renames of a checkpoint save."""
    step = exp / "checkpoints" / "step_00000002"
    os.replace(step, step.parent / (step.name + ".bak"))
    return step


# ---- RR1: read-only preflight ------------------------------------------

def test_refused_resume_does_not_recover_bak(smoke_dataset):
    d, exp = _run(smoke_dataset)
    _crash_swap(exp)
    before = _tree(exp)
    bad = json.loads(json.dumps(d))
    bad["model"]["recurrence_order"] = "sequence"
    with pytest.raises(CheckpointError, match="Cannot resume"):
        Trainer(config_from_dict(bad), experiment_dir=exp, resume=True)
    assert _tree(exp) == before                      # .bak still where it was


def test_valid_resume_recovers_bak_after_validation(smoke_dataset):
    d, exp = _run(smoke_dataset)
    step = _crash_swap(exp)
    t = Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    assert t.start_step == 2
    assert (step / "checkpoint.json").is_file()
    assert not step.with_name(step.name + ".bak").exists()


def test_scan_is_read_only(smoke_dataset):
    _, exp = _run(smoke_dataset)
    _crash_swap(exp)
    before = _tree(exp)
    inv = scan_checkpoints(exp / "checkpoints")
    assert inv["latest"].recover_from is not None and inv["any"]
    assert _tree(exp) == before


@pytest.mark.parametrize("damage", ["no_index", "only_bak"])
def test_fresh_run_sees_checkpoints_without_index(smoke_dataset, damage):
    d, exp = _run(smoke_dataset)
    (exp / "checkpoints" / "latest.json").unlink()
    if damage == "only_bak":
        _crash_swap(exp)
    before = _tree(exp)
    with pytest.raises(CheckpointError, match="already holds checkpoints"):
        Trainer(config_from_dict(d), experiment_dir=exp)
    assert _tree(exp) == before


def test_resume_refuses_to_guess_without_index(smoke_dataset):
    d, exp = _run(smoke_dataset)
    (exp / "checkpoints" / "latest.json").unlink()
    with pytest.raises(CheckpointError, match="refusing to guess"):
        Trainer(config_from_dict(d), experiment_dir=exp, resume=True)


def test_resume_refuses_newer_unindexed_checkpoint(smoke_dataset):
    d, exp = _run(smoke_dataset)
    ck = exp / "checkpoints"
    import shutil
    shutil.copytree(ck / "step_00000002", ck / "step_00000004")
    with pytest.raises(CheckpointError, match="newer complete"):
        Trainer(config_from_dict(d), experiment_dir=exp, resume=True)


def test_checkpoint_records_data_provenance(smoke_dataset):
    _, exp = _run(smoke_dataset)
    meta = json.loads((exp / "checkpoints" / "step_00000002" / "checkpoint.json")
                      .read_text(encoding="utf-8"))
    prov = meta["data_provenance"]
    assert len(prov["tokenizer_sha256"]) == 64 and prov["train_tokens"] > 0


def _edit_meta(exp, fn):
    p = exp / "checkpoints" / "step_00000002" / "checkpoint.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    fn(meta)
    p.write_text(json.dumps(meta), encoding="utf-8")


def test_resume_refuses_changed_data(smoke_dataset):
    d, exp = _run(smoke_dataset)
    _edit_meta(exp, lambda m: m["data_provenance"].update(tokenizer_sha256="0" * 64))
    before = _tree(exp)
    with pytest.raises(CheckpointError, match="tokenizer_sha256"):
        Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    assert _tree(exp) == before


def test_resume_of_pre_provenance_checkpoint_warns(smoke_dataset, capsys):
    d, exp = _run(smoke_dataset)
    _edit_meta(exp, lambda m: m.pop("data_provenance"))
    Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    assert "cannot be verified" in capsys.readouterr().out


def test_second_writer_refused_and_lock_released(smoke_dataset):
    d, exp = _run(smoke_dataset)
    assert not (exp / Trainer.WRITER_LOCK).exists()         # released after success
    (exp / Trainer.WRITER_LOCK).write_text("pid=1 host=other", encoding="utf-8")
    before = _tree(exp)
    # round 3: refused at construction, before loading or recovering anything
    with pytest.raises(CheckpointError, match="locked by another writer"):
        Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    assert _tree(exp) == before                       # foreign lock untouched


def test_lock_released_on_exception(smoke_dataset, monkeypatch):
    d = _cfg(smoke_dataset)
    exp = smoke_dataset["tmp_path"] / "boom"
    t = Trainer(config_from_dict(d), experiment_dir=exp)
    monkeypatch.setattr(t, "_train_unlocked", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        t.train()
    assert not (exp / Trainer.WRITER_LOCK).exists()


def test_manager_latest_still_recovers(smoke_dataset):
    """Inference/eval callers keep the documented recovering latest()."""
    _, exp = _run(smoke_dataset)
    step = _crash_swap(exp)
    assert CheckpointManager(exp / "checkpoints").latest() == step
    assert (step / "checkpoint.json").is_file()


# ---- RR2: weight vs KV dtype under autocast ------------------------------

class _ByteTok:
    def encode(self, s):
        return list(s.encode())

    def decode(self, t):
        return bytes(i % 256 for i in t).decode(errors="replace")


@pytest.mark.parametrize("autocast", [False, True])
def test_profile_separates_weight_and_kv_dtype(tiny_cfg, autocast):
    cfg = clone_config(tiny_cfg, model={"architecture": "domovoi_recurrent_v1",
                                        "recurrence_depth": 2})
    torch.manual_seed(0)
    m = build_model(cfg.model).eval()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        prof = profile_model(m, cfg, _ByteTok(), torch.device("cpu"),
                             max_new_tokens=4, context_sweep=False)
    kv = "torch.bfloat16" if autocast else "torch.float32"
    assert prof["weight_dtype"] == "torch.float32" and prof["kv_dtype"] == kv
    assert prof["autocast"] is autocast
    inf, est = prof["inference"], prof["analytic_estimates"]
    assert inf["kv_dtype"] == kv and inf["weight_dtype"] == "torch.float32"
    assert est["bytes_per_param"] == 4 and est["bytes_per_kv"] == (2 if autocast else 4)
    # the KV estimate now equals the bytes the generated cache really holds
    assert inf["kv_cache_logical_bytes"] == \
        est["kv_cache_bytes_per_token"] * inf["final_sequence_length"]
    assert est["weight_bytes"] == prof["measured_parameter_bytes"]
