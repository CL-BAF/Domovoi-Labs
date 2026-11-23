"""Regression tests for Codex's fourth review (POST_AUDIT_FIXES.txt, "Round 4").

  R4-1 (High)   Ctrl+C inside an optimizer update must not save a rescue
                checkpoint whose weights/AdamW state are ahead of its step.
  R4-2 (High)   A non-finite loss, gradient or validation loss makes the run
                FAILED, before the update where possible, with no checkpoint.
  R4-3 (Medium) Resume refuses changes to trajectory-defining training
                settings; only a longer schedule is possible, by explicit
                opt-in, and it is recorded as a protocol deviation.
  R4-4 (Medium) Metric rows replayed after a crash are reconciled per
                session instead of duplicated in the learning curve.

The integration tests compare an interrupted-and-resumed run with an
uninterrupted one on final weights, sampler (data-generator) position and
reconciled metric history, as Codex suggested.  All four defects were
reproduced before the fix (post_audit/repro_round4_before.txt).  CPU only.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

import pytest
import torch

from domovoi.config import config_from_dict
from domovoi.experiments import read_metrics, read_sessions
from domovoi.training import CheckpointError, CheckpointManager, Trainer
from domovoi.training.trainer import TRAJECTORY_FIELDS


def _tree(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): ("dir" if p.is_dir() else
                                       hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*"))}


def _cfg(smoke_dataset, **training):
    d = json.loads(json.dumps(smoke_dataset["config_dict"]))
    d["model"].update({"architecture": "domovoi_recurrent_v1", "recurrence_depth": 2})
    d["training"].update({"steps": 6, "checkpoint_interval": 2, "eval_interval": 100,
                          "log_interval": 1, **training})
    return config_from_dict(d)


def _reference(smoke_dataset, steps):
    """Uninterrupted run: (trainer, reconciled metrics, final weights)."""
    exp = smoke_dataset["tmp_path"] / f"ref{steps}"
    t = Trainer(_cfg(smoke_dataset, steps=steps), experiment_dir=exp)
    t.train()
    return t, read_metrics(exp), {k: v.clone() for k, v in t.model.state_dict().items()}


def _assert_same_run(ref, ref_rows, ref_weights, t, rows):
    assert [r["step"] for r in rows] == [r["step"] for r in ref_rows]
    for a, b in zip(ref_rows, rows):
        assert abs(a["train_loss"] - b["train_loss"]) < 1e-6, (a, b)
        assert a["tokens_seen"] == b["tokens_seen"]
    for k, v in ref_weights.items():
        assert torch.allclose(v, t.model.state_dict()[k], atol=1e-6), k
    # sampler position: the data generator is in the same state
    assert torch.equal(ref.data_generator.get_state(), t.data_generator.get_state())


# ---- R4-1: interrupt inside the optimizer update ---------------------------

def test_interrupt_inside_update_writes_no_rescue_and_resume_matches(smoke_dataset):
    ref, ref_rows, ref_w = _reference(smoke_dataset, 6)
    exp = smoke_dataset["tmp_path"] / "torn"
    t = Trainer(_cfg(smoke_dataset), experiment_dir=exp)
    real, n = t.optimizer.step, {"k": 0}

    def step_then_interrupt(*a, **k):
        out = real(*a, **k)
        n["k"] += 1
        if n["k"] == 3:                  # the 3rd update HAS been applied
            raise KeyboardInterrupt
        return out

    t.optimizer.step = step_then_interrupt
    s = t.train()
    assert s["status"] == "INTERRUPTED" and s["final_val_loss"] is None
    assert s["recovery_checkpoint"] == "step_00000002"
    assert s["steps_completed"] == 2 and s["tokens_trained"] == 2 * t.tokens_per_step()
    ck = exp / "checkpoints" / "step_00000002"
    meta = json.loads((ck / "checkpoint.json").read_text(encoding="utf-8"))
    assert meta["step"] == 2 and "interrupted" not in meta     # the committed one
    adam = torch.load(ck / "optimizer.pt", weights_only=False)["state"]
    assert {float(s["step"]) for s in adam.values()} == {2.0}  # was {3.0} before
    # resume from the committed checkpoint reproduces the uninterrupted run
    r = Trainer(_cfg(smoke_dataset), experiment_dir=exp, resume=True)
    assert r.start_step == 2
    r.train()
    _assert_same_run(ref, ref_rows, ref_w, r, read_metrics(exp))


# ---- R4-2: non-finite values --------------------------------------------------

def test_nan_loss_fails_before_the_update(smoke_dataset):
    exp = smoke_dataset["tmp_path"] / "nanloss"
    t = Trainer(_cfg(smoke_dataset, steps=2), experiment_dir=exp)
    real, c = t.model.forward, {"k": 0}

    def nan_forward(*a, **k):
        out = real(*a, **k)
        c["k"] += 1
        return out._replace(loss=out.loss * float("nan")) if c["k"] == 2 else out

    t.model.forward = nan_forward
    s = t.train()
    assert s["status"] == "FAILED" and "non-finite training loss" in s["failure"]
    assert s["final_val_loss"] is None
    assert CheckpointManager(exp / "checkpoints").all() == []      # nothing saved
    assert all(torch.isfinite(p).all() for p in t.model.parameters())


def test_nan_gradient_fails_before_the_update(smoke_dataset):
    exp = smoke_dataset["tmp_path"] / "nangrad"
    t = Trainer(_cfg(smoke_dataset, steps=2), experiment_dir=exp)
    real, c = t.model.forward, {"k": 0}
    snapshot = {}

    def poisoned_forward(*a, **k):
        c["k"] += 1
        if c["k"] == 2:          # step 2: weights are post-step-1; poison grads
            snapshot.update({n: p.detach().clone() for n, p in t.model.named_parameters()})
            t.model.wte.weight.register_hook(lambda g: g * float("nan"))
        return real(*a, **k)

    t.model.forward = poisoned_forward
    s = t.train()
    assert s["status"] == "FAILED" and "gradient norm" in s["failure"]
    # the update was NOT applied: weights are exactly the post-step-1 weights
    for n, p in t.model.named_parameters():
        assert torch.equal(p, snapshot[n]), n


def test_nan_validation_fails_and_is_recorded(smoke_dataset):
    exp = smoke_dataset["tmp_path"] / "nanval"
    t = Trainer(_cfg(smoke_dataset, steps=4, eval_interval=2), experiment_dir=exp)
    t.evaluate = lambda n: {"val_loss": float("nan"), "val_ppl": float("nan"),
                            "n_batches": n}
    s = t.train()
    assert s["status"] == "FAILED" and "validation" in s["failure"]
    rows = read_metrics(exp, "val_metrics.jsonl")
    assert rows[-1]["step"] == 2 and math.isnan(rows[-1]["val_loss"])
    assert CheckpointManager(exp / "checkpoints").all() == []


# ---- R4-3: trajectory-defining settings ----------------------------------------

_CHANGES = {"steps": 8, "batch_size": 3, "gradient_accumulation": 2, "context_length": 16,
            "learning_rate": 5e-4, "min_lr_ratio": 0.2, "weight_decay": 0.0,
            "betas": [0.8, 0.95], "grad_clip": 0.5, "warmup_steps": 2,
            "precision": "bf16", "seed": 7, "eval_batches": 3,
            "sampler": "disjoint_segments", "pilot_stop_checks": True,
            "pilot_calibrated_seconds": 180.0, "deterministic_cuda": True,
            "verify_checkpoint_writes": True}


def test_every_trajectory_field_is_covered():
    assert set(_CHANGES) == set(TRAJECTORY_FIELDS)


@pytest.mark.parametrize("field", sorted(_CHANGES))
def test_resume_refuses_changed_training_protocol(smoke_dataset, field):
    exp = smoke_dataset["tmp_path"] / "proto"
    initial = {"steps": 2}
    if field == "pilot_calibrated_seconds":
        initial.update(pilot_stop_checks=True, pilot_calibrated_seconds=120.0)
    Trainer(_cfg(smoke_dataset, **initial), experiment_dir=exp).train()
    before = _tree(exp)
    with pytest.raises(CheckpointError, match="training protocol"):
        Trainer(_cfg(smoke_dataset, **{**initial, field: _CHANGES[field]}),
                experiment_dir=exp, resume=True)
    assert _tree(exp) == before


def test_interval_changes_are_allowed(smoke_dataset):
    exp = smoke_dataset["tmp_path"] / "intervals"
    Trainer(_cfg(smoke_dataset, steps=2), experiment_dir=exp).train()
    t = Trainer(_cfg(smoke_dataset, steps=2, log_interval=5, eval_interval=7,
                     checkpoint_interval=9), experiment_dir=exp, resume=True)
    assert t.start_step == 2
    t.close()


def test_extend_steps_is_opt_in_and_recorded(smoke_dataset):
    exp = smoke_dataset["tmp_path"] / "extend"
    Trainer(_cfg(smoke_dataset, steps=2), experiment_dir=exp).train()
    with pytest.raises(CheckpointError, match="training protocol"):
        Trainer(_cfg(smoke_dataset, steps=3), experiment_dir=exp, resume=True)
    with pytest.raises(CheckpointError, match="batch_size"):     # only steps may change
        Trainer(_cfg(smoke_dataset, steps=3, batch_size=3), experiment_dir=exp,
                resume=True, extend_steps=True)
    with pytest.raises(CheckpointError, match="steps"):          # and only upward
        Trainer(_cfg(smoke_dataset, steps=1), experiment_dir=exp, resume=True,
                extend_steps=True)
    s = Trainer(_cfg(smoke_dataset, steps=3), experiment_dir=exp, resume=True,
                extend_steps=True).train()
    dev = [{"field": "steps", "from": 2, "to": 3, "at_step": 2}]
    assert s["protocol_deviations"] == dev
    meta = json.loads((exp / "checkpoints" / "step_00000003" / "checkpoint.json")
                      .read_text(encoding="utf-8"))
    assert meta["protocol_deviations"] == dev                   # carried on disk


# ---- R4-4: metric history across a crash ---------------------------------------

def test_crash_then_resume_reconciles_metrics_and_matches(smoke_dataset):
    """Codex's scenario: the save at step 4 fails after rows 1-4 were logged;
    the committed checkpoint is step 2; resume replays 3 and 4."""
    ref, ref_rows, ref_w = _reference(smoke_dataset, 4)
    exp = smoke_dataset["tmp_path"] / "crash"
    t = Trainer(_cfg(smoke_dataset, steps=4), experiment_dir=exp)
    real = t._save_checkpoint

    def failing_save(step, extra=None):
        if step == 4:
            raise OSError("simulated save failure")
        return real(step, extra)

    t._save_checkpoint = failing_save
    with pytest.raises(OSError):
        t.train()
    r = Trainer(_cfg(smoke_dataset, steps=4), experiment_dir=exp, resume=True)
    r.train()
    raw = [json.loads(l)["step"] for l in
           (exp / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert raw == [1, 2, 3, 4, 3, 4]                    # append-only evidence kept
    rows = read_metrics(exp)
    assert [(x["step"], x["attempt"]) for x in rows] == [(1, 1), (2, 1), (3, 2), (4, 2)]
    assert [(s["attempt"], s["start_step"]) for s in read_sessions(exp)] == [(1, 0), (2, 2)]
    _assert_same_run(ref, ref_rows, ref_w, r, rows)


def test_legacy_metrics_without_sessions_read_unchanged(tmp_path):
    rows = [{"step": i, "train_loss": 1.0 / i} for i in (1, 2, 3)]
    (tmp_path / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert read_metrics(tmp_path) == rows
