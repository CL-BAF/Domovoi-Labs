"""Checkpoint save / load / resume, including bit-exact resume through the
real Trainer.train() API."""

import json

import pytest
import torch

from domovoi.config import config_from_dict
from domovoi.models import build_model
from domovoi.training import (CheckpointManager, CheckpointError, Trainer,
                              build_optimizer, build_scheduler,
                              load_checkpoint, save_checkpoint)


def _model_and_optim(tiny_cfg):
    model = build_model(tiny_cfg.model)
    opt = build_optimizer(model, 1e-3, 0.1, (0.9, 0.95))
    sched = build_scheduler(opt, 1, 10, 0.1)
    return model, opt, sched


def test_save_load_roundtrip(tiny_cfg, tmp_path):
    model, opt, sched = _model_and_optim(tiny_cfg)
    x = torch.randint(0, tiny_cfg.model.vocab_size, (2, 8))
    out = model(x, targets=x)
    out.loss.backward()
    opt.step()

    save_checkpoint(tmp_path / "ckpt", model, opt, sched, step=5,
                     tokens_seen=1000, config=tiny_cfg)

    model2, opt2, sched2 = _model_and_optim(tiny_cfg)
    meta, rng = load_checkpoint(tmp_path / "ckpt", model2, opt2, sched2,
                               restore_rng=False)
    assert meta["step"] == 5 and meta["tokens_seen"] == 1000

    model.eval()
    model2.eval()
    with torch.no_grad():
        a = model(x).logits
        b = model2(x).logits
    assert torch.allclose(a, b, atol=1e-6), "weights differ after load"


def test_load_missing_dir_raises(tiny_cfg, tmp_path):
    model, opt, sched = _model_and_optim(tiny_cfg)
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "nope", model)


def test_manager_latest(tiny_cfg, tmp_path):
    mgr = CheckpointManager(tmp_path / "ckpts")
    model, opt, sched = _model_and_optim(tiny_cfg)
    mgr.save(10, model=model, optimizer=opt, scheduler=sched,
             tokens_seen=0, config=tiny_cfg)
    assert mgr.latest() is not None
    mgr.save(20, model=model, optimizer=opt, scheduler=sched,
             tokens_seen=0, config=tiny_cfg)
    meta, _ = load_checkpoint(mgr.latest(), model, restore_rng=False)
    assert meta["step"] == 20
    assert len(mgr.all()) == 2


def _train_steps(cfg, exp_dir, steps):
    cfg.training.steps = steps
    trainer = Trainer(cfg, experiment_dir=exp_dir)
    summary = trainer.train()
    return trainer, summary


def _read_metrics(exp_dir):
    return [json.loads(l) for l in
            (exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def test_resume_is_bit_exact(smoke_dataset, tmp_path):
    """Run A: 6 uninterrupted steps.  Run B: crashes (non-interrupt
    exception, e.g. power loss / OOM) during step 4 after the periodic
    checkpoint at step 3, then resumes with a fresh process state to 6.
    Losses and weights must match exactly.  Resuming means continuing the
    SAME schedule: the step count must not change between sessions."""
    # ---- uninterrupted reference run --------------------------------
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    cfg_ref = config_from_dict(smoke_dataset["config_dict"])
    cfg_ref.training.log_interval = 1
    cfg_ref.training.eval_interval = 6
    cfg_ref.training.checkpoint_interval = 100
    _, summary_ref = _train_steps(cfg_ref, ref_dir, 6)
    ref_metrics = _read_metrics(ref_dir)

    # ---- crashed run: periodic checkpoint at 3, dies during step 4 ----
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    cfg_b = config_from_dict(smoke_dataset["config_dict"])
    cfg_b.training.steps = 6                 # same horizon as the reference
    cfg_b.training.log_interval = 1
    cfg_b.training.eval_interval = 100
    cfg_b.training.checkpoint_interval = 3
    trainer_b = Trainer(cfg_b, experiment_dir=exp_dir)
    original = trainer_b.model.forward

    def crashing_forward(*args, **kwargs):
        if not crashing_forward.done:
            crashing_forward.calls += 1
            if crashing_forward.calls == 4:   # during step 4
                crashing_forward.done = True
                raise RuntimeError("simulated crash")
        return original(*args, **kwargs)

    crashing_forward.calls = 0
    crashing_forward.done = False
    trainer_b.model.forward = crashing_forward
    with pytest.raises(RuntimeError, match="simulated crash"):
        trainer_b.train()
    assert (exp_dir / "checkpoints" / "step_00000003").is_dir()

    # fresh process-like state before resume (different ambient RNG)
    torch.manual_seed(9999)
    cfg_c = config_from_dict(smoke_dataset["config_dict"])
    cfg_c.training.steps = 6
    cfg_c.training.log_interval = 1
    cfg_c.training.eval_interval = 100
    cfg_c.training.checkpoint_interval = 100
    trainer_c = Trainer(cfg_c, experiment_dir=exp_dir, resume=True)
    assert trainer_c.start_step == 3
    trainer_c.train()                               # phase 2: 3 -> 6
    resumed_metrics = _read_metrics(exp_dir)

    # loss trajectories identical
    assert len(ref_metrics) == len(resumed_metrics) == 6
    for i, (r, s) in enumerate(zip(ref_metrics, resumed_metrics)):
        assert abs(r["train_loss"] - s["train_loss"]) < 1e-6, \
            f"resume diverged at step {i + 1}: {r['train_loss']} vs {s['train_loss']}"
        assert r["tokens_seen"] == s["tokens_seen"]

    # weights identical
    from domovoi.training import build_model_from_checkpoint
    model_ref, _, _ = build_model_from_checkpoint(ref_dir / "checkpoints" / "step_00000006")
    model_res, _, _ = build_model_from_checkpoint(exp_dir / "checkpoints" / "step_00000006")
    for k, v in model_ref.state_dict().items():
        assert torch.allclose(v, model_res.state_dict()[k], atol=1e-6), \
            f"weights differ after resume: {k}"


def test_mid_step_interrupt_rescue_is_exact(smoke_dataset, tmp_path):
    """Ctrl+C in the middle of a step must save a rescue checkpoint that
    references only fully completed steps and rolls RNG back to the last
    clean boundary; resume from it must reproduce the uninterrupted
    trajectory exactly (the interrupted step replays, its data draw is
    not lost)."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    cfg = config_from_dict(smoke_dataset["config_dict"])
    cfg.training.steps = 6
    cfg.training.log_interval = 1
    cfg.training.eval_interval = 100
    cfg.training.checkpoint_interval = 100  # the rescue ckpt is the only one

    trainer = Trainer(cfg, experiment_dir=exp_dir)
    original = trainer.model.forward
    raised = {"done": False}
    calls = {"n": 0}

    def exploding_forward(*args, **kwargs):
        # calls 1-3 = training steps 0-2; raise inside training step 3,
        # after its batch draw consumed generator state but before it ran
        calls["n"] += 1
        if calls["n"] == 4 and not raised["done"]:
            raised["done"] = True
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    trainer.model.forward = exploding_forward
    summary = trainer.train()

    tps = trainer.tokens_per_step()
    assert summary["status"] == "INTERRUPTED"
    assert summary["steps_completed"] == 3
    assert summary["tokens_trained"] == 3 * tps

    rescue = exp_dir / "checkpoints" / "step_00000003"
    meta = json.loads((rescue / "checkpoint.json").read_text(encoding="utf-8"))
    assert meta["step"] == 3
    assert meta["tokens_seen"] == 3 * tps

    # ---- uninterrupted reference over the same 6 steps ----------------
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    cfg_ref = config_from_dict(smoke_dataset["config_dict"])
    cfg_ref.training.log_interval = 1
    cfg_ref.training.eval_interval = 100
    cfg_ref.training.checkpoint_interval = 100
    Trainer(cfg_ref, experiment_dir=ref_dir).train()
    ref_metrics = _read_metrics(ref_dir)

    # ---- resume from the rescue checkpoint to completion ---------------
    cfg_r = config_from_dict(smoke_dataset["config_dict"])
    cfg_r.training.steps = 6          # resume target: 3 -> 6
    cfg_r.training.log_interval = 1
    cfg_r.training.eval_interval = 100
    cfg_r.training.checkpoint_interval = 100
    trainer_r = Trainer(cfg_r, experiment_dir=exp_dir, resume=True)
    assert trainer_r.start_step == 3
    summary_r = trainer_r.train()
    assert summary_r["steps_completed"] == 6
    assert summary_r["tokens_trained"] == 6 * tps

    # the resumed steps replay the reference trajectory exactly
    resumed_metrics = _read_metrics(exp_dir)[-3:]
    assert len(resumed_metrics) == 3
    for i, (r, s) in enumerate(zip(ref_metrics[3:], resumed_metrics)):
        assert abs(r["train_loss"] - s["train_loss"]) < 1e-6, \
            f"rescue-resume diverged at step {i + 4}: " \
            f"{r['train_loss']} vs {s['train_loss']}"
        assert r["tokens_seen"] == s["tokens_seen"]


def test_interrupt_during_validation_rescue_is_exact(smoke_dataset, tmp_path):
    """Ctrl+C inside the periodic VALIDATION window — after the step's
    optimizer update but before logging/validation/checkpointing completed —
    is the widest interrupt window in the loop.  The rescue checkpoint must
    reference the completed step with weights, accounting and RNG that are
    mutually consistent, and resume must reproduce the uninterrupted
    trajectory exactly (the step must NOT be double-applied)."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    cfg = config_from_dict(smoke_dataset["config_dict"])
    cfg.training.steps = 6
    cfg.training.log_interval = 1
    cfg.training.eval_interval = 1          # validation after EVERY step
    cfg.training.checkpoint_interval = 100  # the rescue ckpt is the only one

    trainer = Trainer(cfg, experiment_dir=exp_dir)
    original_eval = trainer.evaluate
    calls = {"n": 0}

    def exploding_eval(n_batches):
        calls["n"] += 1
        if calls["n"] == 4:                 # validation after step 4
            raise KeyboardInterrupt
        return original_eval(n_batches)

    trainer.evaluate = exploding_eval
    summary = trainer.train()
    tps = trainer.tokens_per_step()
    assert summary["status"] == "INTERRUPTED"
    assert summary["steps_completed"] == 4
    assert summary["tokens_trained"] == 4 * tps

    rescue = exp_dir / "checkpoints" / "step_00000004"
    meta = json.loads((rescue / "checkpoint.json").read_text(encoding="utf-8"))
    assert meta["step"] == 4
    assert meta["tokens_seen"] == 4 * tps

    # ---- uninterrupted reference over the same 6 steps ------------------
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    cfg_ref = config_from_dict(smoke_dataset["config_dict"])
    cfg_ref.training.steps = 6
    cfg_ref.training.log_interval = 1
    cfg_ref.training.eval_interval = 1
    cfg_ref.training.checkpoint_interval = 100
    Trainer(cfg_ref, experiment_dir=ref_dir).train()
    ref_metrics = _read_metrics(ref_dir)

    # ---- resume from the rescue checkpoint to completion -----------------
    cfg_r = config_from_dict(smoke_dataset["config_dict"])
    cfg_r.training.steps = 6
    cfg_r.training.log_interval = 1
    cfg_r.training.eval_interval = 1
    cfg_r.training.checkpoint_interval = 100
    trainer_r = Trainer(cfg_r, experiment_dir=exp_dir, resume=True)
    assert trainer_r.start_step == 4
    summary_r = trainer_r.train()
    assert summary_r["steps_completed"] == 6
    assert summary_r["tokens_trained"] == 6 * tps

    # steps 5 and 6 replay the reference trajectory exactly (a double-applied
    # step-4 update would diverge the losses here)
    resumed_metrics = _read_metrics(exp_dir)
    assert len(resumed_metrics) == 6
    for i, (r, s) in enumerate(zip(ref_metrics[4:], resumed_metrics[-2:])):
        assert abs(r["train_loss"] - s["train_loss"]) < 1e-6, \
            f"interrupt-in-validation resume diverged at step {i + 5}: " \
            f"{r['train_loss']} vs {s['train_loss']}"
        assert r["tokens_seen"] == s["tokens_seen"]

    # weights identical
    from domovoi.training import build_model_from_checkpoint
    model_ref, _, _ = build_model_from_checkpoint(ref_dir / "checkpoints" / "step_00000006")
    model_res, _, _ = build_model_from_checkpoint(exp_dir / "checkpoints" / "step_00000006")
    for k, v in model_ref.state_dict().items():
        assert torch.allclose(v, model_res.state_dict()[k], atol=1e-6), \
            f"weights differ after interrupt-in-validation resume: {k}"


def test_resume_completed_run_keeps_summary(smoke_dataset, tmp_path):
    """--resume on an already-complete run runs zero steps; it must NOT
    overwrite summary.json with session-only wall clock, 0 tok/s and an
    eval-only VRAM peak.  The original end-to-end measurement survives."""
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    cfg = config_from_dict(smoke_dataset["config_dict"])
    cfg.training.steps = 3
    cfg.training.log_interval = 1
    cfg.training.eval_interval = 3
    cfg.training.checkpoint_interval = 3
    Trainer(cfg, experiment_dir=exp_dir).train()
    original = (exp_dir / "summary.json").read_text(encoding="utf-8")

    trainer_r = Trainer(cfg, experiment_dir=exp_dir, resume=True)
    assert trainer_r.start_step == 3
    summary = trainer_r.train()
    assert summary["status"] == "COMPLETED"
    assert summary["avg_tok_per_s"] > 0        # the original run's number
    assert (exp_dir / "summary.json").read_text(encoding="utf-8") == original


def test_scaler_state_roundtrip(tiny_cfg, tmp_path):
    """GradScaler state must survive save/load so fp16 resume is bit-exact
    (a restarted loss scale changes which steps overflow and diverges the
    trajectory).  A disabled scaler's state_dict is empty — that must
    round-trip too without breaking the load."""
    model, opt, sched = _model_and_optim(tiny_cfg)
    scaler = torch.amp.GradScaler("cpu")   # enabled: has a real state_dict
    sd = scaler.state_dict()
    sd["scale"] = 4096.0
    sd["growth_interval"] = 37
    scaler.load_state_dict(sd)

    save_checkpoint(tmp_path / "ckpt", model, opt, sched, step=5,
                    tokens_seen=100, config=tiny_cfg, scaler=scaler)
    scaler2 = torch.amp.GradScaler("cpu")
    load_checkpoint(tmp_path / "ckpt", model, opt, sched, scaler=scaler2,
                    restore_rng=False)
    assert scaler2.state_dict()["scale"] == 4096.0
    assert scaler2.state_dict()["growth_interval"] == 37

    # disabled scaler (fp32/bf16 runs) persists an empty state: loading it
    # is a clean no-op, and a checkpoint saved that way still resumes
    save_checkpoint(tmp_path / "ckpt_fp32", model, opt, sched, step=5,
                    tokens_seen=100, config=tiny_cfg,
                    scaler=torch.amp.GradScaler("cpu", enabled=False))
    scaler3 = torch.amp.GradScaler("cpu")
    load_checkpoint(tmp_path / "ckpt_fp32", model, opt, sched, scaler=scaler3,
                    restore_rng=False)
    assert scaler3.state_dict()["scale"] == 65536.0   # untouched default


def test_manager_recovers_checkpoint_after_crashed_swap(tiny_cfg, tmp_path):
    """A crash between the two renames of a save leaves the previous
    checkpoint aside as `*.bak`; latest() must recover it instead of
    silently training from scratch."""
    import os

    mgr = CheckpointManager(tmp_path / "ckpts")
    model, opt, sched = _model_and_optim(tiny_cfg)
    mgr.save(10, model=model, optimizer=opt, scheduler=sched,
             tokens_seen=0, config=tiny_cfg)
    step_dir = tmp_path / "ckpts" / "step_00000010"
    os.replace(step_dir, step_dir.parent / (step_dir.name + ".bak"))

    latest = mgr.latest()
    assert latest is not None
    assert latest.name == "step_00000010"
    assert (latest / "checkpoint.json").is_file()


def test_checkpoint_records_config_and_step(tiny_cfg, tmp_path):
    model, opt, sched = _model_and_optim(tiny_cfg)
    save_checkpoint(tmp_path / "ckpt", model, opt, sched, step=7,
                    tokens_seen=42, config=tiny_cfg,
                    extra={"git_commit": "abc123"})
    meta = json.loads((tmp_path / "ckpt" / "checkpoint.json").read_text(encoding="utf-8"))
    assert meta["config"]["model"]["d_model"] == tiny_cfg.model.d_model
    assert meta["git_commit"] == "abc123"