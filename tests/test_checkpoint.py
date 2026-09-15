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
    mgr.save(10, model=model, optimizer=opt, scheduler=sched, step=10,
             tokens_seen=0, config=tiny_cfg)
    assert mgr.latest() is not None
    mgr.save(20, model=model, optimizer=opt, scheduler=sched, step=20,
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
    """Run A: 6 uninterrupted steps.  Run B: 3 steps, then resume with a
    fresh process state to 6.  Losses and weights must match exactly."""
    # ---- uninterrupted reference run --------------------------------
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    cfg_ref = config_from_dict(smoke_dataset["config_dict"])
    cfg_ref.training.log_interval = 1
    cfg_ref.training.eval_interval = 6
    cfg_ref.training.checkpoint_interval = 100
    _, summary_ref = _train_steps(cfg_ref, ref_dir, 6)
    ref_metrics = _read_metrics(ref_dir)

    # ---- interrupted + resumed run ------------------------------------
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    cfg_b = config_from_dict(smoke_dataset["config_dict"])
    cfg_b.training.log_interval = 1
    cfg_b.training.eval_interval = 100
    cfg_b.training.checkpoint_interval = 3
    _, _ = _train_steps(cfg_b, exp_dir, 3)          # phase 1: stops at 3
    assert (exp_dir / "checkpoints" / "step_00000003").is_dir()

    # fresh process-like state before resume (different ambient RNG)
    torch.manual_seed(9999)
    cfg_c = config_from_dict(smoke_dataset["config_dict"])
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


def test_checkpoint_records_config_and_step(tiny_cfg, tmp_path):
    model, opt, sched = _model_and_optim(tiny_cfg)
    save_checkpoint(tmp_path / "ckpt", model, opt, sched, step=7,
                    tokens_seen=42, config=tiny_cfg,
                    extra={"git_commit": "abc123"})
    meta = json.loads((tmp_path / "ckpt" / "checkpoint.json").read_text(encoding="utf-8"))
    assert meta["config"]["model"]["d_model"] == tiny_cfg.model.d_model
    assert meta["git_commit"] == "abc123"