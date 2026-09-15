"""End-to-end smoke test: the full pipeline on CPU, cheap.

1. prepare tiny synthetic data + train a tokenizer
2. train a few steps (with grad accumulation + checkpointing)
3. save checkpoint
4. reload the checkpoint (independent of the trainer)
5. resume from the checkpoint for one more step
6. generate text
7. run a tiny evaluation (val_ppl + toy MC)
8. write an experiment directory with metadata, summary, evals
9. generate a human-readable report from the stored artifacts

Runs in a couple of seconds on CPU.
"""

import json

import torch

from domovoi.config import config_from_dict
from domovoi.datasets import prepare_dataset
from domovoi.experiments import create_experiment, generate_report, set_status, update_metadata
from domovoi.inference import generate
from domovoi.evals import (MultipleChoiceEval, build_task, run_evals)
from domovoi.training import Trainer, build_model_from_checkpoint


def test_end_to_end_smoke(smoke_dataset, tmp_path):
    # ---------- 1. data prepared in the fixture (synthetic) ------------
    cfg = config_from_dict(smoke_dataset["config_dict"])
    cfg.training.steps = 4
    cfg.training.log_interval = 1
    cfg.training.eval_interval = 4
    cfg.training.eval_batches = 2
    cfg.training.checkpoint_interval = 2
    cfg.output.experiments_dir = str(tmp_path / "experiments")

    env = {"git_commit": "smoketest", "git_dirty": False,
           "python": "test", "torch": "test", "cuda_available": False}
    exp_dir = create_experiment(cfg.output.experiments_dir, cfg, env,
                                tokenizer_meta=cfg.tokenizer.__dict__)
    assert exp_dir.name == "DML-0001"

    # ---------- 2-3. train + checkpoint --------------------------------
    # pinned to CPU: this is the cheap CPU smoke test of the full pipeline
    # (a GPU may be present on this machine; the e2e GPU path is exercised
    # by the real DML-0001 run)
    cpu = torch.device("cpu")
    trainer = Trainer(cfg, experiment_dir=exp_dir, device=cpu)
    update_metadata(exp_dir, {"parameters": trainer.n_params,
                              "tokenizer": trainer.tokenizer.metadata()})
    summary = trainer.train()
    assert summary["status"] == "COMPLETED"
    assert (exp_dir / "metrics.jsonl").is_file()
    assert (exp_dir / "val_metrics.jsonl").is_file()
    assert (exp_dir / "checkpoints" / "step_00000004").is_dir()
    assert summary["tokens_trained"] == 4 * trainer.tokens_per_step()

    # ---------- 4. reload checkpoint independently -----------------------
    ckpt = exp_dir / "checkpoints" / "step_00000004"
    model2, cfg2, meta2 = build_model_from_checkpoint(ckpt)
    assert cfg2.model.d_model == cfg.model.d_model
    model2.eval()

    # weights actually loaded (not a fresh init): identical logits
    trainer.model.eval()
    ids = trainer.tokenizer.encode("alpha beta gamma.")
    x = torch.tensor([ids[:16]], dtype=torch.long)
    with torch.no_grad():
        a = trainer.model(x).logits
        b = model2(x).logits
    assert torch.allclose(a, b, atol=1e-5), "reloaded model differs"

    # ---------- 5. resume for one more step -----------------------------
    cfg.training.steps = 5
    trainer_r = Trainer(cfg, experiment_dir=exp_dir, resume=True, device=cpu)
    assert trainer_r.start_step == 4
    summary_r = trainer_r.train()
    assert summary_r["status"] == "COMPLETED"
    assert summary_r["steps_completed"] == 5

    # ---------- 6. generate --------------------------------------------
    gen = generate(trainer_r.model, "alpha beta", trainer_r.tokenizer,
                   max_new_tokens=8, temperature=0.0)
    assert gen["n_new_tokens"] == 8
    (exp_dir / "generation.json").write_text(json.dumps(gen, indent=2),
                                             encoding="utf-8")

    # ---------- 7. tiny evaluation ---------------------------------------
    val_ppl = build_task("val_ppl").run(
        trainer_r.model, trainer_r.tokenizer, torch.device("cpu"),
        training_config=cfg, n_batches=2)
    assert val_ppl["val_loss"] > 0

    # toy MC eval with synthetic data to avoid network access
    mc_dir = tmp_path / "evals" / "toy_mc"
    mc_dir.mkdir(parents=True)
    mc_dir.joinpath("data.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "1", "context": "alpha beta", "choices": ["gamma", "delta"], "label": 0},
        {"id": "2", "context": "alpha beta", "choices": ["delta", "gamma"], "label": 1},
    ]) + "\n", encoding="utf-8")
    mc_dir.joinpath("meta.json").write_text(json.dumps(
        {"source": "synthetic", "sha256": "0" * 64}), encoding="utf-8")
    mc = MultipleChoiceEval("toy_mc", "Toy MC").run(
        trainer_r.model, trainer_r.tokenizer, torch.device("cpu"),
        data_root=tmp_path / "evals")
    assert mc["n"] == 2

    evals = {"tasks": {"val_ppl": val_ppl, "toy_mc": mc}}
    (exp_dir / "evals.json").write_text(json.dumps(evals, indent=2),
                                         encoding="utf-8")

    # ---------- 8. finalize metadata + status ---------------------------
    update_metadata(exp_dir, {
        "training_tokens": summary_r["tokens_trained"],
        "checkpoints": ["step_00000002", "step_00000004", "step_00000005"],
    })
    set_status(exp_dir, cfg.experiment.status)

    # ---------- 9. report -------------------------------------------------
    report = generate_report(exp_dir)
    assert "DML-0001" in report
    assert "BASELINE" in report
    assert "Val" in report or "Validation" in report
    assert (exp_dir / "report.md").is_file()

    # report regenerates identically from stored data
    assert generate_report(exp_dir) == report