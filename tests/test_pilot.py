import copy
import importlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from domovoi.config import config_from_dict
from domovoi.datasets import DomovoiTokenizer
from domovoi.datasets.loader import DisjointSegmentSampler
from domovoi.evals import run_evals
from domovoi.experiments.metrics_log import read_metrics
from domovoi.models import build_model
from domovoi.training import Trainer
from domovoi.training.pilot_limits import PilotLimits, PilotStopError


def test_disjoint_sampler_has_no_overlap_and_rejects_short_corpus():
    sampler = DisjointSegmentSampler(23, context=4, seed=7, required=4)
    first = sampler.take(1)
    rest = sampler.take(3)
    assert sorted(first + rest) == [0, 5, 10, 15]
    sampler.restore(1)
    assert sampler.take(3) == rest
    with pytest.raises(RuntimeError, match="exhausted"):
        sampler.take(1)
    with pytest.raises(ValueError, match="4 complete segments"):
        DisjointSegmentSampler(23, context=4, seed=7, required=5)


def test_interrupted_segment_run_matches_uninterrupted(smoke_dataset):
    raw = copy.deepcopy(smoke_dataset["config_dict"])
    raw["training"].update(sampler="disjoint_segments", steps=4, gradient_accumulation=2,
                           context_length=16, pilot_stop_checks=True)
    cfg = config_from_dict(raw)
    root = smoke_dataset["tmp_path"]
    clean = Trainer(cfg, experiment_dir=root / "clean")
    clean.train()
    interrupted = Trainer(cfg, experiment_dir=root / "interrupted")
    get_batch = interrupted.get_train_batch

    def interrupt_after_consuming_first_batch_of_step_three():
        batch = get_batch()
        if interrupted.segment_sampler.position == 10:
            raise KeyboardInterrupt
        return batch

    interrupted.get_train_batch = interrupt_after_consuming_first_batch_of_step_three
    assert interrupted.train()["status"] == "INTERRUPTED"
    resumed = Trainer(cfg, experiment_dir=root / "interrupted", resume=True)
    assert resumed.start_step == 2
    assert resumed.segment_sampler.position == 8
    resumed.train()
    assert resumed.segment_sampler.position == clean.segment_sampler.position == 16
    for key, param in clean.model.state_dict().items():
        assert torch.equal(param, resumed.model.state_dict()[key])
    for name in ("metrics.jsonl", "val_metrics.jsonl"):
        def rows(directory):
            return [{k: v for k, v in row.items() if k not in ("attempt", "tok_per_s", "elapsed_s")}
                    for row in read_metrics(directory, name)]

        assert rows(root / "clean") == rows(root / "interrupted")


def test_full_split_nll_counts_document_targets_and_partial_windows(smoke_dataset):
    raw = copy.deepcopy(smoke_dataset["config_dict"])
    raw["training"]["context_length"] = 3
    cfg = config_from_dict(raw)
    directory = smoke_dataset["tmp_path"]
    packed = directory / "test.bin"
    ends = directory / "test_ends.json"
    tokens = [1, 2, 3, 4, 5, 1, 2, 7, 8, 9]
    np.array(tokens, dtype=np.uint16).tofile(packed)
    ends.write_text("[5, 7, 8, 10]", encoding="utf-8")
    meta_path = directory / "meta.json"
    meta = copy.deepcopy(smoke_dataset["meta"])
    meta["files"].update(test_bin=str(packed), test_document_ends=str(ends))
    meta["tokens"]["test"] = len(tokens)
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    cfg.dataset.processed_dir = str(directory)
    tokenizer = DomovoiTokenizer(cfg.tokenizer.path)

    class PredictNext(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(*x.shape, 10)
            logits.scatter_(-1, ((x + 1) % 10).unsqueeze(-1), 4.0)
            return SimpleNamespace(logits=logits)

    task = run_evals(PredictNext(), tokenizer, torch.device("cpu"),
                     ["test_nll"], training_config=cfg)["tasks"]["test_nll"]
    assert task["n_documents"] == 4
    assert task["n"] == 6
    assert task["nll"] == pytest.approx(math.log(math.exp(4) + 9) - 4)
    ends.write_text("[5, 7, 7, 10]", encoding="utf-8")
    with pytest.raises(ValueError, match="strictly increasing"):
        run_evals(PredictNext(), tokenizer, torch.device("cpu"),
                  ["test_nll"], training_config=cfg)


def test_early_exit_nll_ends_at_full_model_nll(smoke_dataset):
    raw = copy.deepcopy(smoke_dataset["config_dict"])
    raw["model"].update(architecture="domovoi_recurrent_v1", recurrence_depth=2)
    cfg = config_from_dict(raw)
    meta = copy.deepcopy(smoke_dataset["meta"])
    directory = smoke_dataset["tmp_path"]
    cfg.dataset.processed_dir = str(directory)
    ends = directory / "val_ends.json"
    ends.write_text(json.dumps([meta["tokens"]["val"]]), encoding="utf-8")
    meta["files"]["val_document_ends"] = str(ends)
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    model = build_model(cfg.model)
    tokenizer = DomovoiTokenizer(cfg.tokenizer.path)
    task = run_evals(model, tokenizer, torch.device("cpu"),
                     ["val_nll_exits"], training_config=cfg)["tasks"]["val_nll_exits"]
    assert model.training
    assert len(task["application_nll"]) == 4
    assert task["application_nll"][-1] == pytest.approx(task["nll"])


def test_full_split_cli_keeps_checkpoint_results_and_rejects_interim_test(
        smoke_dataset, monkeypatch):
    train_cli = importlib.import_module("scripts.train")
    eval_cli = importlib.import_module("scripts.evaluate")
    raw = copy.deepcopy(smoke_dataset["config_dict"])
    directory = smoke_dataset["tmp_path"]
    raw["output"]["experiments_dir"] = str(directory / "experiments")
    raw["training"].update(sampler="disjoint_segments", pilot_stop_checks=True,
                           extra_checkpoint_steps=[1])
    manifest = Path(raw["dataset"]["processed_dir"]) / "meta.json"
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    val_ends = directory / "validation-ends.json"
    val_ends.write_text(json.dumps([meta["tokens"]["val"]]), encoding="utf-8")
    test_bin = directory / "test.bin"
    test_ends = directory / "test-ends.json"
    np.array([1, 2, 3, 4, 5], dtype=np.uint16).tofile(test_bin)
    test_ends.write_text("[5]", encoding="utf-8")
    meta["files"].update(val_document_ends=str(val_ends), test_bin=str(test_bin),
                         test_document_ends=str(test_ends))
    meta["tokens"]["test"] = 5
    manifest.write_text(json.dumps(meta), encoding="utf-8")
    config = directory / "config.yaml"
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config)])
    assert train_cli.main() == 0
    exp = directory / "experiments" / "DML-0001"

    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--experiment", str(exp),
                                    "--checkpoint-step", "1", "--tasks", "test_nll"])
    with pytest.raises(SystemExit, match="completed final checkpoint"):
        eval_cli.main()
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--experiment", str(exp),
                                    "--checkpoint-step", "1", "--tasks", "val_nll"])
    assert eval_cli.main() == 0
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--experiment", str(exp), "--tasks", "test_nll"])
    assert eval_cli.main() == 0
    assert (exp / "evals-step-00000001-val_nll.json").is_file()
    result = json.loads((exp / "evals-step-00000003-test_nll.json").read_text(encoding="utf-8"))
    assert result["tasks"]["test_nll"]["n"] == 4


def test_pilot_stop_thresholds_and_state():
    limits = PilotLimits(warmup_steps=0, total_steps=1000, calibrated_seconds=100)
    for step in range(1, 250):
        limits.check_update(step, 1 if step <= 50 else 100, 1)
    resumed = PilotLimits(0, 1000, 100)
    resumed.load_state_dict(limits.state_dict())
    with pytest.raises(PilotStopError, match="200 consecutive"):
        resumed.check_update(250, 100, 1)
    limits = PilotLimits(0, 1000, 100)
    for step in range(1, 50):
        limits.check_update(step, 1, 101)
    with pytest.raises(PilotStopError, match="50 consecutive"):
        limits.check_update(50, 1, 101)
    with pytest.raises(PilotStopError, match="12 GiB"):
        limits.check_resources(199, 100, 12 * 2**30 + 1)
    limits.check_resources(199, 100, None, check_projection=True)
    with pytest.raises(PilotStopError, match="runtime"):
        limits.check_resources(200, 100, None, check_projection=True)
    expired = PilotLimits(0, 1000, 100, deadline=time.monotonic() - 1)
    with pytest.raises(PilotStopError, match="aggregate"):
        expired.check_resources(1, 0, None)


def test_pilot_time_stop_does_not_commit_failed_step(smoke_dataset):
    raw = copy.deepcopy(smoke_dataset["config_dict"])
    raw["training"]["pilot_stop_checks"] = True
    root = smoke_dataset["tmp_path"] / "expired"
    trainer = Trainer(config_from_dict(raw), experiment_dir=root,
                      pilot_deadline=time.monotonic() - 1)
    result = trainer.train()
    assert result["status"] == "FAILED"
    assert result["failure_code"] == "pilot_time_limit"
    assert result["steps_completed"] == 0
    assert trainer.ckpt.latest() is None


def test_campaign_launcher_shares_deadline_and_stops_on_failure(tmp_path, tiny_model_cfg_dict,
                                                            monkeypatch):
    runner = importlib.import_module("scripts.run_pilot")
    raw = copy.deepcopy(tiny_model_cfg_dict)
    raw["training"].update(sampler="disjoint_segments", pilot_stop_checks=True,
                           pilot_calibrated_seconds=100)
    paths = [tmp_path / name for name in ("first.yaml", "second.yaml", "third.yaml")]
    for path in paths:
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    calls = []

    def run(args):
        calls.append(args)
        return SimpleNamespace(returncode=0 if len(calls) == 1 else 1)

    monkeypatch.setattr(sys, "argv", ["run_pilot.py", *map(str, paths)])
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main() == 1
    assert [call[3] for call in calls] == list(map(str, paths[:2]))
    assert calls[0][-1] == calls[1][-1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_cuda_profile_timing_and_peak_across_repeats(tiny_cfg, monkeypatch):
    profiler = importlib.import_module("domovoi.profiling.profiler")
    device = torch.device("cuda")
    model = build_model(tiny_cfg.model).to(device).eval()
    generate = profiler.generate
    synchronize = torch.cuda.synchronize
    calls = 0
    sync_calls = 0
    injected_peak = 0

    def count_synchronizations(device):
        nonlocal sync_calls
        sync_calls += 1
        synchronize(device)

    def inject_first_timed_peak(*args, **kwargs):
        nonlocal calls, injected_peak
        calls += 1
        if calls == 2:
            scratch = torch.empty(32 * 2**20, dtype=torch.uint8, device=device)
            injected_peak = torch.cuda.memory_allocated(device) / 2**20
            del scratch
        response = generate(*args, **kwargs)
        assert response["prefill_s"] > 0
        assert response["ttft_s"] >= response["prefill_s"]
        assert response["decode_tok_per_s"] > 0
        return response

    tokenizer = SimpleNamespace(decode=lambda _: "")
    monkeypatch.setattr(profiler, "generate", inject_first_timed_peak)
    monkeypatch.setattr(torch.cuda, "synchronize", count_synchronizations)
    profile = profiler.profile_inference(model, tokenizer, device, prompt_ids=[1, 2, 3],
                                         max_new_tokens=4, n_repeats=2)
    assert calls == 3
    assert sync_calls >= 3 * calls
    assert profile["peak_vram_allocated_mb"] >= round(injected_peak, 2)
