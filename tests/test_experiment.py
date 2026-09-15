"""Experiment registry: ID allocation, metadata, statuses, reports."""

import json

import pytest

from domovoi.config import config_from_dict
from domovoi.experiments import (allocate_experiment_id, create_experiment,
                                 generate_report, list_experiments,
                                 load_metadata, set_status, update_metadata)


def test_allocate_and_create(tmp_path, tiny_model_cfg_dict):
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    env = {"git_commit": "deadbeef", "git_dirty": False, "torch": "x"}

    assert allocate_experiment_id(tmp_path) == "DML-0001"
    exp_dir = create_experiment(tmp_path, cfg, env)
    assert exp_dir.name == "DML-0001"
    assert (exp_dir / "metadata.json").is_file()
    assert (exp_dir / "config.yaml").is_file()

    meta = load_metadata(exp_dir)
    assert meta["status"] == "RUNNING"
    assert meta["git_commit"] == "deadbeef"
    assert meta["config"]["model"]["vocab_size"] == 256

    assert allocate_experiment_id(tmp_path) == "DML-0002"


def test_create_refuses_overwrite(tmp_path, tiny_model_cfg_dict):
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    cfg.output.experiment_id = "DML-0001"
    create_experiment(tmp_path, cfg, {})
    with pytest.raises(FileExistsError):
        create_experiment(tmp_path, cfg, {})


def test_status_labels(tmp_path, tiny_model_cfg_dict):
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    exp_dir = create_experiment(tmp_path, cfg, {})
    set_status(exp_dir, "PROMISING", notes="beat baseline")
    assert load_metadata(exp_dir)["status"] == "PROMISING"
    with pytest.raises(ValueError):
        set_status(exp_dir, "NOT_A_STATUS")


def test_update_metadata_merge(tmp_path, tiny_model_cfg_dict):
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    exp_dir = create_experiment(tmp_path, cfg, {})
    update_metadata(exp_dir, {"training_tokens": 123, "conclusion": "ok"})
    meta = load_metadata(exp_dir)
    assert meta["training_tokens"] == 123 and meta["conclusion"] == "ok"
    assert meta["experiment_id"] == "DML-0001"  # untouched by merge


def test_report_reproducible(tmp_path, tiny_model_cfg_dict):
    """Report generation: written from stored artifacts only, and
    regenerating it produces byte-identical output."""
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    exp_dir = create_experiment(tmp_path, cfg, {})
    set_status(exp_dir, "BASELINE")

    (exp_dir / "summary.json").write_text(json.dumps({
        "status": "COMPLETED", "steps_completed": 3, "tokens_trained": 192,
        "final_train_loss": 5.5, "final_val_loss": 5.6, "final_val_ppl": 270.4,
        "wall_clock_s": 1.2, "avg_tok_per_s": 160.0, "vram": {},
    }), encoding="utf-8")
    (exp_dir / "evals.json").write_text(json.dumps({"tasks": {
        "val_ppl": {"display_name": "Validation perplexity",
                    "val_loss": 5.6, "val_ppl": 270.4, "n_batches": 2},
        "hellaswag": {"display_name": "HellaSwag (validation)",
                      "acc": 0.25, "acc_norm": 0.26, "chance": 0.25, "n": 100},
    }}), encoding="utf-8")

    text1 = generate_report(exp_dir)
    assert "DML-0001" in text1
    assert "HellaSwag" in text1
    assert "0.26" in text1
    # regenerating from the same stored data is byte-identical
    text2 = generate_report(exp_dir)
    assert text1 == text2
    assert (exp_dir / "report.md").is_file()


def test_report_comparison_section(tmp_path, tiny_model_cfg_dict):
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    base = create_experiment(tmp_path, cfg, {})
    cand = create_experiment(tmp_path, cfg, {})
    for d, acc in ((base, 0.25), (cand, 0.30)):
        (d / "evals.json").write_text(json.dumps({"tasks": {
            "hellaswag": {"display_name": "HellaSwag", "acc": acc,
                          "acc_norm": acc, "chance": 0.25, "n": 100},
        }}), encoding="utf-8")
    text = generate_report(cand, baseline_dir=base)
    assert "Comparison" in text
    assert "improvement" in text
    assert "regression" not in text


def test_list_experiments(tmp_path, tiny_model_cfg_dict):
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    create_experiment(tmp_path, cfg, {})
    create_experiment(tmp_path, cfg, {})
    ids = [e["experiment_id"] for e in list_experiments(tmp_path)]
    assert ids == ["DML-0001", "DML-0002"]
    # non-experiment dirs are ignored
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    assert len(list_experiments(tmp_path)) == 2


def test_failed_experiments_are_kept(tmp_path, tiny_model_cfg_dict):
    """A FAILED experiment still lists (never deleted)."""
    cfg = config_from_dict(tiny_model_cfg_dict)
    cfg.output.experiments_dir = str(tmp_path)
    exp_dir = create_experiment(tmp_path, cfg, {})
    set_status(exp_dir, "FAILED")
    assert list_experiments(tmp_path)[0]["status"] == "FAILED"