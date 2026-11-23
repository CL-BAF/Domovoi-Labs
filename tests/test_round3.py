"""Regression tests for Codex's third review (POST_AUDIT_FIXES.txt, "Round 3").

  R3-H  The writer lock is taken first and held across validation, loading,
        recovery, the CLI's metadata and status writes, and training.  A
        damaged *.bak is never renamed into place; a lock refusal leaves the
        experiment untouched.
  R3-M  Resume verifies the packed .bin files the loader actually reads
        (sha256 and token counts from the opened files), not only the
        manifest.

The damaged-backup, lost-lock and modified-.bin cases were reproduced
before the fix (post_audit/repro_round3_before.txt).  CPU only.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest
import yaml

from domovoi.config import config_from_dict
from domovoi.training import CheckpointError, Trainer

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _tree(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): ("dir" if p.is_dir() else
                                       hashlib.sha256(p.read_bytes()).hexdigest())
            for p in sorted(root.rglob("*"))}


def _cfg(smoke_dataset):
    d = json.loads(json.dumps(smoke_dataset["config_dict"]))
    d["model"].update({"architecture": "domovoi_recurrent_v1", "recurrence_depth": 2})
    d["training"].update(steps=2, checkpoint_interval=2, eval_interval=2)
    return d


def _run(smoke_dataset, name="exp"):
    d = _cfg(smoke_dataset)
    exp = smoke_dataset["tmp_path"] / name
    Trainer(config_from_dict(d), experiment_dir=exp).train()
    return d, exp


def _crash_swap(exp) -> pathlib.Path:
    step = exp / "checkpoints" / "step_00000002"
    bak = step.with_name(step.name + ".bak")
    os.replace(step, bak)
    return bak


# ---- R3-H: lock first, load in place, recover last ------------------------

@pytest.mark.parametrize("damaged", ["optimizer.pt", "model.safetensors", "scheduler.pt"])
def test_damaged_backup_is_not_moved_into_place(smoke_dataset, damaged):
    d, exp = _run(smoke_dataset)
    bak = _crash_swap(exp)
    (bak / damaged).write_bytes(b"damaged")
    before = _tree(exp)
    with pytest.raises(Exception):
        Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    assert _tree(exp) == before          # .bak still a .bak, no lock left behind


def test_lock_is_held_from_construction_until_close(smoke_dataset):
    d, exp = _run(smoke_dataset)
    t = Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    lock = exp / Trainer.WRITER_LOCK
    assert lock.is_file()                                  # held before train()
    with pytest.raises(CheckpointError, match="locked by another writer"):
        Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    t.train(release_lock=False)
    assert lock.is_file()                                  # still held for the CLI's writes
    t.close()
    assert not lock.exists()


def test_close_never_deletes_another_writers_lock(smoke_dataset):
    d, exp = _run(smoke_dataset)
    t = Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    lock = exp / Trainer.WRITER_LOCK
    lock.write_text("pid=1 host=someone-else", encoding="utf-8")   # e.g. a manual takeover
    t.close()
    assert lock.read_text(encoding="utf-8") == "pid=1 host=someone-else"


def test_resume_without_existing_dir_creates_nothing(smoke_dataset):
    d = _cfg(smoke_dataset)
    missing = smoke_dataset["tmp_path"] / "missing"
    with pytest.raises(CheckpointError):
        Trainer(config_from_dict(d), experiment_dir=missing, resume=True)
    assert not missing.exists()


def _train_script(cfg_path, exp):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py"), "--config",
                           str(cfg_path), "--resume", "--experiment-dir", str(exp)],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_cli_resume_that_loses_the_lock_leaves_experiment_untouched(smoke_dataset):
    d, exp = _run(smoke_dataset)
    (exp / "metadata.json").write_text(json.dumps(
        {"experiment_id": "DML-0001", "status": "RUNNING"}), encoding="utf-8")
    (exp / Trainer.WRITER_LOCK).write_text("pid=1 host=active-run", encoding="utf-8")
    d["training"]["steps"] = 4
    cfg_path = smoke_dataset["tmp_path"] / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(d), encoding="utf-8")
    before = _tree(exp)
    r = _train_script(cfg_path, exp)
    assert r.returncode != 0 and "locked by another writer" in r.stderr
    assert _tree(exp) == before          # no metadata merge, status still RUNNING
    assert json.loads((exp / "metadata.json").read_text(encoding="utf-8"))["status"] == "RUNNING"


def test_cli_successful_resume_releases_lock_after_status(smoke_dataset):
    d, exp = _run(smoke_dataset)
    (exp / "metadata.json").write_text(json.dumps(
        {"experiment_id": "DML-0002", "status": "RUNNING"}), encoding="utf-8")
    cfg_path = smoke_dataset["tmp_path"] / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(d), encoding="utf-8")
    r = _train_script(cfg_path, exp)
    assert r.returncode == 0, r.stderr
    # on success the CLI writes the research label from the config
    status = json.loads((exp / "metadata.json").read_text(encoding="utf-8"))["status"]
    assert status == d["experiment"]["status"]
    assert not (exp / Trainer.WRITER_LOCK).exists()


# ---- R3-M: packed inputs ----------------------------------------------------

def _train_bin(smoke_dataset) -> pathlib.Path:
    return pathlib.Path(smoke_dataset["meta"]["files"]["train_bin"])


def test_checkpoint_records_packed_file_hashes_and_actual_counts(smoke_dataset):
    _, exp = _run(smoke_dataset)
    prov = json.loads((exp / "checkpoints" / "step_00000002" / "checkpoint.json")
                      .read_text(encoding="utf-8"))["data_provenance"]
    raw = _train_bin(smoke_dataset).read_bytes()
    assert prov["train_bin_sha256"] == hashlib.sha256(raw).hexdigest()
    assert prov["train_tokens"] == len(raw) // 2          # uint16 tokens in the file


def test_resume_refuses_modified_packed_file_with_same_manifest(smoke_dataset):
    d, exp = _run(smoke_dataset)
    gc.collect()                              # Windows: drop earlier read-only maps
    data = np.memmap(_train_bin(smoke_dataset), dtype=np.uint16, mode="r+")
    data[:50] = (data[:50] + 1) % 256         # same size, same manifest, different tokens
    data.flush()
    del data
    before = _tree(exp)
    with pytest.raises(CheckpointError, match="train_bin_sha256"):
        Trainer(config_from_dict(d), experiment_dir=exp, resume=True)
    assert _tree(exp) == before


def test_packed_file_disagreeing_with_manifest_count_is_refused(smoke_dataset):
    d = _cfg(smoke_dataset)
    gc.collect()
    with open(_train_bin(smoke_dataset), "ab") as f:
        f.write(b"\x00\x00")                  # one extra token the manifest does not know
    with pytest.raises(RuntimeError, match="meta.json records"):
        Trainer(config_from_dict(d), experiment_dir=smoke_dataset["tmp_path"] / "x")


def test_resume_of_older_provenance_format_warns(smoke_dataset, capsys):
    d, exp = _run(smoke_dataset)
    p = exp / "checkpoints" / "step_00000002" / "checkpoint.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    for k in ("train_bin_sha256", "val_bin_sha256"):          # round-2 format
        meta["data_provenance"].pop(k)
    p.write_text(json.dumps(meta), encoding="utf-8")
    Trainer(config_from_dict(d), experiment_dir=exp, resume=True).close()
    out = capsys.readouterr().out
    assert "train_bin_sha256" in out and "cannot be verified" in out
