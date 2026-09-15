"""Experiment registry.

Every run gets a unique directory `experiments/DML-XXXX/` containing all
artifacts of that experiment: metadata.json, config copy, checkpoints,
metrics, eval results, profile and report.  IDs are allocated by scanning
the experiments directory (no central mutable state to corrupt).

Failed experiments are never deleted — they are research data.  The status
labels come from the project spec:

    RUNNING (in progress)
    FAILED / INVALID / INTERRUPTED (terminal, unsuccessful)
    BASELINE / PROMISING / REJECTED / PROMOTED (research judgement)
    COMPLETED (pipeline succeeded; research label applied separately or
               left as the config's default status)
"""

from __future__ import annotations

import json
import pathlib
import re
import time
from typing import Optional

from ..config import Config

ID_PATTERN = re.compile(r"^DML-(\d{4})$")


def experiments_root(base_dir: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(base_dir)


def list_experiments(base_dir: str | pathlib.Path) -> list[dict]:
    """All experiment metadata, oldest ID first."""
    root = experiments_root(base_dir)
    out = []
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        meta_path = d / "metadata.json"
        if d.is_dir() and ID_PATTERN.match(d.name) and meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out.append({"dir": d, **meta})
    return out


def allocate_experiment_id(base_dir: str | pathlib.Path) -> str:
    """Next free DML-XXXX id."""
    root = experiments_root(base_dir)
    if not root.is_dir():
        return "DML-0001"
    used = []
    for d in root.iterdir():
        m = ID_PATTERN.match(d.name)
        if m:
            used.append(int(m.group(1)))
    return f"DML-{max(used, default=0) + 1:04d}"


def create_experiment(
    base_dir: str | pathlib.Path,
    config: Config,
    env_info: dict,
    tokenizer_meta: Optional[dict] = None,
    n_params: Optional[int] = None,
) -> pathlib.Path:
    """Allocate an ID and write the initial metadata.json (status RUNNING).

    The metadata grows as the run progresses: summary, evals, profile and
    report steps each merge their results back in via update_metadata()."""
    exp_id = config.output.experiment_id or allocate_experiment_id(base_dir)
    if not ID_PATTERN.match(exp_id):
        raise ValueError(f"Experiment id must match DML-XXXX; got {exp_id!r}")
    exp_dir = experiments_root(base_dir) / exp_id
    if exp_dir.exists():
        raise FileExistsError(
            f"Experiment directory {exp_dir} already exists.  Pass a fresh "
            f"output.experiment_id or clear the old run explicitly.")
    exp_dir.mkdir(parents=True)

    metadata = {
        "experiment_id": exp_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": "RUNNING",
        "name": config.experiment.name,
        "notes": config.experiment.notes,
        "git_commit": env_info.get("git_commit"),
        "git_dirty": env_info.get("git_dirty"),
        "config": config.to_dict(),
        "config_source": config.source_path,
        "architecture": config.model.architecture,
        "parameters": n_params,
        "active_parameters": n_params,
        "dataset": {
            "name": config.dataset.name,
            "version": config.dataset.version,
            "source_url": config.dataset.source_url,
        },
        "tokenizer": tokenizer_meta or dict(config.tokenizer.__dict__),
        "seed": config.training.seed,
        "training_tokens": None,
        "env": env_info,
        # artifacts fill in below:
        "summary_file": "summary.json",
        "evals_file": "evals.json",
        "profile_file": "profile.json",
        "report_file": "report.md",
        "checkpoints_dir": "checkpoints",
        "checkpoints": [],
        "final_status": config.experiment.status,
    }
    (exp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    config.save_copy(exp_dir / "config.yaml")
    return exp_dir


def update_metadata(exp_dir: str | pathlib.Path, updates: dict) -> dict:
    """Merge `updates` into metadata.json (read-modify-write)."""
    exp_dir = pathlib.Path(exp_dir)
    meta_path = exp_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(updates)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def set_status(exp_dir: str | pathlib.Path, status: str, notes: str = "") -> dict:
    """Set the experiment's status label.  Statuses are validated so typos
    cannot silently corrupt the registry."""
    from ..config import STATUSES
    if status not in STATUSES:
        raise ValueError(f"Unknown status {status!r}; expected one of {STATUSES}")
    updates = {"status": status, "status_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if notes:
        updates["status_notes"] = notes
    return update_metadata(exp_dir, updates)


def load_metadata(exp_dir_or_id: str, base_dir: str | pathlib.Path = "experiments") -> dict:
    """Load metadata by path or by bare ID (e.g. 'DML-0001')."""
    p = pathlib.Path(exp_dir_or_id)
    if not p.is_dir():
        p = experiments_root(base_dir) / str(exp_dir_or_id)
    meta_path = p / "metadata.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No experiment metadata at {meta_path}. "
            f"Known: {[e['experiment_id'] for e in list_experiments(base_dir)]}")
    return json.loads(meta_path.read_text(encoding="utf-8"))