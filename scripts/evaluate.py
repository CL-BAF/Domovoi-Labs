"""Run evaluation tasks on a trained experiment checkpoint.

Usage:
    python scripts/evaluate.py --experiment DML-0001
    python scripts/evaluate.py --experiment DML-0001 --tasks hellaswag,piqa
    python scripts/evaluate.py --experiment DML-0001 --limit 200

Writes machine-readable results to experiments/<ID>/evals.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.config import config_from_dict
from domovoi.datasets import DomovoiTokenizer
from domovoi.evals import DEFAULT_TASKS, run_evals
from domovoi.experiments import load_metadata, update_metadata
from domovoi.training import CheckpointManager
from domovoi.utils import detect_device
from domovoi.training.checkpoint import build_model_from_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True,
                    help="experiment ID (DML-0001) or experiment dir")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                    help="comma-separated eval task names")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples per task (fast research mode)")
    ap.add_argument("--val-batches", type=int, default=50)
    args = ap.parse_args()

    meta = load_metadata(args.experiment)
    exp_dir = Path(args.experiment) if Path(args.experiment).is_dir() \
        else Path("experiments") / meta["experiment_id"]
    ckpts = CheckpointManager(exp_dir / "checkpoints")
    ckpt_dir = ckpts.latest()
    if ckpt_dir is None:
        raise SystemExit(f"No checkpoint found under {exp_dir / 'checkpoints'}")

    model, cfg, ckpt_meta = build_model_from_checkpoint(ckpt_dir)
    device = detect_device()
    model = model.to(device)
    model.eval()
    tokenizer = DomovoiTokenizer(cfg.tokenizer.path)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[eval] experiment {meta['experiment_id']} checkpoint {ckpt_dir.name}")
    print(f"[eval] tasks: {tasks}")
    started = time.time()
    results = run_evals(
        model, tokenizer, device, tasks,
        limit=args.limit,
        training_config=cfg,
        n_val_batches=args.val_batches,
    )
    results.update({
        "experiment_id": meta["experiment_id"],
        "checkpoint": str(ckpt_dir),
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - started, 1),
        "limit": args.limit,
        "val_batches": args.val_batches,
    })
    out = exp_dir / "evals.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    update_metadata(exp_dir, {"evals_file": str(out)})
    print(f"[eval] results written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())