"""Train an experiment end to end.

Usage:
    python scripts/train.py --config configs/baseline_small.yaml
    python scripts/train.py --config configs/baseline_small.yaml \
        --resume --experiment-dir experiments/DML-0001

--resume always requires --experiment-dir: without it a fresh experiment
would be allocated and training would silently restart from step 0.

Exit code: 0 only when the run COMPLETED.  An interrupted run exits 1 so
automation never mistakes a partial model for a finished one.

Flow: allocate experiment ID → write metadata (status RUNNING, git commit,
env, dataset, tokenizer, seed) → train (checkpoints, metrics, val loss) →
write summary.json → set status COMPLETED (or FAILED / INTERRUPTED).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.config import load_config
from domovoi.datasets import DomovoiTokenizer
from domovoi.experiments import create_experiment, set_status, update_metadata
from domovoi.profiling import count_parameters
from domovoi.training import CheckpointManager, Trainer
from domovoi.utils import collect_env_info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="experiment config yaml")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in the experiment dir")
    ap.add_argument("--experiment-dir", default=None,
                    help="resume a specific experiment dir instead of allocating a new ID")
    ap.add_argument("--extend-steps", action="store_true",
                    help="with --resume: allow ONLY a larger training.steps "
                         "(recorded as a protocol deviation; such a run is not "
                         "a single controlled trajectory)")
    ap.add_argument("--pilot-deadline", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--stop-file", type=Path,
                    help="stop at a clean step boundary when this file exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.training.deterministic_cuda:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.pilot_deadline is not None and not cfg.training.pilot_stop_checks:
        ap.error("--pilot-deadline requires enabled pilot stop checks")
    env = collect_env_info()

    if args.resume and not args.experiment_dir:
        raise SystemExit(
            "--resume requires --experiment-dir: resuming without it would "
            "allocate a brand-new experiment and silently train from "
            "step 0.\nUsage: python scripts/train.py --config CONFIG "
            "--resume --experiment-dir experiments/DML-XXXX")
    if args.extend_steps and not args.resume:
        raise SystemExit("--extend-steps is only meaningful with --resume.")
    if args.experiment_dir and not args.resume:
        # a fresh run inside an existing experiment would append a second
        # training history to its metrics/checkpoints and overwrite its status
        raise SystemExit(
            "--experiment-dir is only valid with --resume. A fresh run always "
            "gets a newly allocated experiment ID.")

    # ---- experiment bookkeeping ---------------------------------------
    if args.experiment_dir:
        exp_dir = Path(args.experiment_dir)
    else:
        exp_dir = create_experiment(cfg.output.experiments_dir, cfg, env)
    exp_id = exp_dir.name
    print(f"[train] experiment {exp_id}")
    print(f"[train] git commit {env['git_commit']}{' (dirty)' if env['git_dirty'] else ''}")

    status = "FAILED"
    failure = ""
    trainer_ready = False   # resume pre-flight (Trainer.__init__) passed
    trainer = None          # holds the experiment's writer lock once built
    try:
        trainer = Trainer(cfg, experiment_dir=exp_dir, resume=args.resume,
                          extend_steps=args.extend_steps, pilot_deadline=args.pilot_deadline,
                          stop_file=args.stop_file)
        trainer_ready = True
        counts = count_parameters(trainer.model)
        update_metadata(exp_dir, {
            "parameters": trainer.n_params,
            "active_parameters": trainer.n_params,
            # parameter uses per token and block applications per forward;
            # for the baseline these equal `parameters` and `n_layers`
            "applied_parameters": counts["applied_per_token"],
            "effective_depth": counts["effective_depth"],
            "tokenizer": trainer.tokenizer.metadata(),
            "dataset": {
                "name": cfg.dataset.name,
                "version": cfg.dataset.version,
                "source_url": cfg.dataset.source_url,
                "processed_dir": cfg.dataset.processed_dir,
            },
        })
        print(f"[train] model: {cfg.model.architecture} "
              f"({trainer.n_params:,} params)")
        print(f"[train] device: {trainer.device}"
              + (f", {cfg.training.precision}" if cfg.training.precision != 'fp32' else ""))

        # keep the writer lock through the metadata/status writes below
        summary = trainer.train(release_lock=False)

        # record final checkpoints + summary in metadata
        ckpts = CheckpointManager(exp_dir / "checkpoints").all() \
            if (exp_dir / "checkpoints").is_dir() else []
        update_metadata(exp_dir, {
            "training_tokens": summary["tokens_trained"],
            "checkpoints": [c.name for c in ckpts],
        })
        status = "COMPLETED" if summary["status"] == "COMPLETED" else summary["status"]
        failure = summary.get("failure") or ""
        val_loss = summary["final_val_loss"]
        print(f"[train] done: {summary['steps_completed']} steps, "
              f"{summary['tokens_trained']:,} tokens, "
              f"val loss {format(val_loss, '.4f') if val_loss is not None else 'unavailable'}")

    except KeyboardInterrupt:
        status = "INTERRUPTED"
        print("[train] interrupted by user")
    except Exception:
        traceback.print_exc()
        status = "FAILED"
    finally:
        # On success the *research* status label (BASELINE/PROMISING/...)
        # comes from the config; failures never pretend otherwise.
        try:
            if args.resume and not trainer_ready:
                # the resume was refused (incompatible, or another writer
                # holds the lock) before it touched the existing experiment:
                # leave its recorded status exactly as is
                print(f"[train] resume not started; status of {exp_id} left unchanged")
            elif status == "COMPLETED":
                set_status(exp_dir, cfg.experiment.status,
                           notes="pipeline completed; research label from config")
            else:
                set_status(exp_dir, status, notes=failure)
        finally:
            # the status write above happened under this run's writer lock
            if trainer is not None:
                trainer.close()
    # an INTERRUPTED run is a partial model: exit 1 so pipelines/CI never
    # treat it as a finished training step
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
