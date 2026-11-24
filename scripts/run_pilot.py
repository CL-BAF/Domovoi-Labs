"""Run calibrated pilot arms serially under one aggregate time budget."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.experiments.launch import run_research, start_training_child
from domovoi.config import load_config


def main() -> int:
    return run_research("run_pilot.py", _run)


def _run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("configs", type=Path, nargs="+", help="calibrated arm/seed YAML files")
    args = ap.parse_args(argv)
    paths = [path.resolve() for path in args.configs]
    if len(set(paths)) != len(paths):
        ap.error("each pilot config must appear once")
    durations = []
    for path in paths:
        cfg = load_config(path)
        training = cfg.training
        if (training.sampler != "disjoint_segments" or not training.pilot_stop_checks
                or training.pilot_calibrated_seconds is None or cfg.output.experiment_id is not None):
            ap.error(f"{path} requires disjoint segments, calibrated stop checks, and a fresh ID")
        durations.append(training.pilot_calibrated_seconds)
    if not torch.cuda.is_available():
        ap.error("the calibrated pilot requires a CUDA-enabled PyTorch environment")
    deadline = time.monotonic() + 2 * sum(durations)
    for path in paths:
        if time.monotonic() >= deadline:
            print("Pilot stopped at its aggregate time limit.", file=sys.stderr)
            return 1
        child = start_training_child(["--config", str(path), "--pilot-deadline", str(deadline)])
        try:
            code = child.wait()
        except KeyboardInterrupt:
            child.wait()  # The same terminal interrupt reaches the training worker.
            raise
        if code:
            print(f"Pilot paused after {path.name} exited {code}. "
                  "Inspect its status and failure reason before starting any remaining arms.",
                  file=sys.stderr)
            return 130 if code == 130 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
