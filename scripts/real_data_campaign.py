"""Create and run an isolated, time-bounded paired Domovoi experiment."""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Use python -I -S -B scripts/experiment_protocol.py run for research.")

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.config import clone_config, load_config
from domovoi.datasets import load_processed_meta
from domovoi.experiments.launch import start_training_child
from domovoi.experiments.runtime import active_runtime
from domovoi.utils.env import collect_env_info
from scripts.plan_recurrent_experiments import PILOT_ARMS, build_arms

REPO = Path(__file__).resolve().parent.parent
SAMPLE_FIELDS = ("timestamp", "temperature.gpu", "utilization.gpu", "memory.used", "memory.free", "power.draw")


def sample_gpu():
    runtime = active_runtime()
    if "nvidia-smi" not in runtime.tools or not runtime.gpu_device:
        raise RuntimeError("The campaign requires locked GPU telemetry.")
    out = subprocess.check_output([str(runtime.tools["nvidia-smi"]), "-i", runtime.gpu_device,
                                   "--query-gpu=" + ",".join(SAMPLE_FIELDS),
                                   "--format=csv,noheader,nounits"], text=True, timeout=10)
    return dict(zip(SAMPLE_FIELDS, (x.strip() for x in out.strip().split(","))))


def _campaign(args):
    meta = load_processed_meta(args.data)
    env = collect_env_info(REPO)
    if env["git_commit"] == "unknown" or env["git_dirty"]:
        raise RuntimeError("A clean, committed implementation source is required.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16 CUDA device is required.")
    if args.steps < 1 or args.seconds < 1 or not 2 <= args.first_id <= 9997:
        raise ValueError("Specify positive run limits and a first ID from 2 through 9997.")
    needed = args.steps * 32 * 513
    if meta["tokens"]["train"] < needed:
        raise ValueError(f"Train split has fewer than {needed:,} tokens for disjoint segments.")
    if shutil.disk_usage(args.root.parent).free < 10 * 2**30:
        raise RuntimeError("At least 10 GiB of free checkpoint storage is required.")
    args.root.mkdir(exist_ok=False)
    (args.root / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (args.root / "data-manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (args.root / "runtime-lock.json").write_text(
        json.dumps(active_runtime().lock, indent=2), encoding="utf-8")
    base = clone_config(
        load_config(REPO / "configs/baseline_small.yaml"),
        model={"vocab_size": meta["tokens"]["vocab_size"]},
        dataset={"name": meta["dataset"]["name"], "version": meta["dataset"]["version"],
                 "processed_dir": str(args.data), "source_url": "local-pinned-corpus"},
        tokenizer={"path": meta["tokenizer"]["path"], "version": meta["tokenizer"]["version"]},
        output={"experiments_dir": str(args.root)},
        training={"steps": args.steps, "seed": args.seed, "gradient_accumulation": 2,
                  "sampler": "disjoint_segments", "pilot_stop_checks": True,
                  "synchronize_cuda_timing": True,
                  "deterministic_cuda": True, "verify_checkpoint_writes": True,
                  "warmup_steps": min(200, max(1, args.steps // 20)),
                  "checkpoint_interval": 50 if args.steps <= 200 else 500,
                  "eval_interval": 50 if args.steps <= 200 else 250, "eval_batches": 25,
                  "log_interval": 10, "extra_checkpoint_steps": [
                      step for step in (610, 1526, 3052) if step < args.steps]},
    )
    stop = args.root / "STOP"
    deadline = time.monotonic() + args.seconds
    results = []
    for index, (name, cfg) in enumerate(build_arms(base, only=PILOT_ARMS), args.first_id):
        if stop.exists() or time.monotonic() >= deadline:
            break
        run_id = f"DML-{index:04d}"
        cfg.output.experiment_id = run_id
        cfg.experiment.name = f"{args.root.name}-{name}"
        cfg.experiment.status = "COMPLETED"
        if args.calibrated_seconds:
            cfg.training.pilot_calibrated_seconds = args.calibrated_seconds[name]
        path = args.root / f"{run_id}-{name}.yaml"
        path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
        child_args = ["--config", str(path), "--stop-file", str(stop),
                      "--pilot-deadline", str(deadline)]
        print(f"{run_id} {name}: {path}", flush=True)
        with (args.root / f"{run_id}.console.log").open("w", encoding="utf-8") as log, \
                (args.root / f"{run_id}.gpu.jsonl").open("w", encoding="utf-8") as gpu:
            child = start_training_child(child_args, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    if not stop.exists():
                        try:
                            reading = sample_gpu()
                            gpu.write(json.dumps(reading) + "\n")
                            gpu.flush()
                            if float(reading["temperature.gpu"]) >= 88:
                                stop.write_text("GPU reached 88 C", encoding="utf-8")
                            if shutil.disk_usage(args.root).free < 10 * 2**30:
                                stop.write_text("Checkpoint disk below 10 GiB free", encoding="utf-8")
                        except (subprocess.SubprocessError, OSError, ValueError) as exc:
                            stop.write_text(f"Monitoring failed: {exc}", encoding="utf-8")
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            except KeyboardInterrupt:
                stop.write_text("Operator interrupt", encoding="utf-8")
                child.wait()
        results.append({"arm": name, "id": run_id, "config": str(path), "trainer_argv": child_args,
                        "exit_code": child.returncode})
        (args.root / "campaign.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if child.returncode:
            print(f"Campaign stopped after {run_id} exited {child.returncode}.", flush=True)
            return 130 if child.returncode == 130 or stop.exists() else 1
    if len(results) == len(PILOT_ARMS):
        return 0
    return 130 if stop.exists() else 1


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--root", type=lambda s: Path(s).resolve(), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--first-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--calibrated-seconds", type=json.loads,
                        help="arm to full-run duration mapping, measured during calibration")
    return _campaign(parser.parse_args(argv))
