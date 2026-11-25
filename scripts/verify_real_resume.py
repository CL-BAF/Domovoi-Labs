"""Exercise clean-boundary checkpoint recovery on the packed TinyStories data."""

if __name__ == "__main__":
    raise SystemExit("Use python -I -S -B scripts/experiment_protocol.py run for research.")

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.config import clone_config, load_config
from domovoi.experiments.metrics_log import read_metrics
from domovoi.training import Trainer


def state_digest(trainer):
    digest = hashlib.sha256()
    for value in trainer.model.state_dict().values():
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _verify(args):
    args.out.mkdir(exist_ok=False)
    cfg = clone_config(load_config(args.config), training={
        "steps": 4, "warmup_steps": 1, "eval_interval": 2, "eval_batches": 2,
        "checkpoint_interval": 2, "log_interval": 1, "extra_checkpoint_steps": [],
        "pilot_calibrated_seconds": None})
    stop = args.out / "STOP"
    interrupted_dir = args.out / "interrupted"
    interrupted = Trainer(cfg, experiment_dir=interrupted_dir, stop_file=stop)
    save = interrupted._save_checkpoint

    def stop_after_commit(step, *pos, **kw):
        save(step, *pos, **kw)
        if step == 2:
            stop.touch()

    interrupted._save_checkpoint = stop_after_commit
    summary = interrupted.train()
    if summary["status"] != "INTERRUPTED" or summary["steps_completed"] != 2:
        raise RuntimeError(f"Expected clean interruption at step two: {summary}")
    checkpoint = interrupted.ckpt.latest()
    resumed = Trainer(cfg, experiment_dir=interrupted_dir, resume=True)
    try:
        if resumed.start_step != 2 or resumed.segment_sampler.position != 64:
            raise RuntimeError("Restored checkpoint has the wrong step or segment position")
        if state_digest(resumed) != state_digest(interrupted):
            raise RuntimeError("Model tensors changed across checkpoint loading")
        first, second = interrupted.optimizer.state_dict(), resumed.optimizer.state_dict()
        if first["param_groups"] != second["param_groups"] or first["state"].keys() != second["state"].keys():
            raise RuntimeError("AdamW optimizer structure differs after restore")
        for key, state in first["state"].items():
            for name, value in state.items():
                if not torch.equal(value.cpu(), second["state"][key][name].cpu()):
                    raise RuntimeError(f"AdamW {name} differs after restore")
        resumed_result = resumed.train()
    finally:
        resumed.close()
    clean = Trainer(cfg, experiment_dir=args.out / "clean")
    clean_result = clean.train()
    if any(r["status"] != "COMPLETED" for r in (resumed_result, clean_result)):
        raise RuntimeError("A smoke branch did not complete")
    if resumed.segment_sampler.position != clean.segment_sampler.position:
        raise RuntimeError("Segment stream drifted on resume")
    diff = max((a - b).abs().max().item()
               for a, b in zip(resumed.model.parameters(), clean.model.parameters()))
    resumed_steps = [m["step"] for m in read_metrics(interrupted_dir)]
    if resumed_steps != [1, 2, 3, 4] or diff > 5e-5:
        raise RuntimeError(f"Resume trajectory drift: steps={resumed_steps}, weights={diff}")
    report = {"interruption_step": 2, "restore_checkpoint": checkpoint.name,
              "weights_and_adamw_restore_exact": True, "reconciled_steps": resumed_steps,
              "max_final_weight_delta": diff,
              "final_sampler_position": resumed.segment_sampler.position}
    (args.out / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


def _run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    _verify(parser.parse_args(argv))
    return 0
