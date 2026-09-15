"""Profile a trained experiment checkpoint: parameters, checkpoint size,
real CUDA VRAM peaks, prefill/decode speed, TTFT, context-length VRAM sweep.

Usage:
    python scripts/profile_model.py --experiment DML-0001
    python scripts/profile_model.py --experiment DML-0001 --context-sweep 128 256 512
    python scripts/profile_model.py --experiment DML-0001 --no-sweep

Writes machine-readable results to experiments/<ID>/profile.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.datasets import DomovoiTokenizer
from domovoi.experiments import load_metadata, update_metadata
from domovoi.profiling import profile_checkpoint
from domovoi.training import CheckpointManager
from domovoi.training.checkpoint import build_model_from_checkpoint
from domovoi.utils import detect_device


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--context-sweep", type=int, nargs="*", default=None,
                    help="context lengths for the VRAM-vs-context sweep; "
                         "defaults to a ladder up to the model context")
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()

    meta = load_metadata(args.experiment)
    exp_dir = Path(args.experiment) if Path(args.experiment).is_dir() \
        else Path("experiments") / meta["experiment_id"]
    ckpt_dir = CheckpointManager(exp_dir / "checkpoints").latest()
    if ckpt_dir is None:
        raise SystemExit(f"No checkpoint found under {exp_dir / 'checkpoints'}")

    device = detect_device()
    print(f"[profile] device: {device}")
    if device.type != "cuda":
        print("[profile] NOTE: CUDA unavailable — VRAM fields will be null "
              "(never estimated).")

    _, cfg, _ = build_model_from_checkpoint(ckpt_dir)
    tokenizer = DomovoiTokenizer(cfg.tokenizer.path)

    lengths = args.context_sweep
    if lengths is None and not args.no_sweep:
        cap = cfg.model.context_length
        ladder = [64, 128, 256, 512]
        lengths = [l for l in ladder if l <= cap] or [cap]

    profile = profile_checkpoint(
        ckpt_dir, tokenizer, device,
        max_new_tokens=args.max_new_tokens,
        context_sweep=(not args.no_sweep),
        context_lengths=lengths,
    )
    profile["experiment_id"] = meta["experiment_id"]
    out = exp_dir / "profile.json"
    out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    update_metadata(exp_dir, {"profile_file": str(out)})

    inf = profile["inference"]
    print(f"[profile] params       : {profile['parameters']['total']:,}")
    print(f"[profile] ckpt size    : {profile['checkpoint_size_mb']} MiB")
    print(f"[profile] decode tok/s : {inf['decode_tok_per_s_median']}")
    print(f"[profile] ttft         : {inf['ttft_s_median'] * 1000:.1f} ms")
    if inf.get("peak_vram_allocated_mb") is not None:
        print(f"[profile] peak VRAM    : {inf['peak_vram_allocated_mb']} MiB allocated, "
              f"{inf['peak_vram_reserved_mb']} MiB reserved")
    print(f"[profile] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())