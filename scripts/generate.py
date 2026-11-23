"""Generate text from a checkpoint (independent of training).

Usage:
    python scripts/generate.py --checkpoint experiments/DML-0001/checkpoints/step_00003000 \
        --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8 --top-p 0.95

Prints the generated text plus measured generation speed and peak VRAM.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.datasets import DomovoiTokenizer, verify_vocab
from domovoi.inference import generate, load_model_for_inference


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="checkpoint directory")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="0 = greedy/deterministic")
    ap.add_argument("--top-k", type=int, default=0, help="0 = disabled")
    ap.add_argument("--top-p", type=float, default=1.0, help="1.0 = disabled")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-to", default=None,
                    help="optional path to write the machine-readable result json")
    args = ap.parse_args()

    model, cfg, meta, device = load_model_for_inference(args.checkpoint)
    tokenizer = DomovoiTokenizer(cfg.tokenizer.path) if cfg.tokenizer.path else None
    if tokenizer is None:
        raise SystemExit(
            "Checkpoint config has no tokenizer path; the tokenizer must be "
            "available where it was when training ran.")
    verify_vocab(tokenizer, cfg.model.vocab_size)

    result = generate(
        model, args.prompt, tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        device=device,
    )
    print("=" * 72)
    print(result["text"])
    print("=" * 72)
    print(f"prompt tokens : {result['n_prompt_tokens']}")
    print(f"new tokens    : {result['n_new_tokens']}")
    if result["ttft_s"] is not None:    # None when no token could be sampled
        print(f"ttft          : {result['ttft_s'] * 1000:.1f} ms")
    if result.get("decode_tok_per_s"):
        print(f"decode speed  : {result['decode_tok_per_s']:.1f} tok/s")
    if "peak_vram_allocated_mb" in result:
        print(f"peak VRAM     : {result['peak_vram_allocated_mb']} MiB allocated "
              f"({result['peak_vram_reserved_mb']} MiB reserved)")

    if args.save_to:
        Path(args.save_to).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[generate] result saved to {args.save_to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())