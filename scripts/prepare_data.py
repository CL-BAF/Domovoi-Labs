"""Prepare a training dataset: download → deterministic split → tokenize → pack.

Usage:
    python scripts/prepare_data.py --config configs/baseline_small.yaml
    python scripts/prepare_data.py --config configs/baseline_small.yaml --force

The corpus, tokenizer and processed binaries are cached in data/; the
recorded sha256 hashes let the pipeline detect any drift.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.config import load_config
from domovoi.datasets import prepare_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="experiment config yaml")
    ap.add_argument("--force", action="store_true",
                    help="re-run even if cached processed data exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    meta = prepare_dataset(
        name=cfg.dataset.name,
        source_url=cfg.dataset.source_url,
        version=cfg.dataset.version,
        train_fraction=cfg.dataset.train_fraction,
        processed_dir=cfg.dataset.processed_dir,
        tokenizer_path=cfg.tokenizer.path,
        vocab_size=cfg.model.vocab_size,
        force=args.force,
    )
    print(f"[prepare] dataset     : {meta['dataset']['name']} ({meta['dataset']['version']})")
    print(f"[prepare] raw sha256  : {meta['dataset']['raw_sha256'][:16]}...")
    print(f"[prepare] train tokens: {meta['tokens']['train']:,}")
    print(f"[prepare] val tokens  : {meta['tokens']['val']:,}")
    print(f"[prepare] vocab size  : {meta['tokens']['vocab_size']}")
    print(f"[prepare] tokenizer  : {meta['tokenizer']['path']}")
    actual = meta["tokens"]["vocab_size"]
    if actual != cfg.model.vocab_size:
        # The BPE trainer stops early when the corpus runs out of merge
        # candidates.  Training would fail at verify_vocab; fail here with
        # an actionable message instead.
        print(f"[prepare] ERROR: trained vocab ({actual}) != config "
              f"model.vocab_size ({cfg.model.vocab_size}). "
              f"Update model.vocab_size in {args.config} to {actual} "
              f"and re-run.")
        return 1
    print("[prepare] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())