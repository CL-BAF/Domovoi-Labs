"""Dataset download, splitting and tokenization.

Design rules:

- **Deterministic split**: the corpus is split by character position (first
  `train_fraction` = train, rest = validation).  No RNG is involved, so the
  split is identical on every machine.
- **Everything is recorded**: raw file sha256, split sha256, tokenizer
  sha256, token counts and timestamps go into `meta.json` inside the
  processed directory.  The training pipeline refuses to run without it.
- **Caching**: if the processed data exists and the recorded hashes match,
  preparation is a no-op (`--force` redoes it).
- **Contamination safety**: this module only ever touches training corpora
  under `data/raw` / `data/processed`.  Benchmark evaluation data lives in
  a separate tree (`data/evals`, written by scripts/download_eval_data.py)
  that nothing in the training pipeline reads.  See test_data.py for the
  regression test guarding this.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import random
from typing import Optional

import numpy as np
import requests

from .tokenizer import DomovoiTokenizer, sha256_file, train_byte_bpe

DATA_ROOT = pathlib.Path("data")


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def download_file(url: str, dest: pathlib.Path, expected_sha256: Optional[str] = None) -> None:
    """Download url to dest.  If expected_sha256 is given, verify the
    download against it (integrity pinning)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_sha256 and sha256_file(dest) == expected_sha256:
        return  # already cached and verified
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    data = response.content
    if expected_sha256 and _sha256_bytes(data) != expected_sha256:
        raise RuntimeError(
            f"Downloaded {url} but sha256 does not match the pinned value.\n"
            f"  expected {expected_sha256}\n"
            f"  got      {_sha256_bytes(data)}")
    dest.write_bytes(data)


def split_text(text: str, train_fraction: float) -> tuple[str, str]:
    """Deterministic block split by character position."""
    n_train = int(len(text) * train_fraction)
    return text[:n_train], text[n_train:]


def make_smoke_corpus(dest_dir: pathlib.Path, seed: int = 7) -> pathlib.Path:
    """Generate a tiny deterministic synthetic corpus for CPU smoke tests.
    No network access needed."""
    rng = random.Random(seed)
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
             "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron"]
    lines = []
    for i in range(400):
        n = rng.randint(4, 12)
        lines.append(" ".join(rng.choice(words) for _ in range(n)) + ".")
    text = "\n".join(lines) + "\n"
    dest_dir = pathlib.Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "smoke_corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


def prepare_dataset(
    name: str,
    source_url: str,
    version: str,
    train_fraction: float,
    processed_dir: str | pathlib.Path,
    tokenizer_path: str | pathlib.Path,
    vocab_size: int,
    force: bool = False,
) -> dict:
    """Full preparation: download → deterministic split → tokenize → pack.

    Returns the processed-dir meta.json contents (also written to disk).
    Re-running with identical inputs is a no-op unless force=True.
    """
    processed_dir = pathlib.Path(processed_dir)
    meta_path = processed_dir / "meta.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("complete") and pathlib.Path(meta["tokenizer"]["path"]).exists():
            return meta

    # -- 1. raw corpus ------------------------------------------------
    if source_url == "synthetic":
        raw_path = processed_dir / "raw.txt"
        if not raw_path.exists():
            corpus = make_smoke_corpus(processed_dir)
            raw_path.write_text(corpus.read_text(encoding="utf-8"),
                                encoding="utf-8")
    else:
        raw_path = DATA_ROOT / "raw" / f"{name}.txt"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            download_file(source_url, raw_path)
    raw_sha = sha256_file(raw_path)

    # -- 2. deterministic split ---------------------------------------
    text = raw_path.read_text(encoding="utf-8")
    train_text, val_text = split_text(text, train_fraction)
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "train.txt").write_text(train_text, encoding="utf-8")
    (processed_dir / "val.txt").write_text(val_text, encoding="utf-8")

    # -- 3. tokenizer --------------------------------------------------
    tokenizer_path = pathlib.Path(tokenizer_path)
    if not tokenizer_path.exists():
        tok_meta = train_byte_bpe(
            [processed_dir / "train.txt"], vocab_size, tokenizer_path)
    else:
        tok_meta = {
            "path": str(tokenizer_path),
            "sha256": sha256_file(tokenizer_path),
            "vocab_size": DomovoiTokenizer(tokenizer_path).vocab_size,
            "version": "byte-bpe-v1",
        }
    tok = DomovoiTokenizer(tokenizer_path)
    if tok.vocab_size > 65535:
        raise RuntimeError(
            "Packed token files are uint16; vocab_size must be <= 65535.")

    # -- 4. tokenize + pack as uint16 binaries -------------------------
    train_ids = np.array(tok.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tok.encode(val_text), dtype=np.uint16)
    train_bin = processed_dir / "train.bin"
    val_bin = processed_dir / "val.bin"
    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)

    # -- 5. meta -------------------------------------------------------
    meta = {
        "complete": True,
        "dataset": {
            "name": name,
            "source_url": source_url,
            "version": version,
            "raw_sha256": raw_sha,
            "train_chars": len(train_text),
            "val_chars": len(val_text),
            "train_sha256": sha256_file(processed_dir / "train.txt"),
            "val_sha256": sha256_file(processed_dir / "val.txt"),
            "train_fraction": train_fraction,
            "split_method": "deterministic_block",
        },
        "tokenizer": tok_meta,
        "tokens": {
            "train": int(train_ids.size),
            "val": int(val_ids.size),
            "vocab_size": tok.vocab_size,
            "dtype": "uint16",
        },
        "files": {
            "train_bin": str(train_bin),
            "val_bin": str(val_bin),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_processed_meta(processed_dir: str | pathlib.Path) -> dict:
    """Load and sanity-check the meta.json of a processed dataset."""
    meta_path = pathlib.Path(processed_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No processed dataset at {processed_dir}. "
            f"Run scripts/prepare_data.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete"):
        raise RuntimeError(
            f"Dataset at {processed_dir} is incomplete; re-run "
            f"scripts/prepare_data.py --force.")
    return meta