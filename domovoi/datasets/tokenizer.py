"""Tokenizer wrapper.

The baseline tokenizer is a byte-level BPE trained on the training corpus
with the `tokenizers` library.  It is deliberately independent of Hugging
Face model abstractions: the wrapper here only needs `encode` / `decode` /
`vocab_size`, so swapping in a different tokenizer type later is trivial.

Determinism: byte-level BPE training in `tokenizers` is deterministic given
the same input text and hyperparameters (no RNG is involved).  We still
record the sha256 of the saved tokenizer file so any drift is detectable.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Iterable

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

TOKENIZER_VERSION = "byte-bpe-v1"


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_byte_bpe(
    corpus_paths: Iterable[str | pathlib.Path],
    vocab_size: int,
    out_path: str | pathlib.Path,
) -> dict:
    """Train a byte-level BPE tokenizer over one or more plain-text corpora.

    Returns metadata: {path, sha256, vocab_size, version}.
    """
    if vocab_size < 256:
        raise ValueError(f"vocab_size must be >= 256 (byte alphabet); got {vocab_size}")
    tok = Tokenizer(models.BPE())
    # ByteLevel pre-tokenizer: every input is representable (byte fallback),
    # no <unk> needed, and decode(encode(x)) == x always holds.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train([str(p) for p in corpus_paths], trainer)
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    return {
        "path": str(out_path),
        "sha256": sha256_file(out_path),
        "vocab_size": tok.get_vocab_size(),
        "version": TOKENIZER_VERSION,
    }


class DomovoiTokenizer:
    """Minimal, architecture-independent tokenizer interface."""

    def __init__(self, path: str | pathlib.Path):
        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)
        self.version = TOKENIZER_VERSION
        self.vocab_size = self.tokenizer.get_vocab_size()
        self.sha256 = sha256_file(self.path)

    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def metadata(self) -> dict:
        return {
            "type": "byte_bpe",
            "path": self.path,
            "sha256": self.sha256,
            "vocab_size": self.vocab_size,
            "version": self.version,
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_metadata(cls, meta: dict) -> "DomovoiTokenizer":
        tok = cls(meta["path"])
        if tok.sha256 != meta.get("sha256"):
            raise RuntimeError(
                f"Tokenizer file {tok.path} has sha256 {tok.sha256} but metadata "
                f"records {meta.get('sha256')}. The tokenizer changed after the "
                f"experiment; results are not comparable.")
        return tok


def verify_vocab(tokenizer: DomovoiTokenizer, vocab_size: int) -> None:
    """Fail loudly when the tokenizer and the model disagree about vocab
    size — a silent mismatch would corrupt training."""
    if tokenizer.vocab_size != vocab_size:
        raise RuntimeError(
            f"Tokenizer vocab size ({tokenizer.vocab_size}) does not match "
            f"model.vocab_size ({vocab_size}). Re-train the tokenizer or fix "
            f"the config.")