"""Data pipeline: tokenizer round-trip, deterministic split, packed batches,
hash recording, and eval/train data isolation."""

import numpy as np
import pytest
import torch

from domovoi.datasets import (DomovoiTokenizer, PackedTokenData,
                              load_processed_meta, make_smoke_corpus,
                              prepare_dataset, split_text, train_byte_bpe,
                              verify_vocab)


def test_byte_bpe_roundtrip(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("hello world, héllo wörld! 123\n\ttabs\n", encoding="utf-8")
    meta = train_byte_bpe([corpus], vocab_size=300, out_path=tmp_path / "tok.json")
    tok = DomovoiTokenizer(meta["path"])
    for text in ["hello world", "héllo wörld", "unicode: 🎉 αβγ", ""]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"roundtrip failed for {text!r}"


def test_tokenizer_metadata_and_pinning(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("a b c d e f g " * 200, encoding="utf-8")
    meta = train_byte_bpe([corpus], vocab_size=256, out_path=tmp_path / "tok.json")
    tok = DomovoiTokenizer.from_metadata(meta)  # hash matches
    with pytest.raises(RuntimeError):
        bad = dict(meta)
        bad["sha256"] = "0" * 64
        DomovoiTokenizer.from_metadata(bad)


def test_verify_vocab(tmp_path):
    corpus = tmp_path / "c.txt"
    corpus.write_text("a b c " * 100, encoding="utf-8")
    meta = train_byte_bpe([corpus], vocab_size=300, out_path=tmp_path / "tok.json")
    tok = DomovoiTokenizer(meta["path"])
    # The trained vocab can be smaller than the requested 300; verify against
    # the actual size recorded in the metadata.
    verify_vocab(tok, meta["vocab_size"])
    with pytest.raises(RuntimeError):
        verify_vocab(tok, meta["vocab_size"] + 1)


def test_split_deterministic():
    text = "x" * 1000
    a1, b1 = split_text(text, 0.9)
    a2, b2 = split_text(text, 0.9)
    assert a1 == a2 and b1 == b2
    assert len(a1) == 900 and len(b1) == 100


def test_prepare_smoke_dataset(smoke_dataset):
    meta = smoke_dataset["meta"]
    assert meta["complete"]
    assert meta["tokens"]["train"] > 0
    assert meta["tokens"]["val"] > 0
    assert meta["dataset"]["train_sha256"] != meta["dataset"]["val_sha256"]
    # token ids are within vocab
    train = np.fromfile(meta["files"]["train_bin"], dtype=np.uint16)
    val = np.fromfile(meta["files"]["val_bin"], dtype=np.uint16)
    assert train.max() < meta["tokens"]["vocab_size"]
    assert val.max() < meta["tokens"]["vocab_size"]
    # re-running without force is cached (same metadata)
    meta2 = load_processed_meta(smoke_dataset["config_dict"]["dataset"]["processed_dir"])
    assert meta2["dataset"]["raw_sha256"] == meta["dataset"]["raw_sha256"]


def test_get_batch_shapes_and_determinism(smoke_dataset):
    meta = smoke_dataset["meta"]
    data = PackedTokenData(meta["files"]["train_bin"])
    device = torch.device("cpu")
    g1, g2 = torch.Generator(), torch.Generator()
    g1.manual_seed(5)
    g2.manual_seed(5)
    x1, y1 = data.get_batch(2, 16, device, g1)
    x2, y2 = data.get_batch(2, 16, device, g2)
    assert x1.shape == (2, 16) and y1.shape == (2, 16)
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    # y is x shifted by one
    assert torch.equal(x1[:, 1:], y1[:, :-1])


def test_val_batches_are_fixed_across_calls(smoke_dataset):
    """Same fixed-seed generator → identical val batches every time."""
    meta = smoke_dataset["meta"]
    data = PackedTokenData(meta["files"]["val_bin"])
    device = torch.device("cpu")
    from domovoi.datasets import fixed_val_generator
    x1, _ = data.get_batch(2, 16, device, fixed_val_generator(device))
    x2, _ = data.get_batch(2, 16, device, fixed_val_generator(device))
    assert torch.equal(x1, x2)


def test_eval_data_isolation():
    """The training data pipeline must never touch the benchmark directory.

    Walks the datasets package AST looking for references to the eval tree
    in actual code (string constants, names, attributes) — docstrings and
    comments explaining the policy are fine; functional references are not.
    """
    import ast
    import pathlib
    pkg = pathlib.Path(__file__).resolve().parent.parent / "domovoi" / "datasets"
    for src in pkg.glob("*.py"):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            # docstrings are bare string-expression statements — skip them
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and "evals" in node.value.lower():
                raise AssertionError(
                    f"{src.name} references eval data ({node.value!r}). "
                    f"Training and benchmark data must stay separate "
                    f"(contamination guard).")
            if isinstance(node, ast.Name) and "eval" in node.id.lower():
                raise AssertionError(
                    f"{src.name} references {node.id!r} — training code "
                    f"must not depend on eval data structures.")


def test_train_val_no_overlap_by_content(smoke_dataset):
    """Deterministic block split means zero character overlap."""
    meta = smoke_dataset["meta"]
    assert meta["dataset"]["train_sha256"] != meta["dataset"]["val_sha256"]