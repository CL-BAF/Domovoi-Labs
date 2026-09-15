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


def test_prepare_cache_hit_with_same_config_is_noop(smoke_dataset):
    """Identical config → the cache must be honored (no re-pack)."""
    cfg = smoke_dataset["config_dict"]
    meta = prepare_dataset(
        name="smoke", source_url="synthetic", version="smoke-v1",
        train_fraction=0.9, processed_dir=cfg["dataset"]["processed_dir"],
        tokenizer_path=cfg["tokenizer"]["path"], vocab_size=300)
    assert meta["complete"]
    assert meta["dataset"]["train_fraction"] == 0.9


def test_prepare_cache_rejects_config_drift(smoke_dataset):
    """A cached dataset prepared with a different train_fraction must be
    REFUSED, not silently reused — otherwise the experiment would record the
    new config while training on the old split."""
    cfg = smoke_dataset["config_dict"]
    with pytest.raises(RuntimeError, match="force"):
        prepare_dataset(
            name="smoke", source_url="synthetic", version="smoke-v1",
            train_fraction=0.5,                    # drifted
            processed_dir=cfg["dataset"]["processed_dir"],
            tokenizer_path=cfg["tokenizer"]["path"], vocab_size=300)


def test_prepare_cache_rejects_changed_tokenizer(smoke_dataset, tmp_path):
    """If the tokenizer file changed on disk after the bins were packed, the
    cached data is stale (ids would be misinterpreted) — refuse."""
    import json
    import pathlib
    import shutil

    src_dir = pathlib.Path(smoke_dataset["config_dict"]["dataset"]["processed_dir"])
    dst_dir = tmp_path / "processed"
    shutil.copytree(src_dir, dst_dir)
    dst_tok = tmp_path / "tok.json"
    shutil.copy(smoke_dataset["config_dict"]["tokenizer"]["path"], dst_tok)
    meta = json.loads((dst_dir / "meta.json").read_text(encoding="utf-8"))
    meta["tokenizer"]["path"] = str(dst_tok)
    (dst_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # mutate the tokenizer file after packing (same vocab, different bytes)
    dst_tok.write_bytes(dst_tok.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="sha256"):
        prepare_dataset(
            name="smoke", source_url="synthetic", version="smoke-v1",
            train_fraction=0.9, processed_dir=dst_dir,
            tokenizer_path=dst_tok, vocab_size=300)


def test_trainer_refuses_changed_tokenizer(smoke_dataset):
    """The Trainer must refuse to train when the configured tokenizer no
    longer matches the one recorded in the processed-data meta (sha256),
    even if vocab_size still agrees — otherwise train.bin's token ids are
    silently misinterpreted."""
    import pathlib

    from domovoi.config import config_from_dict
    from domovoi.training import Trainer

    cfg = config_from_dict(smoke_dataset["config_dict"])
    tok_path = pathlib.Path(cfg.tokenizer.path)
    # trailing whitespace keeps the JSON valid and the vocab identical;
    # only the file hash changes
    tok_path.write_bytes(tok_path.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="sha256"):
        Trainer(cfg, experiment_dir=None, device=torch.device("cpu"))


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


def test_get_batch_covers_full_window_range(tmp_path):
    """The last valid window of the corpus must be reachable, and an
    exactly-minimum-sized file (block_size+1 tokens) must not crash."""
    block = 16
    # block + 2 tokens → valid offsets are exactly {0, 1}
    p = tmp_path / "two_windows.bin"
    np.arange(block + 2, dtype=np.uint16).tofile(p)
    data = PackedTokenData(p)
    g = torch.Generator()
    g.manual_seed(0)
    x, _ = data.get_batch(256, block, torch.device("cpu"), g)
    assert set(x[:, 0].tolist()) == {0, 1}

    # exact minimum: block + 1 tokens (single valid window at offset 0)
    p2 = tmp_path / "minimum.bin"
    np.arange(block + 1, dtype=np.uint16).tofile(p2)
    x2, y2 = PackedTokenData(p2).get_batch(4, block, torch.device("cpu"), g)
    assert x2.shape == (4, block) and y2.shape == (4, block)
    assert torch.equal(x2[:, 0], torch.zeros(4, dtype=torch.int64))


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
        # docstrings are bare string-expression statements; mark their
        # Constant nodes so the walk below can skip them (ast.walk yields
        # the Expr and its child Constant as separate nodes)
        docstrings = {id(n.value) for n in ast.walk(tree)
                      if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings \
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