import json
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from domovoi.datasets import streaming
from domovoi.config import config_from_dict
from domovoi.training import Trainer
from domovoi.training.checkpoint import CheckpointError


def spec(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("Bonjour café 雪<|endoftext|>Repeated Story<|endoftext|>   "
                     "<|endoftext|>A new train document.", encoding="utf-8")
    val.write_text(" repeated story <|endoftext|>Independent validation example.", encoding="utf-8")
    return {"name": "fixture", "version": "v1", "vocab_size": 300,
            "sources": [{"id": "fixture", "revision": "v1", "kind": "text",
                         "path": str(p), "split": split}
                        for p, split in ((train, "train"), (val, "val"))]}


def test_streaming_dedup_splits_eos_and_tokenizer(tmp_path):
    settings = spec(tmp_path)
    out = tmp_path / "packed"
    result = streaming.prepare_streaming(settings, out)
    assert result["documents"] == {"val": 2, "train": 2}
    stats = result["ingestion"][-1]
    assert (stats["invalid"], stats["exact_duplicates"]) == (1, 1)
    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
    for split in ("val", "train"):
        tokens = np.fromfile(result["files"][f"{split}_bin"], dtype="<u2")
        ends = json.loads(Path(result["files"][f"{split}_document_ends"]).read_text())
        assert len(tokens) == ends[-1] == result["tokens"][split]
        assert (tokens[np.array(ends) - 1] == result["tokenizer"]["eos_id"]).all()
    assert "Bonjour café 雪" in tokenizer.decode(np.fromfile(out / "train.bin", dtype="<u2").tolist())
    with pytest.raises(FileExistsError):
        streaming.prepare_streaming(settings, out)


def test_resume_after_packing_interruption_and_input_drift(tmp_path, monkeypatch):
    settings = spec(tmp_path)
    expected = streaming.prepare_streaming(settings, tmp_path / "expected")
    pack = streaming._pack

    def interrupt(db, staged, split, *args):
        if split == "train":
            raise KeyboardInterrupt
        return pack(db, staged, split, *args)

    monkeypatch.setattr(streaming, "_pack", interrupt)
    out = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt):
        streaming.prepare_streaming(settings, out)
    assert not out.exists()
    monkeypatch.setattr(streaming, "_pack", pack)
    with pytest.raises(ValueError, match="staged run"):
        streaming.prepare_streaming({**settings, "vocab_size": 320}, out)
    result = streaming.prepare_streaming(settings, out)
    assert result["packed_sha256"] == expected["packed_sha256"]
    assert result["tokenizer"]["sha256"] == expected["tokenizer"]["sha256"]


def test_parquet_schema_document_read_and_hash_partition(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "documents.parquet"
    pq.write_table(pa.table({"body": [None, " ", "a document", "another document"]}), path,
                   row_group_size=2)
    assert list(streaming.documents(path, "parquet", "body")) == [
        None, " ", "a document", "another document"]
    with pytest.raises(ValueError, match="Missing text column"):
        next(streaming.documents(path, "parquet"))
    settings = spec(tmp_path)
    settings["sources"].append({"id": "extra", "revision": "v1", "kind": "parquet",
                                "path": str(path), "split": "hash", "column": "body",
                                "test_permille": 1000, "val_permille": 0})
    result = streaming.prepare_streaming(settings, tmp_path / "packed")
    assert result["documents"]["test"] == 2
    assert result["ingestion"][-1]["invalid"] == 2


def test_bad_utf8_and_source_hash_fail_closed(tmp_path):
    settings = spec(tmp_path)
    train = Path(settings["sources"][0]["path"])
    train.write_bytes(b"Invalid UTF8 \xff contents")
    with pytest.raises(UnicodeDecodeError):
        streaming.prepare_streaming(settings, tmp_path / "bad")
    settings["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        streaming.prepare_streaming(settings, tmp_path / "hash")


def test_mixture_records_document_quotas(tmp_path):
    settings = spec(tmp_path)
    extra = tmp_path / "extra.txt"
    extra.write_text("another designated train source <|endoftext|> one more story", encoding="utf-8")
    settings["sources"].append({"id": "extra", "revision": "v1", "kind": "text",
                                "path": str(extra), "split": "train"})
    settings["mixture"] = {"document_quotas": {"fixture": 2, "extra": 1}}
    result = streaming.prepare_streaming(settings, tmp_path / "mixed")
    assert result["documents"]["train"] == 3
    assert result["mixture"]["document_weights"] == {"fixture": 2 / 3, "extra": 1 / 3}


def test_verified_checkpoint_rejects_corrupted_optimizer(smoke_dataset):
    settings = smoke_dataset["config_dict"]
    settings["training"]["verify_checkpoint_writes"] = True
    settings["training"]["synchronize_cuda_timing"] = True
    exp = smoke_dataset["tmp_path"] / "verified"
    cfg = config_from_dict(settings)
    trainer = Trainer(cfg, experiment_dir=exp)
    assert trainer.train()["status"] == "COMPLETED"
    checkpoint = trainer.ckpt.latest()
    state = checkpoint / "optimizer.pt"
    with state.open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(CheckpointError, match="checksum mismatch"):
        Trainer(cfg, experiment_dir=exp, resume=True)
