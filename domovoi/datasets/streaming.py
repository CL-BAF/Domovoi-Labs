"""Streaming document readers and restartable, versioned token packing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import unicodedata
from contextlib import closing
from itertools import islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from .tokenizer import sha256_file

EOS = "<|endoftext|>"
SPLITS = ("test", "val", "train")
BATCH_SIZE = 512
MAX_DOCUMENT_CHARS = 1_000_000


def documents(path: Path, kind: str, column: str = "text", delimiter: str = EOS):
    if kind == "text":
        with path.open(encoding="utf-8-sig", errors="strict") as source:
            pending = ""
            for block in iter(lambda: source.read(2**16), ""):
                *items, pending = (pending + block).split(delimiter)
                yield from items
                if len(pending) > MAX_DOCUMENT_CHARS:
                    raise ValueError(f"No document boundary within {MAX_DOCUMENT_CHARS} chars in {path}")
            if pending.strip():
                yield pending
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if column not in parquet.schema_arrow.names:
        raise ValueError(f"Missing text column {column!r} in {path}")
    dtype = parquet.schema_arrow.field(column).type
    if not (pa.types.is_string(dtype) or pa.types.is_large_string(dtype)):
        raise ValueError(f"{path}: {column!r} must be a string column")
    article = []
    chars = 0
    for batch in parquet.iter_batches(batch_size=256, columns=[column]):
        for value in batch.column(0).to_pylist():
            if kind == "parquet":
                yield value
                continue
            if value is None or not value.strip():
                continue
            if re.fullmatch(r"=\s+[^=].*?\s+=", value.strip()) and article:
                yield "\n".join(article)
                article, chars = [], 0
            article.append(value.strip())
            chars += len(value)
            if chars > MAX_DOCUMENT_CHARS:
                raise ValueError(f"WikiText article exceeds the configured reader bound in {path}")
    if article:
        yield "\n".join(article)


def document_key(text: str) -> bytes:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def _atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
    temp.replace(path)


def _pin_sources(spec: dict) -> list[dict]:
    sources = []
    for source in spec["sources"]:
        if source["kind"] not in ("text", "parquet", "wikitext"):
            raise ValueError(f"Unknown source reader {source['kind']!r}")
        if source["split"] not in (*SPLITS, "hash"):
            raise ValueError(f"Unknown split {source['split']!r}")
        if not source.get("id") or not source.get("revision"):
            raise ValueError("Each source requires an ID and revision.")
        limit = source.get("max_documents")
        if limit is not None and (type(limit) is not int or limit <= 0):
            raise ValueError("max_documents must be a positive integer.")
        if source["split"] == "hash":
            fractions = (source.get("test_permille", 10), source.get("val_permille", 10))
            if any(type(x) is not int or x < 0 for x in fractions) or sum(fractions) > 1000:
                raise ValueError("Hash split fractions must be nonnegative integer permille summing to at most 1000.")
        path = Path(source["path"]).resolve(strict=True)
        size = path.stat().st_size
        digest = sha256_file(path)
        if source.get("sha256", digest) != digest:
            raise ValueError(f"Source checksum mismatch: {path}")
        sources.append({**source, "path": str(path), "bytes": size, "sha256": digest})
    if len({s["path"] for s in sources}) != len(sources):
        raise ValueError("A raw file must appear only once in the manifest.")
    return sorted(sources, key=lambda s: (*SPLITS, "hash").index(s["split"]))


def _ingest(db, sources):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS docs (
            digest BLOB UNIQUE, split TEXT, source INTEGER, text TEXT);
        CREATE TABLE IF NOT EXISTS progress (
            source INTEGER PRIMARY KEY, scanned INTEGER, invalid INTEGER,
            duplicates INTEGER, complete INTEGER);
    """)
    for i, source in enumerate(sources):
        scanned, invalid, duplicates, complete = db.execute(
            "SELECT scanned, invalid, duplicates, complete FROM progress WHERE source=?", (i,)
        ).fetchone() or (0, 0, 0, 0)
        if complete:
            continue
        records = documents(Path(source["path"]), source["kind"],
                            source.get("column", "text"), source.get("delimiter", EOS))
        allow = source.get("max_documents")
        for value in islice(records, scanned, allow):
            scanned += 1
            if not isinstance(value, str) or not (value := value.strip()) or len(value) > MAX_DOCUMENT_CHARS:
                invalid += 1
            else:
                key = document_key(value)
                split = source["split"]
                if split == "hash":
                    bucket = int.from_bytes(key[:8], "big") % 1000
                    test = source.get("test_permille", 10)
                    val = source.get("val_permille", 10)
                    split = "test" if bucket < test else "val" if bucket < test + val else "train"
                if db.execute("INSERT OR IGNORE INTO docs VALUES (?,?,?,?)",
                              (key, split, i, value)).rowcount == 0:
                    duplicates += 1
            if scanned % 4096 == 0:
                db.execute("INSERT OR REPLACE INTO progress VALUES (?,?,?,?,0)",
                           (i, scanned, invalid, duplicates))
                db.commit()
            if scanned % 200_000 == 0:
                print(f"{source['id']} {source['split']}: {scanned:,} documents read", flush=True)
        db.execute("INSERT OR REPLACE INTO progress VALUES (?,?,?,?,1)",
                   (i, scanned, invalid, duplicates))
        db.commit()
        print(f"Ingested {scanned:,} documents from {Path(source['path']).name}", flush=True)
    db.execute("CREATE INDEX IF NOT EXISTS docs_split ON docs(split)")


def _make_tokenizer(db, path, spec):
    if path.is_file():
        return Tokenizer.from_file(str(path))
    vocab = spec.get("vocab_size", 4096)
    max_bytes = spec.get("tokenizer_sample_bytes", 16 * 2**20)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    remaining = max_bytes
    sample = path.parent.parent / "tokenizer-sample.txt"
    selection = " AND rowid IN (SELECT doc FROM mixture_train)" if spec.get("mixture") else ""
    with sample.open("wb") as output:
        for (value,) in db.execute("SELECT text FROM docs WHERE split='train'" + selection + " ORDER BY rowid"):
            data = value.encode("utf-8") + b"\n"
            if len(data) > remaining:
                break
            output.write(data)
            remaining -= len(data)
    tok.train([str(sample)], trainers.BpeTrainer(
        vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[EOS], show_progress=False))
    if not 257 <= tok.get_vocab_size() <= 65536:
        raise ValueError("Tokenizer vocabulary does not fit uint16 or the byte alphabet.")
    for value in ("Hello, world!", " café\n雪 😀"):
        if tok.decode(tok.encode(value).ids) != value:
            raise ValueError("Byte-level tokenizer failed the Unicode round-trip check.")
    temp = path.with_suffix(".tmp")
    tok.save(str(temp))
    temp.replace(path)
    return tok


def _pack(db, staged, split, tokenizer, budget):
    # Journal offsets are committed only after the corresponding bytes reach disk.
    db.execute("""CREATE TABLE IF NOT EXISTS packing (
        split TEXT PRIMARY KEY, rowid_done INTEGER, tokens INTEGER, docs INTEGER,
        complete INTEGER)""")
    done, count, n_docs, complete = db.execute(
        "SELECT rowid_done,tokens,docs,complete FROM packing WHERE split=?", (split,)
    ).fetchone() or (0, 0, 0, 0)
    tokens_path = staged / f"{split}.bin"
    ends_path = staged / f"{split}.ends"
    if not complete:
        with tokens_path.open("a+b") as tokens, ends_path.open("a+b") as ends:
            if tokens.seek(0, 2) < count * 2 or ends.seek(0, 2) < n_docs * 8:
                raise ValueError("Staged packing files are shorter than their journal.")
            tokens.truncate(count * 2)
            ends.truncate(n_docs * 8)
            eos = tokenizer.token_to_id(EOS)
            selected = ("AND rowid IN (SELECT doc FROM mixture_train) "
                        if split == "train" and db.execute(
                            "SELECT 1 FROM sqlite_master WHERE name='mixture_train'").fetchone()
                        else "")
            while True:
                rows = db.execute("SELECT rowid,text FROM docs WHERE split=? AND rowid>? "
                                  + selected + "ORDER BY rowid LIMIT ?",
                                  (split, done, BATCH_SIZE)).fetchall()
                if not rows or budget is not None and count >= budget:
                    break
                encoded = tokenizer.encode_batch([s for _, s in rows], add_special_tokens=False)
                for (done, _), enc in zip(rows, encoded):
                    ids = enc.ids + [eos]
                    tokens.write(np.asarray(ids, dtype="<u2").tobytes())
                    count += len(ids)
                    n_docs += 1
                    ends.write(count.to_bytes(8, "little"))
                for stream in (tokens, ends):
                    stream.flush()
                    os.fsync(stream.fileno())
                db.execute("INSERT OR REPLACE INTO packing VALUES (?,?,?,?,0)",
                           (split, done, count, n_docs))
                db.commit()
                if n_docs % (BATCH_SIZE * 128) == 0:
                    print(f"Packed {split}: {n_docs:,} documents, {count:,} tokens", flush=True)
        db.execute("INSERT OR REPLACE INTO packing VALUES (?,?,?,?,1)", (split, done, count, n_docs))
        db.commit()
    if count:
        # Write JSON offsets without constructing a full in-memory array.
        path = staged / f"{split}_document_ends.json"
        with ends_path.open("rb") as source, path.open("w", encoding="utf-8") as output:
            output.write("[")
            for i in range(n_docs):
                if i:
                    output.write(",")
                output.write(str(int.from_bytes(source.read(8), "little")))
            output.write("]\n")
    return count, n_docs


def _select_mixture(db, sources, spec):
    mixture = spec.get("mixture")
    if mixture is None:
        return None
    if spec.get("token_budgets", {}).get("train") is not None:
        raise ValueError("Mixture document quotas require packing the entire selected train set.")
    quotas = mixture["document_quotas"]
    if (not quotas or any(type(n) is not int or n < 1 for n in quotas.values())
            or not set(quotas) <= {s["id"] for s in sources}):
        raise ValueError("Mixture requires positive document quotas keyed by source ID.")
    db.execute("CREATE TABLE IF NOT EXISTS mixture_train(doc INTEGER PRIMARY KEY)")
    for source_id, count in quotas.items():
        slots = [i for i, s in enumerate(sources) if s["id"] == source_id]
        sql = ",".join("?" for _ in slots)
        selected = db.execute(f"SELECT COUNT(*) FROM docs WHERE split='train' AND source IN ({sql})",
                              slots).fetchone()[0]
        if selected < count:
            raise ValueError(f"{source_id} has {selected} eligible documents, needs {count}.")
        db.execute(f"INSERT OR IGNORE INTO mixture_train SELECT rowid FROM docs "
                   f"WHERE split='train' AND source IN ({sql}) ORDER BY rowid LIMIT ?",
                   (*slots, count))
    db.commit()
    return {"policy": "first eligible deduplicated documents per source; seeded segment shuffle at train time",
            "document_counts": quotas,
            "document_weights": {k: n / sum(quotas.values()) for k, n in quotas.items()}}


def prepare_streaming(spec: dict, out: Path) -> dict:
    """Write a new, immutable packed version. Re-run with the same spec to resume."""
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"Prepared version already exists: {out}")
    staged = out.with_name(out.name + ".staging")
    staged.mkdir(parents=True, exist_ok=True)
    lock = staged / ".writer.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(str(os.getpid()))
    try:
        return _prepare_locked(spec, out, staged)
    finally:
        lock.unlink()


def _prepare_locked(spec: dict, out: Path, staged: Path) -> dict:
    sources = _pin_sources(spec)
    pinned = {**spec, "sources": sources, "pipeline_version": 1}
    plan = staged / "plan.json"
    if plan.exists() and json.loads(plan.read_text(encoding="utf-8")) != pinned:
        raise ValueError("Input settings or source hashes differ from the staged run.")
    _atomic_json(plan, pinned)
    ready = staged / "ready"
    ready.mkdir(exist_ok=True)
    with closing(sqlite3.connect(staged / "documents.sqlite")) as db:
        _ingest(db, sources)
        mixture = _select_mixture(db, sources, spec)
        tokenizer = _make_tokenizer(db, ready / "tokenizer.json", spec)
        meta = {
            "complete": True, "dataset": {"name": spec["name"], "version": spec["version"]},
            "tokenizer": {"path": str(out / "tokenizer.json"), "version": "stream-byte-bpe-v1",
                          "sha256": sha256_file(ready / "tokenizer.json"),
                          "vocab_size": tokenizer.get_vocab_size(), "eos_id": tokenizer.token_to_id(EOS)},
            "tokens": {"dtype": "uint16", "vocab_size": tokenizer.get_vocab_size()},
            "files": {}, "packed_sha256": {}, "documents": {},
            "ingestion": [{"source": sources[i]["id"], "path": sources[i]["path"],
                           "scanned": n, "invalid": bad, "exact_duplicates": duplicates}
                          for i, n, bad, duplicates, _ in db.execute(
                              "SELECT * FROM progress ORDER BY source")],
        }
        if mixture is not None:
            meta["mixture"] = mixture
        for split in SPLITS:
            count, n_docs = _pack(db, ready, split, tokenizer, spec.get("token_budgets", {}).get(split))
            if not count:
                if split != "test":
                    raise ValueError(f"No usable {split} documents")
                continue
            meta["tokens"][split] = count
            meta["documents"][split] = n_docs
            for key, filename in ((f"{split}_bin", f"{split}.bin"),
                                  (f"{split}_document_ends", f"{split}_document_ends.json")):
                meta["files"][key] = str(out / filename)
                meta["packed_sha256"][key] = sha256_file(ready / filename)
        _atomic_json(ready / "meta.json", meta)
        _atomic_json(ready / "source-manifest.json", pinned)
    ready.rename(out)
    return meta
