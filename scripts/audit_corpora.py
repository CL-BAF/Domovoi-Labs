"""Inventory local TinyStories, WikiText-103 and FineWeb-Edu files read-only."""

import argparse
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.datasets.streaming import _atomic_json, documents
from domovoi.datasets.tokenizer import sha256_file

DATASETS = {
    "TinyStories": ("roneneldan/TinyStories", "CDLA-Sharing-1.0",
                    {"TinyStories-train.txt", "TinyStories-valid.txt"}),
    "WikiText-103": ("Salesforce/wikitext", "CC BY-SA 3.0",
                     {f"wikitext-103-raw-v1/{name}.parquet" for name in (
                         "train-00000-of-00002", "train-00001-of-00002",
                         "validation-00000-of-00001", "test-00000-of-00001")}),
    "FineWeb-Edu": ("HuggingFaceFW/fineweb-edu", "ODC-BY-1.0",
                    {f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)}),
}


def inspect_file(path, marker):
    entry = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if marker.is_file():
        revision, etag, *_ = marker.read_text(encoding="utf-8").splitlines()
        entry.update(revision=revision, upstream_etag=etag, hash_matches_etag=etag == entry["sha256"])
    if path.suffix == ".txt":
        count = empty = 0
        for doc in documents(path, "text"):
            count += 1
            empty += not doc.strip()
        entry.update(records=count, null=0, blank=empty, encoding="strict UTF-8")
    else:
        parquet = pq.ParquetFile(path, page_checksum_verification=True)
        count = null = empty = 0
        for batch in parquet.iter_batches(batch_size=4096, columns=["text"]):
            arr = batch.column(0)
            arr.validate(full=True)
            count += len(arr)
            null += arr.null_count
            empty += pc.sum(pc.equal(pc.utf8_trim_whitespace(arr), "")).as_py() or 0
        if count != parquet.metadata.num_rows:
            raise ValueError("Parquet metadata and decoded row counts disagree")
        entry.update(records=count, null=null, blank=empty, schema=str(parquet.schema_arrow),
                     row_groups=parquet.num_row_groups, columns_scanned=["text"])
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = {}
    for dataset, (repository, license, expected) in DATASETS.items():
        root = args.root / dataset
        all_files = [p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()]
        entry = report[dataset] = {
            "repository": repository, "license": license,
            "missing": sorted(expected - set(all_files)),
            "temporary": sorted(p for p in all_files if p.endswith((".incomplete", ".part", ".tmp", ".lock"))),
            "files": {},
        }
        for relative in sorted(expected & set(all_files)):
            try:
                data = inspect_file(root / relative,
                                    root / ".cache/huggingface/download" / (relative + ".metadata"))
            except (OSError, ValueError) as exc:
                data = {"error": str(exc)}
            entry["files"][relative] = data
            print(dataset, relative, {k: data.get(k) for k in
                                     ("bytes", "records", "blank", "null", "hash_matches_etag", "error")},
                  flush=True)
        _atomic_json(args.report, report)
    return any(e["missing"] or e["temporary"] or any(
        "error" in f or f.get("hash_matches_etag") is False for f in e["files"].values()
    ) for e in report.values())


if __name__ == "__main__":
    sys.exit(main())
