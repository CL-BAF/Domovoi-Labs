"""Download and normalize the fast-research benchmark datasets.

Usage:
    python scripts/download_eval_data.py                # all tasks
    python scripts/download_eval_data.py --tasks hellaswag,piqa
    python scripts/download_eval_data.py --tasks all --limit 1000

Downloads each benchmark into `data/evals/<task>/` as:

    data.jsonl  - normalized records:
                  MC tasks: {"id", "context", "choices": [...], "label": int}
                  GSM8K:    {"id", "question", "answer"}
    meta.json   - source dataset + config + split, sha256 of data.jsonl,
                  sample count, datasets-library version, download date

This is the ONLY place benchmark data enters the repository.  The training
pipeline never reads `data/evals/` (enforced by tests/test_data.py).

Tasks: hellaswag, arc_challenge, piqa, winogrande, gsm8k.
Requires the optional `datasets` package: pip install -e ".[evals]"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EVAL_ROOT = Path("data") / "evals"

# HF dataset name -> (config, split)
SOURCES = {
    "hellaswag": ("hellaswag", None, "validation"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "test"),
    "piqa": ("ybisk/piqa", None, "validation"),
    "winogrande": ("allenai/winogrande", "winogrande_xl", "validation"),
    "gsm8k": ("openai/gsm8k", "main", "test"),
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_hellaswag(rows):
    for i, r in enumerate(rows):
        # lm-eval-harness convention: "Activity: context" then pick ending
        ctx = (r["ctx_a"] + " " + r["ctx_b"]).capitalize()
        context = f"{r['activity_label']}: {ctx}"
        yield {
            "id": r.get("ind", str(i)),
            "context": context.strip(),
            "choices": [e for e in r["endings"]],
            "label": int(r["label"]),
        }


def normalize_arc(rows):
    for i, r in enumerate(rows):
        context = f"Question: {r['question']}\nAnswer:"
        yield {
            "id": r.get("id", str(i)),
            "context": context,
            "choices": r["choices"]["text"],
            "label": r["choices"]["label"].index(r["answerKey"]),
        }


def normalize_piqa(rows):
    for i, r in enumerate(rows):
        context = f"Question: {r['goal']}\nAnswer:"
        yield {
            "id": str(i),
            "context": context,
            "choices": [r["sol1"], r["sol2"]],
            "label": int(r["label"]),
        }


def normalize_winogrande(rows):
    for i, r in enumerate(rows):
        # The sentence contains a "_" placeholder.  Standard scoring (GPT-3
        # paper / lm-eval-harness): score the *suffix after the blank* given
        # prefix-with-option-filled-in, so each choice has its own context.
        parts = r["sentence"].split("_")
        prefix, suffix = parts[0], parts[1] if len(parts) > 1 else ""
        options = [r["option1"], r["option2"]]
        yield {
            "id": str(i),
            "contexts": [prefix + opt for opt in options],
            "choices": [suffix, suffix],
            "label": int(r["answer"]) - 1,
        }


def normalize_gsm8k(rows):
    for i, r in enumerate(rows):
        yield {
            "id": f"gsm8k-test-{i}",
            "question": r["question"],
            "answer": r["answer"],
        }


NORMALIZERS = {
    "hellaswag": normalize_hellaswag,
    "arc_challenge": normalize_arc,
    "piqa": normalize_piqa,
    "winogrande": normalize_winogrande,
    "gsm8k": normalize_gsm8k,
}


def prepare_task(task: str, limit: int | None) -> Path:
    from datasets import load_dataset, __version__ as ds_version

    ds_name, config, split = SOURCES[task]
    print(f"[eval-data] downloading {task}: {ds_name} ({config or '-'}) [{split}]")
    ds = load_dataset(ds_name, config, split=split, trust_remote_code=False)

    normalizer = NORMALIZERS[task]
    records = list(normalizer(ds))
    if limit is not None:
        records = records[:limit]

    task_dir = EVAL_ROOT / task
    task_dir.mkdir(parents=True, exist_ok=True)
    data_path = task_dir / "data.jsonl"
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    data_path.write_text(text, encoding="utf-8")

    meta = {
        "task": task,
        "source": f"{ds_name}/{config}" if config else ds_name,
        "split": split,
        "n_examples": len(records),
        "n_total": len(ds),
        "sha256": _sha256_text(text),
        "datasets_version": ds_version,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "limit": limit,
    }
    (task_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[eval-data] {task}: {len(records)} examples → {data_path}")
    return task_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="all",
                    help="comma-separated task names, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only the first N examples (fast research mode)")
    args = ap.parse_args()

    if args.tasks == "all":
        tasks = list(SOURCES)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in SOURCES]
    if unknown:
        raise SystemExit(f"Unknown eval task(s): {unknown}. Available: {list(SOURCES)}")

    for task in tasks:
        prepare_task(task, args.limit)
    print("[eval-data] done — data recorded with sha256 for contamination tracking")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())