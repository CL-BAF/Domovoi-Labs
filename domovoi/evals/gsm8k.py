"""Fixed GSM8K subset: cheap generation-based math eval.

Uses the first `subset_size` examples of the canonical prepared GSM8K test
file (fixed sample IDs, recorded in the result) so every run scores exactly
the same problems.  Greedy decoding, fixed prompt format, fixed answer
extraction.  A 40M model scores near zero — that is expected and fine; the
point is a comparable gradient across architectures.
"""

from __future__ import annotations

import re
from typing import Optional

import torch

from .base import read_eval_data
from ..inference.generate import generate

DEFAULT_SUBSET = 100
MAX_NEW_TOKENS = 64

# a number: optional sign, comma-grouped digits, optional decimal part.
# The decimal part requires digits after the dot, so "7." extracts as "7"
# and sentence-final periods never leak into the answer.
_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def extract_answer(text: str) -> Optional[str]:
    """Fixed answer extractor: GSM8K gold answers appear after '####'.
    From the *generated* text we take the last number."""
    m = re.search(r"####\s*(.+)", text)
    if m:
        n = _NUM_RE.search(m.group(1))
        return n.group(0).replace(",", "") if n else None
    nums = _NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def answers_match(pred: Optional[str], gold: str) -> bool:
    """Numeric comparison (so '72.0' matches '72'); falls back to string
    equality when either side is not a clean number."""
    if pred is None:
        return False
    try:
        return float(pred) == float(gold)
    except ValueError:
        return pred.strip() == gold.strip()


def gold_answer(answer_field: str) -> str:
    """Extract the canonical numeric gold answer from a GSM8K record."""
    m = re.search(r"####\s*(.+)", answer_field)
    if not m:
        return answer_field.strip()
    n = _NUM_RE.search(m.group(1))
    return n.group(0).replace(",", "") if n else m.group(1).strip()


class GSM8KSubsetEval:
    name = "gsm8k"

    def run(self, model, tokenizer, device, limit: Optional[int] = None,
            data_root=None, **_) -> dict:
        records, meta = read_eval_data("gsm8k")
        n = limit if limit is not None else DEFAULT_SUBSET
        records = records[:n]
        if not records:
            raise RuntimeError("gsm8k eval data is empty.")
        model.eval()
        n_correct = 0
        for rec in records:
            prompt = f"Question: {rec['question']}\nAnswer:"
            result = generate(
                model, prompt, tokenizer,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,  # deterministic decoding
            )
            pred = extract_answer(result["new_text"])
            gold = gold_answer(rec["answer"])
            if answers_match(pred, gold):
                n_correct += 1
        return {
            "task": "gsm8k",
            "display_name": f"GSM8K subset ({len(records)})",
            "metric": "exact_match_accuracy",
            "acc": round(n_correct / len(records), 4),
            "chance": 0.0,
            "n": len(records),
            "example_ids": [r["id"] for r in records],
            "data_sha256": meta.get("sha256"),
            "data_source": meta.get("source"),
            "max_new_tokens": MAX_NEW_TOKENS,
            "decoding": "greedy",
            "limit": limit,
        }