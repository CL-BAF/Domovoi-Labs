"""Evaluation framework.

EVALS.md defines two layers: fast research evals (this module) and full
milestone evals (future).  Each task is a small class with a `run()` that
returns a machine-readable dict; the registry maps task names to classes so
new benchmarks slot in without touching the harness.

Multiple-choice tasks (HellaSwag, ARC-Challenge, PIQA, WinoGrande) are
scored by conditional log-likelihood — the standard lm-evaluation-harness
approach — reporting both raw accuracy (`acc`) and length-normalized
accuracy (`acc_norm`, normalized by choice character length).

Data files live in `data/evals/<task>/data.jsonl` with sidecar `meta.json`
recording source, split, sha256 and sample IDs — prepared by
scripts/download_eval_data.py.  Nothing in the *training* pipeline ever
reads this directory (see tests/test_data.py::test_eval_data_isolation).
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import torch

EVAL_DATA_ROOT = pathlib.Path("data") / "evals"


def read_eval_data(task: str, data_root: pathlib.Path = EVAL_DATA_ROOT) -> tuple[list[dict], dict]:
    """Load a prepared eval dataset: (records, meta).  Fails with a clean
    message if the task has not been downloaded yet."""
    task_dir = pathlib.Path(data_root) / task
    meta_path = task_dir / "meta.json"
    data_path = task_dir / "data.jsonl"
    if not meta_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(
            f"Evaluation data for task '{task}' not found at {task_dir}. "
            f"Run scripts/download_eval_data.py first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    records = [json.loads(l) for l in
               data_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return records, meta


@torch.no_grad()
def choice_logprob(model, context_ids: list[int], choice_ids: list[int],
                   device: torch.device, autocast_dtype: Optional[str] = None) -> float:
    """Sum of log-probs of `choice_ids` given `context_ids` (one sequence)."""
    ids = torch.tensor([context_ids + choice_ids], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type,
                         dtype=getattr(torch, autocast_dtype) if autocast_dtype else torch.float32,
                         enabled=autocast_dtype is not None):
        logits, _, _ = model(ids)
    logits = logits[0].float()  # (T, vocab)
    logprobs = torch.log_softmax(logits, dim=-1)
    n_ctx, n_choice = len(context_ids), len(choice_ids)
    if n_choice == 0:
        return 0.0
    # positions predicting choice tokens are n_ctx-1 .. n_ctx+n_choice-2
    sel = logprobs[n_ctx - 1: n_ctx + n_choice - 1]          # (n_choice, vocab)
    token_lp = sel[torch.arange(n_choice), torch.tensor(choice_ids, device=device)]
    return float(token_lp.sum().item())


class MultipleChoiceEval:
    """Generic log-likelihood multiple-choice evaluation.

    Expected record format (data.jsonl):
        {"id": ..., "context": str, "choices": [str, ...], "label": int}
    """
    name = "mc"

    def __init__(self, task_dir_name: str, display_name: str):
        self.task_dir_name = task_dir_name
        self.display_name = display_name

    def run(self, model, tokenizer, device, limit: Optional[int] = None,
            data_root: pathlib.Path = EVAL_DATA_ROOT, **_) -> dict:
        records, meta = read_eval_data(self.task_dir_name, data_root)
        if limit is not None:
            records = records[:limit]
        if not records:
            raise RuntimeError(f"Eval '{self.task_dir_name}' has no records.")
        model.eval()
        n_correct, n_correct_norm = 0, 0
        for rec in records:
            # Shared context, or per-choice contexts (WinoGrande-style tasks
            # where each option changes the conditioning prefix).
            contexts = rec.get("contexts") or [rec["context"]] * len(rec["choices"])
            scores, scores_norm = [], []
            for ctx, choice in zip(contexts, rec["choices"]):
                ctx_ids = tokenizer.encode(ctx)
                # byte-level BPE: a leading space keeps word boundaries clean
                ch_ids = tokenizer.encode(choice) if choice.startswith(" ") \
                    else tokenizer.encode(" " + choice)
                lp = choice_logprob(model, ctx_ids, ch_ids, device)
                scores.append(lp)
                scores_norm.append(lp / max(1, len(choice)))  # char-length norm
            gold = rec["label"]
            n_correct += int(max(range(len(scores)), key=lambda i: scores[i]) == gold)
            n_correct_norm += int(max(range(len(scores_norm)), key=lambda i: scores_norm[i]) == gold)
        n = len(records)
        n_choices = len(records[0]["choices"])
        return {
            "task": self.task_dir_name,
            "display_name": self.display_name,
            "metric": "accuracy",
            "acc": round(n_correct / n, 4),
            "acc_norm": round(n_correct_norm / n, 4),
            "chance": round(1.0 / n_choices, 4),
            "n": n,
            "data_sha256": meta.get("sha256"),
            "data_source": meta.get("source"),
            "limit": limit,
        }


class HellaSwagEval(MultipleChoiceEval):
    name = "hellaswag"
    def __init__(self):
        super().__init__("hellaswag", "HellaSwag (validation)")


class ARCEval(MultipleChoiceEval):
    name = "arc_challenge"
    def __init__(self):
        super().__init__("arc_challenge", "ARC-Challenge (test)")


class PIQAEval(MultipleChoiceEval):
    name = "piqa"
    def __init__(self):
        super().__init__("piqa", "PIQA (validation)")


class WinoGrandeEval(MultipleChoiceEval):
    name = "winogrande"
    def __init__(self):
        super().__init__("winogrande", "WinoGrande XL (validation)")


# ----------------------------------------------------------------------
# Registry: fast research evals.  Milestone evals (MMLU-Pro, GPQA,
# MATH-500, LiveCodeBench, IFBench, RULER) plug in here later with the
# same interface.
# ----------------------------------------------------------------------
TASKS = {
    "val_ppl": None,  # special-cased by the harness (needs training data)
    "hellaswag": HellaSwagEval,
    "arc_challenge": ARCEval,
    "piqa": PIQAEval,
    "winogrande": WinoGrandeEval,
    "gsm8k": None,    # wired by evals/gsm8k.py below
    "if_mini": None,  # wired by evals/if_mini.py below
}

from .gsm8k import GSM8KSubsetEval       # noqa: E402
from .full_split_nll import FullSplitNLLEval  # noqa: E402
from .if_mini import IFMiniEval           # noqa: E402
from .val_ppl import ValPPLEval          # noqa: E402

TASKS["gsm8k"] = GSM8KSubsetEval
TASKS["if_mini"] = IFMiniEval
TASKS["val_ppl"] = ValPPLEval
TASKS["val_nll"] = FullSplitNLLEval("val")
TASKS["test_nll"] = FullSplitNLLEval("test")
TASKS["val_nll_exits"] = FullSplitNLLEval("val", early_exits=True)

DEFAULT_TASKS = ["val_ppl", "hellaswag", "arc_challenge", "piqa",
                 "winogrande", "gsm8k", "if_mini"]


def build_task(name: str):
    cls = TASKS.get(name)
    if cls is None:
        raise ValueError(f"Unknown eval task {name!r}. Available: {sorted(TASKS)}")
    return cls() if isinstance(cls, type) else cls


def run_evals(
    model, tokenizer, device, tasks: list[str], limit: Optional[int] = None,
    training_config=None, data_root: pathlib.Path = EVAL_DATA_ROOT,
    n_val_batches: Optional[int] = None,
    checkpoint_meta: Optional[dict] = None,
) -> dict:
    """Run a set of eval tasks; returns machine-readable results dict.
    n_val_batches=None lets val_ppl default to the training config's
    eval_batches (directly comparable with training-time validation)."""
    results = {}
    for name in tasks:
        task = build_task(name)
        if name == "val_ppl":
            if training_config is None:
                raise ValueError(
                    "val_ppl requires the training config (dataset paths); "
                    "pass training_config=<Config>.")
            res = task.run(model, tokenizer, device,
                           training_config=training_config,
                           n_batches=n_val_batches)
        elif isinstance(task, FullSplitNLLEval):
            if training_config is None:
                raise ValueError("Full-split NLL requires the checkpoint training config.")
            if limit is not None or n_val_batches is not None:
                raise ValueError("Full-split NLL does not support evaluation limits.")
            res = task.run(model, tokenizer, device, training_config, checkpoint_meta)
        else:
            res = task.run(model, tokenizer, device, limit=limit,
                           data_root=data_root)
        results[name] = res
        metric = res.get("acc", res.get("val_loss", res.get("nll")))
        print(f"[eval] {res.get('display_name', name)}: {metric} "
              f"(n={res.get('n', '?')})", flush=True)
    return {"tasks": results}
