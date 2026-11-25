"""Inspect an architecture from a config, without training (CPU only).

Builds the model with RANDOM weights on CPU, prints exact parameter counts,
closed-form estimates (clearly labelled), and optionally runs an offline
cached-vs-uncached greedy decode smoke test. The workload needs no data,
tokenizer or GPU. Its research claim requires GitHub.

Usage:
    python scripts/inspect_model.py --research-id DRP-NNNN --config configs/baseline_small.yaml
    python scripts/inspect_model.py --research-id DRP-NNNN --config configs/smoke_test.yaml \
        --architecture domovoi_recurrent_v1 --n-layers 2 --depth 4 --smoke
"""

from __future__ import annotations

if __name__ == "__main__":
    raise SystemExit("Use python -I -S -B scripts/experiment_protocol.py run for research.")

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.experiments.launch import run_research
from domovoi.config import clone_config, load_config
from domovoi.models import build_model
from domovoi.profiling import architecture_accounting, count_parameters


@torch.no_grad()
def cache_smoke(model, n_prompt: int = 16, n_new: int = 16, seed: int = 0) -> dict:
    """Greedy-decode random-prompt tokens with and without the KV cache and
    compare logits at every step.  Random weights: tests plumbing, not quality."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, model.cfg.vocab_size, (1, n_prompt), generator=g)
    n_new = min(n_new, model.cfg.context_length - n_prompt)
    t0 = time.perf_counter()
    out = model(ids, use_cache=True)
    cached_logits, seq = [out.logits[0, -1]], ids
    for _ in range(n_new - 1):
        nxt = cached_logits[-1].argmax().view(1, 1)
        seq = torch.cat([seq, nxt], dim=1)
        out = model(nxt, use_cache=True, caches=out.caches)
        cached_logits.append(out.logits[0, -1])
    cached_s = time.perf_counter() - t0
    full = model(seq).logits[0, n_prompt - 1:]          # uncached reference
    diff = (torch.stack(cached_logits) - full).abs().max().item()
    return {"prompt_tokens": n_prompt, "new_tokens": n_new,
            "max_abs_logit_diff_cached_vs_uncached": diff,
            "match_atol_1e-4": diff < 1e-4,
            "cpu_wall_clock_s_cached_decode": round(cached_s, 4),
            "note": "random weights, CPU wall-clock; NOT a GPU measurement"}


def main() -> int:
    return run_research("inspect_model.py", _run)


def _run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--architecture", default=None)
    ap.add_argument("--n-layers", type=int, default=None, help="stored (distinct) blocks")
    ap.add_argument("--depth", type=int, default=None, help="recurrence_depth")
    ap.add_argument("--order", default=None, help="recurrence_order: cycle | sequence")
    ap.add_argument("--smoke", action="store_true", help="run the cached-vs-uncached check")
    ap.add_argument("--out", type=Path, help="save the report to a declared output file")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    overrides = {k: v for k, v in {
        "architecture": args.architecture, "n_layers": args.n_layers,
        "recurrence_depth": args.depth, "recurrence_order": args.order}.items()
        if v is not None}
    if overrides:
        cfg = clone_config(cfg, model=overrides)      # re-validated

    torch.manual_seed(cfg.training.seed)
    model = build_model(cfg.model).eval()             # CPU, random init
    report = {
        "architecture": cfg.model.architecture,
        "measured_parameter_counts": count_parameters(model),
        "param_breakdown": model.param_breakdown(),
        "analytic_estimates_bf16": architecture_accounting(cfg.model),
    }
    if args.smoke:
        report["offline_cache_smoke"] = cache_smoke(model)
    report_json = json.dumps(report, indent=2)
    if args.out is not None:
        with args.out.open("x", encoding="utf-8") as f:
            f.write(report_json + "\n")
    print(report_json)
    return 0 if not args.smoke or report["offline_cache_smoke"]["match_atol_1e-4"] else 1
