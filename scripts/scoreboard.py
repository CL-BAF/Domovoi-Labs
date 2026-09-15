"""Aggregate all experiments into results/scoreboard.md.

Usage:
    python scripts/scoreboard.py

One row per experiment with status, params, tokens, val loss, benchmark
scores, VRAM and speeds — all read from stored JSON artifacts.  The
published-vs-reproduced distinction from EVALS.md applies to external
baselines, which are not in this table yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.experiments import list_experiments

BENCHES = ["hellaswag", "arc_challenge", "piqa", "winogrande", "gsm8k", "if_mini"]


def _load(exp_dir: Path, name: str) -> dict:
    p = exp_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def main() -> int:
    root = Path("experiments")
    exps = list_experiments(root)
    lines = [
        "# Domovoi Labs — experiment scoreboard",
        "",
        "Generated from stored experiment artifacts.  All scores are "
        "Domovoi-harness-reproduced numbers (published external scores live "
        "in BASELINES.md and are marked as reference-only there).",
        "",
        "| Experiment | Status | Arch | Params | Tokens | Val loss | "
        + " | ".join(BENCHES)
        + " | Peak VRAM (MiB) | Train tok/s | Decode tok/s |",
        "|---|---|---|---:|---:|---:|" + "---:|" * len(BENCHES) + "---:|---:|---:|",
    ]
    for e in exps:
        exp_dir = e["dir"]
        summary = _load(exp_dir, "summary.json")
        evals = _load(exp_dir, "evals.json").get("tasks", {})
        profile = _load(exp_dir, "profile.json")
        inf = profile.get("inference", {})
        params = e.get("parameters") or profile.get("parameters", {}).get("total")
        cells = []
        for b in BENCHES:
            r = evals.get(b) or {}
            cells.append(f"{r.get('acc_norm', r.get('acc', '—')):.4f}"
                         if r else "—")
        vram = inf.get("peak_vram_allocated_mb")
        lines.append(
            f"| {e['experiment_id']} | {e['status']} | {e.get('architecture', '—')} "
            f"| {params or '—'} | {summary.get('tokens_trained') or '—'} "
            f"| {summary.get('final_val_loss', '—')} "
            f"| " + " | ".join(cells) +
            f" | {vram if vram is not None else '—'} "
            f"| {summary.get('avg_tok_per_s', '—')} "
            f"| {inf.get('decode_tok_per_s_median', '—')} |")

    out = Path("results") / "scoreboard.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[scoreboard] written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())