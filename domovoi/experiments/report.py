"""Human-readable experiment reports.

`generate_report` builds report.md purely from the experiment's *stored*
result files (metadata.json, summary.json, evals.json, profile.json,
generation.json) — never from live models or re-runs — so any report can be
regenerated identically at any time.

Comparison section: when `baseline_experiment` is given (a metadata dict of
another experiment), the report computes deltas on shared metrics.  The
"next recommended experiment" line comes from metadata's `recommendation`
field (set by the researcher) and is honestly rendered as a placeholder if
absent.
"""

from __future__ import annotations

import json
import pathlib


def _load(exp_dir: pathlib.Path, name: str) -> dict | None:
    p = exp_dir / name
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def _pct(delta: float) -> str:
    return f"{delta:+.1f}%"


def _fmt(x, default="—"):
    if x is None:
        return default
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".") if 0 < abs(x) < 1000 else f"{x:,.0f}"
    return str(x)


def _params_human(n) -> str:
    if n is None:
        return "—"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    return str(n)


def _tokens_human(n) -> str:
    if n is None:
        return "—"
    if n >= 1e9:
        return f"{n / 1e9:.2f}B tokens"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M tokens"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k tokens"
    return f"{n} tokens"


def _compare(result: dict, baseline: dict) -> list[str]:
    """Lines of metric-by-metric comparison between two experiment
    metadata/eval/profile payloads.  Only compares shared keys."""
    lines = []
    r_eval = result.get("evals", {}).get("tasks", {}) or {}
    b_eval = baseline.get("evals", {}).get("tasks", {}) or {}
    for task in sorted(set(r_eval) & set(b_eval)):
        for metric in ("acc_norm", "acc", "val_loss"):
            if metric in r_eval[task] and metric in b_eval[task]:
                a, b = r_eval[task][metric], b_eval[task][metric]
                if b in (0, None) or a is None:
                    continue
                d = (a - b) / abs(b) * 100.0
                lower_better = metric == "val_loss"
                good = (d < 0) if lower_better else (d > 0)
                sign = "+" if d > 0 else ""
                lines.append(
                    f"  {task}.{metric}: {b:.4f} → {a:.4f} ({sign}{d:.1f}% "
                    f"{'improvement' if good else 'regression'})")
                break
    r_prof = (result.get("profile", {}) or {})
    b_prof = (baseline.get("profile", {}) or {})
    ri, bi = r_prof.get("inference", {}), b_prof.get("inference", {})
    if ri.get("peak_vram_allocated_mb") and bi.get("peak_vram_allocated_mb"):
        a, b = ri["peak_vram_allocated_mb"], bi["peak_vram_allocated_mb"]
        d = (a - b) / b * 100.0
        lines.append(f"  peak inference VRAM: {b} MiB → {a} MiB ({_pct(d)})")
    if ri.get("decode_tok_per_s_median") and bi.get("decode_tok_per_s_median"):
        a, b = ri["decode_tok_per_s_median"], bi["decode_tok_per_s_median"]
        d = (a - b) / b * 100.0
        lines.append(f"  decode speed: {b:.1f} → {a:.1f} tok/s ({_pct(d)})")
    return lines or ["  (no shared metrics with the baseline experiment)"]


def generate_report(
    exp_dir: str | pathlib.Path,
    baseline_dir: str | pathlib.Path | None = None,
) -> str:
    """Render report.md content.  Also writes it to the experiment dir and
    returns the markdown text."""
    exp_dir = pathlib.Path(exp_dir)
    meta = json.loads((exp_dir / "metadata.json").read_text(encoding="utf-8"))
    summary = _load(exp_dir, "summary.json") or {}
    evals = _load(exp_dir, "evals.json") or {}
    profile = _load(exp_dir, "profile.json") or {}
    generation = _load(exp_dir, "generation.json")

    ev = evals.get("tasks", {})
    inf = profile.get("inference", {})
    params = profile.get("parameters", {}) or {}
    status = meta.get("status", "—")
    vram = inf.get("peak_vram_allocated_mb")

    lines = []
    lines.append(f"# Experiment {meta['experiment_id']} — {meta.get('name', '')}")
    lines.append("")
    lines.append(f"**Status:** {status}")
    lines.append("")
    lines.append("## Architecture")
    lines.append("")
    lines.append(f"- {meta.get('architecture', '—')} "
                 f"(d_model={meta['config']['model']['d_model']}, "
                 f"layers={meta['config']['model']['n_layers']}, "
                 f"heads={meta['config']['model']['n_heads']}, "
                 f"context={meta['config']['model']['context_length']})")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(f"- total: {_params_human(params.get('total') or meta.get('parameters'))}")
    lines.append(f"- trainable: {_params_human(params.get('trainable'))}")
    lines.append(f"- active per token: {_params_human(params.get('active_per_token') or meta.get('active_parameters'))}")
    if profile.get("checkpoint_size_mb") is not None:
        lines.append(f"- checkpoint size: {profile['checkpoint_size_mb']} MiB")
    lines.append("")
    lines.append("## Training")
    lines.append("")
    lines.append(f"- dataset: {meta.get('dataset', {}).get('name')} "
                 f"({meta.get('dataset', {}).get('version')})")
    lines.append(f"- tokens: {_tokens_human(summary.get('tokens_trained'))}")
    lines.append(f"- seed: {meta.get('seed')}")
    lines.append(f"- git commit: `{meta.get('git_commit', 'unknown')}`"
                 + (" *(dirty working tree)*" if meta.get("git_dirty") else ""))
    lines.append(f"- hardware: {meta.get('env', {}).get('gpu_name') or 'CPU only'}")
    if summary.get("final_val_loss") is not None:
        lines.append(f"- final train loss: {_fmt(summary.get('final_train_loss'))}")
        lines.append(f"- final val loss: {_fmt(summary['final_val_loss'])} "
                     f"(ppl {_fmt(summary.get('final_val_ppl'))})")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Eval | Score | Chance | n |")
    lines.append("|---|---:|---:|---:|")
    order = ["val_ppl", "hellaswag", "arc_challenge", "piqa", "winogrande",
             "gsm8k", "if_mini"]
    for task in order + [t for t in ev if t not in order]:
        r = ev.get(task)
        if not r:
            continue
        score = r.get("acc_norm") if "acc_norm" in r else \
            (r.get("val_loss") if task == "val_ppl" else r.get("acc"))
        label = r.get("display_name", task)
        if task == "val_ppl":
            lines.append(f"| {label} (loss) | {_fmt(score)} | — | "
                         f"{r.get('n_batches', '—')} |")
        else:
            lines.append(f"| {label} | {_fmt(score)} | {_fmt(r.get('chance'), '—')} "
                         f"| {r.get('n', '—')} |")
    if not ev:
        lines.append("| (no evaluation results stored) | | | |")
    lines.append("")
    lines.append("## Efficiency")
    lines.append("")
    if vram is not None:
        lines.append(f"- peak inference VRAM allocated: {vram} MiB")
    else:
        lines.append("- peak inference VRAM allocated: *not measured (no CUDA)*")
    if inf.get("peak_vram_reserved_mb") is not None:
        lines.append(f"- peak inference VRAM reserved: {inf['peak_vram_reserved_mb']} MiB")
    else:
        lines.append("- peak inference VRAM reserved: *not measured*")
    tvram = (summary.get("vram") or {})
    if tvram.get("peak_allocated_mb") is not None:
        lines.append(f"- peak training VRAM allocated: {tvram['peak_allocated_mb']} MiB")
        lines.append(f"- peak training VRAM reserved: {tvram['peak_reserved_mb']} MiB")
    if summary.get("avg_tok_per_s") is not None:
        lines.append(f"- training speed: {summary['avg_tok_per_s']:.0f} tok/s")
    if inf.get("decode_tok_per_s_median") is not None:
        lines.append(f"- decode speed: {inf['decode_tok_per_s_median']:.1f} tok/s")
    if inf.get("prefill_tok_per_s_median") is not None:
        lines.append(f"- prefill speed: {inf['prefill_tok_per_s_median']:.1f} tok/s")
    if inf.get("ttft_s_median") is not None:
        lines.append(f"- time to first token: {inf['ttft_s_median'] * 1000:.1f} ms")
    lines.append("")

    if baseline_dir is not None:
        base_meta = json.loads(
            (pathlib.Path(baseline_dir) / "metadata.json").read_text(encoding="utf-8"))
        lines.append("## Comparison")
        lines.append("")
        lines.append(f"Versus baseline **{base_meta['experiment_id']}** "
                     f"({base_meta.get('architecture')}):")
        lines.append("")
        result_payload = {"evals": evals, "profile": profile}
        base_payload = {
            "evals": _load(pathlib.Path(baseline_dir), "evals.json") or {},
            "profile": _load(pathlib.Path(baseline_dir), "profile.json") or {},
        }
        lines.extend(_compare(result_payload, base_payload))
        lines.append("")

    if generation:
        lines.append("## Generation sample")
        lines.append("")
        prompt = generation["prompt"].replace("\n", " ")
        out = generation["new_text"].replace("\n", " ")
        lines.append(f"> prompt: {prompt!r}")
        lines.append(">")
        lines.append(f"> output: {out!r}")
        lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    env = meta.get("env", {})
    lines.append(f"- python: {env.get('python', '—')}")
    lines.append(f"- torch: {env.get('torch', '—')}")
    lines.append(f"- CUDA: {env.get('cuda_version', 'unavailable (CPU run)')}")
    lines.append(f"- GPU: {env.get('gpu_name', '—')}")
    lines.append(f"- checkpoint: `{meta.get('checkpoints_dir')}/"
                 f"{(meta.get('checkpoints') or ['—'])[-1]}`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(meta.get("conclusion") or
                 "*(to be filled in by the researcher after reviewing the numbers)*")
    lines.append("")
    lines.append("## Next recommended experiment")
    lines.append("")
    lines.append(meta.get("recommendation") or
                 "*(to be filled in by the researcher)*")
    lines.append("")

    text = "\n".join(lines)
    (exp_dir / "report.md").write_text(text, encoding="utf-8")
    return text