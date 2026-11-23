"""Controlled-experiment matrix for domovoi_recurrent_v1 (Experiments A/B/C).

Every arm is derived from ONE base config (default configs/baseline_small.yaml)
with clone_config, so data, tokenizer, optimiser, schedule, seed and token
budget are identical; only model.{architecture, n_layers, recurrence_depth,
recurrence_order} differ.  Costs are closed-form estimates from the config.

This script trains nothing and touches no experiment directory.

WARNING: on the default tinyshakespeare base (~0.3M train tokens, 3000 steps x
8192 tokens ~ 80 epochs) arms with fewer stored parameters may win on
validation loss through regularisation alone.  Results there test the
plumbing, not the hypothesis; see docs/RECURRENT_ARCHITECTURE_RESEARCH.md.

Usage:
    python scripts/plan_recurrent_experiments.py
    python scripts/plan_recurrent_experiments.py --write-configs configs/recurrent
    python scripts/plan_recurrent_experiments.py --pilot --seeds 1234 1235 1236 \
        --base PILOT_BASE.yaml --write-configs configs/pilot
Generated configs never inherit output.experiment_id, and existing files
are only overwritten with --force.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.config import clone_config, load_config
from domovoi.profiling import architecture_accounting

# (arm id, architecture, stored blocks, recurrence depth, order)
ARMS = [
    ("B02", "baseline_transformer", 2, 1, "cycle"),
    ("B04", "baseline_transformer", 4, 1, "cycle"),
    ("B08", "baseline_transformer", 8, 1, "cycle"),
    ("B16", "baseline_transformer", 16, 1, "cycle"),
    ("R02x2", "domovoi_recurrent_v1", 2, 2, "cycle"),
    ("R02x4", "domovoi_recurrent_v1", 2, 4, "cycle"),
    ("R02x8", "domovoi_recurrent_v1", 2, 8, "cycle"),
    ("R01x8", "domovoi_recurrent_v1", 1, 8, "cycle"),
    ("R04x2", "domovoi_recurrent_v1", 4, 2, "cycle"),
    ("R02x4s", "domovoi_recurrent_v1", 2, 4, "sequence"),
]
# R02x1 would be bit-identical to B02 (tested), so B02 serves as depth 1.
COMPARISONS = {
    # depth ablation == param-matched comparison against B02 (same stored weights)
    "C/P2 depth ablation, param-matched to B02": ["B02", "R02x2", "R02x4", "R02x8"],
    "P4 param-matched to B04": ["B04", "R04x2"],
    "F4 compute-matched, effective depth 4": ["B04", "R02x2"],
    "F8 compute-matched, effective depth 8": ["B08", "R04x2", "R02x4", "R01x8"],
    "F16 compute-matched, effective depth 16": ["B16", "R02x8"],
    "O  reuse-order ablation": ["R02x4", "R02x4s"],
}


# the three-arm pilot of docs/DOMOVOI_PILOT_PLAN.md
PILOT_ARMS = ("B02", "R02x4", "B08")


def build_arms(base, only=None, seeds=None):
    """Yield (name, config) per arm and seed.  Every generated config gets
    output.experiment_id = None so each run allocates a fresh ID; a base
    config that pins an ID (e.g. DML-0001) can never be inherited."""
    for arm, arch, n, depth, order in ARMS:
        if only is not None and arm not in only:
            continue
        for seed in (seeds or [base.training.seed]):
            name = arm if seeds is None else f"{arm}_s{seed}"
            cfg = clone_config(base, model={
                "architecture": arch, "n_layers": n,
                "recurrence_depth": depth, "recurrence_order": order},
                training={"seed": seed}, output={"experiment_id": None})
            cfg.experiment.name = f"recurrent_matrix_{name}"
            cfg.experiment.notes = (f"Recurrent matrix arm {arm}: {n} stored block(s) x depth "
                                    f"{depth} ({order}), seed {seed}. Derived from "
                                    f"{base.source_path}.")
            # pipeline success is not a research verdict: baselines keep their
            # label, candidates get the neutral COMPLETED
            cfg.experiment.status = "BASELINE" if arch == "baseline_transformer" else "COMPLETED"
            yield name, cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="configs/baseline_small.yaml")
    ap.add_argument("--write-configs", default=None, metavar="DIR")
    ap.add_argument("--pilot", action="store_true",
                    help=f"only the three pilot arms {PILOT_ARMS}")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="one config per arm per seed (default: the base seed only)")
    ap.add_argument("--force", action="store_true",
                    help="allow --write-configs to overwrite existing files")
    args = ap.parse_args()

    base = load_config(args.base)
    arms = dict(build_arms(base, only=PILOT_ARMS if args.pilot else None,
                           seeds=args.seeds))
    T = base.training.context_length
    print(f"base: {args.base}  d_model={base.model.d_model} ffn={base.model.ffn_dim} "
          f"vocab={base.model.vocab_size} ctx={base.model.context_length}  (ESTIMATES, bf16)")
    print(f"{'arm':12} {'stored':>6} {'depth':>5} {'eff':>4} {'unique M':>9} {'applied M':>9} "
          f"{'weights MiB':>11} {'KV MiB@ctx':>10} {'train GFLOP/tok':>15}")
    for arm, cfg in arms.items():
        a = architecture_accounting(cfg.model, attended_tokens=(T + 1) / 2)
        kv = architecture_accounting(cfg.model)["kv_cache_bytes_full_context"]
        print(f"{arm:12} {cfg.model.n_layers:6d} {cfg.model.recurrence_depth:5d} "
              f"{a['effective_depth']:4d} {a['unique_params'] / 1e6:9.2f} "
              f"{a['applied_params'] / 1e6:9.2f} {a['weight_bytes'] / 2**20:11.1f} "
              f"{kv / 2**20:10.1f} {a['est_train_flops_per_token'] / 1e9:15.3f}")
    print("\ncomparison groups:")
    for name, members in COMPARISONS.items():
        print(f"  {name}: {', '.join(members)}")

    if args.write_configs:
        out = Path(args.write_configs)
        existing = [str(out / f"{arm}.yaml") for arm in arms if (out / f"{arm}.yaml").exists()]
        if existing and not args.force:
            # fail before writing anything: a half-overwritten plan is worse
            print(f"refusing to overwrite existing configs (pass --force): {existing}",
                  file=sys.stderr)
            return 1
        out.mkdir(parents=True, exist_ok=True)
        for arm, cfg in arms.items():
            raw = cfg.to_dict()
            raw.pop("source_path", None)          # not a config section
            (out / f"{arm}.yaml").write_text(
                yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        print(f"\nwrote {len(arms)} configs to {out} (training them requires authorization)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
