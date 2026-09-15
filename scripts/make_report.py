"""Generate a human-readable experiment report from stored result data.

Usage:
    python scripts/make_report.py --experiment DML-0001
    python scripts/make_report.py --experiment DML-0002 --baseline DML-0001

The report is built only from the experiment's stored JSON artifacts, so it
can be regenerated identically at any time (no model, no re-run).

Optional extras (merged into metadata before rendering):
    --conclusion "..."
    --recommendation "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.experiments import generate_report, load_metadata, update_metadata
from domovoi.experiments.registry import load_metadata as _load  # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--baseline", default=None,
                    help="baseline experiment ID for the comparison section")
    ap.add_argument("--conclusion", default=None)
    ap.add_argument("--recommendation", default=None)
    args = ap.parse_args()

    meta = load_metadata(args.experiment)
    exp_dir = Path(args.experiment) if Path(args.experiment).is_dir() \
        else Path("experiments") / meta["experiment_id"]

    updates = {}
    if args.conclusion:
        updates["conclusion"] = args.conclusion
    if args.recommendation:
        updates["recommendation"] = args.recommendation
    if updates:
        update_metadata(exp_dir, updates)

    baseline_dir = None
    if args.baseline:
        base_meta = load_metadata(args.baseline)
        baseline_dir = Path(args.baseline) if Path(args.baseline).is_dir() \
            else Path("experiments") / base_meta["experiment_id"]

    text = generate_report(exp_dir, baseline_dir=baseline_dir)
    print(text)
    print(f"\n[report] written to {exp_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())