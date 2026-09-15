"""Set or inspect an experiment's status label.

Usage:
    python scripts/set_status.py DML-0001 PROMISING --notes "beats baseline on ..."
    python scripts/set_status.py DML-0001

Statuses: RUNNING, FAILED, INVALID, BASELINE, PROMISING, REJECTED,
PROMOTED, COMPLETED, INTERRUPTED.  Failed experiments are never deleted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.experiments import load_metadata, set_status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment", help="experiment ID or dir")
    ap.add_argument("status", nargs="?", default=None)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    meta = load_metadata(args.experiment)
    if args.status is None:
        print(f"{meta['experiment_id']}: {meta['status']}")
        return 0
    set_status(args.experiment, args.status, notes=args.notes)
    print(f"{meta['experiment_id']}: status set to {args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())