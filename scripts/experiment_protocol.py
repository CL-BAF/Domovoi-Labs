"""Inspect protocol state or close a crashed launch without model execution."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.experiments.launch import recover
from domovoi.experiments.protocol import Ledger, ProtocolError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="read the live canonical ledger")
    crash = commands.add_parser("recover", help="publish the terminal state of a dead local launch")
    crash.add_argument("experiment_id")
    crash.add_argument("--note", required=True, help="public, non-sensitive account of the crash")
    args = parser.parse_args()
    try:
        if args.command == "status":
            with Ledger() as ledger:
                head, state = ledger.refresh()
                ledger.verify_history(head)
            print(json.dumps({"head": head, **state}, indent=2))
        else:
            recover(args.experiment_id, args.note)
            print(f"{args.experiment_id} is terminal. Analyse and publish its evidence to advance.")
        return 0
    except (ProtocolError, OSError) as exc:
        print(f"[protocol] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
