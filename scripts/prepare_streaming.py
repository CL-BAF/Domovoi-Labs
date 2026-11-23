"""Prepare a versioned document corpus from a JSON source specification."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from domovoi.datasets.streaming import prepare_streaming


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_streaming(json.loads(args.spec.read_text(encoding="utf-8-sig")), args.out)
    print(json.dumps(manifest, indent=2))
