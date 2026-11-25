"""Trusted research entry point. Use python -I -S -B with a clean checkout."""

from __future__ import annotations

import sys

if __name__ == "__main__" and not (
        sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
    raise SystemExit("Use python -I -S -B scripts/experiment_protocol.py.")

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def plain(path: Path) -> bool:
    info = path.lstat()
    return (not stat.S_ISLNK(info.st_mode) and
            not getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def verify_checkout(root: Path) -> str:
    """Verify raw Git blob bytes and the tree inventory before adding to sys.path."""
    # Local object verification needs no credentials or environment-supplied Git overrides.
    env = {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}
    env["GIT_NO_REPLACE_OBJECTS"] = "1"

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, env=env, timeout=45)

    head = git("rev-parse", "HEAD").decode().strip()
    git("diff-index", "--cached", "--quiet", "HEAD", "--")
    blobs = {}
    directories = set()
    for entry in git("ls-tree", "-rz", head).split(b"\0"):
        if not entry:
            continue
        meta, name = entry.split(b"\t", 1)
        mode, kind, oid = meta.split(b" ")
        relative = name.decode("utf-8")
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError(f"Only plain files are supported in research source: {relative}")
        blobs[relative] = oid.decode()
        directories.update(p.as_posix() for p in Path(relative).parents if p != Path("."))
    seen = set()
    for path, dirs, files in os.walk(root):
        here = Path(path)
        if here == root and ".git" in dirs:
            dirs.remove(".git")
        for name in dirs:
            child = here / name
            if child.relative_to(root).as_posix() not in directories or not plain(child):
                raise ValueError(f"Unexpected source directory: {child.relative_to(root)}")
        for name in files:
            if here == root and name == ".git":
                continue
            file = here / name
            relative = file.relative_to(root).as_posix()
            if relative not in blobs or not plain(file) or not file.is_file():
                raise ValueError(f"Unexpected source file: {relative}")
            content = file.read_bytes()
            digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content)
            if digest.hexdigest() != blobs[relative]:
                raise ValueError(f"Source file differs from its Git blob: {relative}")
            seen.add(relative)
    if seen != blobs.keys():
        raise ValueError("Tracked source files are missing.")
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="read the canonical ledger")
    crash = commands.add_parser("recover", help="close an inactive claim")
    crash.add_argument("experiment_id")
    crash.add_argument("--note", required=True, help="public, non-sensitive evidence of failure")
    for command in ("run", "worker"):
        runner = commands.add_parser(command)
        runner.add_argument("experiment_id")
        runner.add_argument("--workspace", type=Path, required=True)
        runner.add_argument("--dependencies", type=Path, required=True)
        if command == "worker":
            runner.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        source = verify_checkout(ROOT)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[protocol] Unclean source checkout: {exc}", file=sys.stderr)
        return 1
    sys.path.insert(0, str(ROOT))
    from domovoi.experiments.launch import recover, run_approved, run_worker
    from domovoi.experiments.protocol import Ledger, ProtocolError

    try:
        if args.command == "status":
            with Ledger() as ledger:
                head, state = ledger.refresh()
                ledger.verify_history(head)
            print(json.dumps({"head": head, **state}, indent=2))
        elif args.command == "recover":
            recover(args.experiment_id, args.note)
        elif args.command == "run":
            return run_approved(args.experiment_id, source, args.workspace, args.dependencies)
        else:
            return run_worker(args.experiment_id, source, args.workspace, args.dependencies,
                              args.worker_args)
        return 0
    except (ProtocolError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[protocol] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
