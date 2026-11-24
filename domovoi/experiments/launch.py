"""Enforce the research lifecycle at the supported executable boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from .protocol import (
    ID, TERMINAL, Ledger, ProtocolError, git, matches, now, published_implementation,
    read_json, require, slot,
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = {"real_data_campaign.py", "run_pilot.py"}
CONFIG_COMMANDS = {"train.py", "verify_real_resume.py", "inspect_model.py"}


@dataclass(frozen=True)
class Receipt:
    experiment_id: str
    directory: Path
    token: str
    script: str

    @property
    def digest(self) -> str:
        identity = json.dumps([self.token, self.script])
        return hashlib.sha256(identity.encode()).hexdigest()


_active: ContextVar[tuple[Receipt, str] | None] = ContextVar("domovoi_research_claim", default=None)


def receipt_directory(experiment_id: str) -> Path:
    require(matches(ID, experiment_id), "Expected a protocol ID such as DRP-0001.")
    common = git(ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(os.fsdecode(common).strip()) / "domovoi-protocol" / experiment_id


@contextmanager
def local_lock(path: Path):
    """An OS lock is released on process death. Retain its pathname across runs."""
    with path.open("a+b", buffering=0) as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            def acquire():
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            acquire()
        except OSError as exc:
            raise ProtocolError("The local experiment launcher or worker is still active.") from exc
        yield


def write_local(path: Path, data: dict) -> None:
    staging = path.with_suffix(".tmp")
    with staging.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=True)
        output.flush()
        os.fsync(output.fileno())
    staging.replace(path)


def read_receipt(experiment_id: str, directory: Path) -> Receipt:
    path = directory / "claim.json"
    require(path.is_file(), "Recovery requires the original launcher's local claim receipt.")
    data = read_json(path.read_bytes(), path.name)
    require(isinstance(data.get("token"), str) and isinstance(data.get("script"), str),
            "Invalid local claim receipt.")
    return Receipt(experiment_id, directory, data["token"], data["script"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_command(plan: dict, script: str, argv: list[str]) -> None:
    require(Path.cwd().resolve() == ROOT, "Launch from the implementation checkout root.")
    require(plan["command"] == {"script": f"scripts/{script}", "argv": argv},
            "This command does not match the exact committed plan.")
    head = git(ROOT, "rev-parse", "HEAD").decode().strip()
    require(plan["implementation_commit"] == head and
            not git(ROOT, "status", "--porcelain", "--untracked-files=no"),
            "The tracked checkout must be clean at the pinned implementation commit.")
    published_implementation(head)
    if script in CONFIG_COMMANDS:
        config_values = [value for flag, value in zip(argv, argv[1:]) if flag == "--config"]
        config_values += [value.split("=", 1)[1] for value in argv if value.startswith("--config=")]
        require(len(config_values) == 1 and Path(config_values[0]).as_posix() in plan["inputs"],
                "The plan must pin the command's exact config file in inputs.")
    if script == "run_pilot.py":
        require(all(Path(path).as_posix() in plan["inputs"] for path in argv),
                "The plan must pin every pilot arm config in inputs.")
    for path, expected in plan["inputs"].items():
        require(sha256_file(ROOT / path) == expected, f"Planned input has changed: {path}")


def owned_entry(state: dict, receipt: Receipt) -> dict:
    current = slot(state, receipt.experiment_id)
    require(current["state"] == "RUNNING" and current.get("claim_sha256") == receipt.digest and
            state["experiments"][-1]["id"] == receipt.experiment_id,
            "There is no matching active GitHub claim for this local receipt.")
    return current


def settle(ledger: Ledger, receipt: Receipt, outcome: dict) -> None:
    status = outcome.get("state")
    note = outcome.get("execution_note")
    require(isinstance(status, str) and status in TERMINAL and
            isinstance(note, str) and bool(note.strip()), "A terminal status and note are required.")
    local = receipt.directory / "outcome.json"
    with local_lock(receipt.directory / "worker.lock"):
        write_local(local, outcome)
        for attempt in range(3):
            head, state = ledger.refresh()
            current = slot(state, receipt.experiment_id)
            if current.get("claim_sha256") == receipt.digest and current["state"] == status:
                return  # A previous push may have succeeded before its acknowledgement was lost.
            entry = owned_entry(state, receipt)
            entry.update(outcome)
            try:
                ledger.advance(head, state, f"Record {receipt.experiment_id} as {status}")
                return
            except ProtocolError:
                if attempt == 2:
                    raise


def recover(experiment_id: str, note: str) -> None:
    directory = receipt_directory(experiment_id)
    require(directory.is_dir(), "The original local claim directory is unavailable.")
    with local_lock(directory / "launcher.lock"), Ledger() as ledger:
        receipt = read_receipt(experiment_id, directory)
        pending = directory / "outcome.json"
        if pending.is_file():
            outcome = read_json(pending.read_bytes(), pending.name)
        else:
            outcome = {"state": "INTERRUPTED", "execution_note": note,
                       "terminal_recorded_at": now(), "finished_at": None}
        settle(ledger, receipt, outcome)


def start_training_child(argv: list[str], **popen_options) -> subprocess.Popen:
    """Delegate one exact training command inside an already claimed campaign."""
    active = _active.get()
    require(active is not None, "A training worker requires an active campaign claim.")
    receipt, key = active
    require(receipt.script in CAMPAIGNS, "Only a campaign may delegate training.")
    handoff = receipt.directory / "handoff.json"
    require(not handoff.exists(), "A previous campaign handoff was not consumed.")
    write_local(handoff, {"argv": argv})
    child = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "scripts/train.py"),
         "--research-worker", receipt.experiment_id, *argv],
        stdin=subprocess.PIPE, bufsize=0, **popen_options,
    )
    with child.stdin:
        try:
            child.stdin.write((key + "\n").encode())
        except BrokenPipeError:
            pass  # The caller collects the failed child's exit status and logs.
    child.stdin = None
    return child


def _worker(experiment_id: str, script: str, argv: list[str], workload) -> int:
    require(script == "train.py", "Only train.py can receive a campaign handoff.")
    directory = receipt_directory(experiment_id)
    require(directory.is_dir(), "No local campaign receipt exists.")
    with local_lock(directory / "worker.lock"), Ledger() as ledger:
        receipt = read_receipt(experiment_id, directory)
        handoff = directory / "handoff.json"
        require(receipt.script in CAMPAIGNS and handoff.is_file(),
                "No pending one-use campaign handoff.")
        expected = read_json(handoff.read_bytes(), handoff.name)
        key = sys.stdin.readline(66).strip()
        require(len(key) == 64 and argv == expected.get("argv"),
                "Campaign handoff does not match the command.")
        _, state = ledger.refresh()
        worker_sha256 = owned_entry(state, receipt).get("worker_sha256")
        require(isinstance(worker_sha256, str) and
                secrets.compare_digest(hashlib.sha256(key.encode()).hexdigest(), worker_sha256),
                "Worker credential does not match the active GitHub claim.")
        handoff.unlink()
        return workload(argv)


def _launch(experiment_id: str, script: str, argv: list[str], workload) -> int:
    with Ledger() as ledger:
        head, state = ledger.refresh()
        plan = ledger.approved_plan(head, state, experiment_id)
        check_command(plan, script, argv)
        directory = receipt_directory(experiment_id)
        directory.mkdir(parents=True, exist_ok=True)
        with local_lock(directory / "launcher.lock"):
            fresh_head, _ = ledger.refresh()
            require(head == fresh_head, "The GitHub ledger changed during validation. Recheck the plan.")
            receipt = Receipt(experiment_id, directory, secrets.token_hex(32), script)
            # The delegation key stays in memory and the inherited worker pipe.
            # A local recovery receipt alone can never authenticate a worker.
            key = secrets.token_hex(32) if script in CAMPAIGNS else ""
            (directory / "outcome.json").unlink(missing_ok=True)
            write_local(directory / "claim.json", {"token": receipt.token, "script": script})
            slot(state, experiment_id).update(
                state="RUNNING", claim_sha256=receipt.digest,
                started_at=now(), implementation_commit=plan["implementation_commit"], script=script,
                worker_sha256=hashlib.sha256(key.encode()).hexdigest() if key else None,
            )
            commit = ledger.advance(head, state, f"Claim {experiment_id} before execution")
            print(f"[protocol] {experiment_id} claimed on GitHub at {commit}", flush=True)
            context = _active.set((receipt, key))
            code = 1
            try:
                code = workload(argv)
            except KeyboardInterrupt:
                code = 130
            except SystemExit as exc:
                code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
                if exc.code is not None and not isinstance(exc.code, int):
                    print(exc.code, file=sys.stderr)
            finally:
                _active.reset(context)
                status = {0: "COMPLETED", 130: "INTERRUPTED"}.get(code, "FAILED")
                # Save locally before any GitHub operation. A dropped network cannot erase the outcome.
                outcome = {"state": status, "execution_note": f"{script} exit code {code}",
                           "finished_at": now()}
                write_local(directory / "outcome.json", outcome)
                settle(ledger, receipt, outcome)
            return code


def run_research(script: str, workload, argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--help"], ["-h"]):
        return workload(args)
    try:
        require(len(args) >= 2 and args[0] in ("--research-id", "--research-worker"),
                "Start with --research-id DRP-NNNN after the plan is approved on GitHub.")
        require(matches(ID, args[1]), "Expected a protocol ID such as DRP-0001.")
        if args[0] == "--research-worker":
            return _worker(args[1], script, args[2:], workload)
        return _launch(args[1], script, args[2:], workload)
    except (ProtocolError, OSError) as exc:
        print(f"[protocol] {exc}", file=sys.stderr)
        return 1
