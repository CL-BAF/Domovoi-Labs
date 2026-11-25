"""Verify approved research, claim execution, and publish evidenced outcomes."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import secrets
import subprocess
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from scripts.experiment_protocol import plain, verify_checkout
from .protocol import (
    DIGEST, ID, Ledger, ProtocolError, git, matches, now, published_implementation,
    read_json, repo_path, require, slot, valid_completion,
)
from .runtime import Runtime, hash_file, prepare, supervisor, workload_environment

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = {"real_data_campaign.py", "run_pilot.py"}
BUILTINS = CAMPAIGNS | {"train.py", "evaluate.py", "generate.py", "profile_model.py",
                        "inspect_model.py", "verify_real_resume.py"}


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


@dataclass(frozen=True)
class Campaign:
    receipt: Receipt
    delegation_key: str
    runtime: Runtime
    operator_env: dict


_active: ContextVar[Campaign | None] = ContextVar("domovoi_research_claim", default=None)


def receipt_directory(experiment_id: str) -> Path:
    require(matches(ID, experiment_id), "Expected a protocol ID such as DRP-0001.")
    common = git(ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(os.fsdecode(common).strip()) / "domovoi-protocol" / experiment_id


@contextmanager
def local_lock(path: Path):
    with path.open("a+b", buffering=0) as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ProtocolError("The local launcher or worker is still active.") from exc
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
    require(path.is_file() and plain(path), "Recovery requires the original claim receipt.")
    data = read_json(path.read_bytes(), path.name)
    require(set(data) == {"token", "script"} and matches(DIGEST, data["token"])
            and isinstance(data["script"], str) and bool(data["script"]),
            "Invalid local claim receipt.")
    return Receipt(experiment_id, directory, data["token"], data["script"])


def check_command(plan: dict, source: str) -> None:
    require(plan["implementation_commit"] == source,
            "The checkout must be at the pinned implementation commit.")
    published_implementation(source)
    path = Path(plan["command"]["script"])
    require((path.parent == Path("scripts") and path.name in BUILTINS) or
            (path.parent == Path("research_workloads") and path.suffix == ".py"),
            "Use a supported command or a tracked research_workloads/ Python adapter.")
    for relative, expected in plan["inputs"].items():
        file = ROOT / relative
        require(file.is_file() and plain(file) and hash_file(file) == expected,
                f"Pinned source/input differs: {relative}")


def owned_entry(state: dict, receipt: Receipt) -> dict:
    current = slot(state, receipt.experiment_id)
    require(current["state"] == "RUNNING" and current.get("claim_sha256") == receipt.digest
            and state["experiments"][-1]["id"] == receipt.experiment_id,
            "There is no matching active GitHub claim for this local receipt.")
    return current


def _valid_outcome(outcome: dict) -> bool:
    return (outcome.get("state") in ("COMPLETED", "FAILED", "INTERRUPTED")
            and isinstance(outcome.get("execution_note"), str) and bool(outcome["execution_note"].strip())
            and set(outcome) <= {"state", "execution_note", "finished_at", "terminal_recorded_at",
                                 "exit_code", "execution_evidence", "completion_token"})


def settle(ledger: Ledger, receipt: Receipt, outcome: dict) -> None:
    require(_valid_outcome(outcome), "Invalid terminal execution record.")
    status = outcome["state"]
    with local_lock(receipt.directory / "worker.lock"):
        for attempt in range(3):
            head, state = ledger.refresh()
            current = slot(state, receipt.experiment_id)
            if current.get("claim_sha256") == receipt.digest and current["state"] == status:
                require(all(current.get(k) == v for k, v in outcome.items()),
                        "GitHub already has a different terminal receipt.")
                return
            entry = owned_entry(state, receipt)
            proposed = {**entry, **outcome}
            if status == "COMPLETED":
                require(valid_completion(proposed), "COMPLETED requires original execution proof.")
            entry.update(outcome)
            try:
                ledger.advance(head, state, f"Record {receipt.experiment_id} as {status}")
                return
            except ProtocolError:
                if attempt == 2:
                    raise


def recover(experiment_id: str, note: str) -> None:
    require(isinstance(note, str) and bool(note.strip()), "Recovery requires a public evidence note.")
    directory = receipt_directory(experiment_id)
    require(directory.is_dir(), "The original local claim directory is unavailable.")
    with local_lock(directory / "launcher.lock"), Ledger() as ledger:
        receipt = read_receipt(experiment_id, directory)
        _, state = ledger.refresh()
        current = slot(state, experiment_id)
        require(current.get("claim_sha256") == receipt.digest, "Local claim does not match GitHub.")
        if current["state"] != "RUNNING":
            require(current["state"] in ("COMPLETED", "FAILED", "INTERRUPTED"),
                    "Recovery requires an active or already terminal claim.")
            return
        pending = directory / "outcome.json"
        outcome = {"state": "INTERRUPTED", "execution_note": note,
                   "terminal_recorded_at": now(), "finished_at": None}
        if pending.is_file():
            require(plain(pending), "Unsafe pending outcome path.")
            saved = read_json(pending.read_bytes(), pending.name)
            require(_valid_outcome(saved) and "completion_token" not in saved,
                    "Malformed local outcome; preserve it and seek owner review.")
            if saved["state"] == "FAILED":
                outcome["state"] = "FAILED"
            # A pending success is evidence to archive, never authority to publish success.
        settle(ledger, receipt, outcome)
        write_local(directory / "recovery.json", outcome)


def _invoke(script: str, argv: list[str]) -> int:
    module = runpy.run_path(str(ROOT / script), run_name="_domovoi_approved_workload")
    run = module["_run"] if script.startswith("scripts/") else module["main"]
    result = run(argv)
    require(type(result) is int, "Workload must return an integer exit code.")
    return result


def _expand(args: list[str], runtime: Runtime) -> list[str]:
    prefixes = {"@source/": runtime.source, "@work/": runtime.workspace}
    expanded = []
    for arg in args:
        for prefix, root in prefixes.items():
            if arg.startswith(prefix):
                relative = arg[len(prefix):]
                require(repo_path(relative), "Invalid source or workspace argument path.")
                arg = str(root / relative)
                break
        expanded.append(arg)
    return expanded


def _collect_completion(runtime: Runtime, plan: dict) -> dict:
    evidence = {}
    for relative in plan["execution"]["completion_artifacts"]:
        file = runtime.workspace / relative
        require(file.is_file() and plain(file) and file.stat().st_size > 0
                and file.resolve() == file, f"Missing or unsafe completion evidence: {relative}")
        evidence[relative] = hash_file(file)
    return evidence


def run_approved(experiment_id: str, source: str, workspace: Path, dependencies: Path) -> int:
    directory = receipt_directory(experiment_id)
    with Ledger() as ledger:
        head, state = ledger.refresh()
        plan = ledger.approved_plan(head, state, experiment_id)
        check_command(plan, source)
        runtime = prepare(ROOT, workspace, dependencies, plan)
        argv = _expand(plan["command"]["argv"], runtime)
        directory.mkdir(parents=True, exist_ok=True)
        with local_lock(directory / "launcher.lock"):
            fresh_head, _ = ledger.refresh()
            require(head == fresh_head, "The GitHub ledger changed during validation.")
            require(verify_checkout(ROOT) == source, "The reviewed checkout changed.")
            script = plan["command"]["script"]
            receipt = Receipt(experiment_id, directory, secrets.token_hex(32), Path(script).name)
            delegation = secrets.token_hex(32) if receipt.script in CAMPAIGNS else ""
            completion = secrets.token_hex(32)
            write_local(directory / "claim.json", {"token": receipt.token, "script": receipt.script})
            slot(state, experiment_id).update(
                state="RUNNING", claim_sha256=receipt.digest, started_at=now(),
                implementation_commit=source, script=receipt.script,
                worker_sha256=hashlib.sha256(delegation.encode()).hexdigest() if delegation else None,
                completion_sha256=hashlib.sha256(completion.encode()).hexdigest(),
            )
            commit = ledger.advance(head, state, f"Claim {experiment_id} before execution")
            print(f"[protocol] {experiment_id} claimed on GitHub at {commit}", flush=True)
            context = _active.set(Campaign(receipt, delegation, runtime, dict(os.environ)))
            code, evidence = 1, {}
            try:
                for path in runtime.temporary_dirs:
                    path.mkdir(parents=True, exist_ok=True)
                with workload_environment(runtime, source):
                    result = _invoke(script, argv)
                if result == 0:
                    evidence = _collect_completion(runtime, plan)
                code = result
            except KeyboardInterrupt:
                code = 130
            finally:
                _active.reset(context)
                status = {0: "COMPLETED", 130: "INTERRUPTED"}.get(code, "FAILED")
                outcome = {"state": status, "execution_note": f"{script} exit code {code}",
                           "finished_at": now(), "exit_code": code}
                if status == "COMPLETED":
                    outcome["execution_evidence"] = evidence
                write_local(directory / "outcome.json", outcome)
                # The preimage exists only in this launcher until a successful terminal push.
                if status == "COMPLETED":
                    outcome["completion_token"] = completion
                settle(ledger, receipt, outcome)
            return code


def start_training_child(argv: list[str], **options) -> subprocess.Popen:
    campaign = _active.get()
    require(campaign is not None and campaign.receipt.script in CAMPAIGNS,
            "A training worker requires an active campaign.")
    receipt, runtime = campaign.receipt, campaign.runtime
    with supervisor():
        handoff = receipt.directory / "handoff.json"
        require(not handoff.exists(), "A previous campaign handoff was not consumed.")
        write_local(handoff, {"argv": argv})
        child = subprocess.Popen(
            [sys.executable, "-I", "-S", "-B", "-u", str(ROOT / "scripts/experiment_protocol.py"),
             "worker", "--workspace", str(runtime.workspace), "--dependencies",
             str(runtime.dependencies), receipt.experiment_id, *argv],
            stdin=subprocess.PIPE, bufsize=0, cwd=runtime.workspace, env=campaign.operator_env,
            **options,
        )
        with child.stdin:
            try:
                child.stdin.write((campaign.delegation_key + "\n").encode())
            except BrokenPipeError:
                pass
        child.stdin = None
        return child


def run_worker(experiment_id: str, source: str, workspace: Path,
               dependencies: Path, argv: list[str]) -> int:
    directory = receipt_directory(experiment_id)
    require(directory.is_dir(), "No local campaign receipt exists.")
    with local_lock(directory / "worker.lock"), Ledger() as ledger:
        receipt = read_receipt(experiment_id, directory)
        handoff = directory / "handoff.json"
        require(receipt.script in CAMPAIGNS and handoff.is_file() and plain(handoff),
                "No pending one-use campaign handoff.")
        expected = read_json(handoff.read_bytes(), handoff.name)
        key = sys.stdin.readline(66).strip()
        require(matches(DIGEST, key) and argv == expected.get("argv"), "Invalid campaign handoff.")
        _, state = ledger.refresh()
        entry = owned_entry(state, receipt)
        worker_digest = entry.get("worker_sha256")
        require(matches(DIGEST, worker_digest) and secrets.compare_digest(
                hashlib.sha256(key.encode()).hexdigest(), worker_digest),
                "Worker credential does not match the active GitHub claim.")
        plan = ledger.documented_plan(entry["plan_commit"], experiment_id)
        check_command(plan, source)
        runtime = prepare(ROOT, workspace, dependencies, plan, worker=True)
        require(verify_checkout(ROOT) == source, "The reviewed checkout changed.")
        handoff.unlink()
        with workload_environment(runtime, source):
            return _invoke("scripts/train.py", argv)


def run_research(script: str, workload, argv: list[str] | None = None) -> int:
    """Retain importable command Help; live workloads use the isolated bootstrap."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args in (["--help"], ["-h"]):
        return workload(args)
    print("[protocol] Use python -I -S -B scripts/experiment_protocol.py run.", file=sys.stderr)
    return 1
