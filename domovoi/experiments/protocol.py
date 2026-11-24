"""Read and advance the canonical GitHub experiment ledger."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

RESEARCH_URL = "https://github.com/CL-BAF/Domovoi-research.git"
LEDGER_REF = "refs/heads/research/real-data-calibration"
IMPLEMENTATION_API = "https://api.github.com/repos/CL-BAF/Domovoi-Labs/git/commits/"
STATE = "experiments/protocol/state.json"
INDEX = "experiments/ARCHIVE-INDEX.md"
ID = re.compile(r"DRP-[0-9]{4,}")
SHA = re.compile(r"[0-9a-f]{40}")
DIGEST = re.compile(r"[0-9a-f]{64}")
TERMINAL = {"COMPLETED", "FAILED", "INTERRUPTED"}
PHASES = {"PROPOSED", "DOCUMENTED", "APPROVED", "RUNNING", "ANALYSED", "PUBLISHED"} | TERMINAL
NEXT = {
    "PROPOSED": {"DOCUMENTED"},
    "DOCUMENTED": {"APPROVED"},
    "APPROVED": {"DOCUMENTED", "RUNNING"},
    "RUNNING": TERMINAL,
    "COMPLETED": {"ANALYSED"},
    "FAILED": {"ANALYSED"},
    "INTERRUPTED": {"ANALYSED"},
    "ANALYSED": {"PUBLISHED"},
    "PUBLISHED": set(),
}
PLAN_FIELDS = (
    "question", "rationale", "problem", "implementation_changes", "design_choice",
    "hypothesis", "mathematics", "expected_and_alternative_outcomes", "baselines",
    "dataset_tokenizer_config", "environment", "evaluation", "success_and_failure",
    "compute_and_memory", "exact_reproduction_commands",
)
RESULT_FIELDS = (
    "execution_history", "actual_outcomes", "hypothesis_comparison", "capability_loss_speed_memory",
    "improvements_and_regressions", "numerical_analysis", "observations",
    "possible_explanations", "supporting_evidence", "competing_hypotheses", "uncertainty",
    "unexpected_results_and_failures", "lessons", "recommended_next_experiment",
)


class ProtocolError(RuntimeError):
    """An experiment cannot advance safely."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(condition, reason: str) -> None:
    if not condition:
        raise ProtocolError(reason)


def matches(pattern: re.Pattern, value) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def repo_path(value) -> bool:
    return (isinstance(value, str) and bool(value) and
            not any(c in value for c in "\\:\r\n\x00") and
            all(part not in ("", ".", "..") for part in value.split("/")))


def git(repo: Path, *args, input_bytes: bytes | None = None) -> bytes:
    # A launcher must never fall back to cached state or wait for an interactive login.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "never"}
    try:
        result = subprocess.run(["git", *args], cwd=repo, input=input_bytes,
                                capture_output=True, env=env, timeout=45, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ProtocolError(f"Git {args[0]} failed. Check GitHub connectivity and credentials.") from exc
    return result.stdout


def read_json(data: bytes, location: str) -> dict:
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ProtocolError(f"Invalid JSON in {location}.") from exc
    require(isinstance(value, dict), f"{location} must be a JSON object.")
    return value


def require_fields(data: dict, fields: tuple[str, ...], label: str) -> None:
    for field in fields:
        value = data.get(field)
        require(isinstance(value, str) and bool(value.strip()), f"{label} needs {field}.")


def record_path(experiment_id: str) -> str:
    return f"experiments/protocol/records/{experiment_id}.md"


def plan_path(experiment_id: str) -> str:
    return f"experiments/protocol/plans/{experiment_id}.json"


def result_path(experiment_id: str) -> str:
    return f"experiments/protocol/results/{experiment_id}.json"


def slot(state: dict, experiment_id: str) -> dict:
    for item in state["experiments"]:
        if item["id"] == experiment_id:
            return item
    raise ProtocolError(f"{experiment_id} is absent from the canonical GitHub ledger.")


def published_implementation(commit: str) -> None:
    try:
        request = Request(IMPLEMENTATION_API + commit, headers={
            "Accept": "application/vnd.github+json", "User-Agent": "Domovoi-Research-Protocol",
        })
        with urlopen(request, timeout=15) as response:
            found = read_json(response.read(256_000), "GitHub implementation commit")
    except OSError as exc:
        raise ProtocolError("The pinned implementation commit could not be verified on GitHub.") from exc
    require(found.get("sha") == commit, "GitHub returned a different implementation commit.")


class Ledger:
    """Fetch into isolated Git storage. A normal fast-forward push is the atomic claim."""

    def __init__(self):
        self._temp = TemporaryDirectory(prefix="domovoi-protocol-")
        self.repo = Path(self._temp.name)

    def __enter__(self):
        try:
            git(self.repo, "init", "--bare", "--quiet")
        except BaseException:
            self._temp.cleanup()
            raise
        return self

    def __exit__(self, *_):
        self._temp.cleanup()

    def refresh(self) -> tuple[str, dict]:
        git(self.repo, "fetch", "--quiet", "--no-tags", RESEARCH_URL, LEDGER_REF)
        head = git(self.repo, "rev-parse", "FETCH_HEAD").decode().strip()
        return head, self.state_at(head)

    def blob(self, commit: str, path: str) -> bytes:
        return git(self.repo, "show", f"{commit}:{path}")

    def state_at(self, commit: str) -> dict:
        state = read_json(self.blob(commit, STATE), STATE)
        rows = state.get("experiments")
        require(type(state.get("schema")) is int and state["schema"] == 1 and
                isinstance(rows, list), "Unsupported or invalid canonical ledger.")
        for number, row in enumerate(rows, 1):
            require(isinstance(row, dict) and row.get("id") == f"DRP-{number:04d}" and
                    isinstance(row.get("state"), str) and row["state"] in PHASES,
                    "Ledger IDs must be consecutive and each row must have a valid phase.")
        return state

    def ancestor(self, earlier, later: str) -> str:
        require(matches(SHA, earlier), "A full Git commit SHA is required in the ledger.")
        try:
            git(self.repo, "merge-base", "--is-ancestor", earlier, later)
        except ProtocolError as exc:
            raise ProtocolError(f"Pinned commit {earlier} is not reachable from {later}.") from exc
        return earlier

    def linked(self, commit: str, experiment_id: str) -> bytes:
        path = record_path(experiment_id)
        record = self.blob(commit, path)
        require(record.strip(), f"{path} is empty.")
        for index in ("README.md", INDEX):
            link = path if index == "README.md" else path.removeprefix("experiments/")
            require(f"]({link})".encode() in self.blob(commit, index),
                    f"{index} must link to {path}.")
        return record

    def verify_history(self, head: str) -> None:
        commits = git(self.repo, "log", "--first-parent", "--reverse", "--format=%H",
                      head, "--", STATE).decode().splitlines()
        previous = []
        require(bool(commits), "The ledger needs a committed bootstrap.")
        for commit in commits:
            rows = self.state_at(commit)["experiments"]
            require(len(previous) <= len(rows) <= len(previous) + 1,
                    "The ledger cannot delete records or propose more than one at a time.")
            if len(rows) > len(previous):
                require(rows[:-1] == previous and all(row["state"] == "PUBLISHED" for row in previous)
                        and rows[-1]["state"] == "PROPOSED",
                        "Publish the open experiment before proposing another.")
            elif rows != previous:
                old, new = previous[-1], rows[-1]
                require(rows[:-1] == previous[:-1] and new["state"] in NEXT[old["state"]],
                        "The ledger rewrote a previous experiment or skipped a lifecycle phase.")
                if new["state"] == "RUNNING":
                    require(new.get("plan_commit") == old.get("plan_commit") and
                            matches(DIGEST, new.get("claim_sha256")) and
                            matches(SHA, new.get("implementation_commit")),
                            "RUNNING needs the approved plan and a valid claim identity.")
                if old["state"] in {"RUNNING", *TERMINAL, "ANALYSED"}:
                    require(all(new.get(key) == old.get(key) for key in
                                ("plan_commit", "claim_sha256", "implementation_commit",
                                 "started_at", "script", "worker_sha256")),
                            "The ledger changed the identity of a claimed experiment.")
            previous = rows

    def approved_plan(self, head: str, state: dict, experiment_id: str) -> dict:
        require(bool(state["experiments"]) and state["experiments"][-1]["id"] == experiment_id,
                "Only the last ledger entry can be launched.")
        entry = slot(state, experiment_id)
        require(entry["state"] == "APPROVED", f"{experiment_id} is {entry['state']}, not APPROVED.")
        self.verify_history(head)
        for previous in state["experiments"][:-1]:
            self.verify_publication(head, previous)
        pinned = self.ancestor(entry.get("plan_commit"), head)
        prior = slot(self.state_at(pinned), experiment_id)
        require(pinned != head and prior["state"] == "DOCUMENTED",
                "Approval must pin an earlier DOCUMENTED revision.")
        path = plan_path(experiment_id)
        saved = self.blob(pinned, path)
        require(saved == self.blob(head, path) and
                self.linked(pinned, experiment_id) == self.linked(head, experiment_id),
                "The approved plan changed. Document and approve the new version before launch.")
        data = read_json(saved, path)
        require(data.get("id") == experiment_id and matches(SHA, data.get("implementation_commit")),
                "The plan needs the approved ID and a full implementation commit SHA.")
        require_fields(data, PLAN_FIELDS, "Plan")
        command = data.get("command")
        require(isinstance(command, dict) and isinstance(command.get("script"), str)
                and isinstance(command.get("argv"), list)
                and all(isinstance(x, str) for x in command["argv"]),
                "The plan must contain one structured command with script and argv.")
        inputs = data.get("inputs")
        require(isinstance(inputs, dict) and all(repo_path(k) and matches(DIGEST, v)
                                                 for k, v in inputs.items()),
                "Plan inputs must map checkout-relative files to SHA-256 digests.")
        return data

    def verify_publication(self, head: str, entry: dict) -> None:
        experiment_id = entry["id"]
        require(entry["state"] == "PUBLISHED",
                f"Publish the outcome of {experiment_id} ({entry['state']}) first.")
        analysed = self.ancestor(entry.get("analysis_commit"), head)
        proof = slot(self.state_at(analysed), experiment_id)
        require(proof["state"] == "ANALYSED", f"{experiment_id} needs a pinned ANALYSED revision.")
        terminal = self.ancestor(proof.get("terminal_commit"), analysed)
        start = self.ancestor(entry.get("plan_commit"), terminal)
        finished = slot(self.state_at(terminal), experiment_id)
        require(finished["state"] in TERMINAL and finished.get("plan_commit") == start and
                matches(DIGEST, finished.get("claim_sha256")) and
                entry.get("terminal_commit") == terminal and
                proof.get("plan_commit") == start and
                entry.get("outcome") == proof.get("outcome") == finished["state"],
                f"{experiment_id} needs a matching terminal claim, outcome, and analysis.")
        path = result_path(experiment_id)
        raw = self.blob(analysed, path)
        result = read_json(raw, path)
        require(result.get("id") == experiment_id and result.get("outcome") == finished["state"],
                f"{experiment_id} result does not match the terminal outcome.")
        require_fields(result, RESULT_FIELDS, "Result")
        evidence = result.get("evidence")
        require(isinstance(evidence, list) and bool(evidence) and all(repo_path(p) for p in evidence),
                f"{experiment_id} needs repository-relative evidence paths.")
        for path in evidence:
            before = git(self.repo, "rev-parse", "--verify", f"{analysed}:{path}")
            after = git(self.repo, "rev-parse", "--verify", f"{head}:{path}")
            require(before == after, f"{experiment_id} published evidence changed: {path}")
        updated = self.linked(analysed, experiment_id)
        require(raw == self.blob(head, result_path(experiment_id)) and
                updated != self.linked(start, experiment_id) and
                updated == self.linked(head, experiment_id) and
                self.blob(terminal, "README.md") != self.blob(analysed, "README.md"),
                f"{experiment_id} needs a preserved result, updated original record and README.")

    def advance(self, parent: str, state: dict, message: str) -> str:
        git(self.repo, "read-tree", parent)
        raw = (json.dumps(state, indent=2, ensure_ascii=True) + "\n").encode()
        blob = git(self.repo, "hash-object", "-w", "--stdin", input_bytes=raw).decode().strip()
        git(self.repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{STATE}")
        tree = git(self.repo, "write-tree").decode().strip()
        commit = git(self.repo, "-c", "user.name=Domovoi Experiment Protocol", "-c",
                     "user.email=domovoi-experiment-protocol@users.noreply.github.com",
                     "commit-tree", tree, "-p", parent, "-m", message).decode().strip()
        git(self.repo, "push", "--quiet", RESEARCH_URL, f"{commit}:{LEDGER_REF}")
        return commit
