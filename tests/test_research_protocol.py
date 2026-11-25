"""No live GitHub claims, model training or scientific workloads in these tests."""

import hashlib
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import pytest

from domovoi.experiments import launch, protocol
from domovoi.experiments.protocol import Ledger, ProtocolError, git, slot
from domovoi.experiments.runtime import runtime_identity
from scripts.experiment_protocol import verify_checkout

ID = "DRP-0001"
GATED = ("train", "evaluate", "profile_model", "generate", "inspect_model",
         "verify_real_resume", "run_pilot", "real_data_campaign")
ROOT = Path(__file__).resolve().parents[1]
SCRIPT = "research_workloads/writer.py"
WRITE = "from pathlib import Path\ndef main(argv):\n    Path('out').mkdir(exist_ok=True)\n    Path('out/receipt.json').write_text('fixture output')\n    return 0\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Fixtures:
    def __init__(self, tmp, monkeypatch):
        self.remote, self.editor, self.source, self.workspace, self.deps = (
            tmp / name for name in ("remote", "editor", "src", "work", "deps"))
        for path in (self.remote, self.editor, self.source, self.workspace, self.deps):
            path.mkdir()
        git(self.remote, "init", "--bare", "--quiet")
        for path in (self.editor, self.source):
            git(path, "init", "--quiet")
            git(path, "config", "core.autocrlf", "false")
        self.put(self.source, ".gitignore", "cache/\n")
        self.put(self.deps, "fixture_dependency.py", "value = 1\n")
        self.put(self.source, "runtime.json",
                 {**runtime_identity(self.deps), "gpu": None, "tools": {}, "environment": {}})
        self.put(self.source, SCRIPT, WRITE)
        for role in protocol.ANALYSIS_ROLES:
            self.put(self.source, f"analysis/{role}.txt", f"frozen {role}")
        self.source_sha = self.commit(self.source, "fixture code")
        self.plan = {name: "frozen fixture" for name in protocol.PLAN_FIELDS}
        self.plan.update(
            id=ID, implementation_commit=self.source_sha, command={"script": SCRIPT, "argv": []},
            execution={"runtime_lock": "runtime.json", "output_dirs": ["out"],
                       "temporary_dirs": ["scratch"], "assets": {},
                       "completion_artifacts": ["out/receipt.json"],
                       "analysis": {role: f"analysis/{role}.txt" for role in protocol.ANALYSIS_ROLES}},
        )
        self.pin_inputs()
        self.rows = []
        self.publish("bootstrap")
        monkeypatch.setattr(protocol, "RESEARCH_URL", str(self.remote))
        monkeypatch.setattr(launch, "ROOT", self.source)
        monkeypatch.setattr(launch, "published_implementation", lambda sha: None)

    @staticmethod
    def put(root, path, payload):
        file = root / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes((payload if isinstance(payload, str) else json.dumps(payload)).encode())

    @staticmethod
    def commit(root, note):
        git(root, "add", ".")
        git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "--quiet", "-m", note)
        return git(root, "rev-parse", "HEAD").decode().strip()

    def pin_inputs(self):
        execution = self.plan["execution"]
        paths = [execution["runtime_lock"], *execution["analysis"].values(),
                 self.plan["command"]["script"]]
        self.plan["inputs"] = {p: digest((self.source / p).read_bytes()) for p in paths}

    def publish(self, message):
        self.put(self.editor, protocol.STATE, {"schema": 1, "experiments": self.rows})
        commit = self.commit(self.editor, message)
        git(self.editor, "push", "--quiet", str(self.remote), f"HEAD:{protocol.LEDGER_REF}")
        return commit

    def sync(self):
        git(self.editor, "fetch", "--quiet", str(self.remote), protocol.LEDGER_REF)
        git(self.editor, "merge", "--quiet", "--ff-only", "FETCH_HEAD")
        self.rows = json.loads((self.editor / protocol.STATE).read_bytes())["experiments"]
        return git(self.editor, "rev-parse", "HEAD").decode().strip()

    def approve(self, id=ID):
        self.rows.append({"id": id, "state": "PROPOSED"})
        self.publish("proposed")
        self.put(self.editor, protocol.plan_path(id), {**self.plan, "id": id})
        self.put(self.editor, protocol.record_path(id), "# Fixture")
        for prefix, doc in (("", "README.md"), ("experiments/", protocol.INDEX)):
            file = self.editor / doc
            before = file.read_text(encoding="utf-8") if file.exists() else ""
            self.put(self.editor, doc,
                     before + f"\n[{id}]({protocol.record_path(id).removeprefix(prefix)})\n")
        self.rows[-1]["state"] = "DOCUMENTED"
        documented = self.publish("documented")
        self.rows[-1].update(state="APPROVED", plan_commit=documented)
        self.publish("approved")

    def invoke(self, id=ID):
        return launch.run_approved(id, self.source_sha, self.workspace, self.deps)

    def manual_claim(self, token="a" * 64, script=Path(SCRIPT).name, key=None):
        directory = launch.receipt_directory(ID)
        directory.mkdir(parents=True)
        receipt = launch.Receipt(ID, directory, token, script)
        launch.write_local(directory / "claim.json", {"token": token, "script": script})
        with Ledger() as ledger:
            head, state = ledger.refresh()
            slot(state, ID).update(state="RUNNING", claim_sha256=receipt.digest,
                                   implementation_commit=self.source_sha, script=script,
                                   completion_sha256=digest(b"fixture proof"),
                                   worker_sha256=digest(key.encode()) if key else None)
            ledger.advance(head, state, "fixture claim")
        return receipt

    def publish_result(self, *, changed_metric=False):
        terminal = self.sync()
        evidence = "measurements.json"
        result = {name: "fixture" for name in protocol.RESULT_FIELDS}
        pins = protocol.analysis_receipt(self.plan)
        if changed_metric:
            pins["primary_metric"]["sha256"] = "0" * 64
        self.put(self.editor, protocol.result_path(ID), {
            **result, "id": ID, "outcome": self.rows[-1]["state"], "evidence": [evidence],
            "analysis_sources": pins, "analysis_implementation_commit": self.source_sha,
        })
        self.put(self.editor, evidence, {"fixture": True})
        self.put(self.editor, protocol.record_path(ID), "# Fixture\n\nEvidence")
        file = self.editor / "README.md"
        self.put(self.editor, "README.md", file.read_text(encoding="utf-8") + "\nPublished.\n")
        self.rows[-1].update(state="ANALYSED", terminal_commit=terminal, outcome=self.rows[-1]["state"])
        analysed = self.publish("analysed")
        self.rows[-1].update(state="PUBLISHED", analysis_commit=analysed)
        self.publish("published")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    return Fixtures(tmp_path, monkeypatch)


@pytest.mark.parametrize("name", GATED)
def test_direct_cli_refuses_before_importing_workload(name, tmp_path):
    result = subprocess.run([sys.executable, "-I", "-S", "-B",
                             str(ROOT / "scripts" / f"{name}.py"), "--research-id", ID],
                            cwd=tmp_path, capture_output=True)
    assert result.returncode and b"experiment_protocol.py run" in result.stderr
    assert not list(tmp_path.iterdir())


def test_valid_external_adapter_and_published_prerequisite(fixture, tmp_path):
    fixture.approve()
    assert fixture.invoke() == 0
    fixture.sync()
    terminal = fixture.rows[-1]
    assert terminal["state"] == "COMPLETED" and protocol.valid_completion(terminal)
    with pytest.raises(ProtocolError, match="not APPROVED"):
        fixture.invoke()
    fixture.publish_result()
    fixture.approve("DRP-0002")
    fixture.workspace = tmp_path / "fresh"
    fixture.workspace.mkdir()
    assert fixture.invoke("DRP-0002") == 0


@pytest.mark.parametrize("file", ["surprise.py", "cache/ignored.py", "working-copy.pyc"])
def test_untracked_source_even_ignored_is_rejected(fixture, file):
    fixture.approve()
    fixture.put(fixture.source, file, "raise RuntimeError('must not be imported')")
    with pytest.raises(ValueError, match="Unexpected source"):
        fixture.invoke()
    fixture.sync()
    assert fixture.rows[-1]["state"] == "APPROVED"


def test_bootstrap_refuses_untracked_module_before_inserting_source(tmp_path):
    git(tmp_path, "init", "--quiet")
    git(tmp_path, "config", "core.autocrlf", "false")
    path = tmp_path / "scripts/experiment_protocol.py"
    path.parent.mkdir()
    path.write_bytes((ROOT / "scripts/experiment_protocol.py").read_bytes())
    Fixtures.commit(tmp_path, "bootstrap fixture")
    (tmp_path / "scripts/os.py").write_text("raise RuntimeError('untracked import')", encoding="utf-8")
    result = subprocess.run([sys.executable, "-I", "-S", "-B", str(path), "status"],
                            capture_output=True, timeout=30)
    assert result.returncode and b"Unexpected source file" in result.stderr
    assert b"untracked import" not in result.stderr


def test_isolated_bootstrap_and_spawned_campaign_worker(fixture):
    # Use the real isolated CLI and handoff with a fixture-only ledger and no trainer.
    for path in ("scripts/__init__.py", "scripts/experiment_protocol.py", "domovoi/__init__.py",
                 "domovoi/experiments/__init__.py", "domovoi/experiments/launch.py",
                 "domovoi/experiments/runtime.py", "domovoi/experiments/protocol.py"):
        payload = (ROOT / path).read_text(encoding="utf-8")
        if path.endswith("/protocol.py"):
            payload = payload.replace(f"RESEARCH_URL = {json.dumps('https://github.com/CL-BAF/Domovoi-research.git')}",
                                      f"RESEARCH_URL = {str(fixture.remote)!r}")
            payload += "\n\ndef published_implementation(commit):\n    pass\n"
        fixture.put(fixture.source, path, payload)
    fixture.put(fixture.source, "scripts/run_pilot.py",
                "from domovoi.experiments.launch import start_training_child\n"
                "def _run(argv):\n    return start_training_child(argv).wait(timeout=35)\n")
    fixture.put(fixture.source, "scripts/train.py", WRITE.replace("def main", "def _run"))
    fixture.source_sha = fixture.commit(fixture.source, "isolated subprocess fixture")
    fixture.plan.update(implementation_commit=fixture.source_sha,
                        command={"script": "scripts/run_pilot.py", "argv": ["--fixture"]})
    fixture.pin_inputs()
    fixture.approve()
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(fixture.source / "scripts/experiment_protocol.py"),
         "run", ID, "--workspace", str(fixture.workspace), "--dependencies", str(fixture.deps)],
        env=dict(os.environ), capture_output=True, text=True, timeout=65,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    fixture.sync()
    assert protocol.valid_completion(fixture.rows[-1])
    assert (fixture.workspace / "out/receipt.json").read_text() == "fixture output"


def test_source_blob_check_ignores_git_assume_unchanged(fixture):
    git(fixture.source, "update-index", "--assume-unchanged", SCRIPT)
    fixture.put(fixture.source, SCRIPT, "unexpected payload")
    with pytest.raises(ValueError, match="differs from its Git blob"):
        verify_checkout(fixture.source)


@pytest.mark.parametrize("mutation", ["dependency", "extra_dependency", "workspace", "metric", "offline"])
def test_drift_or_outage_does_not_claim(fixture, mutation, monkeypatch):
    if mutation == "metric":
        fixture.plan["execution"]["analysis"].pop("primary_metric")
    fixture.approve()
    if mutation == "dependency":
        fixture.put(fixture.deps, "fixture_dependency.py", "value = 2\n")
    if mutation == "extra_dependency":
        fixture.put(fixture.deps, "shadow.py", "value = 2\n")
    if mutation == "workspace":
        fixture.put(fixture.workspace, "unreviewed.txt", "mutable evidence")
    if mutation == "offline":
        monkeypatch.setattr(protocol, "RESEARCH_URL", str(fixture.remote / "missing"))
    with pytest.raises(ProtocolError):
        fixture.invoke()
    assert not (fixture.source / ".git/domovoi-protocol").exists()


@pytest.mark.parametrize("action", [
    "Path('../secret.txt').read_text()", "Path('unapproved').write_text('x')",
    "__import__('subprocess').run(['unapproved'])",
    "with __import__('concurrent.futures', fromlist=['ThreadPoolExecutor']).ThreadPoolExecutor() as pool:"
    "\n        pool.submit(Path('unknown.txt').read_bytes).result()",
])
def test_approved_adapter_has_bounded_python_io(fixture, action):
    fixture.put(fixture.source, SCRIPT, "from pathlib import Path\ndef main(argv):\n    " +
                action + "\n    return 0\n")
    fixture.source_sha = fixture.commit(fixture.source, "locked fixture workload")
    fixture.plan["implementation_commit"] = fixture.source_sha
    fixture.pin_inputs()
    fixture.approve()
    with pytest.raises(ProtocolError, match="refused|locked tool"):
        fixture.invoke()
    fixture.sync()
    assert fixture.rows[-1]["state"] == "FAILED"


def test_successful_return_requires_declared_output_receipt(fixture):
    fixture.put(fixture.source, SCRIPT, "def main(argv):\n    return 0\n")
    fixture.source_sha = fixture.commit(fixture.source, "empty callback")
    fixture.plan["implementation_commit"] = fixture.source_sha
    fixture.pin_inputs()
    fixture.approve()
    with pytest.raises(ProtocolError, match="Missing or unsafe completion evidence"):
        fixture.invoke()
    fixture.sync()
    assert fixture.rows[-1]["state"] == "FAILED"


def test_two_fast_forward_claims_of_same_parent_cannot_both_win(fixture):
    fixture.approve()
    with Ledger() as first, Ledger() as second:
        snapshots = [ledger.refresh() for ledger in (first, second)]

        def claim(ledger, snapshot, value):
            head, state = snapshot
            slot(state, ID).update(state="RUNNING", claim_sha256=value)
            try:
                ledger.advance(head, state, "concurrent fixture claim")
            except ProtocolError:
                return False
            return True

        with ThreadPoolExecutor(2) as pool:
            attempts = [pool.submit(claim, ledger, snap, str(i) * 64)
                        for i, (ledger, snap) in enumerate(zip((first, second), snapshots))]
            assert sorted(a.result() for a in attempts) == [False, True]


@pytest.mark.parametrize("malformed", [False, True])
def test_local_recovery_cannot_forge_success_or_overwrite_evidence(fixture, malformed):
    fixture.approve()
    receipt = fixture.manual_claim()
    pending = receipt.directory / "outcome.json"
    data = {"state": "COMPLETED", "execution_note": "made-up record", "exit_code": 0}
    if malformed:
        data["claim_sha256"] = "9" * 64
    launch.write_local(pending, data)
    original = pending.read_bytes()
    if malformed:
        with pytest.raises(ProtocolError, match="Malformed local outcome"):
            launch.recover(ID, "evidence missing")
    else:
        with launch.local_lock(receipt.directory / "worker.lock"):
            with pytest.raises(ProtocolError, match="still active"):
                launch.recover(ID, "evidence missing")
        launch.recover(ID, "evidence missing")
    fixture.sync()
    assert fixture.rows[-1]["state"] == ("RUNNING" if malformed else "INTERRUPTED")
    assert pending.read_bytes() == original


def test_network_failure_preserves_local_success_but_manual_close_is_interrupted(fixture, monkeypatch):
    fixture.approve()
    advance = Ledger.advance

    def unavailable(self, head, state, note):
        if note.startswith("Record"):
            raise ProtocolError("test network failure")
        return advance(self, head, state, note)

    with monkeypatch.context() as network:
        network.setattr(Ledger, "advance", unavailable)
        with pytest.raises(ProtocolError, match="test network failure"):
            fixture.invoke()
    pending = launch.receipt_directory(ID) / "outcome.json"
    original = pending.read_bytes()
    assert json.loads(original)["state"] == "COMPLETED"
    assert "completion_token" not in json.loads(original)
    launch.recover(ID, "terminal push failed")
    fixture.sync()
    assert fixture.rows[-1]["state"] == "INTERRUPTED"
    assert pending.read_bytes() == original


def test_handwritten_remote_completion_is_refused_by_history_verifier(fixture):
    fixture.approve()
    fixture.manual_claim()
    fixture.sync()
    fixture.rows[-1].update(state="COMPLETED", exit_code=0, completion_token="b" * 64,
                           execution_evidence={"out/receipt.json": "1" * 64})
    fixture.publish("fabricated result")
    with Ledger() as ledger:
        head, _ = ledger.refresh()
        with pytest.raises(ProtocolError, match="original execution proof"):
            ledger.verify_history(head)


def test_posthoc_metric_pin_is_refused(fixture):
    fixture.approve()
    assert fixture.invoke() == 0
    fixture.publish_result(changed_metric=True)
    fixture.approve("DRP-0002")
    with pytest.raises(ProtocolError, match="original approved source pins"):
        fixture.invoke("DRP-0002")


def test_worker_handoff_uses_active_claim_and_is_single_use(fixture, monkeypatch):
    fixture.put(fixture.source, "scripts/run_pilot.py", "def _run(argv): return 0\n")
    fixture.put(fixture.source, "scripts/train.py", WRITE.replace("main(argv)", "_run(argv)"))
    fixture.source_sha = fixture.commit(fixture.source, "worker source")
    fixture.plan.update(command={"script": "scripts/run_pilot.py", "argv": []},
                        implementation_commit=fixture.source_sha)
    fixture.pin_inputs()
    fixture.approve()
    key = "1" * 64
    receipt = fixture.manual_claim(script="run_pilot.py", key=key)
    handoff = receipt.directory / "handoff.json"
    launch.write_local(handoff, {"argv": []})
    monkeypatch.setattr(sys, "stdin", io.StringIO("2" * 64 + "\n"))
    with pytest.raises(ProtocolError, match="credential"):
        launch.run_worker(ID, fixture.source_sha, fixture.workspace, fixture.deps, [])
    monkeypatch.setattr(sys, "stdin", io.StringIO(key + "\n"))
    assert launch.run_worker(ID, fixture.source_sha, fixture.workspace, fixture.deps, []) == 0
    assert not handoff.exists()
    with pytest.raises(ProtocolError, match="one-use"):
        launch.run_worker(ID, fixture.source_sha, fixture.workspace, fixture.deps, [])
