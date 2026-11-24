"""Exercise the launch boundary with synthetic callbacks and an isolated Git remote."""

import hashlib
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from domovoi.experiments import launch, protocol
from domovoi.experiments.protocol import Ledger, ProtocolError, git, slot

ID = "DRP-0001"
ARGS = ["--config", "config.yaml"]
GATED = ("train", "evaluate", "profile_model", "generate", "inspect_model",
         "verify_real_resume", "run_pilot", "real_data_campaign")


class RemoteFixture:
    def __init__(self, tmp_path, monkeypatch):
        self.remote = tmp_path / "remote"
        self.editor = tmp_path / "editor"
        self.local = tmp_path / "implementation"
        for path in (self.remote, self.editor, self.local):
            path.mkdir()
        git(self.remote, "init", "--bare", "--quiet")
        git(self.editor, "init", "--quiet")
        git(self.local, "init", "--quiet")
        self.put("config.yaml", "test: fixture\n", self.local)
        self.implementation = self.commit("fixture implementation", self.local)
        self.plan = {field: "synthetic verification fixture" for field in protocol.PLAN_FIELDS}
        self.plan.update(
            id=ID, implementation_commit=self.implementation,
            command={"script": "scripts/train.py", "argv": ARGS},
            inputs={"config.yaml": hashlib.sha256(b"test: fixture\n").hexdigest()},
        )
        monkeypatch.setattr(protocol, "RESEARCH_URL", str(self.remote))
        monkeypatch.setattr(launch, "ROOT", self.local)
        monkeypatch.chdir(self.local)
        monkeypatch.setattr(launch, "published_implementation",
                            lambda sha: sha == self.implementation)
        self.rows = []
        self.publish("bootstrap")

    def put(self, path, data, repo=None):
        target = (repo or self.editor) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((data if isinstance(data, str) else json.dumps(data)).encode())

    def commit(self, message, repo):
        git(repo, "add", ".")
        git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "--quiet", "-m", message)
        return git(repo, "rev-parse", "HEAD").decode().strip()

    def publish(self, message):
        self.put(protocol.STATE, {"schema": 1, "experiments": self.rows})
        head = self.commit(message, self.editor)
        git(self.editor, "push", "--quiet", str(self.remote), f"HEAD:{protocol.LEDGER_REF}")
        return head

    def sync(self):
        git(self.editor, "fetch", "--quiet", str(self.remote), protocol.LEDGER_REF)
        git(self.editor, "merge", "--quiet", "--ff-only", "FETCH_HEAD")
        self.rows = json.loads((self.editor / protocol.STATE).read_bytes())["experiments"]
        return git(self.editor, "rev-parse", "HEAD").decode().strip()

    def approve(self, experiment_id=ID):
        self.rows.append({"id": experiment_id, "state": "PROPOSED"})
        self.publish("propose " + experiment_id)
        self.put(protocol.plan_path(experiment_id), {**self.plan, "id": experiment_id})
        self.put(protocol.record_path(experiment_id), "# Plan\n")
        for prefix, doc in (("", "README.md"), ("experiments/", protocol.INDEX)):
            previous = self.editor / doc
            text = previous.read_text(encoding="utf-8") if previous.exists() else ""
            path = protocol.record_path(experiment_id).removeprefix(prefix)
            self.put(doc, text + f"\n[{experiment_id}]({path})\n")
        self.rows[-1]["state"] = "DOCUMENTED"
        documented = self.publish("document " + experiment_id)
        self.rows[-1].update(state="APPROVED", plan_commit=documented)
        return self.publish("approve " + experiment_id)

    def analyse_and_publish(self):
        terminal = self.sync()
        row = self.rows[-1]
        result = {field: "synthetic callback result" for field in protocol.RESULT_FIELDS}
        self.put(protocol.result_path(ID), {**result, "id": ID, "outcome": row["state"],
                                           "evidence": ["measurements.json"]})
        self.put("measurements.json", {"fixture": True})
        self.put(protocol.record_path(ID), "# Plan\n\n# Result\nCallback finished.")
        self.put("README.md", (self.editor / "README.md").read_text() + "\nPublished summary.\n")
        row.update(state="ANALYSED", outcome=row["state"], terminal_commit=terminal)
        analysed = self.publish("analyse")
        row.update(state="PUBLISHED", analysis_commit=analysed)
        self.publish("publish")


@pytest.fixture
def remote(tmp_path, monkeypatch):
    return RemoteFixture(tmp_path, monkeypatch)


@pytest.mark.parametrize("name", GATED)
def test_direct_cli_refuses_without_research_id(name, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / "scripts" / f"{name}.py"),
                             "--config", "missing"], cwd=tmp_path, capture_output=True)
    assert result.returncode == 1
    assert b"Start with --research-id" in result.stderr
    assert not list(tmp_path.iterdir())


def test_launch_publishes_terminal_and_only_unblocks_after_analysis(remote):
    remote.approve()
    observed = []

    def workload(args):
        with Ledger() as ledger:
            _, state = ledger.refresh()
        observed.append((args, slot(state, ID)["state"]))
        return 0

    assert launch.run_research("train.py", workload, ["--research-id", ID, *ARGS]) == 0
    assert observed == [(ARGS, "RUNNING")]
    assert launch.run_research("train.py", workload, ["--research-id", ID, *ARGS]) == 1
    remote.sync()
    assert remote.rows[-1]["state"] == "COMPLETED"
    candidate = remote.rows[-1]
    with Ledger() as ledger:
        head, _ = ledger.refresh()
        with pytest.raises(ProtocolError, match="Publish the outcome"):
            ledger.verify_publication(head, candidate)
    remote.analyse_and_publish()
    remote.approve("DRP-0002")
    assert launch.run_research("train.py", workload, ["--research-id", "DRP-0002", *ARGS]) == 0


@pytest.mark.parametrize("mismatch", ["argv", "script", "input", "dirty", "plan", "absent", "offline"])
def test_mismatches_and_outage_never_call_workload(remote, mismatch, monkeypatch):
    if mismatch != "absent":
        remote.approve()
    if mismatch == "dirty":
        remote.put("config.yaml", "changed", remote.local)
    if mismatch == "input":
        remote.put("config.yaml", "changed", remote.local)
        implementation = remote.commit("changed input", remote.local)
        remote.plan["implementation_commit"] = implementation
        # Make the source match a new documented plan, leaving the old file digest.
        remote.rows[-1]["state"] = "DOCUMENTED"
        remote.put(protocol.plan_path(ID), remote.plan)
        pinned = remote.publish("repin plan")
        remote.rows[-1].update(state="APPROVED", plan_commit=pinned)
        remote.publish("approve again")
    if mismatch == "plan":
        remote.put(protocol.plan_path(ID), {**remote.plan, "question": "edited after approval"})
        remote.publish("edit the approved plan")
    if mismatch == "offline":
        monkeypatch.setattr(protocol, "RESEARCH_URL", str(remote.remote / "missing"))
    script = "generate.py" if mismatch == "script" else "train.py"
    args = [*ARGS, "--resume"] if mismatch == "argv" else ARGS
    result = launch.run_research(script, lambda _: pytest.fail("workload started"),
                                 ["--research-id", ID, *args])
    assert result == 1
    assert not (remote.local / ".git/domovoi-protocol").exists()


def test_two_clones_cannot_claim_the_same_remote_parent(remote):
    remote.approve()
    with Ledger() as first, Ledger() as second:
        snapshots = [ledger.refresh() for ledger in (first, second)]

        def claim(ledger, snapshot, token):
            parent, state = snapshot
            slot(state, ID).update(state="RUNNING", claim_sha256=token)
            try:
                ledger.advance(parent, state, "concurrent claim")
            except ProtocolError:
                return False
            return True

        with ThreadPoolExecutor(2) as pool:
            attempts = [pool.submit(claim, ledger, snap, str(i) * 64)
                        for i, (ledger, snap) in enumerate(zip((first, second), snapshots), 1)]
        assert sorted(attempt.result() for attempt in attempts) == [False, True]


@pytest.mark.parametrize("pending", [False, True])
def test_crash_recovery_closes_only_owned_inactive_claim(remote, pending):
    remote.approve()
    directory = launch.receipt_directory(ID)
    directory.mkdir(parents=True)
    receipt = launch.Receipt(ID, directory, "test recovery token", "train.py")
    launch.write_local(directory / "claim.json", {"token": receipt.token, "script": receipt.script})
    if pending:
        launch.write_local(directory / "outcome.json",
                           {"state": "FAILED", "execution_note": "saved before network failure"})
    with Ledger() as ledger:
        head, state = ledger.refresh()
        slot(state, ID).update(state="RUNNING", claim_sha256=receipt.digest)
        ledger.advance(head, state, "claim before simulated crash")
    with launch.local_lock(directory / "worker.lock"):
        with pytest.raises(ProtocolError, match="still active"):
            launch.recover(ID, "record this crash")
    launch.recover(ID, "record this crash")
    remote.sync()
    assert remote.rows[-1]["state"] == ("FAILED" if pending else "INTERRUPTED")


def test_worker_requires_active_remote_claim_and_consumes_handoff(remote, monkeypatch):
    remote.approve()
    directory = launch.receipt_directory(ID)
    directory.mkdir(parents=True)
    receipt = launch.Receipt(ID, directory, "parent token", "run_pilot.py")
    launch.write_local(directory / "claim.json", {"token": receipt.token, "script": receipt.script})
    key = "0" * 64
    launch.write_local(directory / "handoff.json", {"argv": ARGS})
    args = ["--research-worker", ID, *ARGS]
    monkeypatch.setattr(sys, "stdin", io.StringIO(key + "\n"))
    assert launch.run_research("train.py", lambda _: pytest.fail("unclaimed worker"), args) == 1
    with Ledger() as ledger:
        head, state = ledger.refresh()
        slot(state, ID).update(state="RUNNING", claim_sha256=receipt.digest,
                               worker_sha256=hashlib.sha256(key.encode()).hexdigest())
        ledger.advance(head, state, "claim parent")
    monkeypatch.setattr(sys, "stdin", io.StringIO("1" * 64 + "\n"))
    assert launch.run_research("train.py", lambda _: pytest.fail("wrong credential"), args) == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO(key + "\n"))
    assert launch.run_research("train.py", lambda a: 0 if a == ARGS else 1, args) == 0
    assert launch.run_research("train.py", lambda _: pytest.fail("worker replay"), args) == 1


def test_spawned_campaign_worker_accepts_only_its_parent_handoff(remote):
    source = Path(__file__).resolve().parents[1]
    remote.put("scripts/train.py", f"""import sys
from pathlib import Path
sys.path.insert(0, {str(source)!r})
from domovoi.experiments import launch, protocol
launch.ROOT = Path({str(remote.local)!r})
protocol.RESEARCH_URL = {str(remote.remote)!r}
def work(argv):
    print('verified worker ' + ' '.join(argv))
    return 0
raise SystemExit(launch.run_research('train.py', work))
""", remote.local)
    remote.plan["command"] = {"script": "scripts/real_data_campaign.py", "argv": []}
    remote.approve()

    def campaign(_):
        child = launch.start_training_child(ARGS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, _ = child.communicate(timeout=20)
        assert child.returncode == 0, output
        assert b"verified worker --config config.yaml" in output
        return 0

    assert launch.run_research("real_data_campaign.py", campaign, ["--research-id", ID]) == 0


def test_lifecycle_history_catches_removed_run(remote):
    remote.approve()
    remote.rows = []
    head = remote.publish("silently remove the approved run")
    with Ledger() as ledger:
        ledger.refresh()
        with pytest.raises(ProtocolError, match="cannot delete"):
            ledger.verify_history(head)


def test_outage_after_launch_saves_pending_result_and_recovery_publishes_it(remote, monkeypatch):
    remote.approve()
    valid_url = protocol.RESEARCH_URL

    def disconnect(_):
        monkeypatch.setattr(protocol, "RESEARCH_URL", str(remote.remote / "offline"))
        return 0

    assert launch.run_research("train.py", disconnect, ["--research-id", ID, *ARGS]) == 1
    pending = launch.receipt_directory(ID) / "outcome.json"
    assert json.loads(pending.read_bytes())["state"] == "COMPLETED"
    monkeypatch.setattr(protocol, "RESEARCH_URL", valid_url)
    remote.sync()
    assert remote.rows[-1]["state"] == "RUNNING"
    launch.recover(ID, "publish stored outcome")
    remote.sync()
    assert remote.rows[-1]["state"] == "COMPLETED"
