# Experiment protocol

Every new research experiment needs a committed, pushed and approved plan before execution. Every earlier experiment needs a committed, pushed result and interpretation before the next experiment can start. Both rules apply to training, architecture tests, benchmarking, profiling, evaluation, hyperparameter changes, optimization, repeats and failures.

Domovoi Labs owns the code. [Domovoi Research](https://github.com/CL-BAF/Domovoi-research) owns the authoritative records. The canonical execution ledger is `experiments/protocol/state.json` on the research branch `research/real-data-calibration`. Keep that ref available across PR review and merge. A change to the canonical ref requires a coordinated code and ledger migration.

## Lifecycle

| Phase | Required record |
|---|---|
| PROPOSED | Reserve the next `DRP-NNNN` in the canonical ledger. Keep DML run and checkpoint IDs as artifact IDs. |
| DOCUMENTED | Commit and push the complete plan JSON and original Markdown record. Link the record from the research README and archive index. |
| APPROVED | Review the exact command, implementation commit, inputs and decision criteria. Commit and push approval with the full DOCUMENTED research commit as `plan_commit`. |
| RUNNING | The launcher verifies the GitHub records, creates a local recovery receipt and pushes a single global claim before calling the workload. |
| COMPLETED, FAILED or INTERRUPTED | The launcher records the execution outcome locally and publishes it to the ledger. Recover crashed claims with the command below. |
| ANALYSED | Add the evidence, results and interpretation to the original Markdown record and structured result JSON. Update the research README and pin the terminal commit. |
| PUBLISHED | Push the ledger entry with the full `analysis_commit`. The launcher checks the published evidence before it accepts another experiment. |

See the [research-side protocol and record format](https://github.com/CL-BAF/Domovoi-research/blob/research/real-data-calibration/docs/EXPERIMENT_PROTOCOL.md) for the plan and result fields. A fixed, reviewed comparison may contain serial arms or controls within one plan. Adding an arm, changing the command or repeating the experiment requires a new plan. A trajectory change is distinct from administrative recovery and must be documented explicitly.

## Guarded commands

Run from the root of a clean tracked implementation checkout at the approved commit. Keep mutable configs under the checkout and include their SHA-256 hashes in the plan. Include checkpoint, dataset and tokenizer digests and any other material inputs in the plan. The launcher's exact argument comparison excludes the leading `--research-id DRP-NNNN` selector. For example, with an approval for that precise argv, use the following command.

```text
python scripts/train.py --research-id DRP-0001 --config configs/baseline_small.yaml
```

`train.py` (including resume), `evaluate.py`, `profile_model.py`, `generate.py`, `inspect_model.py`, `verify_real_resume.py`, `run_pilot.py` and `real_data_campaign.py` all use the same gate. Campaign supervisors delegate one exact training command at a time with a single-use local handoff tied to the active GitHub claim. Their predetermined arms retain separate training processes.

The gate requires Git, GitHub network access and authenticated fast-forward pushes to the canonical research ref. It fetches the live ref, verifies its lifecycle history and pinned records, checks the exact implementation SHA on GitHub, and checks the declared input bytes. The RUNNING transition must be acknowledged by GitHub before the workload executes. A competing claim based on the same revision cannot also fast-forward the ref.

## Crash and outage recovery

```text
python scripts/experiment_protocol.py status
python scripts/experiment_protocol.py recover DRP-0001 --note "Launcher died after the committed checkpoint"
```

If GitHub is unavailable before launch, preserve the plan and push it when connectivity returns. If final publication of the execution state fails, preserve the console, artifacts and the receipt under the Git common directory at `domovoi-protocol/DRP-NNNN/`. Recover from the original checkout after the launcher and any worker have stopped. The command republishes the pending local terminal outcome or records INTERRUPTED if the process died before saving one. It verifies the owner against the live GitHub claim and leaves artifacts alone. It cannot train, take over another machine's claim or start a different experiment.

If the original receipt is irrecoverable, a repository reviewer must establish that the original worker has stopped, then publish an audited terminal correction on the canonical ref with the same claim identity and an explanation of missing evidence. Inspect existing checkpoints and logs without modifying them. Further compute, including a checkpoint continuation after terminal closure, requires a new published plan linked to the original.

## Enforcement scope

The gate covers the checked-in CLI entry points on revisions that contain it. The low-level Python APIs remain available for correctness tests and development, and old revisions and custom scripts can execute code independently. Those paths are subject to the same publication rule. GitHub repository writers and administrators remain trusted to protect the canonical ref against force-pushes and deletion. The gate checks required records, immutable identities, command arguments and declared hashes. Reviewers judge scientific adequacy and ensure the pin list covers the workload, including dependencies, host state and every material artifact. Do not edit code or input files while an experiment is running.
