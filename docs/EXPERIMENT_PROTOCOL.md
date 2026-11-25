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

Use a dedicated clean checkout of the approved commit. On Windows, clone with `-c core.autocrlf=false` so tracked files match the Git blob bytes. Use separate existing directories for the workspace and dependency tree. The launcher rejects untracked files, including ignored files and caches, anywhere in source. The workspace starts with only pinned assets and empty declared output/temporary paths. Keep the interpreter and dependencies outside both source and workspace. For an approved plan, the command is:

```text
python -I -S -B scripts/experiment_protocol.py run DRP-NNNN --workspace <workspace> --dependencies <dependency-tree>
```

The plan supplies the exact script and argv. Use separate argument values such as `--config`, `@source/configs/example.yaml`, `--out`, `@work/out/results.json`. The bootstrap resolves these paths against its explicit roots. `train.py`, `evaluate.py`, `profile_model.py`, `generate.py`, `inspect_model.py`, `verify_real_resume.py`, `run_pilot.py` and `real_data_campaign.py` use this entry point. Campaign supervisors delegate one training command at a time with a single-use handoff tied to the active GitHub claim. A tracked, pinned `research_workloads/<name>.py` can expose `main(argv: list[str]) -> int` for an external Python harness. Review the adapter and full framework lock, and write its evidence to the declared output paths. There is no arbitrary external command passthrough.

For evaluation, profiling and inspection set `--out` to a new declared file. For generation use `--save-to`. Keep imported checkpoint/run inputs read-only. A new resume plan can use a reviewed source adapter to copy the pinned run artifacts into a declared writable run directory before calling the trainer. Pin the exact copy list and include the continued run's final artifacts. A declared output directory starts without imported data.

The gate requires Git, GitHub network access and authenticated fast-forward pushes to the canonical research ref. It fetches the live ref, verifies its lifecycle history and pinned records, checks the exact implementation SHA on GitHub, and checks the declared input bytes. The RUNNING transition must be acknowledged by GitHub before the workload executes. A competing claim based on the same revision cannot also fast-forward the ref.

## Execution and analysis manifest

The plan's `inputs` maps source-relative paths to SHA-256. Include the command, configurations, runtime lock and each analysis file. The `execution` object includes these fields.

| Field | Required content |
|---|---|
| `runtime_lock` | Source-relative JSON file pinned under `inputs` |
| `analysis` | Source-relative paths for `evaluation`, `scoring`, `statistics`, `primary_metric`, `dataset_generation` and `transforms`, each pinned under `inputs` |
| `assets` | Workspace-relative SHA-256 files, including source data, packed inputs, tokenizers or imported checkpoints used by the workload |
| `output_dirs`, `temporary_dirs` | Nonempty lists of disjoint relative writable directories |
| `completion_artifacts` | Nonempty list of expected nonempty artifact files under `output_dirs` |

The lock pins `python`, `python_sha256`, `base_python_sha256`, `platform`, `machine` and `dependencies_sha256`. Obtain these fields through `domovoi.experiments.runtime.runtime_identity` on the intended bootstrap interpreter and exact dependency tree, without a workload. The fingerprint includes paths and file bytes throughout the dependency tree. Add `environment` with only reviewed thread/CUDA settings and `tools` with SHA-256 of any permitted `nvidia-smi` executable. Use `gpu: null` for CPU, or provide `uuid`, `name`, `driver` and `cuda_runtime` for the selected GPU together with the telemetry executable. The launcher disables site startup hooks, removes ambient environment settings and uses offline flags for supported packages. This lock assumes a trusted Python standard library, OS, native runtime and reviewed package supply chain. Store provisioning records and a wheel/environment lock alongside the plan to reproduce the byte fingerprint.

Published results carry `analysis_implementation_commit` and `analysis_sources` with the path and SHA-256 of each role. The verifier requires exact agreement with the preapproved plan. This is provenance checking. The review also has to inspect that the analysis actually used those sources. There is no amendment workflow in this revision. A changed primary endpoint needs a new reviewed plan before further execution, with any post-hoc analysis labelled exploratory.

## Crash and outage recovery

```text
python -I -S -B scripts/experiment_protocol.py status
python -I -S -B scripts/experiment_protocol.py recover DRP-NNNN --note "Launcher died after the committed checkpoint"
```

If GitHub is unavailable before launch, preserve the plan and push it when connectivity returns. On interruption, preserve the console, artifacts and the receipt under the Git common directory at `domovoi-protocol/DRP-NNNN/`. Recover from the original checkout after the launcher and worker have stopped. The RUNNING claim commits the hash of an in-memory completion credential. The live launcher can publish COMPLETED only with the credential, zero exit status and hashes of all declared completion artifacts. Manual recovery can publish FAILED with the supplied note and a valid pending failure receipt, or INTERRUPTED if positive execution proof is unavailable. If a terminal push already landed, recovery returns with the remote history intact. `outcome.json`, artifacts and a separate recovery receipt preserve the evidence.

If the original receipt is lost or malformed, a named owner must establish that the worker has stopped and review the evidence and proposed FAILED or INTERRUPTED closure. Keep the prior claim identity and original evidence. Further compute, including checkpoint continuation after terminal closure, requires a new plan linked to the original.

## Enforcement scope

The launcher verifies source, dependencies and declared workspace inputs before execution. Python audit checks cover common file, network and subprocess operations in the approved workload and its threads. This is a guard for reviewed code. Native calls, pre-opened descriptors, a malicious adapter or host administrator, concurrent external mutation, old revisions, direct library imports and separate processes remain outside that control. Use restricted host accounts, read-only inputs and disjoint workspaces with controlled access. The launcher trusts the canonical GitHub history and authorized writers. The [proposed owner policy](RESEARCH_APPROVAL_POLICY.md) describes the extra controls needed before treating that history as a human approval signal.
