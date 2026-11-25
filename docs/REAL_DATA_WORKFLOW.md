# Real-data workflow

The NAS holds immutable raw files. Local NVMe holds the SQLite staging journal,
packed corpus versions and experiment artifacts. Each output path names a fresh
version or campaign.

## Inventory and preparation

Use the CUDA environment with the project's optional `corpora` dependencies.
Audit the available raw files without writing to the NAS:

```powershell
python scripts/audit_corpora.py --root <raw-datasets> --report <new-report.json>
python scripts/prepare_streaming.py --spec <source-spec.json> --out <new-packed-version>
```

A source spec records the dataset name, version, requested vocabulary, and
`sources` array. Each source specifies `id`, `revision`, `path`, `kind`,
`split`, and optionally an expected SHA-256. The readers are `text` with a
document separator (default `<|endoftext|>`), `parquet` with a selectable
`column` (default `text`), and `wikitext` which groups rows by top-level
article headings within each shard. Preserve official splits with `train`,
`val` and `test`; for a source without splits, use `hash` with
`test_permille` and `val_permille` (default 10 each). Held-out official
documents enter the deduplication index before training documents.

Normalized exact deduplication uses Unicode normalization, case folding and
whitespace normalization. Similar but unequal documents can survive. Use
`max_documents` per source and `token_budgets` per split for preview versions.
Token budgets stop at a batch boundary, so realized counts can be slightly
larger. The tokenizer trains on a bounded, train-only sample and checks a
Unicode round trip. Packing writes uint16 token IDs with EOS and document-end
offsets. For mixtures, `mixture.document_quotas` names positive document
quotas by source ID. The manifest records the selected document proportions;
they do not imply equal token proportions. Training shuffles disjoint
segments by seed after packing. Keep evaluation benchmark files outside
preparation specs.

Rerun the same preparation command after an interruption. The SQLite journal
skips completed ingestion and packing work and truncates uncommitted output
bytes before appending. Check the pid in the staging lock if a hard kill left
one behind. Remove the lock only after verifying the writer is gone. A
complete, published packed directory is refused on subsequent runs.

## Calibration and bounded campaign

Run from the exact clean implementation commit with Git on the process PATH.
Publish each approved plan and the previous result under the
[experiment protocol](EXPERIMENT_PROTOCOL.md). Pin `scripts/real_data_campaign.py`
or `scripts/verify_real_resume.py` and the exact arguments in the plan. Declare
packed data and any imported checkpoints as read-only workspace assets, and
declare the writable campaign or verification directories.

```powershell
python -I -S -B scripts/experiment_protocol.py run DRP-NNNN --workspace <workspace> --dependencies <dependency-tree>
```

Each campaign uses a single CUDA device and a shared deadline, records the
implementation commit, stage config and dependencies, and creates distinct
experiment IDs. The GPU monitor appends temperature, utilization, device
memory and power samples. A monitored thermal, free-space or operator stop
creates the campaign's `STOP` file. The trainer checks it at update
boundaries and closes the active run. The controller leaves the other arms
pending when a run fails or is stopped.

Read `DML-xxxx.console.log`, `DML-xxxx.gpu.jsonl`, and the matching
experiment's checkpoints, metrics and summary. Close and publish the original
interruption before planning further compute. An approved continuation uses the
original YAML and pins `scripts/train.py` with `--resume --experiment-dir <run-dir>`.
Use a new stop-file path and a freshly bounded
monotonic deadline if running under a new supervisor. Calibration figures
describe training. They do not qualify an inference workload.
