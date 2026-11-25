# Domovoi Labs

Research laboratory for **capability-per-GiB** language models: how close can
a from-scratch model get to modern 27B-class capability inside a hard
**4 GiB GPU-memory budget**?

Phase 0 (this repository's current milestone) is **the laboratory itself**:
a boring, conventional Transformer baseline plus the complete pipeline —
data → training → checkpoints → inference → evaluation → profiling →
experiment tracking → reports — that all future architecture research will
run on. No Domovoi architecture has been invented yet, by design.

Read the spec first: **RESEARCH.md** (goal + rules), **BASELINES.md**
(comparison models), **EVALS.md** (evaluation layers).

## Research publication gate

Before using a research launcher, commit and push its experiment plan and
approval to [Domovoi Research](https://github.com/CL-BAF/Domovoi-research).
Publish the preceding experiment's results and interpretation before starting
another. The launcher verifies both on GitHub. See the
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md) and [contribution guide](CONTRIBUTING.md).
Run approved commands through the isolated bootstrap. The exact script and
arguments come from the approved plan. Keep source, dependencies and data
in separate directories. For CPU correctness work on the new reference
instruments, see [CPU reference instruments](docs/CPU_INSTRUMENTS.md).

```text
python -I -S -B scripts/experiment_protocol.py run DRP-NNNN --workspace <workspace> --dependencies <dependency-tree>
```

## Setup

Requires Python 3.10+ (developed on 3.13).

```
# PyTorch with CUDA (if you have an NVIDIA GPU):
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
# CPU-only works too:
python -m pip install torch

# The rest of the pipeline:
python -m pip install numpy pyyaml tqdm requests safetensors tokenizers

# Optional — benchmark data downloads for the eval suite:
python -m pip install datasets

# Dev / tests:
python -m pip install pytest
```

(or `python -m pip install -e ".[evals,dev]"` from this directory)

## The pipeline

```
dataset ─► prepare/tokenize ─► configure experiment ─► train ─► checkpoint
   ─► resume ─► load ─► generate ─► evaluate ─► profile (VRAM/speed)
   ─► compare vs baseline ─► structured results ─► report
```

### 1. Prepare data

```
python scripts/prepare_data.py --config configs/baseline_small.yaml
```

Downloads tinyshakespeare, splits it deterministically (first 90% train /
last 10% val, sha256 recorded), trains a byte-level BPE tokenizer, packs
tokens into `data/processed/tinyshakespeare/*.bin` with a `meta.json`
provenance record.

### 2. Train

Use the guarded command with a plan for `scripts/train.py` and the pinned config.

Allocates an experiment ID (`DML-0001`, ...), records git commit +
environment, trains (~41M-parameter baseline Transformer) with bf16 AMP,
writes `metrics.jsonl` / `val_metrics.jsonl`, periodic + final checkpoints,
and `summary.json`. Ctrl+C marks the run INTERRUPTED. Checkpoint recovery
uses the last complete state when an optimizer update was interrupted.

### 3. Resume

Use a reviewed resume plan to stage a copy of the pinned run into a declared writable directory.

Resume restores model, optimizer, scheduler, step count, RNG states and the
data-batch generator. Exact CUDA trajectory replay was verified with the
pinned deterministic stack. See the [CUDA recovery record](https://github.com/CL-BAF/Domovoi-research/blob/research/real-data-calibration/findings/CUDA-RESUME-INCIDENT.md).

### 4. Generate

Use a plan for `scripts/generate.py` that pins the checkpoint, prompt and decoding arguments.

Independent of training; prints text plus measured TTFT, decode tok/s and
peak CUDA VRAM.

### 5. Evaluate

Use a plan for `scripts/evaluate.py` with provisioned and hashed evaluation assets, and give `--out` a declared file.

Fast research evals (EVALS.md §1): validation perplexity, HellaSwag,
ARC-Challenge, PIQA, WinoGrande, GSM8K subset, IF mini-suite. Results are
machine-readable JSON with data hashes; benchmark data is stored strictly
separate from training data.

### 6. Profile

Use a plan for `scripts/profile_model.py` with a pinned checkpoint, runtime lock and declared `--out` file.

Parameter counts, checkpoint size, real CUDA peak VRAM (allocated +
reserved), prefill/decode speed, TTFT, and the VRAM-vs-context sweep. On
CPU, VRAM fields are null — never estimated.

### 7. Report + scoreboard

```
python scripts/make_report.py --experiment DML-0001 \
    --conclusion "..." --recommendation "..."
python scripts/scoreboard.py
```

Reports are regenerated only from stored artifacts, so they are always
reproducible. `results/scoreboard.md` aggregates all experiments.

## Tests

```
python -m pytest tests/ -v
```

All tests run on CPU, offline, in seconds — including a full end-to-end
smoke test (tiny data → train → checkpoint → reload → resume → generate →
eval → experiment metadata → report).

## Experiment tracking

Each run gets `experiments/DML-XXXX/` containing `metadata.json` (status,
git commit, config, env, dataset, tokenizer, seed, params), the config
copy, metrics, `evals.json`, `profile.json`, `report.md` and (gitignored)
`checkpoints/`. Statuses: RUNNING, FAILED, INVALID, INTERRUPTED, COMPLETED,
BASELINE, PROMISING, REJECTED, PROMOTED. **Failed experiments are never
deleted — they are research data.**

## Repository layout

```
configs/         experiment configs (YAML, validated)
domovoi/         the pipeline package (see HANDOFF.md for details)
scripts/         CLI entry points (one per pipeline stage)
experiments/     experiment artifacts (DML-XXXX dirs)
results/         scoreboard
tests/           pytest suite
docs/            design notes
```

## Roadmap and status

See ROADMAP.md for milestone state and HANDOFF.md for the current
development snapshot and next steps.
