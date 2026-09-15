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

```
python scripts/train.py --config configs/baseline_small.yaml
```

Allocates an experiment ID (`DML-0001`, ...), records git commit +
environment, trains (~41M-parameter baseline Transformer) with bf16 AMP,
writes `metrics.jsonl` / `val_metrics.jsonl`, periodic + final checkpoints,
and `summary.json`. Ctrl+C saves a rescue checkpoint and marks the run
INTERRUPTED.

### 3. Resume

```
python scripts/train.py --config configs/baseline_small.yaml --resume
```

Resume is **bit-exact**: model, optimizer, scheduler, step count, RNG
states and the data-batch generator are all restored (verified by test).

### 4. Generate

```
python scripts/generate.py \
    --checkpoint experiments/DML-0001/checkpoints/step_00003000 \
    --prompt "Once upon a time" --max-new-tokens 200 \
    --temperature 0.8 --top-p 0.95
```

Independent of training; prints text plus measured TTFT, decode tok/s and
peak CUDA VRAM.

### 5. Evaluate

```
python scripts/download_eval_data.py --tasks all --limit 500   # once; needs `datasets`
python scripts/evaluate.py --experiment DML-0001 --limit 500
```

Fast research evals (EVALS.md §1): validation perplexity, HellaSwag,
ARC-Challenge, PIQA, WinoGrande, GSM8K subset, IF mini-suite. Results are
machine-readable JSON with data hashes; benchmark data is stored strictly
separate from training data.

### 6. Profile

```
python scripts/profile_model.py --experiment DML-0001
```

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