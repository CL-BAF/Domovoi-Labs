# Domovoi Labs Handoff

## Current state

Building milestone **M0: the research laboratory** (pipeline before
architecture research). All core modules are written and the dependency
install (PyTorch CUDA) was in progress at the time of this snapshot. Tests
and the first real end-to-end run are the immediate next work. This file
is updated after every major milestone — do not let it go stale.

## Current Git commit

(See `git log -1` — this file is updated per milestone; treat the latest
commit on main as the source of truth.)

## Completed work

- Spec files in repo root: RESEARCH.md, BASELINES.md, EVALS.md
- Config system: `domovoi/config.py` (typed dataclasses, YAML, strict
  validation, clone/merge for ablations)
- Baseline model: `domovoi/models/baseline_transformer.py` — plain
  PyTorch pre-LN decoder-only Transformer, SDPA causal attention, KV cache,
  optional tied embeddings, GPT-2-style init. No Hugging Face model
  abstractions.
- Data: `domovoi/datasets/` — deterministic block split, byte-level BPE
  tokenizer wrapper (sha256-pinned), packed uint16 binaries, memory-mapped
  batch loader with restorable RNG stream, synthetic smoke corpus
- Training: `domovoi/training/` — Trainer (grad accumulation, clipping,
  warmup+cosine, bf16/fp16/fp32 AMP, periodic eval/checkpoints,
  KeyboardInterrupt rescue), AdamW param groups, safetensors checkpoints
  (model/optimizer/scheduler/RNG, atomic directory swap), bit-exact resume
- Inference: `domovoi/inference/generate.py` — independent checkpoint→model
  path, temperature/top-k/top-p, KV-cache decode, TTFT + prefill/decode
  speed measured, real CUDA peak VRAM
- Evals: `domovoi/evals/` — registry with val_ppl, HellaSwag, ARC-Challenge,
  PIQA, WinoGrande (per-choice contexts), GSM8K subset, IF mini-suite;
  machine-readable JSON; milestone-eval slots reserved
- Profiling: `domovoi/profiling/` — param counts, checkpoint size, real CUDA
  VRAM stats, inference speed medians, VRAM-vs-context sweep; CPU reports
  null VRAM (never fabricated)
- Experiments: `domovoi/experiments/` — DML-XXXX ID allocation, metadata
  (git commit, env, dataset, tokenizer, seed, params, status), status
  labels, markdown reports built only from stored artifacts, scoreboard
- Scripts: prepare_data, train, generate, evaluate, profile_model,
  download_eval_data, make_report, set_status, scoreboard
- Tests written for: model, masking, data, checkpoints/resume, generation,
  config, experiments, evals, profiling, and a full CPU end-to-end smoke test

## Active work

- PyTorch (cu128) install for the RTX 2050 4 GiB machine
- Running the full test suite; fixing failures
- First real end-to-end run → experiment DML-0001

## Known issues

- Nothing verified by execution yet at the time of this snapshot — tests
  exist but torch was still installing. Do not trust any "works" claim
  until the test suite is green.
- `scripts/download_eval_data.py` requires the optional `datasets` package
  (`pip install -e ".[evals]"`) and network access; core tests never touch
  the network.

## Failed approaches

(none yet)

## Important design decisions

- **Plain PyTorch model** — deliberately no HF model abstractions, so future
  architectures are easy to write and modify.
- **Config travels inside checkpoints** — inference/eval/profiling rebuild
  the model from `checkpoint.json`, never from a config file path.
- **Bit-exact resume** — RNG states including the data-batch generator are
  checkpointed; resumed runs reproduce the uninterrupted trajectory.
- **Deterministic validation** — val batches come from a fixed-seed
  generator, so val loss is comparable across steps/runs.
- **Contamination isolation** — benchmark data lives only in
  `data/evals/`, written by one script; the training pipeline never reads
  it (guarded by test_eval_data_isolation).
- **VRAM honesty** — CPU runs report null VRAM; CUDA numbers come from
  `torch.cuda.max_memory_allocated/reserved`, never from parameter-count
  estimates. The 4 GiB qualification can only ever be claimed from real
  measurements.
- **Experiments are append-only** — failed runs stay on disk as research
  data; only checkpoints are git-ignored, metadata/results/reports are
  committed.

## Repository structure

```
configs/            baseline_small.yaml, smoke_test.yaml
data/                raw/, processed/, tokenizers/, evals/  (gitignored)
domovoi/
  config.py          typed+validated experiment config
  models/            baseline_transformer.py (+ registry)
  datasets/          prepare.py, tokenizer.py, loader.py
  training/          trainer.py, checkpoint.py, optim.py
  inference/         generate.py
  evals/             base.py, val_ppl.py, gsm8k.py, if_mini.py
  profiling/         profiler.py
  experiments/       registry.py, report.py
  utils/             seeding.py, env.py
scripts/             prepare_data, train, generate, evaluate,
                     profile_model, download_eval_data, make_report,
                     set_status, scoreboard
experiments/         DML-XXXX/ (metadata, config, metrics, evals, report;
                     checkpoints/ gitignored)
results/             scoreboard.md
tests/               pytest suite incl. end-to-end smoke test
docs/                pipeline notes
```

## How to run training

```
python scripts/prepare_data.py --config configs/baseline_small.yaml
python scripts/train.py --config configs/baseline_small.yaml
python scripts/train.py --config configs/baseline_small.yaml --resume
```

## How to run inference

```
python scripts/generate.py --checkpoint experiments/DML-0001/checkpoints/step_00003000 \
    --prompt "Once upon a time" --max-new-tokens 200 --temperature 0.8
```

## How to run evaluations

```
python scripts/download_eval_data.py --tasks all --limit 500
python scripts/evaluate.py --experiment DML-0001 --limit 500
```

## How to run tests

```
python -m pytest tests/ -v          # all tests, CPU, no network
```

## Latest experiment

None recorded yet (DML-0001 pending first successful run).

## Latest benchmark results

None yet. Do not invent any.

## Hardware limitations

- Development GPU: NVIDIA RTX 2050, 4 GiB VRAM — exactly the Domovoi target
  budget, so training must use modest batch sizes (baseline config:
  batch 16 × context 512 with bf16 AMP).
- CPU fallback works; all tests are CPU-only and network-free.

## Immediate next steps

1. Finish torch install; run `python -m pytest tests/ -v` and fix failures.
2. `python scripts/prepare_data.py --config configs/baseline_small.yaml`.
3. Train baseline (configs/baseline_small.yaml) → DML-0001.
4. `python scripts/download_eval_data.py --tasks all --limit 500`
   (network permitting), then `python scripts/evaluate.py --experiment DML-0001`.
5. `python scripts/profile_model.py --experiment DML-0001`.
6. `python scripts/make_report.py --experiment DML-0001`.
7. Three review passes (correctness / research integrity / code quality).
8. Update ROADMAP.md checkboxes and this file; commit.

## Longer-term next steps

- Freeze the fast-research eval protocol (fixed sample IDs via --limit and
  recorded sha256).
- Add scaling-study configs (~20M / ~50M / ~100M) via clone_config.
- Begin architecture research (M1) against DML-0001 as the control.

## Files that must be read before continuing

- RESEARCH.md, BASELINES.md, EVALS.md (project spec)
- ROADMAP.md (milestone state)
- domovoi/config.py (config schema)
- domovoi/training/trainer.py (training loop)
- domovoi/evals/base.py (eval registry + scoring)
- domovoi/experiments/registry.py (experiment ID/metadata rules)