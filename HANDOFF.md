# Domovoi Labs Handoff

## Current state

Building milestone **M0: the research laboratory** (pipeline before
architecture research). All core modules are written, the full test
suite is GREEN (65 tests, torch 2.11.0+cu128, CUDA verified on the
RTX 2050), and the first real training run (DML-0001, baseline ~41M
Transformer on tinyshakespeare) is executing on the GPU. The three
review passes are running as an adversarial verification workflow.
This file is updated after every major milestone — do not let it go
stale.

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

- DML-0001 training in progress (3000 steps, ~17.5k tok/s, expected
  wall ~25 min); afterwards: download eval data, evaluate, profile,
  report, scoreboard
- Three-pass review workflow running (correctness / research
  integrity / code quality, each finding adversarially verified)

## Known issues

- `scripts/download_eval_data.py` requires the optional `datasets` package
  (`pip install -e ".[evals]"`) and network access; core tests never touch
  the network. (Install was in flight at snapshot time.)
- Resuming with a *different* `training.steps` than the checkpointed run
  deliberately changes the LR-schedule horizon; the trainer now prints a
  loud warning — such a continuation is NOT comparable to an uninterrupted
  run and the tests pin the horizon.

## Failed approaches

- First torch cu128 install attempt died mid-download (flaky DNS +
  a Windows file lock at ~1.8/2.75 GB). Fixed with pip
  `--timeout 90 --retries 20` in a background shell.
- Executing the suite exposed that several "written" paths had never
  actually run: `load_checkpoint(restore_rng=False)` returned a dummy
  dict so the data-generator state was NEVER restored on resume;
  `CheckpointManager.save` dropped `step`; `generate()` could emit a
  token at position ctx_len (outside the window); the GSM8K number
  regex kept trailing periods (`'7.'` vs `'7'`); the Trainer unpacked a
  3-field ModelOutput into 2. All fixed with regression tests.
- A test scenario that "resumed" a 3-step run into a 6-step schedule
  was wrong by construction — bit-exact resume only holds for the same
  schedule horizon. The test now simulates a mid-run crash instead.

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

DML-0001 (baseline_small_v1) — training in progress at snapshot time.
See `experiments/DML-0001/` once complete: metadata.json, summary.json,
metrics.jsonl, val_metrics.jsonl, evals.json, profile.json, report.md.
Will hold real measured numbers only — nothing in this file or the
report is ever estimated.

## Latest benchmark results

None yet. Do not invent any.

## Hardware limitations

- Development GPU: NVIDIA RTX 2050, 4 GiB VRAM — exactly the Domovoi target
  budget, so training must use modest batch sizes (baseline config:
  batch 16 × context 512 with bf16 AMP).
- CPU fallback works; all tests are CPU-only and network-free.

## Immediate next steps

1. Wait for DML-0001 training to finish (~step 3000).
2. `python scripts/download_eval_data.py --tasks all --limit 500`
   (network permitting), then `python scripts/evaluate.py --experiment DML-0001`.
3. `python scripts/profile_model.py --experiment DML-0001`.
4. `python scripts/make_report.py --experiment DML-0001` and
   `python scripts/scoreboard.py`.
5. Triage the three-pass review findings; fix the confirmed ones and
   re-run the full test suite.
6. Update ROADMAP.md checkboxes and this file; commit.

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