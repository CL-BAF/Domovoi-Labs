# Domovoi Labs — pipeline design notes

Short notes explaining *why* the laboratory is shaped this way. The
authoritative spec lives in RESEARCH.md / BASELINES.md / EVALS.md.

## Why a boring baseline Transformer first

Every future architecture claim ("better capability per GiB") is only as
trustworthy as the control it is compared against. The baseline
(`domovoi/models/baseline_transformer.py`) is a conventional pre-LN
decoder-only Transformer because (a) its behaviour is well understood,
(b) any deviation in results is attributable to the new architecture, not to
a pipeline difference, and (c) it is plain PyTorch with zero framework
magic, so swapping in research architectures is a one-file change
registered in `domovoi/models/__init__.py`.

## Config-in-checkpoint

Checkpoints carry the full config that produced them (`checkpoint.json`).
Inference, evaluation and profiling rebuild the model from the checkpoint
alone. This removes an entire class of "wrong config at eval time" bugs and
lets old experiments be re-evaluated exactly as they were trained.

## Bit-exact resume

Resume restores: weights, optimizer, LR scheduler, step counter, tokens
seen, python/numpy/torch/CUDA RNG, *and* the data-batch offset generator.
Consequently, `train(steps=N)` interrupted and resumed equals
`train(steps=N)` uninterrupted — verified bit-for-bit in
tests/test_checkpoint.py. This is what makes interruption-safe
experimentation honest.

## Determinism policy

- Data split: positional block split, no RNG — identical on every machine.
- Tokenizer: byte-level BPE; training is deterministic; file is sha256
  pinned and verified at load time.
- Training batches: seeded generator saved/restored across checkpoints.
- Validation batches: constant seed → same batches every call, so val
  loss is comparable across steps, checkpoints and runs.
- Generation: seedable sampling; temperature 0 is exactly greedy.

## Contamination policy

- Benchmark data enters only via `scripts/download_eval_data.py` into
  `data/evals/` (gitignored).
- The training data pipeline (`domovoi/datasets/`) never references that
  tree — enforced by `tests/test_data.py::test_eval_data_isolation`.
- Every eval result records the sha256 of the exact data file scored and
  the sample count/IDs, so silent benchmark changes are detectable.

## VRAM honesty rules

1. Peak VRAM is only ever read from `torch.cuda.max_memory_allocated` /
   `max_memory_reserved` after the measured operation.
2. CPU runs store `null` VRAM, labelled `cuda: false`.
3. No code path may claim the 4 GiB qualification from theoretical weight
   sizes; the claim requires a measured qualifying run.

## Experiment artifact contract

```
experiments/DML-XXXX/
  metadata.json     id, status, timestamps, git commit, config, env,
                    dataset + tokenizer provenance, params, seed
  config.yaml       exact YAML the run consumed
  metrics.jsonl     train loss / lr / tok/s per log interval
  val_metrics.jsonl validation loss per eval interval
  summary.json      final training summary (tokens, losses, VRAM, wall time)
  evals.json        machine-readable eval results (+ data hashes)
  profile.json      params, ckpt size, VRAM, prefill/decode/TTFT, ctx sweep
  generation.json   sample generation with speed stats
  report.md         human-readable report (rebuildable from the above)
  checkpoints/      step_XXXXXXXX dirs (gitignored)
```

Reports regenerate from stored artifacts only — never from live models —
so any report can be reproduced or audited later.