# Domovoi Labs — Roadmap

**Status markers:** ✅ completed (tested) · 🔄 active · ⛔ blocked · ⬜ planned

Milestone M0 (the pipeline laboratory) precedes all architecture research.
Nothing is marked complete until it has actually been run and tested.

---

## M0.1 — Repository foundation 🔄

- ✅ Spec files in repo: RESEARCH.md, BASELINES.md, EVALS.md
- ✅ Package layout `domovoi/` with submodules (models, training, inference,
  evals, profiling, datasets, utils, experiments)
- ✅ Config schema (YAML, validated, typed) — `configs/*.yaml`
- ✅ pyproject.toml with pinned dependency set
- ✅ .gitignore (data, checkpoints, artifacts excluded; experiment metadata kept)
- ⬜ README with full setup + usage instructions

## M0.2 — Baseline model ⬜

- ✅ `domovoi/models/baseline_transformer.py` — plain-PyTorch pre-LN
  decoder-only Transformer (~41M params at baseline_small config)
- ✅ Causal masking via F.scaled_dot_product_attention
- ✅ KV cache for incremental decoding
- ✅ Tied embedding option
- ⬜ Forward/masking tests passing on CPU
- ⬜ Parameter count verified against breakdown

## M0.3 — Training pipeline ⬜

- ✅ Trainer: grad accumulation, clipping, warmup+cosine, AMP (bf16/fp16/fp32)
- ✅ Deterministic seeding + RNG-state checkpointing (bit-exact resume)
- ✅ Checkpoints: safetensors weights, optimizer/scheduler/RNG state, atomic save
- ✅ CheckpointManager (step_XXXXXXXX + latest.json)
- ✅ Metrics: train loss, val loss, tokens, tok/s, lr → JSONL
- ✅ CPU fallback; clean failure messages
- ⬜ Training tests passing (save/load/resume bit-exact)
- ⬜ First real run on CUDA (RTX 2050, 4 GiB) recorded as an experiment

## M0.4 — Data pipeline ⬜

- ✅ Deterministic download + block split + hash recording (meta.json)
- ✅ Byte-level BPE tokenizer wrapper (deterministic, sha256-pinned)
- ✅ Packed uint16 token binaries, memory-mapped batch loader
- ✅ Synthetic smoke corpus (no network needed for tests)
- ✅ Contamination guard: eval data tree fully separate from training data
- ⬜ tinyshakespeare prepared and hashed
- ⬜ Data tests passing

## M0.5 — Inference ⬜

- ✅ Independent checkpoint → model path (config travels in checkpoint)
- ✅ Generation: temperature, top-k, top-p, max tokens, seed
- ✅ KV-cache incremental decode
- ✅ TTFT, prefill tok/s, decode tok/s measured (not estimated)
- ✅ Generation script (scripts/generate.py)
- ⬜ Generation tests passing

## M0.6 — Profiling ⬜

- ✅ Parameter counts (total/trainable/active)
- ✅ Checkpoint size on disk
- ✅ Real CUDA max-memory-allocated/reserved statistics
- ✅ Prefill/decode/TTFT medians; VRAM-vs-context sweep
- ✅ CPU runs report null VRAM explicitly (never fabricated)
- ⬜ Profiling tests passing
- ⬜ Baseline profile recorded into experiment DML-0001

## M0.7 — Evaluations ⬜

- ✅ Modular eval registry (val_ppl, HellaSwag, ARC-C, PIQA, WinoGrande,
  GSM8K subset, IF mini-suite)
- ✅ MC log-likelihood scoring (acc + acc_norm, per-choice contexts for
  WinoGrande)
- ✅ Machine-readable JSON results with data sha256 + sample IDs
- ✅ Benchmark download/normalization script with version pinning
- ✅ Milestone-eval slots reserved (MMLU-Pro, GPQA, MATH-500, LCB, IFBench,
  RULER)
- ⬜ Eval tests passing (synthetic, no network)
- ⬜ Real benchmark data downloaded; first eval results recorded

## M0.8 — Experiment tracking ⬜

- ✅ DML-XXXX ID allocation from directory scan
- ✅ metadata.json: id, timestamps, git commit, config, env, dataset,
  tokenizer, seed, params, status
- ✅ Statuses: RUNNING/FAILED/INVALID/INTERRUPTED/COMPLETED +
  BASELINE/PROMISING/REJECTED/PROMOTED
- ✅ Failed experiments never deleted
- ✅ Human-readable reports generated only from stored artifacts
- ✅ Scoreboard aggregation (results/scoreboard.md)
- ⬜ Registry/report tests passing

## M0.9 — End-to-end validation ⬜

- ⬜ CPU smoke test (data → train → checkpoint → resume → generate →
  eval → experiment → report) passing
- ⬜ Real run: prepare tinyshakespeare → train baseline on CUDA →
  evaluate → profile → report as DML-0001
- ⬜ Three review passes (correctness, research integrity, code quality)

## M0.10 — Documentation and cleanup ⬜

- ⬜ README complete (setup, commands, structure)
- ⬜ HANDOFF.md current
- ⬜ docs/ pipeline overview
- ⬜ Full test suite green

## M1 — Architecture research begins ⬜

Blocked until every M0 box is ✅. The first research experiments compare
against DML-0001 (baseline Transformer) under identical evaluation
conditions. Candidate directions live in RESEARCH.md §2.

---

## Change log

- 2026-09-15: Initial roadmap drafted during M0 implementation. Milestones
  flip to ✅ only after their tests actually pass.