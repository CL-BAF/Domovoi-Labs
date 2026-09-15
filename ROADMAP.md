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
- ✅ README with full setup + usage instructions

## M0.2 — Baseline model ⬜

- ✅ `domovoi/models/baseline_transformer.py` — plain-PyTorch pre-LN
  decoder-only Transformer (~41M params at baseline_small config)
- ✅ Causal masking via F.scaled_dot_product_attention
- ✅ KV cache for incremental decoding
- ✅ Tied embedding option
- ✅ Forward/masking tests passing on CPU
- ✅ Parameter count verified against breakdown

## M0.3 — Training pipeline ⬜

- ✅ Trainer: grad accumulation, clipping, warmup+cosine, AMP (bf16/fp16/fp32)
- ✅ Deterministic seeding + RNG-state checkpointing (bit-exact resume)
- ✅ Checkpoints: safetensors weights, optimizer/scheduler/RNG state, atomic save
- ✅ CheckpointManager (step_XXXXXXXX + latest.json)
- ✅ Metrics: train loss, val loss, tokens, tok/s, lr → JSONL
- ✅ CPU fallback; clean failure messages
- ✅ Training tests passing (save/load/resume bit-exact, crash + rescue
  mid-step interrupt replay the uninterrupted trajectory exactly)
- 🔄 First real run on CUDA (RTX 2050, 4 GiB) recorded as DML-0001 —
  STOPPED at the operator's request at ~step 520/3000 (loss 7.4→3.4,
  val ppl ~104 at step 500). No further local GPU training will run
  without explicit operator approval; completing the run (CPU-only finish,
  cloud GPU, or abandoning it) is the operator's decision.

## M0.4 — Data pipeline ⬜

- ✅ Deterministic download + block split + hash recording (meta.json)
- ✅ Byte-level BPE tokenizer wrapper (deterministic, sha256-pinned)
- ✅ Packed uint16 token binaries, memory-mapped batch loader
- ✅ Synthetic smoke corpus (no network needed for tests)
- ✅ Contamination guard: eval data tree fully separate from training data
  (AST-enforced by tests)
- ✅ tinyshakespeare prepared and hashed (307,607 train / 38,425 val
  tokens, vocab 4096)
- ✅ Data tests passing

## M0.5 — Inference ⬜

- ✅ Independent checkpoint → model path (config travels in checkpoint)
- ✅ Generation: temperature, top-k, top-p, max tokens, seed
- ✅ KV-cache incremental decode
- ✅ TTFT, prefill tok/s, decode tok/s measured (not estimated)
- ✅ Generation script (scripts/generate.py)
- ✅ Generation tests passing (context-window invariant: prompt + new
  tokens never exceed the window)

## M0.6 — Profiling ⬜

- ✅ Parameter counts (total/trainable/active)
- ✅ Checkpoint size on disk
- ✅ Real CUDA max-memory-allocated/reserved statistics
- ✅ Prefill/decode/TTFT medians; VRAM-vs-context sweep
- ✅ CPU runs report null VRAM explicitly (never fabricated)
- ✅ Profiling tests passing
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
- ✅ Eval tests passing (synthetic, no network)
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
- ✅ Registry/report tests passing

## M0.9 — End-to-end validation 🔄

- ✅ CPU smoke test (data → train → checkpoint → resume → generate →
  eval → experiment → report) passing
- ⛔ Real run (DML-0001): training halted at the operator's request
  (~step 520/3000 on the laptop GPU). Registry status write was also
  declined by the operator — DML-0001's recorded status awaits their
  decision. Remaining: finish or abandon the run, download eval data,
  evaluate, profile, report.
- ✅ Three review passes (correctness, research integrity, code quality)
  run as an adversarial verification workflow: 10 confirmed findings
  (3 high / 6 medium / 1 low), all fixed with regression tests;
  deferred findings triaged (see HANDOFF)

## M0.10 — Documentation and cleanup 🔄

- ✅ README complete (setup, commands, structure)
- 🔄 HANDOFF.md current
- ✅ docs/ pipeline overview
- ✅ Full test suite green (75 tests, CPU + CUDA-verified model code)

## M1 — Architecture research begins ⬜

Blocked until every M0 box is ✅. The first research experiments compare
against DML-0001 (baseline Transformer) under identical evaluation
conditions. Candidate directions live in RESEARCH.md §2.

---

## Change log

- 2026-09-15: Initial roadmap drafted during M0 implementation. Milestones
  flip to ✅ only after their tests actually pass.
- 2026-09-15: Full test suite green with torch 2.11.0+cu128 on the RTX
  2050 (65 tests). Executing the suite surfaced and fixed real bugs:
  rng.pt never restored on resume, CheckpointManager dropping `step`,
  generate() emitting a token beyond the context window, GSM8K number
  regex keeping trailing periods, trainer ModelOutput unpacking.
  DML-0001 real training run started.
- 2026-09-15: Operator halted the DML-0001 GPU run (~step 520/3000) and
  declined further local GPU training / eval downloads / registry writes
  without approval. Three-pass adversarial review completed: 10 confirmed
  findings, all fixed (rescue-checkpoint double-apply window during
  validation [high], prepare-cache config drift [high, 2 findings], eval
  tokenizer-hash verification [high], GradScaler not checkpointed,
  decode-throughput inflation, summary clobber on zero-progress resume,
  val_ppl agreement claims, batch-loader off-by-one) — suite now 75 tests
  green, CPU-only. Known remaining (deferred, low): no staleness
  heartbeat for processes killed without a Python exception.