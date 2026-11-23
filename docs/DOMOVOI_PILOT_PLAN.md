# Domovoi pilot plan: does shared-core recurrence buy anything at 512 context?

Status: **PLAN ONLY. Nothing here has been run.** Every step marked
[AUTHORISATION REQUIRED] needs the project owner's explicit approval:
GPU training, any dataset download, new experiment IDs and status changes.
Written 2026-11-23, after the post-audit fixes (POST_AUDIT_FIXES.txt).

This plan **supersedes the thresholds in
`docs/RECURRENT_ARCHITECTURE_RESEARCH.md` sections 8-9 for the pilot.**
Those sections describe the full matrix and the 8K frontier question,
which the pilot cannot answer. It starts from Codex's proposed three-arm
falsification experiment (docs/CODEX_ARCHITECTURE_REVIEW.md, "Smallest
falsification experiment") and adopts its metric and margins. Where this
plan differs, the reason is stated.

## 1. Question and scope

The pilot asks two questions at 512 context, on one corpus, at one
width:

- **Q1 (does depth help at fixed storage?)** With the same 2 stored blocks,
  does applying them 4 times (R02x4) beat applying them once (B02)?
- **Q2 (what does tying cost at fixed compute?)** At the same effective
  depth 8, and therefore about the same training FLOPs, how much of the
  advantage of 8 independent blocks (B08) does R02x4 keep while storing 4x
  fewer block weights?

It **cannot** answer the project question: capability per GiB of measured
inference memory at 8K, at the 4 GiB target. That needs 8K positions, a
conventional width/depth frontier and reference-GPU measurements (section 12).
A positive pilot only justifies those next steps.

## 2. The three arms and why each is necessary

All three use the `baseline_small` width: d_model 640, 10 heads, ffn_dim
1728, tied embeddings, learned absolute positions with 512 rows, and
dropout 0. Only the `model.architecture`, `n_layers` and `recurrence_*`
fields differ.

| Arm | Architecture | Stored blocks | Depth | Effective depth A | Unique params | Applied params / token | Est. train GFLOP / token |
|---|---|---|---|---|---|---|---|
| **B02** | baseline_transformer | 2 | 1 | 2 | 10,665,856 | 10,665,856 | 0.066 |
| **R02x4** | domovoi_recurrent_v1 (cycle) | 2 | 4 | 8 | 10,665,856 | 33,812,224 | 0.216 |
| **B08** | baseline_transformer | 8 | 1 | 8 | 33,812,224 | 33,812,224 | 0.216 |

Parameter counts are exact: the closed form is verified against tensor
counts in tests. FLOPs are analytic estimates (accounting.py: matmuls plus
causal attention at the average attended length 256.5, training = 3x
forward). Generate the configs with
`scripts/plan_recurrent_experiments.py --pilot --seeds 1234 1235 1236 --base <pilot base>`.

- **B02 is the parameter-matched control.** It has exactly R02x4's stored
  weights and differs only in how often they are applied. Without it,
  "R02x4 is good" cannot be attributed to recurrence rather than to the
  width/vocabulary/data setup.
- **R02x4 is the candidate.** Depth 4 is the middle of the brief's 1/2/4/8
  ladder. Cycle order is the default; the order ablation (R02x4s) is
  deferred because it only matters if R02x4 works at all.
- **B08 is the compute-matched control.** It has the same applied
  parameters, the same effective depth and about the same training and
  inference FLOPs as R02x4, but 8 untied blocks. It measures the price of
  tying. Without it a win over B02 could simply be "more compute helps",
  which is not news.

Not included, and why: a conventional arm matched to R02x4 on BOTH stored
parameters and total FLOPs (B02 trained on about 3.3x the tokens).
Codex lists it as optional. It answers "should extra FLOPs go to data
or to depth?", a different question, and it needs 3.3x more unique data.
Add it only if Q1 is positive.

**Initialisation caveat (Codex).** Using the same integer seed gives the
same data order but not matched initial weights across shapes, because
modules of different sizes consume the RNG differently. B08 and R02x4
share A = 8, so their residual-projection init scale 0.02/sqrt(2A)
matches. B02 (A = 2) has a larger scale than R02x4. This is a confound of
the B02-vs-R02x4 contrast; it is reported, not corrected, in the pilot.

## 3. Comparisons

| Contrast | Matched on | Differs in | Reads as |
|---|---|---|---|
| B02 vs R02x4 | stored parameters (10.67M), data, tokens, schedule | applied depth (2 vs 8), training FLOPs (3.3x) | Q1: benefit of reusing the same weights deeper |
| B08 vs R02x4 | effective depth, applied parameters, est. FLOPs, data, tokens | stored parameters (33.8M vs 10.7M), optimizer state, gradient statistics | Q2: cost of tying |
| Gap closure G = (L_B02 - L_R02x4) / (L_B02 - L_B08) | | | fraction of B08's gain over B02 that R02x4 recovers |

Matching analytic FLOPs is not the same as matching wall time or
optimizer work: R02x4 updates 3.2x fewer parameters per step and may run
at a different speed. Wall time and measured throughput are reported
alongside, never substituted.

## 4. Training corpus (versioned) [AUTHORISATION REQUIRED to download]

tinyshakespeare is **not** used. It has about 0.3M training tokens. At the
old budget every arm sees it about 80 times, so lighter arms can win
through regularisation alone (RESEARCH doc section 7).

**Proposed corpus: FineWeb-Edu, a deterministic document subset** (data card:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu; the ODC-BY
licence is as stated on the card, to be re-checked at download).
Fallback with the same procedure: DCLM-Baseline 1.0
(https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0, CC BY 4.0
per its card). Both were proposed in the Codex review; this plan adds the
exact procedure.

Versioning and preparation:

1. Pin the dataset repository **commit hash** at download time and record
   it, together with the exact shard file names and their SHA-256 hashes,
   in a manifest (`data/manifests/pilot_fineweb_edu.json`). No revision
   hash is stated here because none has been looked up. It must come from
   the actual download, not from this document.
2. Read documents in the shard's stored order and take the first documents
   until the train split holds **at least 110M tokens**. Then take the next
   ~3M tokens for validation and the next ~3M for a held-out test split.
   The split is at **document level**, before packing. Record document
   counts and IDs. The size alone does NOT prevent repeated windows: the
   current loader draws random offsets with replacement, so windows can
   overlap or repeat whatever the corpus size. "No repeated windows" holds
   only with the segment sampler of prerequisite P9 (section 5). 110M
   tokens give about 214,000 disjoint 513-token segments; the budget
   needs 195,328 (6,104 steps x 32 sequences).
3. Exact-deduplicate documents across splits (hash of normalised text).
   Record how many documents were removed.
4. Train **one** byte-level BPE tokenizer, vocab 4096 like the current
   pipeline, on the **train split only**. It is shared by all arms. Record
   its SHA-256 (the Trainer already refuses packed bins made with a
   different tokenizer).
5. Pack with an EOS token between documents. Save document boundaries.
6. The test split is untouched until the final evaluation. It is not used
   for any decision before then.

Consequence: **DML-0001 cannot be the control for the pilot.** It used a
different corpus and tokenizer. B02 and B08 are the controls. DML-0001
remains a historical pipeline-validation record and is not modified. This
partly reverses the original reason for keeping the old pipeline, that
DML-0001 would be the control. The pipeline is still kept so that B02/B08
and R02x4 share every non-architecture line of code.

## 5. Training budget and hyperparameters (identical for all arms)

| Setting | Value | Note |
|---|---|---|
| Tokens per run | 100,007,936 (6,104 steps x 16,384) | about 9.4 tokens/param for B02 and R02x4, 3.0 for B08 |
| Tokens per update | 16,384 = micro-batch 16 x 512 x accumulation 2 | micro-batch identical to DML-0001 (fits a 4 GiB card, see section 9) |
| Context | 512 (learned positions, unchanged) | the positional-encoding change is a separate decision (section 12) |
| Optimizer | AdamW (0.9, 0.95), weight decay 0.1 on matrices only | as in training/optim.py |
| LR | 6e-4 peak, 200 warmup steps, cosine to 10% | the same for every arm; no per-arm tuning |
| Grad clip | 1.0 | pre-clip norm logged (clip_grad_norm_ returns it) |
| Precision | bf16 autocast, FP32 master weights | |
| Seeds | 1234, 1235, 1236 | the same seed gives every arm the same batch order (data generator seeded from seed+1), so differences are **paired** |
| Data order | the packed train split cut into disjoint 513-token segments (512 inputs + 1 target), visited once in a seed-determined permutation, without replacement (prerequisite P9) | each training window is then used at most once and windows never overlap. The current random-offset loader cannot guarantee this, so the pilot does not run on it |

**Why 100M tokens.** Codex proposed 10M per arm. That is about 1 token per
stored parameter for B02 and 0.3 for B08, which measures early learning
speed rather than what the arms can represent. The analytic cost of all 9
runs at 100M is small (section 10), so the budget is raised tenfold, and
the 10M, 25M and 50M points are kept as **intermediate evaluations**
rather than separate runs. The curve shows whether the ranking changes
with training. It is still well below 20 tokens per parameter for B08, so
conclusions are explicitly **undertrained-regime** conclusions.

**Symmetric tuning rule.** No arm is tuned. If ANY run of any arm meets a
divergence condition (section 8, conditions 1-3), the pilot stops (rule 6). A re-run uses the same LR grid {3e-4, 6e-4, 1.2e-3} for
**all three arms** on seed 1234, choosing by validation loss, then repeats
all seeds at the chosen LR. The re-run is labelled as tuned.

## 6. Evaluation (identical conditions for all arms)

- **Primary metric: token-weighted negative log-likelihood (nats/token) on
  the held-out TEST split.** Score each within-document next-token target
  once in windows of up to 512 targets, including the last partial window
  and EOS. The first token of each document has no target score. Consecutive
  windows share only the input token needed to predict the next target.
  Run `scripts/evaluate.py --experiment RUN --tasks test_nll` once at the
  end of training. The evaluator uses FP32, rejects unfinished checkpoints
  and protocol deviations, and records packed-data, boundary and tokenizer hashes.
- **Curves:** the existing fixed-window validation loss every 250 steps, and
  a full validation-split NLL near 10M, 25M, 50M and 100M tokens. Set
  `training.extra_checkpoint_steps: [610, 1526, 3052]` to retain the
  intermediate checkpoints. Run `scripts/evaluate.py --experiment RUN
  --checkpoint-step STEP --tasks val_nll`. Each checkpoint's full-split
  results are stored separately.
- **Uncertainty:** paired per-seed differences
  d_s = NLL_B02,s - NLL_R02x4,s and e_s = NLL_R02x4,s - NLL_B08,s; the
  mean, min and max over the 3 seeds; and a document-level bootstrap
  interval (1,000 resamples of test documents) per run, reported
  separately. Three seeds give a **screening result**, not a significance
  test, and are labelled as such.
- **Diagnostics (R02x4):** run the `val_nll_exits` task for early-exit NLL
  after each of the 8 applications
  (`application_hidden_states` -> ln_f -> lm_head). These are diagnostics
  only, because the head was trained for the final depth. Also pre-clip
  grad norm, and the per-application residual-stream norm.
- **Sanity:** greedy samples from a fixed set of 5 prompts, for humans to
  read. They are not scored.

## 7. Measurements

| What | How | Label |
|---|---|---|
| Peak training memory | `Trainer.vram_stats()`: torch.cuda max_memory_allocated/reserved over the run | measured (CUDA) |
| Training throughput | tokens per second per logged window plus whole-run wall clock | measured |
| Inference peak memory | `profile_model`: batch 1, the timed runs' peak (max over all timed runs since the post-audit fix), plus a whole-GPU snapshot | measured (CUDA) |
| Inference throughput | synchronised prefill tok/s, TTFT (prefill + first sample), decode tok/s over exactly the timed decode tokens | measured |
| Context sweep | `scripts/profile_model.py --context-sweep 64 128 256 448`: exact prompts of those lengths, each followed by profile_model's sweep default of 8 new tokens (prompt cap 512 - 8; none of these rows is capped) | measured |
| KV cache | exact logical and retained-storage bytes from the cache tensors | exact count |
| Weights | exact parameter bytes; checkpoint size on disk | exact count |
| FLOPs | `analytic_estimates` | estimate, labelled |

All inference measurements run on the **same GPU, driver, torch build and
dtype** for all arms, with no other training job running. Each profile
records device, dtype, torch/CUDA versions and the device-wide memory
snapshot. Profiles currently run in the checkpoint's FP32. A bf16
inference measurement needs a small flag that does not exist yet
(prerequisite P6). **None of these numbers are compared with the 4,096 MiB
target.** The pilot models are 20-130 MiB of weights, so meeting a 4 GiB
limit at 512 context says nothing about the target configuration.

## 8. Early-termination conditions (fixed in advance)

Termination is only for **failure or invalidity**, never for "looks good".
Stopping early on a favourable result would bias the outcome.

Per run, stop and record status FAILED with the reason in notes when:

1. The loss is NaN or inf at any step.
2. After warmup, the 50-step mean training loss exceeds 1.5x its running
   minimum for 200 consecutive steps (divergence).
3. The pre-clip grad norm exceeds 100 for 50 consecutive steps.
4. The measured peak training memory exceeds 12 GiB. The 16 GB card
   already had about 3 GB in use by other processes when checked.
5. Measured throughput in the first 200 steps projects the run beyond 3x
   its calibrated estimate (section 10). This points to a setup problem.

For the pilot as a whole:

6. Any single run of any arm fails (conditions 1-3) at the shared LR: stop
   the pilot and report. A divergence in one seed is already evidence that
   the shared LR is not safe for that arm, and continuing would compare a
   partial arm with complete ones. A re-run follows the symmetric tuning
   rule (section 5), never per-arm tuning. Conditions 4-5 (memory,
   projected time) stop only the affected run, which is then investigated
   as a setup problem before the pilot continues.
7. The total measured GPU time exceeds 2x the calibrated estimate: stop
   and ask for re-authorisation.

## 9. Hardware constraint (development GPU: RTX 5080, 16 GB)

- **Training memory, estimate with an empirical anchor.** DML-0001
  trained the 10-application baseline (d 640, bf16 autocast, micro-batch
  16x512) on an RTX 2050 whose recorded total memory is 4,095.5 MB. All
  pilot arms have at most 8 applications at the same width and
  micro-batch, so they should fit well inside 16 GB. Activation memory
  scales with applications, so R02x4 and B08 should be about equal and B02
  smaller. Weights, gradients and AdamW states take about 16 bytes per
  stored parameter: about 0.17 GB for B02/R02x4 and 0.54 GB for B08.
  **These are estimates. The calibration step measures them.**
- **Final target (unchanged): 4,096 MiB inference VRAM** on the reference
  setup (batch 1, no offload, >= 8 tok/s, usable 8K). The pilot does not
  test it.

## 10. Computational requirements (estimates)

Analytic training FLOPs at 100,007,936 tokens per run (accounting.py):

| Arm | FLOP per run | x 3 seeds |
|---|---|---|
| B02 | 6.59e15 | 1.98e16 |
| R02x4 | 2.16e16 | 6.49e16 |
| B08 | 2.16e16 | 6.49e16 |
| **Total** | | **1.50e17** |

**Throughput anchor (measured, different GPU).** DML-0001 logged a median
of 17,402 tokens/s on the RTX 2050, for an estimated 0.266 GFLOP/token.
That is about **4.6 TFLOP/s effective**. At that rate the 9 runs would take
**about 9.0 GPU-hours**, plus about 5-10% for evaluation. The RTX 5080 is
expected to be faster, but **no RTX 5080 throughput has been measured**.
Small models (B02) usually reach lower utilisation, so per-arm times do not
scale exactly with FLOPs.

**Step 0, calibration [AUTHORISATION REQUIRED, GPU].** Run 200 steps of each
arm (seed 1234) on the RTX 5080 into throwaway experiment IDs marked
INVALID. Record tokens/s and peak training memory, then replace the
estimate above with a measured one before starting the pilot. The cost is
about 3 x 200 x 16,384 = 9.8M tokens, a few minutes.

Disk: about 0.5 GB of raw text for about 116M tokens (unmeasured; it
depends on bytes per token of the new tokenizer), a few hundred MB of
packed bins, and 9 runs x (checkpoints of 41-130 MB FP32 safetensors, plus
optimizer state) for a few GB.

## 11. Decision criteria (pre-registered; frozen before the first pilot run)

Let d_s = NLL_B02 - NLL_R02x4 (positive means R02x4 is better) and
g_s = d_s / (NLL_B02 - NLL_B08), per seed s, on the final test NLL.

**Control validity.** The B08 contrast is interpretable only if B08 beats
B02 by at least **0.05 nats/token in all 3 seeds**. If it does not, Q2 is
**inconclusive** (the control gap is too small) and only Q1 is reported.

**Advance** (recommend the next stage, section 12) if ALL of these hold:
- A1: d_s >= **0.02 nats/token** in all 3 seeds (Codex's margin);
- A2: mean gap closure >= **0.50**, with every seed above 0.25;
- A3: no early-termination event in any arm;
- A4: measured inference memory. Field: `context_sweep[i]["peak_vram_allocated_mb"]`
  of `profile.json` for the row with `target_tokens == 448`, produced by
  `scripts/profile_model.py` on the same GPU with FP32 weights and no
  autocast (so `weight_dtype` and `kv_dtype` are both float32). Each arm's
  seed-1234 final checkpoint is profiled **3 times in separate
  processes**. The noise N_arm is that arm's max - min over the 3 values.
  A4 holds if mean(B08) - mean(R02x4) >= max(3 x max(N_B08, N_R02x4),
  16 MiB). For scale, the exact FP32 weight bytes differ by about 88 MiB
  (129.0 vs 40.7 MiB) and the KV caches are equal, so A4 checks that the
  stored-weight saving actually shows up in the measured peak. Decode
  throughput is reported alongside. A slower R02x4 is a cost, not a
  disqualifier, at this scale.

**Abandon fixed sharing at this scale** if EITHER holds:
- B1: d_s < 0.02 in all 3 seeds (depth does not help at fixed storage); or
- B2: mean gap closure < **0.25** with every seed below 0.50, and the
  control is valid.

**Inconclusive** otherwise. Add seeds 1237 and 1238 for all three arms (5
pairs) and apply the same rules once more. In that second evaluation
every rule is read over all 5 seeds, with no other change:
- "in all 3 seeds" becomes "in all 5 seeds" (control validity, A1, B1);
- "every seed above 0.25" / "every seed below 0.50" means all 5 seeds
  (A2, B2);
- every "mean" is over the 5 seeds (A2, B2);
- A3 covers all 15 runs, and A4 is not re-measured (it does not depend on
  the seed count).
If the result is still mixed, report it as inconclusive. Do not add more
seeds.

Abandoning would be honest grounds to drop `domovoi_recurrent_v1`. An
advance only means "worth the 8K and frontier work". It is **not** evidence
that the architecture beats a conventional Transformer at the target.

## 12. What the pilot deliberately does not decide

- **Positional encoding for 8K.** The 512-row learned table is NOT silently
  replaced. Options are a larger learned table (adds 4.9M parameters at
  8K; old checkpoints would not load), RoPE or ALiBi. Any change applies
  to **both** families, needs a new versioned baseline and a
  512-context ablation first, and is a separate owner decision (Codex
  review, "Training data and positional migration").
- **The memory frontier.** Beating B08 per stored byte is not the
  hypothesis. The hypothesis needs a conventional width/depth sweep at
  equal **measured** peak memory at 8K, where the KV cache (which grows
  with A) dominates (docs/RECURRENT_ARCHITECTURE_RESEARCH.md section 5).
- **Order ablation, R01x8 and depth 8 or 16:** only after an advance.

## 13. Prerequisites before Step 0 (all [AUTHORISATION REQUIRED] unless noted)

- **P1:** git, so every run records a commit hash. The source tree
  currently has no .git.
- **P2:** a CUDA build of torch in a **separate** environment for the
  RTX 5080. The audited `.venv` is CPU-only and should stay as the CPU test
  reference. Run `python -m pytest -q tests/test_pilot.py -k cuda_profile`
  in the CUDA environment before calibration.
- **P3:** the corpus download and manifest (section 4), plus a preparation
  path for parquet shards. The current `prepare_data.py` handles
  single-file text sources.
- **P4:** the pilot base config (512 context, baseline_small width, the
  section 5 settings, the new dataset/tokenizer paths), then
  `plan_recurrent_experiments.py --pilot --seeds 1234 1235 1236`.
- **P5 (implemented):** full-split NLL and early-exit evaluation (section 6).
  Corpus preparation must supply `files.val_bin` and `files.test_bin` plus
  `files.val_document_ends` and `files.test_document_ends` in `meta.json`.
  Each boundary file is a JSON array of strictly increasing exclusive token
  offsets, including the end of the split. Record the corresponding packed
  token counts under `tokens.val` and `tokens.test`.
- **P6** (code, safe to write): a bf16 inference option for
  `profile_model.py`, if bf16 inference numbers are wanted.
- **P7:** the owner freezes section 11 (any edits happen before Step 0).
- **P8 (implemented):** set `training.pilot_stop_checks: true`. Set
  `training.pilot_calibrated_seconds` per arm to its projected full-run
  duration after calibration. Omit that duration during calibration itself.
  The checks use pre-clip gradient norms, post-warmup 50-step loss windows,
  and CUDA peak allocated bytes. The throughput projection uses elapsed
  training time through step 200. The stop reason is stored in the summary.
  Launch calibrated configs with `scripts/run_pilot.py CONFIG [CONFIG ...]`.
  It runs serially, stops launching arms after a failure, and applies the
  aggregate limit to elapsed campaign time (including setup and evaluation).
  A resource failure requires inspection before launching any remaining arms.
  Exclude runs with `protocol_deviations` and read learning curves through
  `domovoi.experiments.read_metrics`.
- **P9 (implemented):** set `training.sampler: disjoint_segments` in the
  pilot base config. The sampler rejects an insufficient corpus and saves
  its position in each checkpoint.
