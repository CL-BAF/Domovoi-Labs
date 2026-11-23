# Research proposal: Domovoi Shared-Core Recurrent Transformer (`domovoi_recurrent_v1`)

Status: **implemented and unit-tested on CPU; not trained; no capability
result exists.**  This document describes the experiment, not a finding.
Line-by-line code documentation is in `ARCHITECTURE_IMPLEMENTATION_EXPLAINED.txt`.

---

## 1. Central hypothesis

> A model that repeatedly applies shared learned computation may obtain
> greater useful capability per unit of stored parameter memory than a
> conventional Transformer.

Operationally, for this project (RESEARCH.md: "capability per GiB" inside a
4096 MiB inference budget):

**H1.** At equal *measured peak inference memory* (weights + KV cache +
activations, at the target context), a shared-core recurrent model reaches
lower held-out loss / higher benchmark scores than the best conventional
Transformer of the same width.

**H0 (what we expect to see if the idea fails).** Recurrent arms land on or
above the conventional Transformer's loss-vs-memory frontier: sharing saves
weight memory, but the saving is matched or exceeded by lost capacity and/or
by the extra KV cache that recurrence creates.

The hypothesis is deliberately stated in terms of *measured memory*, not
parameter count. Parameter count is not the quantity the project optimises.

## 2. What the architecture is

`n_layers` distinct pre-LN Transformer blocks (identical in design to the
baseline's) are applied `recurrence_depth` times each. Effective depth is
`A = n_layers × recurrence_depth`. Two reuse orders:

| order      | schedule for 3 blocks, depth 2 | prior name (Takase & Kiyono) |
|------------|--------------------------------|------------------------------|
| `cycle`    | 0 1 2 0 1 2                    | CYCLE                        |
| `sequence` | 0 0 1 1 2 2                    | SEQUENCE                     |

Everything else is identical to the DML-0001 baseline: learned absolute
positions, LayerNorm, GELU MLP, biases, tied embeddings, GPT-2-style init
(residual projections scaled by `1/sqrt(2A)`). With `recurrence_depth = 1`
the model is bit-identical to the baseline (tested).

Three different "memory" mechanisms, which must not be confused:

1. **Hidden-state recurrence across depth.** Implemented. The residual
   stream is the recurrent state, passed from one block application to the
   next within a single token's forward computation.
2. **Memory across sequence positions.** Ordinary causal self-attention over
   the KV cache, unchanged from the baseline. *Every application of a block
   has its own keys/values*, so the KV cache grows with `A`, not with
   `n_layers`.
3. **Persistent memory across generated tokens** (a compact state carried
   forward in place of, or in addition to, the KV cache). **Not implemented.**
   Nothing in v1 carries state between tokens except the KV cache.

## 3. Related work

Verified this session (title, authors and identifier checked against arXiv
or the venue via web search on 2026-11-23):

- **Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser — "Universal Transformers"**,
  ICLR 2019, arXiv:1807.03819. One block applied recurrently over depth,
  with a per-position halting mechanism (ACT). Closest conceptual ancestor
  of `cycle` with `n_layers = 1`.
- **Dabre & Fujita — "Recurrent Stacking of Layers for Compact Neural
  Machine Translation Models"**, AAAI 2019, arXiv:1807.05353. Reports that
  one layer stacked 6 times approaches a 6-layer model in NMT quality.
- **Lan et al. — "ALBERT: A Lite BERT for Self-supervised Learning of
  Language Representations"**, ICLR 2020, arXiv:1909.11942. All-layer
  parameter sharing in an encoder; parameter count no longer grows with
  depth; reports a small quality cost.
- **Takase & Kiyono — "Lessons on Parameter Sharing across Layers in
  Transformers"**, SustaiNLP 2023, arXiv:2104.06022. Defines SEQUENCE, CYCLE
  and CYCLE (REV) assignments of M parameter sets to N layers. Our
  `recurrence_order` names follow this paper.
- **Graves — "Adaptive Computation Time for Recurrent Neural Networks"**,
  arXiv:1603.08983. Learned halting; reference for any future adaptive-depth
  variant.
- **Bae et al. — "Relaxed Recursive Transformers: Effective Parameter
  Sharing with Layer-wise LoRA"**, ICLR 2025, arXiv:2410.20672. A looped
  block with per-depth low-rank deltas. Converts pretrained models rather
  than training natively, which is relevant to our native-fit rule.
- **Geiping et al. — "Scaling up Test-Time Compute with Latent Reasoning: A
  Recurrent Depth Approach"**, arXiv:2502.05171 (NeurIPS 2025). A 3.5B model
  trained from scratch with a recurrent core iterated a variable number of
  times at test time.
- **Saunshi, Dikkala, Li, Kumar, Reddi — "Reasoning with Latent Thoughts: On
  the Power of Looped Transformers"**, ICLR 2025, arXiv:2502.17416. On
  synthetic reasoning tasks, a k-layer model looped L times nearly matches a
  kL-layer model.
- **Bae et al. — "Mixture-of-Recursions: Learning Dynamic Recursive Depths
  for Adaptive Token-Level Computation"**, arXiv:2507.10524. Per-token
  routed recursion depth. Explicitly discusses per-recursion KV caching
  versus KV sharing across recursions, which is the same issue as §2 point 2.

Used but **not verified this session (verify before citing externally):**

- Kaplan et al., "Scaling Laws for Neural Language Models", arXiv:2001.08361.
  Only used for the FLOP-counting convention (2 FLOPs per multiply-add,
  backward ≈ 2× forward).
- Hoffmann et al., "Training Compute-Optimal Large Language Models"
  ("Chinchilla"), arXiv:2203.15556. Only used for the general point that
  tokens-per-parameter matters when comparing models of different sizes.
- Radford et al. 2019, GPT-2 technical report. The `1/sqrt(2·depth)`
  residual-init scaling is attributed to it in the existing baseline code.
- EdgeFormer (arXiv:2202.07959) was mentioned in a Hugging Face comment as
  earlier work on LoRA-relaxed sharing. **Not checked.** Do not cite until
  verified.

Summary of the literature: sharing reliably reduces stored parameters, and
several papers report that looping recovers much of the quality of the
corresponding unshared deep model, especially on reasoning-like tasks. None
of these results is evidence for *our* question: capability per GiB of
measured inference memory, at 8K context, for tiny models trained from
scratch.

## 4. Expected mechanisms (speculative; not demonstrated)

- **Depth for free in memory:** more sequential refinement steps without
  more stored weights. Weight memory is `O(n_layers)` while depth is `O(A)`.
- **Iterative refinement:** applying the same function repeatedly may favour
  algorithm-like, iterative computation (the reasoning claims in Saunshi et
  al. and Geiping et al.).
- **Regularisation:** fewer free parameters for the same depth. This is a
  *confound* as well as a mechanism (§7).

Mechanisms that work **against** the hypothesis, all derived from the
implementation and not speculative:

- KV cache ∝ effective depth `A`. At 8K context the KV cache already
  dominates weights for every arm below (§6).
- Compute and latency ∝ `A`. Decode tokens/s falls roughly in proportion to
  the extra applications.
- Capacity: a block shared across `r` positions must serve all of them.

## 5. Baselines and comparisons

- **M1 control (per ROADMAP.md): DML-0001**, `baseline_transformer`,
  10 layers, d=640, 41,527,680 parameters. Its run was stopped at about step
  520/3000; its status is the operator's decision and is untouched here.
- **Matrix ladder.** Conventional baselines B02/B04/B08/B16 with the same
  width as the recurrent arms. These are needed because DML-0001's depth (10)
  is not divisible by the planned recurrence depths 4 and 8.

Note: RESEARCH.md defines M1 as "beat the strongest native-fit baseline on
efficiency", while ROADMAP.md defines M1 as "architecture research begins".
This proposal does not resolve that conflict and **does not claim any
milestone**.

### Planned arms (generated by `scripts/plan_recurrent_experiments.py`)

All arms: d_model 640, 10 heads, ffn 1728, vocab 4096, context 512, and the
identical data/tokenizer/optimiser/schedule/token budget from
`configs/baseline_small.yaml`. Numbers are closed-form estimates, bf16.

| arm    | stored blocks | depth | eff. depth | unique params | applied params | weights MiB | KV MiB @512 | train GFLOP/token |
|--------|--------------:|------:|-----------:|--------------:|---------------:|------------:|------------:|------------------:|
| B02    | 2  | 1 | 2  | 10.67M | 10.67M | 20.3  | 2.5  | 0.066 |
| B04    | 4  | 1 | 4  | 18.38M | 18.38M | 35.1  | 5.0  | 0.116 |
| B08    | 8  | 1 | 8  | 33.81M | 33.81M | 64.5  | 10.0 | 0.216 |
| B16    | 16 | 1 | 16 | 64.67M | 64.67M | 123.4 | 20.0 | 0.417 |
| R02x2  | 2  | 2 | 4  | 10.67M | 18.38M | 20.3  | 5.0  | 0.116 |
| R02x4  | 2  | 4 | 8  | 10.67M | 33.81M | 20.3  | 10.0 | 0.216 |
| R02x8  | 2  | 8 | 16 | 10.67M | 64.67M | 20.3  | 20.0 | 0.417 |
| R01x8  | 1  | 8 | 8  | 6.81M  | 33.81M | 13.0  | 10.0 | 0.216 |
| R04x2  | 4  | 2 | 8  | 18.38M | 33.81M | 35.1  | 10.0 | 0.216 |
| R02x4s | 2  | 4 (sequence) | 8 | 10.67M | 33.81M | 20.3 | 10.0 | 0.216 |

`R02x1` would be bit-identical to B02, so B02 serves as depth 1 of the
depth ablation.

### Comparison groups

| group | arms | question |
|---|---|---|
| **C / P2**: depth ablation, param-matched | B02, R02x2, R02x4, R02x8 | At fixed stored weights, does more recurrence help, and with what returns? |
| **P4**: param-matched | B04, R04x2 | Same question at a larger stored size. |
| **F4 / F8 / F16**: compute-matched | {B04, R02x2}, {B08, R04x2, R02x4, R01x8}, {B16, R02x8} | At equal FLOPs per token, how much of the unshared model's quality does sharing keep? |
| **O**: reuse order | R02x4 vs R02x4s | Does cycle vs sequence matter? |

**Comparison that is not in the matrix: equal stored parameters AND equal
training FLOPs.** An example is B02 trained on 4× the tokens versus R02x4.
On tinyshakespeare this would mean about 320 epochs, which is meaningless.
It belongs in the matrix once a larger corpus is authorised.

### The 8K-context view (derived, bf16, `context_length` overridden to 8192)

| arm | weights MiB | KV cache MiB @8K | weights + KV MiB |
|---|---:|---:|---:|
| DML-0001 (B10) | 88.6 | 200.0 | 288.6 |
| B02 | 29.7 | 40.0 | 69.7 |
| B04 | 44.4 | 80.0 | 124.4 |
| B08 | 73.9 | 160.0 | 233.9 |
| B16 | 132.7 | 320.0 | 452.7 |
| R02x4 | 29.7 | 160.0 | 189.7 |
| R02x8 | 29.7 | 320.0 | 349.7 |
| R01x8 | 22.4 | 160.0 | 182.4 |

This is the most important derived fact in the proposal. **At 8K context
and this model size, the KV cache, not the weights, dominates memory, and
recurrence multiplies the KV cache.** R02x8 stores 4.5× fewer weight bytes
than B16 but has the same KV cache. At 8K it saves only 23% of weights+KV
compared with B16. Any capability-per-GiB claim must be made on measured
memory at the target context. At much larger widths the weights share grows
again, so this balance depends on scale and must be re-derived for any
scaled-up configuration.

The 8K rows also require the learned position table to grow from 512 to
8192 rows. That is a new confound for every arm, and the existing baseline
has never been trained beyond 512 tokens.

## 6. Required resources (estimates)

- Per-run token budget (baseline_small): 3000 steps × 8192 tokens =
  24.6M tokens.
- DML-0001 measured about 17.5k tokens/s on the RTX 2050 (from its
  metrics.jsonl), so a full 10-layer run takes about 23 minutes. Cost scales
  roughly with effective depth. The ten arms sum to effective depth 82, about
  8.2 DML-0001 runs, or **about 3.2 GPU-hours per seed and about 10
  GPU-hours for 3 seeds** on that GPU. This is a rough estimate; it ignores
  head cost, and B16/R02x8 may need gradient accumulation to fit 4 GiB
  during training.
- A larger corpus (§7) increases this in proportion to the number of tokens.
- All of this requires the operator's authorisation (GPU training, dataset
  downloads, new experiment IDs).

## 7. Methodology

1. **Dataset: fix this first.** The configured tinyshakespeare split has
   about 307k training tokens. 24.6M training tokens is about 80 epochs.
   DML-0001's train/val gap was already widening at about 13 epochs (train
   3.41 vs val 4.65 at step 500). In that regime, arms with fewer stored
   parameters can win on validation loss *through regularisation alone*,
   which would falsely support H1. **A matrix run on tinyshakespeare can
   validate the pipeline but cannot support any capability conclusion.**
   Requirement: a corpus large enough that every arm trains for at most
   about 1 epoch. Choosing and downloading it needs authorisation.
2. **Token-matched training.** Every arm sees the same tokens in the same
   order (same seed for the data generator), with the same optimiser,
   schedule and batch. Compute-matched comparisons are then also equal in
   training FLOPs.
3. **Hyperparameters.** Use one shared learning rate for all arms by
   default. If any arm diverges, run the same small LR sweep (for example
   {3e-4, 6e-4, 1.2e-3}) for *every* arm and report best-of-sweep for all.
   Record the reason in the experiment notes. Deep recurrence (A = 16) may
   want a lower LR; that is a finding, not something to tune silently.
4. **Seeds.** At least 3 seeds per arm. Treat any difference smaller than
   twice the pooled seed standard deviation as no effect.
5. **Measurements per arm:** final val loss/ppl (fixed validation batches),
   training tok/s, measured peak training VRAM, measured inference peak VRAM,
   prefill/decode tok/s and TTFT (batch 1), context sweep, checkpoint size,
   unique/applied parameters, and analytic FLOPs. GPU memory is measured on
   CUDA only; CPU runs report null.
6. **Diagnostics:** per-iteration early-exit loss, from
   `application_hidden_states` through `ln_f` and `lm_head`. This shows
   whether later iterations actually lower loss.
7. **Pre-registration.** Freeze the thresholds below (owner to confirm)
   *before* the first matrix run. Do not change them after seeing results.

## 8. Success criteria (proposed; to be frozen before any run)

> **Post-audit note (2026-11-23):** for the first controlled run (the
> three-arm pilot B02 / R02x4 / B08 at 512 context), these criteria and
> those in section 9 are **superseded by `docs/DOMOVOI_PILOT_PLAN.md`
> section 11**. The pilot uses paired per-seed differences, labelled as
> screening evidence, in place of the "twice the pooled seed SD" rule in
> section 7. Sections 8-9 still describe the later full-matrix and 8K
> frontier stage.

All on held-out loss from a ≤1-epoch corpus, mean of 3 or more seeds:

- **S1 (depth helps):** in C/P2, val loss improves from B02 → R02x2 → R02x4
  by more than noise at each step.
- **S2 (sharing keeps most of the quality):** in F8, the gap closure
  `(L_B02 − L_R02x4) / (L_B02 − L_B08) ≥ 0.5`. In words, recurrence recovers
  at least half of the loss improvement that 4× more stored weights would buy.
- **S3 (the actual project question):** at least one recurrent arm lies
  *below* the conventional frontier (B02–B04–B08–B16, interpolated) on a plot
  of val loss against **measured** peak inference memory at the target
  context, by more than seed noise.
- **S4 (usability):** that arm's measured decode speed on the reference GPU
  stays at or above the project's 8 tok/s floor when scaled to the
  configuration in question.

H1 is supported only if S3 holds. S1 and S2 without S3 mean "sharing works
but does not buy capability per GiB at this scale".

## 9. Failure criteria (any one is enough to abandon or redesign v1)

- **F1:** in F8, gap closure < 0.25.
- **F2:** no recurrent arm is below the baseline frontier at equal measured
  memory at 8K (S3 fails), in 3 or more seeds.
- **F3:** R02x4 → R02x8 gives no improvement beyond noise *and* early-exit
  diagnostics show that later iterations do not reduce loss.
- **F4:** recurrent arms diverge at the shared LR and need per-arm tuning
  that the baselines do not, so the method is fragile.

If F2 holds mainly because of the KV cache, the next direction is a KV
reduction specific to recurrence (§10), not more depth.

## 10. Follow-up directions (not implemented; interfaces only)

The `application_schedule()` hook isolates "which block runs at which
depth". All of the following can be built on it without touching the
baseline:

- **KV sharing across recursions** (reuse the first iteration's K/V in later
  iterations, as discussed in Mixture-of-Recursions). This directly attacks
  the 8K KV problem, at a possible quality cost.
- **Input re-injection / prelude–core–coda structure** (as in Geiping et
  al.): untied first/last blocks and a looped middle.
- **Per-iteration relaxation:** small per-depth LoRA deltas or per-depth
  LayerNorm/scales (Relaxed Recursive Transformers), as a middle ground
  between shared and untied.
- **Adaptive depth** (ACT / per-token routing): needs a schedule that varies
  per token, which changes cache semantics.
- **Rotary/relative positions** to remove the learned-position-table
  confound before any 8K experiment.
- **Native low-bit shared weights.** Shared weights are a small fraction of
  memory at 8K for these sizes, so this matters mainly at larger widths.

## 11. Honest summary

The implementation is in place and passes 63 dedicated CPU tests (138 total
with the pre-existing suite). No training, evaluation or GPU measurement
has been performed. **The architecture has been implemented, but no
meaningful capability improvement has yet been demonstrated.** The derived
8K memory table already shows the main risk: recurrence trades weight
memory for KV-cache memory one-for-one in effective depth.
