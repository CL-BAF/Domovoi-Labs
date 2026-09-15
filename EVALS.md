# Domovoi Labs — Evaluation Specification

**Status:** Initial evaluation design  
**Date:** 2026-09-15

Domovoi needs two evaluation layers:

1. **Fast research evals** for hundreds or thousands of small architecture experiments.
2. **Milestone evals** for results that are worth comparing publicly.

Running GPQA and LiveCodeBench on every 20M-parameter experiment would waste time. Using only perplexity would also be useless because it does not tell us whether the model became more capable.

---

# 1. Fast research evaluation

Run this on every serious architecture candidate.

## Quality

- validation cross-entropy / perplexity on a fixed held-out corpus
- HellaSwag
- ARC-Challenge
- PIQA
- WinoGrande
- a fixed small GSM8K subset
- a fixed instruction-following mini-suite

Use the same dataset versions and fixed sample IDs for every run.

## Efficiency

Measure:

- total parameters
- active parameters per token
- checkpoint size
- peak GPU memory allocated
- peak GPU memory reserved
- tokens/second during prefill
- tokens/second during decode
- time to first token
- context-length memory growth
- training tokens/second
- training peak VRAM
- estimated energy per generated token where reliable measurement is available

## Scaling test

Promising architectures should be trained at several small sizes using the same token budget ratios, for example:

- ~20M
- ~50M
- ~100M
- ~300M

The point is to detect whether an improvement survives scaling instead of winning only at one tiny configuration.

---

# 2. Full milestone evaluation

Run this when an architecture clearly beats the current research baseline.

## General knowledge and reasoning

### MMLU-Pro

Why:
- difficult multi-domain benchmark
- more reasoning-focused than original MMLU
- 12,000+ questions across 14 domains
- less sensitive to prompt variation than original MMLU according to its authors

Metric: accuracy.

Official repository:
https://github.com/TIGER-AI-Lab/MMLU-Pro

### GPQA Diamond

Why:
- difficult graduate-level science reasoning
- used in the official Qwen3.8-27B results

Metric: accuracy.

Official repository:
https://github.com/idavidrein/gpqa

---

## Mathematical reasoning

Use two levels:

### GSM8K

Use during early development because weak models can still produce a useful performance gradient.

### MATH-500

Use for milestone models when GSM8K begins to saturate.

Metric: exact-answer accuracy with a fixed answer extractor.

---

## Coding

### LiveCodeBench

Why:
- execution-based coding evaluation
- designed to reduce contamination
- used in Qwen3.8's published evaluation

Metric: pass@1 for the fixed benchmark release used by the project.

Official repository:
https://github.com/LiveCodeBench/LiveCodeBench

For very small prototypes, do not treat a near-zero LiveCodeBench score as a reason to discard an architecture if the lower-cost research evals are improving.

---

## Instruction following

### IFBench

Why:
- verifiable instruction constraints
- harder out-of-distribution instruction-following cases
- official evaluation code is available

Metric: official IFBench score.

Official repository:
https://github.com/allenai/IFBench

---

## Long-context behaviour

### RULER

Test at:

- 2K
- 4K
- 8K

Later add:

- 16K
- 32K

The primary 4 GiB track only requires 8K initially.

Record both accuracy and VRAM growth. A model that handles 8K correctly but exceeds 4 GiB does not qualify.

---

# 3. Optional later evaluations

These are useful once the core model is capable enough:

- HLE — frontier multidisciplinary reasoning
- BFCL — tool/function calling
- SWE-bench / SWE-bench Pro — software engineering
- Terminal Bench — agentic terminal work
- multilingual benchmarks
- safety/refusal evaluations
- factuality/hallucination evaluations
- vision benchmarks if Domovoi becomes multimodal

Do not add expensive benchmarks just because they are popular. Each benchmark must answer a research question.

---

# 4. Domovoi Capability Index

A single number is useful for tracking progress, but raw benchmark percentages cannot simply be averaged because each benchmark has different difficulty and chance levels.

For each core benchmark:

**Normalized score = (Domovoi score - chance floor) / (Qwen3.8-27B score - chance floor)**

Clamp only for dashboard display, not raw research analysis.

For benchmarks without a meaningful random-chance floor, use 0.

The **Domovoi Capability Index (DCI)** is the weighted mean of normalized category scores.

Initial category weights:

| Category | Weight |
|---|---:|
| General knowledge/reasoning | 25% |
| Scientific reasoning | 20% |
| Mathematics | 15% |
| Coding | 20% |
| Instruction following | 10% |
| Long context | 10% |

These weights should be frozen before official milestone comparisons.

Qwen3.8-27B is defined as **1.00 / 100%** on the target index once we have run or reconstructed all required reference evaluations.

---

# 5. Efficiency metrics

Capability alone is not the point of Domovoi.

Report at minimum:

### Capability per GiB

`DCI / peak inference VRAM in GiB`

### Capability per active billion parameters

`DCI / active parameters (billions)`

### Decode efficiency

- output tokens/second
- joules/output token where measurable

### Context efficiency

Plot:

`peak VRAM vs context length`

at fixed prompt/output settings.

A model can therefore be considered a useful result even before it approaches Qwen3.8-27B if it materially improves capability-per-GiB.

---

# 6. Standard inference conditions

Unless a benchmark requires otherwise:

- batch size: 1
- no retrieval
- no external tools
- no CPU/SSD weight offload for native-fit results
- fixed prompt template per benchmark
- deterministic decoding where the benchmark expects it
- reasoning budget fixed across directly compared models
- same maximum output tokens
- same benchmark version/sample set

If a model has an internal recurrent state or architecture-specific memory, that memory counts toward the 4 GiB device budget if it resides on GPU.

---

# 7. Repetition and uncertainty

For deterministic multiple-choice benchmarks:
- one fixed run is acceptable after the harness is validated

For sampling/reasoning evaluations:
- use at least 3 seeds for milestone results where practical
- report mean and standard deviation

For tiny experiments:
- repeat the best candidates with multiple training seeds before concluding that an architecture change is real

---

# 8. Contamination rules

- benchmark test examples must not intentionally enter training data
- maintain a list of all benchmark datasets and hashes
- run exact/near-duplicate checks against controlled training corpora where possible
- mark any suspected contamination
- never silently replace a benchmark version after results have been published

---

# 9. First scoreboard

The first dashboard should contain:

| Model | Peak VRAM | Params | MMLU-Pro | GPQA | GSM8K/MATH | LiveCodeBench | IFBench | 8K RULER | DCI | tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXAONE 4.0 1.2B | TBD | 1.2B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3.5-2B | TBD | 2B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3.5-4B <=4GiB compressed | TBD | 4B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3.8-27B target | >4GiB | 27B | TBD | 89.2* | TBD | 90.3* | 79.5* | TBD | 1.00 | TBD |
| Domovoi candidate | <=4GiB | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

`*` = published reference score, not yet reproduced by Domovoi Labs.

The scoreboard should clearly distinguish **published external scores** from **Domovoi-reproduced scores**.

## Sources

- MMLU-Pro: https://github.com/TIGER-AI-Lab/MMLU-Pro
- GPQA: https://github.com/idavidrein/gpqa
- LiveCodeBench: https://github.com/LiveCodeBench/LiveCodeBench
- IFBench: https://github.com/allenai/IFBench
- Qwen3.8-27B: https://huggingface.co/Qwen/Qwen3.8-27B
- EXAONE 4.0: https://github.com/LG-AI-EXAONE/EXAONE-4.0
