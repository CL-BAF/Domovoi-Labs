# Domovoi Labs — Research Specification

**Status:** Initial specification  
**Date:** 2026-09-15

## 1. Research goal

Domovoi Labs is an experimental AI research project focused on increasing useful model capability per unit of memory and compute.

The first major research question is:

> **How close can a model designed from scratch get to modern 27B-class capability while remaining fully usable inside a hard 4 GiB GPU-memory budget?**

The current long-term reference target is **Qwen3.8-27B**. The project is not attempting to compress Qwen3.8-27B itself. The goal is to investigate whether a fundamentally more efficient architecture, training process, memory system, or combination of techniques can achieve similar useful capability with dramatically lower inference requirements.

## 2. Scope

Initial work is **text-first**. Vision, audio and other multimodal capability are out of scope for the first research track.

Research may include:

- new model architectures
- recurrent or state-space components
- sparse computation
- mixture/expert routing
- parameter sharing
- adaptive computation
- alternative attention mechanisms
- external or recurrent memory
- retrieval-assisted architectures
- native low-precision representations
- distillation for training experiments
- improved tokenisation
- better data selection and curriculum design
- automated architecture search
- combinations of the above

The project should not assume that parameter count is a useful proxy for intelligence. The target is **capability under a strict resource budget**.

## 3. Hard inference constraints

A result only qualifies for the primary **4 GiB track** if all of the following are true:

1. **Peak GPU memory is no greater than 4096 MiB.**
2. **No CPU or SSD weight offloading is used** for the qualifying run.
3. The model is fully usable at batch size 1.
4. The model can generate at least **8 output tokens/second** on the reference 4 GiB GPU once the implementation is reasonably optimised.
5. The qualifying result supports at least **8K tokens of usable context**.
6. The model's normal/canonical inference representation must fit the budget.

### Native-fit rule

The primary result must not depend on taking a larger trained model and applying post-training compression purely to force it under 4 GiB.

Architectures that are **designed and trained from the beginning** around sparse, ternary, low-bit, shared, recurrent, or other memory-efficient representations are allowed. In that case, the efficient representation is part of the architecture rather than a post-hoc compression step.

A separate "compressed deployment" track may be maintained, but it must never be presented as the primary native-fit result.

## 4. Development hardware

The 4 GiB limit applies to the **final qualifying inference run**, not to research and training.

Larger GPUs, unified-memory systems, cloud training hardware, clusters, or sponsored compute may be used to:

- train candidate models
- run architecture-search experiments
- perform ablations
- run large benchmark sweeps
- distil knowledge during research
- compare against large reference models

This distinction is intentional: the research target is a low-resource **deployed model**, not low-resource training.

## 5. Capability target

Domovoi will use a benchmark composite rather than parameter count as the definition of "smart".

The long-term target is:

> **Reach at least 95% of the chance-adjusted text capability composite of Qwen3.8-27B while satisfying the 4 GiB native-fit constraints.**

This is the moonshot target, not the expected first milestone.

### Milestone ladder

| Milestone | Target |
|---|---|
| M0 | Reproducible training + evaluation pipeline |
| M1 | Beat the strongest native-fit baseline on efficiency |
| M2 | Beat the strongest native-fit baseline on capability while staying under 4 GiB |
| M3 | Reach 50% of the Qwen3.8-27B normalized capability target |
| M4 | Reach 75% |
| M5 | Reach 90% |
| Moonshot | Reach 95%+ with no major benchmark category collapsing |

A result should not qualify for M4 or above if one major category is less than 60% of the reference target. This prevents a highly specialised model from appearing generally capable because of one unusually strong benchmark.

## 6. What counts as progress

A candidate architecture is useful even if it does not immediately improve total capability.

Publishable/valuable improvements include:

- same capability with less VRAM
- same capability with fewer active parameters
- same capability with lower inference energy
- same capability with lower latency
- better long-context scaling at the same memory
- better reasoning at the same parameter count
- better capability after the same amount of training compute
- better scaling behaviour when moving from tiny to larger prototypes

Every experiment, including failures, should be logged.

## 7. Research integrity

For every reported experiment, record:

- Git commit
- model architecture/config
- parameter count
- active parameter count, if applicable
- canonical weight format
- training dataset(s) and version(s)
- token count
- optimiser and training settings
- hardware
- software versions
- random seed
- inference settings
- peak allocated VRAM
- peak reserved VRAM
- prompt/context length
- prefill speed
- decode speed
- benchmark results
- wall-clock training time
- estimated training compute where available

Benchmark test sets must not intentionally be included in training data.

If contamination is discovered or suspected, the affected result must be marked.

## 8. Initial success definition

The first genuinely interesting Domovoi result is **not** "we trained a language model."

It is:

> **A reproducible architecture or training change that gives measurably better capability-per-GiB than a strong modern baseline under the same evaluation conditions.**

That is the standard every architecture experiment should be judged against.

## 9. Current long-term reference

Qwen3.8-27B is used as the initial long-term target because it is a current, strong 27B-class model with published results across reasoning, coding, instruction following and agentic tasks.

For the initial Domovoi track, only **text capability** is compared. Qwen3.8-27B's vision/video capability is outside the first target.

## Sources

- Qwen3.8-27B model card: https://huggingface.co/Qwen/Qwen3.8-27B
- Qwen3.5-2B: https://huggingface.co/Qwen/Qwen3.5-2B
- Qwen3.5-4B: https://huggingface.co/Qwen/Qwen3.5-4B
- EXAONE 4.0 repository: https://github.com/LG-AI-EXAONE/EXAONE-4.0
