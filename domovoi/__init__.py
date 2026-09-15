"""Domovoi Labs research pipeline.

Package layout:
    domovoi.config        - typed, validated experiment configuration
    domovoi.models        - model architectures (baseline Transformer first)
    domovoi.datasets      - data download / split / tokenization / batching
    domovoi.training      - trainer, optimizer/scheduler, checkpointing
    domovoi.inference     - checkpoint loading and text generation
    domovoi.evals         - fast research evaluation suite
    domovoi.profiling     - parameter / VRAM / speed measurement
    domovoi.experiments   - experiment IDs, metadata, reports
    domovoi.utils         - seeding, environment/version recording
"""

__version__ = "0.1.0"