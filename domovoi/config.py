"""Experiment configuration.

Every experiment-defining value lives in a YAML config file; nothing
experiment-specific is buried in Python code.  This module defines the typed
config schema, validates it, and provides save/merge helpers.

Config schema (see configs/*.yaml):

    experiment:  name, notes, status
    model:       architecture, vocab_size, context_length, d_model,
                 n_layers, n_heads, ffn_dim, dropout, tie_embeddings,
                 recurrence_depth, recurrence_order (both optional)
    tokenizer:   type, path, version
    dataset:     name, source_url, version, train_fraction,
                 processed_dir
    training:    steps, batch_size, gradient_accumulation, context_length,
                 learning_rate, min_lr_ratio, weight_decay, betas,
                 grad_clip, warmup_steps, precision, seed,
                 eval_interval, eval_batches, checkpoint_interval,
                 log_interval
    output:      experiments_dir, experiment_id
"""

from __future__ import annotations

import copy
import dataclasses
import math
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

PRECISIONS = ("bf16", "fp16", "fp32")
STATUSES = ("RUNNING", "FAILED", "INVALID", "BASELINE", "PROMISING",
            "REJECTED", "PROMOTED", "COMPLETED", "INTERRUPTED")


class ConfigError(ValueError):
    """Raised when a config file is missing keys, has bad values, or is
    internally inconsistent.  The message is intended to be shown directly
    to the user."""


@dataclass
class ExperimentSection:
    name: str
    notes: str = ""
    status: str = "BASELINE"


@dataclass
class ModelSection:
    architecture: str = "baseline_transformer"
    vocab_size: int = 4096
    context_length: int = 512
    d_model: int = 640
    n_layers: int = 10
    n_heads: int = 10
    ffn_dim: int = 1728
    dropout: float = 0.0
    tie_embeddings: bool = True
    # Shared-core recurrence (domovoi_recurrent_v1).  The defaults mean "no
    # recurrence", so configs and checkpoints written before these keys
    # existed load unchanged.  For domovoi_recurrent_v1, n_layers counts the
    # *distinct* blocks and each is applied recurrence_depth times.
    recurrence_depth: int = 1
    recurrence_order: str = "cycle"


@dataclass
class TokenizerSection:
    type: str = "byte_bpe"
    path: str = ""
    version: str = "byte-bpe-v1"


@dataclass
class DatasetSection:
    name: str
    source_url: str = "synthetic"
    version: str = "unversioned"
    train_fraction: float = 0.9
    processed_dir: str = ""


@dataclass
class TrainingSection:
    steps: int = 1000
    batch_size: int = 16
    gradient_accumulation: int = 1
    context_length: int = 512
    learning_rate: float = 6.0e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    precision: str = "bf16"
    seed: int = 1234
    eval_interval: int = 250
    eval_batches: int = 25
    checkpoint_interval: int = 500
    log_interval: int = 10
    sampler: str = "random_offsets"
    pilot_stop_checks: bool = False
    pilot_calibrated_seconds: Optional[float] = None
    extra_checkpoint_steps: tuple = ()
    synchronize_cuda_timing: bool = False
    deterministic_cuda: bool = False
    verify_checkpoint_writes: bool = False


@dataclass
class OutputSection:
    experiments_dir: str = "experiments"
    experiment_id: Optional[str] = None


@dataclass
class Config:
    experiment: ExperimentSection
    model: ModelSection
    tokenizer: TokenizerSection
    dataset: DatasetSection
    training: TrainingSection
    output: OutputSection
    source_path: Optional[str] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def validate(self) -> None:
        """Raise ConfigError with a precise message if anything is wrong."""
        # imported here, not at module top: the registry pulls in torch, and
        # the single source of truth for valid names is the registry itself
        from .models import ARCHITECTURES, RECURRENCE_ORDERS

        m, t, d = self.model, self.training, self.dataset

        if m.architecture not in ARCHITECTURES:
            raise ConfigError(
                f"model.architecture must be one of {sorted(ARCHITECTURES)}; "
                f"got {m.architecture!r}.")
        if isinstance(m.recurrence_depth, bool) or not isinstance(m.recurrence_depth, int) \
                or m.recurrence_depth < 1:
            raise ConfigError(
                f"model.recurrence_depth must be an integer >= 1; got {m.recurrence_depth!r}.")
        if m.recurrence_order not in RECURRENCE_ORDERS:
            raise ConfigError(
                f"model.recurrence_order must be one of {RECURRENCE_ORDERS}; "
                f"got {m.recurrence_order!r}.")
        if m.architecture == "baseline_transformer" and (
                m.recurrence_depth != 1 or m.recurrence_order != "cycle"):
            # the baseline would silently ignore these; refuse instead
            raise ConfigError(
                "model.recurrence_depth / recurrence_order only apply to "
                "'domovoi_recurrent_v1'; leave them at their defaults (1, "
                "'cycle') for 'baseline_transformer'.")
        if m.n_heads <= 0 or m.d_model % m.n_heads != 0:
            raise ConfigError(
                f"model.d_model ({m.d_model}) must be divisible by "
                f"model.n_heads ({m.n_heads}).")
        if m.d_model <= 0 or m.n_layers <= 0 or m.ffn_dim <= 0:
            raise ConfigError("model dims must be positive.")
        if m.vocab_size < 256:
            raise ConfigError(
                f"model.vocab_size must be >= 256 (byte alphabet); got {m.vocab_size}.")
        if m.dropout < 0.0 or m.dropout >= 1.0:
            raise ConfigError("model.dropout must be in [0, 1).")
        if not 0.0 < d.train_fraction < 1.0:
            raise ConfigError("dataset.train_fraction must be in (0, 1).")
        if t.steps <= 0:
            raise ConfigError("training.steps must be > 0.")
        checkpoints = t.extra_checkpoint_steps
        if (not isinstance(checkpoints, (list, tuple))
                or any(type(step) is not int or not 1 <= step <= t.steps for step in checkpoints)):
            raise ConfigError("training.extra_checkpoint_steps must contain steps within the run.")
        if t.batch_size <= 0 or t.gradient_accumulation <= 0:
            raise ConfigError("training.batch_size and gradient_accumulation must be > 0.")
        if t.context_length <= 0 or t.context_length > m.context_length:
            raise ConfigError(
                f"training.context_length ({t.context_length}) must be positive and "
                f"<= model.context_length ({m.context_length}).")
        if t.learning_rate <= 0.0:
            raise ConfigError("training.learning_rate must be > 0.")
        if t.precision not in PRECISIONS:
            raise ConfigError(
                f"training.precision must be one of {PRECISIONS}; got {t.precision!r}.")
        if t.sampler not in ("random_offsets", "disjoint_segments"):
            raise ConfigError("training.sampler must be random_offsets or disjoint_segments.")
        if not isinstance(t.pilot_stop_checks, bool):
            raise ConfigError("training.pilot_stop_checks must be a boolean.")
        if not isinstance(t.synchronize_cuda_timing, bool):
            raise ConfigError("training.synchronize_cuda_timing must be a boolean.")
        if not isinstance(t.deterministic_cuda, bool) or not isinstance(t.verify_checkpoint_writes, bool):
            raise ConfigError("Deterministic CUDA and checkpoint verification flags must be booleans.")
        seconds = t.pilot_calibrated_seconds
        if seconds is not None and (
            not t.pilot_stop_checks or type(seconds) not in (int, float)
            or not math.isfinite(seconds) or seconds <= 0
        ):
            raise ConfigError("training.pilot_calibrated_seconds requires enabled pilot stop "
                              "checks and a finite positive duration.")
        if not 0.0 <= t.min_lr_ratio <= 1.0:
            raise ConfigError("training.min_lr_ratio must be in [0, 1].")
        if t.warmup_steps < 0 or t.warmup_steps > t.steps:
            raise ConfigError("training.warmup_steps must be in [0, training.steps].")
        if len(t.betas) != 2 or not all(0.0 < b < 1.0 for b in t.betas):
            raise ConfigError("training.betas must be two floats in (0, 1).")
        if t.grad_clip <= 0.0:
            raise ConfigError("training.grad_clip must be > 0.")
        if self.experiment.status not in STATUSES:
            raise ConfigError(
                f"experiment.status must be one of {STATUSES}; got "
                f"{self.experiment.status!r}.")

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        def clean(obj: Any) -> Any:
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: clean(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, tuple):
                return list(obj)
            return obj
        return clean(self)

    def save_copy(self, path: str | pathlib.Path) -> None:
        """Write the config into the experiment directory so the run can be
        reproduced exactly: the original YAML verbatim when the config was
        loaded from a file, otherwise a dump of the parsed values."""
        if self.source_path is not None:
            text = pathlib.Path(self.source_path).read_text(encoding="utf-8")
        else:
            text = yaml.safe_dump(self.to_dict(), sort_keys=False)
        pathlib.Path(path).write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------
def _section(name: str, raw: dict, cls: type, required: list[str]) -> Any:
    if not isinstance(raw, dict):
        raise ConfigError(f"Config section '{name}' must be a mapping.")
    missing = [k for k in required if k not in raw]
    if missing:
        raise ConfigError(f"Config section '{name}' is missing keys: {missing}.")
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = [k for k in raw if k not in known]
    if unknown:
        raise ConfigError(f"Config section '{name}' has unknown keys: {unknown}.")
    return cls(**raw)


def load_config(path: str | pathlib.Path) -> Config:
    """Load, parse and validate a YAML experiment config."""
    path = pathlib.Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"Could not parse YAML config {path}: {e}") from e
    if not isinstance(raw, dict):
        raise ConfigError(f"Config {path} is empty or not a mapping.")

    for section in ("experiment", "model", "dataset", "training", "output"):
        if section not in raw:
            raise ConfigError(f"Config {path} is missing section '{section}'.")
    raw.setdefault("tokenizer", {})

    cfg = Config(
        experiment=_section("experiment", raw["experiment"], ExperimentSection, ["name"]),
        model=_section("model", raw["model"], ModelSection, ["architecture"]),
        tokenizer=_section("tokenizer", raw["tokenizer"], TokenizerSection, []),
        dataset=_section("dataset", raw["dataset"], DatasetSection, ["name"]),
        training=_section("training", raw["training"], TrainingSection, []),
        output=_section("output", raw["output"], OutputSection, []),
        source_path=str(path),
    )
    # tuples survive YAML lists; betas needs to be a 2-tuple
    cfg.training.betas = tuple(cfg.training.betas)
    cfg.validate()
    return cfg


def config_from_dict(raw: dict) -> Config:
    """Build a Config from an already-parsed dict (e.g. stored in
    metadata.json).  Reuses load_config's validation by round-tripping
    through the same section builder."""
    for section in ("experiment", "model", "dataset", "training", "output"):
        if section not in raw:
            raise ConfigError(f"Config is missing section '{section}'.")
    cfg = Config(
        experiment=_section("experiment", raw["experiment"], ExperimentSection, ["name"]),
        model=_section("model", raw["model"], ModelSection, ["architecture"]),
        tokenizer=_section("tokenizer", raw.get("tokenizer", {}), TokenizerSection, []),
        dataset=_section("dataset", raw["dataset"], DatasetSection, ["name"]),
        training=_section("training", raw["training"], TrainingSection, []),
        output=_section("output", raw["output"], OutputSection, []),
    )
    cfg.training.betas = tuple(cfg.training.betas)
    cfg.validate()
    return cfg


def clone_config(cfg: Config, **overrides: dict) -> Config:
    """Deep-copy a config and apply `overrides` like {'model': {'n_layers': 4}}.
    Used for scaling studies and ablations."""
    raw = copy.deepcopy(cfg.to_dict())
    for section, values in overrides.items():
        if section not in raw:
            raise ConfigError(f"Unknown config section {section!r}")
        raw[section].update(values)
    return config_from_dict(raw)
