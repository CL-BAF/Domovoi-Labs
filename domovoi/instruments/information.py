"""Two-byte information scoring and paired world/seed summaries."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

TEMPERATURES = (0.25, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0, 1.1,
                1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 8.0)
LEAKAGE_TOLERANCE_BITS = 16
LN2 = math.log(2)


@dataclass(frozen=True)
class Calibration:
    world_id: str
    temperature: float
    mean_log2_by_temperature: tuple[float, ...]
    identities: frozenset[int] | None = None

    def __post_init__(self):
        _world_id(self.world_id)
        if self.identities is not None and (not isinstance(self.identities, frozenset)
                or not self.identities or any(type(x) is not int or not 0 <= x < 65536
                                             for x in self.identities)):
            raise ValueError("Invalid calibration identities.")
        if (self.temperature not in TEMPERATURES or
                len(self.mean_log2_by_temperature) != len(TEMPERATURES) or
                not all(math.isfinite(x) and x <= 0 for x in self.mean_log2_by_temperature) or
                self.temperature != TEMPERATURES[int(np.argmax(self.mean_log2_by_temperature))]):
            raise ValueError("Calibration must select the first maximum on the fixed grid.")


@dataclass(frozen=True)
class Information:
    world_id: str
    facts: int
    entropy_bits: int
    representation_bits: int
    k: float
    k_star: float
    density: float
    calibrated_density: float
    temperature: float
    invalid_rate: float
    exact_accuracy: float
    byte_ece: float
    calibrated_byte_ece: float
    possible_leakage: bool


def _world_id(world_id):
    if not isinstance(world_id, str) or not world_id.strip():
        raise ValueError("A world identifier is required.")


def _validated(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 3 or logits.shape[1:] != (2, 256) or not len(logits):
        raise ValueError("Expected logits with shape (facts, 2, 256).")
    if (labels.shape != logits.shape[:2] or labels.dtype.kind not in "iu" or
            np.any(labels < 0) or np.any(labels > 255)):
        raise ValueError("Expected exactly two uint8-valued labels per fact.")
    if not np.isfinite(logits).all():
        raise ValueError("Logits must be finite.")
    return logits, labels


def _log_probs(logits, temperature):
    # Subtract before dividing so extreme finite but equal logits remain uniform.
    with np.errstate(over="ignore"):
        relative = (logits - logits.max(axis=-1, keepdims=True)) / temperature
    return relative - np.log(np.exp(relative).sum(axis=-1, keepdims=True))


def _target_mean(log_probs, labels):
    selected = np.take_along_axis(log_probs, labels[..., None], axis=-1)[..., 0]
    score = float(selected.sum(axis=1).mean() / LN2)
    if not math.isfinite(score):
        raise ValueError("Logit dynamic range exceeded the scoring precision.")
    return score


def _ece(log_probs, labels):
    confidence = np.exp(log_probs.max(axis=-1)).ravel()
    success = (log_probs.argmax(axis=-1) == labels).ravel()
    indices = np.minimum((confidence * 10).astype(int), 9)
    counts = np.bincount(indices, minlength=10)
    confidence_sum = np.bincount(indices, weights=confidence, minlength=10)
    correct_sum = np.bincount(indices, weights=success, minlength=10)
    return float(np.abs(confidence_sum - correct_sum).sum() / counts.sum())


def _identities(values, count: int) -> frozenset[int] | None:
    if values is None:
        return None
    rows = list(values)
    if len(rows) != count or any(type(x) is not int or not 0 <= x < 65536 for x in rows):
        raise ValueError("Expected a valid identity per scored fact row.")
    return frozenset(rows)


def fit_temperature(world_id: str, logits, labels, *, identities=None) -> Calibration:
    """Select on a separate world or a nonoverlapping identity partition."""
    _world_id(world_id)
    logits, labels = _validated(logits, labels)
    scores = tuple(_target_mean(_log_probs(logits, t), labels) for t in TEMPERATURES)
    return Calibration(world_id, TEMPERATURES[int(np.argmax(scores))], scores,
                       _identities(identities, len(labels)))


def score_world(world_id: str, logits, labels, predictions, *, fact_count: int,
                representation_bits: int, calibration: Calibration,
                entropy_per_fact: int = 16, identities=None) -> Information:
    """Score a full independent uniform fact population in frozen order.

    Second-byte logits must be conditional on the first true byte. Free predictions
    are scored separately. Derived/structured facts are excluded from this endpoint.
    """
    _world_id(world_id)
    if not isinstance(calibration, Calibration):
        raise ValueError("Supply a fitted fixed-grid calibration.")
    logits, labels = _validated(logits, labels)
    if type(fact_count) is not int or fact_count != len(labels):
        raise ValueError("Fact count must describe the full scored population.")
    if type(entropy_per_fact) is not int or entropy_per_fact != 16:
        raise ValueError("K applies to independent uniform two-byte facts (16 bits each).")
    if type(representation_bits) is not int or representation_bits <= 0:
        raise ValueError("Supply positive actual whole-representation bits.")
    identity_set = _identities(identities, len(labels))
    same_world = calibration.world_id == world_id
    if same_world and (identity_set is None or calibration.identities is None or
                       not identity_set.isdisjoint(calibration.identities)):
        raise ValueError("Use a disjoint calibration world or disjoint identities.")
    predictions = np.asarray(predictions)
    if predictions.shape != labels.shape or predictions.dtype.kind not in "iu":
        raise ValueError("Free predictions must have two integer slots; use -1 for invalid.")
    invalid = ((predictions < 0) | (predictions > 255)).any(axis=1)
    primary = _log_probs(logits, 1.0)
    calibrated = _log_probs(logits, calibration.temperature)
    k = fact_count * (16 + _target_mean(primary, labels))
    k_star = fact_count * (16 + _target_mean(calibrated, labels))
    # A selected temperature in the scored world can encode up to log2(16) extra bits.
    allowance = 4 if same_world else 0
    threshold = representation_bits + LEAKAGE_TOLERANCE_BITS
    return Information(world_id, fact_count, fact_count * 16, representation_bits,
                       k, k_star, k / representation_bits, k_star / representation_bits,
                       calibration.temperature, float(invalid.mean()),
                       float(((predictions == labels).all(axis=1) & ~invalid).mean()),
                       _ece(primary, labels), _ece(calibrated, labels),
                       k > threshold or k_star > threshold + allowance)


def permuted_labels(labels, seed: int):
    """Preserve the label multiset while breaking the query pairing."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.shape[1] != 2 or len(labels) < 2:
        raise ValueError("Permutation requires at least two two-byte labels.")
    permutation = np.random.default_rng(seed).permutation(len(labels))
    order = np.empty(len(labels), dtype=np.intp)
    order[permutation] = np.roll(permutation, 1)
    return labels[order]


def paired_uncertainty(reference: dict[int, Information], candidate: dict[int, Information],
                       *, resamples: int = 2000, seed: int = 0) -> dict:
    """Describe paired seed/world effects. Three seeds are an instrument check."""
    seeds = sorted(reference)
    if (len(seeds) < 2 or set(seeds) != set(candidate) or
            type(resamples) is not int or resamples < 2):
        raise ValueError("Require identical seed sets and at least two independent pairs.")
    worlds = [reference[s].world_id for s in seeds]
    if len(set(worlds)) != len(seeds) or any(candidate[s].world_id != w for s, w in zip(seeds, worlds)):
        raise ValueError("Pair by world within seed and use independent worlds between seeds.")
    if any(reference[s].facts != candidate[s].facts for s in seeds):
        raise ValueError("Paired arms must score the same fact population.")
    if any(r.possible_leakage for runs in (reference, candidate) for r in runs.values()):
        raise ValueError("Resolve flagged leakage before reporting paired effects.")
    deltas = np.array([candidate[s].calibrated_density - reference[s].calibrated_density
                       for s in seeds])
    sample_indices = np.random.default_rng(seed).integers(len(seeds), size=(resamples, len(seeds)))
    low, high = np.percentile(deltas[sample_indices].mean(axis=1), [2.5, 97.5])
    return {"seeds": seeds, "world_ids": worlds, "differences": deltas.tolist(),
            "mean": float(deltas.mean()), "sample_sd": float(deltas.std(ddof=1)),
            "paired_bootstrap_95": [float(low), float(high)]}
