"""Small hand-checkable fixtures. These tests run no training or benchmarks."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

from domovoi.instruments.bytes import Storage, representation_bytes
from domovoi.instruments.codecs import (
    Format, PackedMatrix, e8_doubled_points, hadamard8, linear_reference, pack,
    residual_books, residual_indices, unpack,
)
from domovoi.instruments.information import (
    LEAKAGE_TOLERANCE_BITS, TEMPERATURES, fit_temperature, paired_uncertainty,
    permuted_labels, score_world,
)
from domovoi.instruments.world import HELDOUT, UNTRAINED, World


FIXTURE_KEY = bytes(range(32))


def test_prf_vectors_and_prompt_separation():
    world = World(FIXTURE_KEY, 1234, 20)
    repeat = World(FIXTURE_KEY, 1234, 20)
    other = World(FIXTURE_KEY, 1235, 20)
    first = world.example(3, 2)
    assert world.id == "036617fb02990c5826671024f03623ea2f00e3780a74f3055a0c9513a747ad56"
    assert first.label == (92, 182)
    assert first == repeat.example(3, 2)
    assert world.id != other.id
    assert [e.label for e in world.random_facts()] != [e.label for e in other.random_facts()]
    assert world.fact_count == len(list(world.random_facts())) == 80
    assert len(first.training_tokens) == 7
    heldout = world.example(3, 2, "heldout")
    assert heldout.query.tokens[1] == HELDOUT
    assert first.query.tokens != heldout.query.tokens and first.label == heldout.label
    untrained = world.example(3, 2, "untrained")
    assert untrained.query.tokens[1] == UNTRAINED and untrained.label != first.label
    assert set(vars(heldout.query)) == {"identity", "attribute", "tokens"}
    with pytest.raises(TypeError, match="must not be serialized"):
        pickle.dumps(world)


def test_structured_split_and_two_hop_labels():
    world = World(FIXTURE_KEY, 3, 21)
    with pytest.raises(ValueError, match="other split"):
        world.example(5, 1, "structured-train")
    heldout = world.example(5, 1, "structured-heldout")
    assert heldout == World(FIXTURE_KEY, 3, 21).example(5, 1, "structured-heldout")
    for row in range(world.identities):
        link = world.example(row, 0, "link")
        target = int.from_bytes(bytes(link.label), "big")
        assert 0 <= target < world.identities and target != row
        probe = world.example(row, 2, "two-hop")
        assert probe.label == world.example(target, 2).label
        assert probe.query.tokens != link.query.tokens


@pytest.mark.parametrize("values", [(b"x", 1, 2, 4), (FIXTURE_KEY, -1, 2, 4),
                                     (FIXTURE_KEY, 1, 65537, 4), (FIXTURE_KEY, 1, 2, True)])
def test_bad_world_schema(values):
    with pytest.raises(ValueError):
        World(*values)


def fixture_labels(world):
    return np.asarray([e.label for e in world.random_facts("heldout")], dtype=np.uint8)


def oracle(labels):
    values = np.zeros((*labels.shape, 256))
    np.put_along_axis(values, labels[..., None], 45, axis=-1)
    return values


def score(world, values, labels, predictions=None, **kw):
    settings = {"fact_count": world.fact_count, "representation_bits": 1024}
    settings.update(kw)
    return score_world(world.id, values, labels, labels if predictions is None else predictions,
                       calibration=fit_temperature("different calibration world", values, labels),
                       **settings)


def test_uniform_oracle_and_excessive_density():
    world = World(FIXTURE_KEY, 1, 8, 1)
    labels = fixture_labels(world)
    result = score(world, np.zeros((*labels.shape, 256)), labels)
    assert result.k == pytest.approx(0) and result.k_star == pytest.approx(0)
    assert result.temperature == TEMPERATURES[0]  # tie rule is fixed
    result = score(world, oracle(labels), labels, representation_bits=16)
    assert result.k == pytest.approx(world.fact_count * 16)
    assert result.exact_accuracy == 1 and result.possible_leakage
    no_alarm = score(world, oracle(labels), labels,
                     representation_bits=world.fact_count * 16 - LEAKAGE_TOLERANCE_BITS)
    assert not no_alarm.possible_leakage


def test_permuted_untrained_overconfidence_and_unrestricted_predictions():
    world = World(FIXTURE_KEY, 20, 8)
    labels = fixture_labels(world)
    logits = oracle(labels)
    permuted = permuted_labels(labels, 9)
    assert sorted(map(tuple, labels)) == sorted(map(tuple, permuted))
    predictions = labels.astype(int)
    predictions[1, 0] = -1
    null = score(world, logits, permuted, predictions)
    assert null.k < 0 and null.k_star > null.k and null.temperature == max(TEMPERATURES)
    assert null.invalid_rate == 1 / len(labels)
    unseen = np.asarray([e.label for e in world.random_facts("untrained")])
    assert score(world, logits, unseen).k < 0


def test_estimator_rejects_wrong_entropy_count_shape_and_calibration_world():
    world = World(FIXTURE_KEY, 1, 8)
    labels = fixture_labels(world)
    values = oracle(labels)
    with pytest.raises(ValueError, match="16 bits"):
        score(world, values, labels, entropy_per_fact=15)
    with pytest.raises(ValueError, match="full scored"):
        score(world, values, labels, fact_count=world.fact_count + 1)
    with pytest.raises(ValueError, match="disjoint"):
        score_world(world.id, values, labels, labels, fact_count=world.fact_count,
                    representation_bits=1024, calibration=fit_temperature(world.id, values, labels))
    with pytest.raises(ValueError, match="finite"):
        score(world, values + np.nan, labels)


def test_same_world_calibration_uses_disjoint_identity_partitions():
    world = World(FIXTURE_KEY, 2, 32)
    calibration_ids, evaluation_ids = world.instrument_partitions()
    assert calibration_ids == (0, 16) and len(evaluation_ids) == 30
    calibration_rows = list(world.random_facts(identities=calibration_ids))
    labels = np.array([r.label for r in calibration_rows])
    cal = fit_temperature(world.id, oracle(labels), labels,
                          identities=[r.query.identity for r in calibration_rows])
    rows = list(world.random_facts(identities=evaluation_ids))
    answers = np.array([r.label for r in rows])
    result = score_world(world.id, oracle(answers), answers, answers, fact_count=len(rows),
                         representation_bits=4096, calibration=cal,
                         identities=[r.query.identity for r in rows])
    assert result.entropy_bits == 30 * 4 * 16 and result.exact_accuracy == 1


def test_seed_level_uncertainty_preserves_world_pairing():
    worlds = [World(FIXTURE_KEY, seed, 4) for seed in (1, 2, 3)]
    runs = {w.seed: score(w, oracle(fixture_labels(w)), fixture_labels(w)) for w in worlds}
    candidates = {i: replace(run, calibrated_density=run.calibrated_density + 0.01 * i)
                  for i, run in runs.items()}
    report = paired_uncertainty(runs, candidates, resamples=200, seed=7)
    assert report == paired_uncertainty(runs, candidates, resamples=200, seed=7)
    assert report["mean"] == pytest.approx(.02)
    assert report["sample_sd"] == pytest.approx(.01)
    candidates[1] = candidates[2]
    with pytest.raises(ValueError, match="Pair by world"):
        paired_uncertainty(runs, candidates)


@pytest.mark.parametrize("fmt", list(Format))
def test_canonical_byte_roundtrip_and_hand_expanded_logits(fmt):
    weights = np.arange(-7, 8, dtype=np.float32).reshape(3, 5) / 7
    options = {"learned_books": residual_books()} if fmt == Format.LEARNED else {}
    packed = pack(weights, fmt, **options)
    rebuilt = PackedMatrix.from_bytes(packed.to_bytes())
    assert rebuilt == packed and rebuilt.to_bytes() == pack(weights, fmt, **options).to_bytes()
    decoded = unpack(rebuilt)
    assert np.isfinite(decoded).all()
    np.testing.assert_array_equal(decoded, unpack(packed))
    # Different summation path, against the decoded reference, not full-precision weights.
    vector = np.array([1, -1, 0.5, 0.75, -0.25])
    logits = [sum(float(x) * float(y) for x, y in zip(row, vector)) for row in decoded]
    np.testing.assert_allclose(linear_reference(vector, rebuilt), logits, rtol=0, atol=1e-12)
    assert sum(packed.byte_counts().values()) == len(packed.to_bytes())
    with pytest.raises(ValueError):
        PackedMatrix.from_bytes(packed.to_bytes()[:-1])


def test_exact_scalar_and_base_three_payloads():
    bf16 = pack(np.array([[1.0, 1 + 1 / 256, 1 + 3 / 256]], dtype=np.float32), Format.BF16)
    assert bf16.payload.hex() == "803f803f823f"  # ties to an even retained mantissa
    assert pack([[-3, -1, 1, 3]], Format.SCALAR2).payload == b"\xe4"
    assert pack([[-7, 0, 7]], Format.INT4).payload == b"\x70\x0e"
    ternary = pack([[-1, 0, 1, 0, 1, 1]], Format.TERNARY)
    assert ternary.payload == bytes((210, 2))
    assert unpack(ternary).tolist() == [[-1, 0, 1, 0, 1, 1]]
    with pytest.raises(ValueError, match="less than 243"):
        replace(ternary, payload=bytes((243, 2)))
    with pytest.raises(ValueError, match="Unused trits"):
        replace(ternary, payload=bytes((210, 5)))


def test_shared_basis_is_rotated_scalar_and_distinct_bases_are_nonseparable():
    rng = np.random.default_rng(42)
    h = hadamard8()
    signals = rng.normal(size=(12, 8))
    shared = residual_books(h, h)
    pair = residual_indices(signals, shared)
    reconstruction = shared[0, pair[:, 0]] + shared[1, pair[:, 1]]
    levels = np.array([-1.5, -.5, .5, 1.5])
    scalar = levels[np.abs((signals @ h)[..., None] - levels).argmin(axis=-1)] @ h.T
    np.testing.assert_allclose(reconstruction, scalar, atol=1e-12, rtol=0)
    different = residual_books()
    pair = residual_indices(signals, different)
    reconstruction = different[0, pair[:, 0]] + different[1, pair[:, 1]]
    assert not np.allclose(reconstruction, scalar)
    directions = np.column_stack((np.eye(8), h))
    dots = abs(directions.T @ directions)
    assert np.count_nonzero(np.triu(dots > 1 - 1e-12, k=1)) == 0  # 16 directions, not 8


def test_e8_exact_cosets_symmetry_and_reserved_index():
    points = e8_doubled_points()
    assert points.shape == (65535, 8) and len(np.unique(points, axis=0)) == len(points)
    assert np.all(points.sum(axis=1) % 4 == 0)
    assert np.all((points % 2) == points[:, :1] % 2)
    np.testing.assert_array_equal(points[1::2], -points[2::2])
    assert np.min((points[1:].astype(int) ** 2).sum(axis=1)) == 8
    with pytest.raises(ValueError, match="reserved"):
        PackedMatrix(Format.E8, (1, 8), b"\xff\xff", b"\x00\x3c")


def test_retained_alignment_scale_and_duplicate_bytes():
    m = pack([[1, 0, -1, 0, 1]], Format.TERNARY)
    evidence = representation_bytes({"mlp": m}, [
        Storage("tied", "embedding", 65536), Storage("qkv_o", "attention", 131072),
        Storage("norms", "norm", 1152), Storage("expanded", "duplicate", 20),
    ])
    assert len(m.payload) == 1 and evidence["coded_matrix_bytes"] == len(m.to_bytes())
    assert evidence["representation_bytes"] == len(m.to_bytes()) + 197760 + 20
    with pytest.raises(ValueError, match="unique"):
        representation_bytes({"mlp": m}, [Storage("mlp", "duplicate", 20)])
