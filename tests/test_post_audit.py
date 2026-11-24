"""Regression tests for the Codex audit findings fixed after the initial
domovoi_recurrent_v1 implementation (see POST_AUDIT_FIXES.txt).

  DML-A01  checkpoint / resume must refuse a different computation graph
  DML-A02  malformed KV caches must fail explicitly
  DML-A03  profiling: exact lengths, real dtype, labelled measurements
  DML-A04  experiment planner must not inherit IDs or overwrite plans
  DML-A05  untied checkpoints must not silently keep a random lm_head
  DML-A08  recurrence_schedule must reject bools

Every test reproduces the audited failure first where that is possible:
the pre-fix code accepted each rejected case below (see
post_audit/repro_before.txt).  CPU only, synthetic data, seconds to run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest
import torch
import yaml
from safetensors.torch import load_file, save_file

from domovoi.config import clone_config, config_from_dict
from domovoi.inference import generate
from domovoi.models import (ApplicationCache, CacheValidationError, build_model,
                            config_signature, model_signature)
from domovoi.models.domovoi_recurrent import recurrence_schedule
from domovoi.profiling import architecture_accounting
from domovoi.profiling import profiler as profiler_mod
from domovoi.profiling.profiler import (profile_context_sweep, profile_inference,
                                        profile_model)
from domovoi.training import (CheckpointError, Trainer, build_optimizer,
                              build_scheduler, load_checkpoint, save_checkpoint)
from domovoi.training.checkpoint import build_model_from_checkpoint

ROOT = pathlib.Path(__file__).resolve().parent.parent


class _ByteTok:
    vocab_size = 256

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


class _WordTok:
    """Codex's probe tokenizer: one token per whitespace word."""

    def encode(self, s):
        return [1 for _ in s.split()]

    def decode(self, t):
        return " ".join("a" for _ in t)


def _rec(cfg, **model):
    base = {"architecture": "domovoi_recurrent_v1"}
    base.update(model)
    return clone_config(cfg, model=base)


def _save(cfg, path, seed=0):
    torch.manual_seed(seed)
    model = build_model(cfg.model).eval()
    opt = build_optimizer(model, 1e-3, 0.1, (0.9, 0.95))
    sched = build_scheduler(opt, 1, 10, 0.1)
    save_checkpoint(path, model, opt, sched, step=1, tokens_seen=8, config=cfg)
    return model


def _tree_hashes(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# =====================================================================
# DML-A01  checkpoint architecture / schedule compatibility
# =====================================================================

def test_checkpoint_records_explicit_schedule(tiny_cfg, tmp_path):
    cfg = _rec(tiny_cfg, recurrence_depth=2, recurrence_order="sequence")
    _save(cfg, tmp_path / "ck")
    meta = json.loads((tmp_path / "ck" / "checkpoint.json").read_text(encoding="utf-8"))
    sig = meta["model_signature"]
    assert sig["architecture"] == "domovoi_recurrent_v1"
    assert sig["recurrence_depth"] == 2 and sig["recurrence_order"] == "sequence"
    assert sig["application_schedule"] == [0, 0, 1, 1]
    assert sig["effective_depth"] == 4


@pytest.mark.parametrize("saved,loaded", [
    # identical tensor shapes, different graph: the sharpest case
    (dict(recurrence_depth=2, recurrence_order="cycle"),
     dict(recurrence_depth=2, recurrence_order="sequence")),
    # Codex probe B: depth-2 cycle loaded as depth-3 sequence
    (dict(recurrence_depth=2, recurrence_order="cycle"),
     dict(recurrence_depth=3, recurrence_order="sequence")),
    (dict(recurrence_depth=4, recurrence_order="cycle"),
     dict(recurrence_depth=1, recurrence_order="cycle")),
])
def test_changed_schedule_rejected_before_weights_load(tiny_cfg, tmp_path, saved, loaded):
    _save(_rec(tiny_cfg, **saved), tmp_path / "ck", seed=1)
    torch.manual_seed(2)
    target = build_model(_rec(tiny_cfg, **loaded).model)
    before = {k: v.clone() for k, v in target.state_dict().items()}
    with pytest.raises(CheckpointError) as e:
        load_checkpoint(tmp_path / "ck", target, restore_rng=False)
    assert "application_schedule" in str(e.value)
    assert "start a new experiment" in str(e.value)      # actionable
    # rejected BEFORE any tensor was copied in
    assert all(torch.equal(v, target.state_dict()[k]) for k, v in before.items())


def test_recurrent_checkpoint_into_same_shape_baseline_rejected(tiny_cfg, tmp_path):
    # tiny_cfg has n_layers=2, so a 2-block baseline has identical tensors
    _save(_rec(tiny_cfg, recurrence_depth=2), tmp_path / "ck")
    with pytest.raises(CheckpointError, match="architecture"):
        load_checkpoint(tmp_path / "ck", build_model(tiny_cfg.model), restore_rng=False)


def test_baseline_checkpoint_into_recurrent_rejected(tiny_cfg, tmp_path):
    _save(tiny_cfg, tmp_path / "ck")
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "ck", build_model(_rec(tiny_cfg).model),
                        restore_rng=False)


def _make_legacy(ckpt_dir):
    """Rewrite checkpoint.json as the pre-recurrence code wrote it: no
    model_signature and no recurrence keys in config.model (DML-0001 era)."""
    p = ckpt_dir / "checkpoint.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta.pop("model_signature")
    meta["config"]["model"].pop("recurrence_depth")
    meta["config"]["model"].pop("recurrence_order")
    p.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def test_legacy_baseline_checkpoint_still_loads(tiny_cfg, tmp_path):
    ref = _save(tiny_cfg, tmp_path / "ck", seed=3)
    _make_legacy(tmp_path / "ck")
    x = torch.randint(0, tiny_cfg.model.vocab_size, (1, 9))
    fresh = build_model(tiny_cfg.model).eval()
    load_checkpoint(tmp_path / "ck", fresh, restore_rng=False)
    rebuilt, cfg, _ = build_model_from_checkpoint(tmp_path / "ck")
    rebuilt.eval()
    with torch.no_grad():
        assert torch.equal(fresh(x).logits, ref(x).logits)
        assert torch.equal(rebuilt(x).logits, ref(x).logits)
    assert cfg.model.recurrence_depth == 1          # defaults filled in
    # ...and a legacy baseline checkpoint is still refused by a recurrent model
    with pytest.raises(CheckpointError):
        load_checkpoint(tmp_path / "ck", build_model(_rec(tiny_cfg, recurrence_depth=2).model),
                        restore_rng=False)


def test_inconsistent_checkpoint_json_rejected(tiny_cfg, tmp_path):
    """A stored schedule that no longer matches what the stored config means
    (e.g. after a code change to recurrence_schedule) must not load."""
    _save(_rec(tiny_cfg, recurrence_depth=2), tmp_path / "ck")
    p = tmp_path / "ck" / "checkpoint.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["model_signature"]["application_schedule"] = [0, 0, 1, 1]
    p.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(CheckpointError, match="inconsistent"):
        build_model_from_checkpoint(tmp_path / "ck")


def test_inference_rebuild_uses_saved_schedule(tiny_cfg, tmp_path):
    cfg = _rec(tiny_cfg, recurrence_depth=3, recurrence_order="sequence")
    ref = _save(cfg, tmp_path / "ck", seed=4)
    rebuilt, _, _ = build_model_from_checkpoint(tmp_path / "ck")
    rebuilt.eval()
    assert rebuilt.application_schedule() == [0, 0, 0, 1, 1, 1]
    x = torch.randint(0, tiny_cfg.model.vocab_size, (1, 7))
    with torch.no_grad():
        assert torch.equal(rebuilt(x).logits, ref(x).logits)


def test_untied_checkpoint_missing_lm_head_rejected(tiny_cfg, tmp_path):
    cfg = clone_config(tiny_cfg, model={"tie_embeddings": False})
    _save(cfg, tmp_path / "ck")
    f = tmp_path / "ck" / "model.safetensors"
    state = load_file(str(f))
    del state["lm_head.weight"]
    save_file(state, str(f))
    with pytest.raises(CheckpointError, match="lm_head.weight"):
        load_checkpoint(tmp_path / "ck", build_model(cfg.model), restore_rng=False)
    with pytest.raises(CheckpointError, match="lm_head.weight"):
        build_model_from_checkpoint(tmp_path / "ck")


# ---- Trainer resume --------------------------------------------------

def _trained_dir(smoke_dataset, **model):
    cfg_dict = json.loads(json.dumps(smoke_dataset["config_dict"]))
    cfg_dict["model"].update({"architecture": "domovoi_recurrent_v1", **model})
    cfg_dict["training"].update(steps=2, checkpoint_interval=2, eval_interval=2)
    cfg = config_from_dict(cfg_dict)
    exp = smoke_dataset["tmp_path"] / "exp"
    Trainer(cfg, experiment_dir=exp).train()
    return cfg_dict, exp


@pytest.mark.parametrize("change", [
    {"recurrence_order": "sequence"},          # same tensor shapes
    {"recurrence_depth": 3},
    {"dropout": 0.1},                          # same graph, different run
])
def test_resume_with_different_model_config_rejected_without_writes(smoke_dataset, change):
    cfg_dict, exp = _trained_dir(smoke_dataset, recurrence_depth=2)
    before = _tree_hashes(exp)
    changed = json.loads(json.dumps(cfg_dict))
    changed["model"].update(change)
    changed["training"]["steps"] = 4
    with pytest.raises(CheckpointError, match="Cannot resume"):
        Trainer(config_from_dict(changed), experiment_dir=exp, resume=True)
    assert _tree_hashes(exp) == before


def test_resume_without_checkpoint_fails_closed(smoke_dataset):
    cfg = config_from_dict(smoke_dataset["config_dict"])
    empty = smoke_dataset["tmp_path"] / "empty_exp"
    with pytest.raises(CheckpointError, match="no checkpoint"):
        Trainer(cfg, experiment_dir=empty, resume=True)
    assert not empty.exists()          # nothing was created


def test_fresh_run_into_dir_with_checkpoints_refused(smoke_dataset):
    cfg_dict, exp = _trained_dir(smoke_dataset, recurrence_depth=2)
    before = _tree_hashes(exp)
    with pytest.raises(CheckpointError, match="already holds checkpoints"):
        Trainer(config_from_dict(cfg_dict), experiment_dir=exp)
    assert _tree_hashes(exp) == before


def test_resume_with_same_model_config_still_works(smoke_dataset):
    cfg_dict, exp = _trained_dir(smoke_dataset, recurrence_depth=2)
    same = json.loads(json.dumps(cfg_dict))
    same["training"]["log_interval"] = 5   # logging-only changes stay allowed
    t = Trainer(config_from_dict(same), experiment_dir=exp, resume=True)
    assert t.start_step == 2
    t.close()


def _run_train_script(*args):
    workload = "import sys; from scripts.train import _run; raise SystemExit(_run(sys.argv[1:]))"
    return subprocess.run([sys.executable, "-c", workload, *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=300)


def test_train_script_refused_resume_leaves_experiment_untouched(smoke_dataset):
    """The scripts/train.py `finally:` used to stamp FAILED on any exception,
    so a refused resume would have rewritten a RUNNING experiment's status
    (the DML-0001 situation).  Now the whole directory stays byte-identical."""
    cfg_dict, exp = _trained_dir(smoke_dataset, recurrence_depth=2)
    (exp / "metadata.json").write_text(json.dumps(
        {"experiment_id": "DML-0001", "status": "RUNNING"}, indent=2), encoding="utf-8")
    before = _tree_hashes(exp)
    changed = json.loads(json.dumps(cfg_dict))
    changed["model"]["recurrence_order"] = "sequence"
    cfg_path = smoke_dataset["tmp_path"] / "changed.yaml"
    cfg_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    r = _run_train_script("--config", str(cfg_path), "--resume", "--experiment-dir", str(exp))
    assert r.returncode != 0
    assert "Cannot resume" in r.stderr
    assert _tree_hashes(exp) == before
    assert json.loads((exp / "metadata.json").read_text(encoding="utf-8"))["status"] == "RUNNING"


def test_train_script_experiment_dir_requires_resume(smoke_dataset):
    cfg_path = smoke_dataset["tmp_path"] / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(smoke_dataset["config_dict"]), encoding="utf-8")
    target = smoke_dataset["tmp_path"] / "some_exp"
    r = _run_train_script("--config", str(cfg_path), "--experiment-dir", str(target))
    assert r.returncode != 0 and "only valid with --resume" in r.stderr
    assert not target.exists()


# =====================================================================
# DML-A02  KV-cache validation
# =====================================================================

@pytest.fixture()
def rec_model(tiny_cfg):
    torch.manual_seed(5)
    return build_model(_rec(tiny_cfg, recurrence_depth=2).model).eval()


def _prefill(model, n=5, batch=1):
    torch.manual_seed(6)
    x = torch.randint(0, model.cfg.vocab_size, (batch, n + 3))
    with torch.no_grad():
        cache = model(x[:, :n], use_cache=True).caches
    return x, cache


def _replace(cache, j, k, v):
    bad = list(cache)
    bad[j] = (k, v)
    return bad


def _malformed(cache, model):
    k1, v1 = cache[1]
    return {
        "empty list": [],
        "too few entries": list(cache)[:-1],
        "too many entries": list(cache) + [cache[0]],
        "None hole": [cache[0], None] + list(cache)[2:],
        "not a list": cache[0],
        "not a pair": [cache[0], (k1,)] + list(cache)[2:],
        "not tensors": _replace(cache, 1, k1.tolist(), v1),
        "3-D tensor": _replace(cache, 1, k1[0], v1[0]),
        "k/v shape mismatch": _replace(cache, 1, k1, v1[:, :, :3]),
        "shorter slot": _replace(cache, 1, k1[:, :, :2], v1[:, :, :2]),
        "longer slot": _replace(cache, 1, torch.cat([k1, k1], 2), torch.cat([v1, v1], 2)),
        "wrong heads": _replace(cache, 1, k1[:, :1], v1[:, :1]),
        "wrong head_dim": _replace(cache, 1, k1[..., :8], v1[..., :8]),
        "wrong batch": _replace(cache, 1, k1.repeat(2, 1, 1, 1), v1.repeat(2, 1, 1, 1)),
        "wrong dtype": _replace(cache, 1, k1.double(), v1.double()),
        "mixed k/v dtype": _replace(cache, 1, k1, v1.double()),
        "wrong device": _replace(cache, 1, torch.empty(k1.shape, device="meta"),
                                 torch.empty(v1.shape, device="meta")),
    }


@pytest.mark.parametrize("case", [
    "empty list", "too few entries", "too many entries", "None hole", "not a list",
    "not a pair", "not tensors", "3-D tensor", "k/v shape mismatch", "shorter slot",
    "longer slot", "wrong heads", "wrong head_dim", "wrong batch", "wrong dtype",
    "mixed k/v dtype", "wrong device"])
def test_malformed_cache_rejected(rec_model, case):
    x, cache = _prefill(rec_model)
    bad = _malformed(cache, rec_model)[case]
    with pytest.raises(CacheValidationError):
        with torch.no_grad():
            rec_model(x[:, 5:6], caches=bad)


def test_cache_errors_are_value_errors_and_name_the_slot(rec_model):
    x, cache = _prefill(rec_model)
    k1, v1 = cache[1]
    with pytest.raises(ValueError, match="cache slot 1"):
        rec_model(x[:, 5:6], caches=_replace(cache, 1, k1[:, :, :2], v1[:, :, :2]))


def test_input_batch_must_match_cache(rec_model):
    x, cache = _prefill(rec_model)
    with pytest.raises(CacheValidationError, match="batch"):
        rec_model(x[:, 5:6].repeat(2, 1), caches=cache)


def test_empty_input_rejected(rec_model):
    with pytest.raises(CacheValidationError):
        rec_model(torch.zeros(1, 0, dtype=torch.long))


def test_context_boundary_with_cache(rec_model):
    ctx = rec_model.cfg.context_length
    x = torch.randint(0, rec_model.cfg.vocab_size, (1, ctx))
    with torch.no_grad():
        cache = rec_model(x[:, :ctx - 1], use_cache=True).caches
        rec_model(x[:, ctx - 1:], caches=cache)          # exactly fills: fine
        full = rec_model(x, use_cache=True).caches
        with pytest.raises(CacheValidationError, match="exceeds context_length"):
            rec_model(x[:, :1], caches=full)


def test_cache_from_other_model_rejected(tiny_cfg, rec_model):
    torch.manual_seed(99)
    other = build_model(_rec(tiny_cfg, recurrence_depth=2).model).eval()
    x, cache = _prefill(rec_model)
    with pytest.raises(CacheValidationError, match="different model"):
        other(x[:, 5:6], caches=cache)


def test_cache_from_other_schedule_rejected(rec_model):
    """Same model, same slot count and shapes, different application order:
    shapes cannot catch it, provenance does."""
    x, cache = _prefill(rec_model)
    rec_model._schedule = (0, 0, 1, 1)        # was (0, 1, 0, 1)
    with pytest.raises(CacheValidationError, match="schedule"):
        rec_model(x[:, 5:6], caches=cache)


def test_returned_cache_is_list_with_provenance(rec_model):
    _, cache = _prefill(rec_model)
    assert isinstance(cache, list) and isinstance(cache, ApplicationCache)
    assert cache.schedule == tuple(rec_model.application_schedule())
    assert cache.owner() is rec_model


def test_plain_list_and_all_none_caches_still_valid(rec_model):
    x, cache = _prefill(rec_model)
    with torch.no_grad():
        full = rec_model(x).logits
        a = rec_model(x[:, 5:], caches=list(cache)).logits        # no provenance
        b = rec_model(x, caches=[None] * rec_model.effective_depth).logits
    assert (a - full[:, 5:]).abs().max() < 1e-5
    assert torch.equal(b, full)


@pytest.mark.parametrize("depth", [1, 2, 4, 8])
@pytest.mark.parametrize("order", ["cycle", "sequence"])
def test_validated_chunked_decoding_matches_uncached(tiny_cfg, depth, order):
    torch.manual_seed(depth)
    m = build_model(_rec(tiny_cfg, recurrence_depth=depth, recurrence_order=order).model).eval()
    x = torch.randint(0, m.cfg.vocab_size, (2, 20))
    with torch.no_grad():
        full = m(x).logits
        cache, p = None, 0
        for k in (1, 6, 1, 1, 11):
            out = m(x[:, p:p + k], caches=cache, use_cache=True)
            assert (out.logits - full[:, p:p + k]).abs().max() < 1e-5
            cache, p = out.caches, p + k


def test_autocast_cache_dtype_accepted(rec_model):
    """Under autocast the cache holds autocast-dtype K/V while weights stay
    fp32; that is the supported mixed-precision path and must validate."""
    x = torch.randint(0, rec_model.cfg.vocab_size, (1, 8))
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        cache = rec_model(x[:, :5], use_cache=True).caches
        assert cache[0][0].dtype == torch.bfloat16
        rec_model(x[:, 5:], caches=cache)
    # ...but the bf16 cache is refused by an fp32 forward without autocast
    with pytest.raises(CacheValidationError, match="dtype"):
        rec_model(x[:, 5:], caches=cache)


def test_cached_generate_matches_uncached_greedy(rec_model):
    """Validated cached generation == recomputing the full prefix each step."""
    tok = _ByteTok()
    r = generate(rec_model, "hello", tok, max_new_tokens=6, temperature=0.0)
    ids = tok.encode("hello")
    with torch.no_grad():
        for _ in range(6):
            ids.append(int(rec_model(torch.tensor([ids])).logits[0, -1].argmax()))
    assert r["new_text"] == tok.decode(ids[5:])


# =====================================================================
# DML-A03  profiling
# =====================================================================

def test_context_sweep_uses_exact_lengths(rec_model):
    rows = profile_context_sweep(rec_model, _WordTok(), torch.device("cpu"), [16, 32], 0)
    assert [(r["target_tokens"], r["actual_prompt_tokens"]) for r in rows] == [(16, 16), (32, 32)]


def test_context_sweep_reserves_decode_headroom(rec_model):
    ctx = rec_model.cfg.context_length
    rows = profile_context_sweep(rec_model, _ByteTok(), torch.device("cpu"), [ctx], 8)
    r = rows[0]
    assert r["requested_tokens"] == ctx and r["target_tokens"] == ctx - 8
    assert r["capped_for_decode_headroom"] is True
    assert r["n_new_tokens"] == 8 and r["decode_tok_per_s"] is not None
    assert r["final_sequence_length"] == ctx - 1     # last sampled token is not fed back
    assert r["context_length"] == ctx and r["effective_depth"] == 4
    assert r["peak_vram_allocated_mb"] is None       # CPU: never estimated


def test_context_sweep_rejects_no_room(rec_model):
    with pytest.raises(ValueError):
        profile_context_sweep(rec_model, _ByteTok(), torch.device("cpu"), [8],
                              rec_model.cfg.context_length)


def test_profile_model_rejects_mismatched_config(tiny_cfg, rec_model):
    wrong = _rec(tiny_cfg, recurrence_depth=2, recurrence_order="sequence")
    with pytest.raises(ValueError, match="application_schedule"):
        profile_model(rec_model, wrong, _ByteTok(), torch.device("cpu"),
                      max_new_tokens=2, context_sweep=False)


def test_profile_model_bytes_use_real_dtype_and_are_labelled(tiny_cfg, rec_model):
    cfg = _rec(tiny_cfg, recurrence_depth=2)
    prof = profile_model(rec_model, cfg, _ByteTok(), torch.device("cpu"),
                         max_new_tokens=4, context_sweep=False)
    est = prof["analytic_estimates"]
    assert est["basis"].startswith("analytic estimate")
    assert est["bytes_per_param"] == 4 and est["bytes_per_kv"] == 4    # fp32 model
    # the analytic count is exact for this dense model, so in the REAL dtype
    # the estimate equals the counted tensor bytes (it was 2x off before)
    assert est["weight_bytes"] == prof["measured_parameter_bytes"]
    assert prof["recurrence"]["application_schedule"] == [0, 1, 0, 1]
    assert prof["weight_dtype"] == prof["kv_dtype"] == "torch.float32"
    inf = prof["inference"]
    assert inf["measurement_basis"]["vram"].startswith("not measured")
    assert inf["peak_vram_allocated_mb"] is None
    # logical KV bytes of the final cache == analytic per-token bytes x length
    assert inf["kv_cache_logical_bytes"] == \
        est["kv_cache_bytes_per_token"] * inf["final_sequence_length"]


def test_prefill_cache_storage_overhead_is_reported(rec_model):
    tok = _ByteTok()
    r = generate(rec_model, "abcdef", tok, max_new_tokens=1, temperature=0.0)
    # prefill-only cache: K/V are views of the fused QKV buffer (1.5x)
    assert r["kv_cache_storage_bytes"] == r["kv_cache_logical_bytes"] * 3 // 2
    r = generate(rec_model, "abcdef", tok, max_new_tokens=3, temperature=0.0)
    # after a decode step torch.cat has produced standalone K/V tensors
    assert r["kv_cache_storage_bytes"] == r["kv_cache_logical_bytes"]


def test_profile_inference_takes_peak_over_all_runs(monkeypatch, rec_model):
    calls = []
    real = profiler_mod.generate

    def spy(*a, **kw):
        calls.append(kw.get("reset_peak", True))
        return real(*a, **kw)

    monkeypatch.setattr(profiler_mod, "generate", spy)
    profile_inference(rec_model, _ByteTok(), torch.device("cpu"), max_new_tokens=3, n_repeats=3)
    # warmup may reset; the timed runs must not, so the peak spans all of them
    assert calls[1:] == [False, False, False]


def test_generate_exact_ids_and_ttft_definition(rec_model):
    tok = _ByteTok()
    ids = [7] * 13
    r = generate(rec_model, None, tok, prompt_ids=ids, max_new_tokens=3, temperature=0.0)
    assert r["n_prompt_tokens"] == 13
    assert r["ttft_s"] >= r["prefill_s"] > 0
    full = [1] * rec_model.cfg.context_length
    r = generate(rec_model, None, tok, prompt_ids=full, max_new_tokens=3)
    assert r["n_new_tokens"] == 0 and r["ttft_s"] is None    # no token: no TTFT


def test_signature_of_config_matches_model(tiny_cfg):
    for depth, order in ((1, "cycle"), (3, "sequence")):
        cfg = _rec(tiny_cfg, recurrence_depth=depth, recurrence_order=order)
        assert config_signature(cfg.model) == model_signature(build_model(cfg.model))
    assert config_signature(tiny_cfg.model) == model_signature(build_model(tiny_cfg.model))


# =====================================================================
# DML-A04  planner   /   DML-A08  schedule input types
# =====================================================================

def _planner():
    spec = importlib.util.spec_from_file_location(
        "plan_recurrent_experiments", ROOT / "scripts" / "plan_recurrent_experiments.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_planner_never_inherits_experiment_id(tiny_cfg):
    base = clone_config(tiny_cfg, output={"experiment_id": "DML-0001"})
    base.source_path = "test"
    arms = dict(_planner().build_arms(base, only=("B02", "R02x4", "B08"),
                                      seeds=[1, 2, 3]))
    assert len(arms) == 9
    assert all(c.output.experiment_id is None for c in arms.values())
    assert {c.training.seed for c in arms.values()} == {1, 2, 3}


def test_planner_refuses_to_overwrite(tmp_path):
    out = tmp_path / "plan"
    cmd = [sys.executable, str(ROOT / "scripts" / "plan_recurrent_experiments.py"),
           "--pilot", "--write-configs", str(out)]
    assert subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 0
    (out / "B02.yaml").write_text("sentinel", encoding="utf-8")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 1 and "refusing to overwrite" in r.stderr
    assert (out / "B02.yaml").read_text(encoding="utf-8") == "sentinel"
    assert subprocess.run(cmd + ["--force"], cwd=ROOT, capture_output=True).returncode == 0
    assert (out / "B02.yaml").read_text(encoding="utf-8") != "sentinel"


@pytest.mark.parametrize("bad", [True, False, 2.0, "2"])
def test_recurrence_schedule_rejects_non_int(bad):
    with pytest.raises(ValueError):
        recurrence_schedule(2, bad, "cycle")
    with pytest.raises(ValueError):
        recurrence_schedule(bad, 2, "cycle")


def test_accounting_labels_its_assumptions(tiny_cfg):
    a = architecture_accounting(tiny_cfg.model, bytes_per_param=4, bytes_per_kv=2)
    assert a["basis"].startswith("analytic estimate")
    assert (a["bytes_per_param"], a["bytes_per_kv"]) == (4, 2)
