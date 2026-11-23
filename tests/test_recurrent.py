"""domovoi_recurrent_v1: shared-core recurrent Transformer.

CPU-only, synthetic data, a few seconds total.  Groups:
  config & validation, construction & sharing, parameter accounting,
  equivalence with the baseline, forward/backward/gradient flow, causality,
  KV cache, determinism & stability, checkpoints, pipeline integration.
"""

import copy
import json
import pathlib

import pytest
import torch

from domovoi.config import ConfigError, clone_config, config_from_dict, load_config
from domovoi.inference import generate
from domovoi.models import (DomovoiRecurrentTransformer, RecurrentTransformerConfig,
                            build_model)
from domovoi.models.domovoi_recurrent import recurrence_schedule
from domovoi.profiling import architecture_accounting, count_parameters, profile_inference
from domovoi.training import build_model_from_checkpoint
from domovoi.training.checkpoint import save_checkpoint
from domovoi.training.optim import build_optimizer, build_scheduler

REPO = pathlib.Path(__file__).resolve().parent.parent
DEPTHS = [1, 2, 4, 8]


def _cfg(tiny_model_cfg_dict, depth=2, order="cycle", n_layers=2, **model):
    d = copy.deepcopy(tiny_model_cfg_dict)
    d["model"].update(architecture="domovoi_recurrent_v1", n_layers=n_layers,
                      recurrence_depth=depth, recurrence_order=order, **model)
    return config_from_dict(d)


def _model(tiny_model_cfg_dict, seed=0, **kw):
    torch.manual_seed(seed)
    return build_model(_cfg(tiny_model_cfg_dict, **kw).model)


class _Tok:
    vocab_size = 256
    def encode(self, text):
        return list(text.encode("utf-8"))
    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


# ---------------------------------------------------------------- config ---
def test_recurrent_config_loads(tiny_model_cfg_dict):
    cfg = _cfg(tiny_model_cfg_dict, depth=4, order="sequence")
    assert cfg.model.recurrence_depth == 4
    assert cfg.model.recurrence_order == "sequence"


@pytest.mark.parametrize("field,value,pat", [
    ("recurrence_depth", 0, "recurrence_depth"),
    ("recurrence_depth", -3, "recurrence_depth"),
    ("recurrence_depth", 2.5, "recurrence_depth"),
    ("recurrence_depth", True, "recurrence_depth"),
    ("recurrence_order", "random", "recurrence_order"),
    ("architecture", "domovoi_recurrent_v2", "domovoi_recurrent_v1"),
])
def test_recurrent_config_invalid(tiny_model_cfg_dict, field, value, pat):
    d = copy.deepcopy(tiny_model_cfg_dict)
    d["model"]["architecture"] = "domovoi_recurrent_v1"
    d["model"][field] = value
    with pytest.raises(ConfigError, match=pat):
        config_from_dict(d)


@pytest.mark.parametrize("field,value", [("recurrence_depth", 2),
                                         ("recurrence_order", "sequence")])
def test_baseline_rejects_recurrence_keys(tiny_model_cfg_dict, field, value):
    """The baseline would silently ignore recurrence settings; refuse them."""
    d = copy.deepcopy(tiny_model_cfg_dict)
    d["model"][field] = value
    with pytest.raises(ConfigError, match="only apply"):
        config_from_dict(d)


def test_pre_recurrence_configs_still_load():
    """Backward compatibility: the DML-0001 YAML and the config stored in its
    metadata predate the recurrence keys and must load with defaults.
    (Read-only: nothing under experiments/ is written.)"""
    cfg = load_config(REPO / "experiments" / "DML-0001" / "config.yaml")
    assert cfg.model.architecture == "baseline_transformer"
    assert (cfg.model.recurrence_depth, cfg.model.recurrence_order) == (1, "cycle")
    meta = json.loads((REPO / "experiments" / "DML-0001" / "metadata.json")
                      .read_text(encoding="utf-8"))
    assert "recurrence_depth" not in meta["config"]["model"]
    assert config_from_dict(meta["config"]).model.recurrence_depth == 1
    for name in ("baseline_small.yaml", "smoke_test.yaml"):
        load_config(REPO / "configs" / name)


def test_clone_config_depth_override(tiny_model_cfg_dict):
    cfg = _cfg(tiny_model_cfg_dict, depth=1)
    cfg8 = clone_config(cfg, model={"recurrence_depth": 8})
    assert cfg8.model.recurrence_depth == 8 and cfg.model.recurrence_depth == 1


# ------------------------------------------------- construction & sharing ---
def test_schedule_orders():
    assert recurrence_schedule(3, 2, "cycle") == [0, 1, 2, 0, 1, 2]
    assert recurrence_schedule(3, 2, "sequence") == [0, 0, 1, 1, 2, 2]
    with pytest.raises(ValueError):
        recurrence_schedule(3, 0, "cycle")
    with pytest.raises(ValueError):
        recurrence_schedule(0, 2, "cycle")
    with pytest.raises(ValueError):
        recurrence_schedule(3, 2, "zigzag")


def test_direct_construction_validates():
    with pytest.raises(ValueError):
        DomovoiRecurrentTransformer(RecurrentTransformerConfig(
            vocab_size=256, context_length=16, d_model=16, n_layers=2,
            n_heads=2, ffn_dim=32, recurrence_depth=0))


@pytest.mark.parametrize("order", ["cycle", "sequence"])
def test_reuse_calls_the_same_modules(tiny_model_cfg_dict, order):
    """Every application must invoke one of the n_layers stored modules, in
    schedule order: reuse of real parameters, not copies."""
    model = _model(tiny_model_cfg_dict, depth=3, order=order, n_layers=2)
    assert len(model.blocks) == 2
    called = []
    for i, blk in enumerate(model.blocks):
        blk.register_forward_hook(lambda mod, inp, out, i=i: called.append(i))
    model(torch.randint(0, 256, (1, 8)))
    assert called == recurrence_schedule(2, 3, order)


def test_no_parameter_copies(tiny_model_cfg_dict):
    """Depth does not add tensors: same tensor count, same storage, same
    state_dict keys as depth 1."""
    m1 = _model(tiny_model_cfg_dict, depth=1)
    m8 = _model(tiny_model_cfg_dict, depth=8)
    assert len(list(m8.parameters())) == len(list(m1.parameters()))
    assert list(m8.state_dict()) == list(m1.state_dict())
    ptrs = [p.data_ptr() for p in m8.parameters()]
    assert len(ptrs) == len(set(ptrs))
    assert not any(k.startswith("blocks.2.") for k in m8.state_dict())


# ---------------------------------------------------- parameter accounting ---
@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("tie", [True, False])
def test_param_counts_exact(tiny_model_cfg_dict, depth, tie):
    cfg = _cfg(tiny_model_cfg_dict, depth=depth, tie_embeddings=tie)
    model = build_model(cfg.model)
    actual = sum(p.numel() for p in model.parameters())
    acc = architecture_accounting(cfg.model)
    counts = count_parameters(model)
    b = model.param_breakdown()
    per_block = sum(p.numel() for p in model.blocks[0].parameters())
    assert acc["unique_params"] == actual == counts["total"] == b["total"]
    assert acc["block_params"] == per_block
    assert counts["applied_per_token"] == acc["applied_params"] \
        == actual + (depth - 1) * 2 * per_block
    assert counts["effective_depth"] == acc["effective_depth"] == b["effective_depth"] == 2 * depth
    assert b["applied_blocks"] == per_block * 2 * depth


def test_unique_params_independent_of_depth(tiny_model_cfg_dict):
    n = {d: count_parameters(_model(tiny_model_cfg_dict, depth=d))["total"] for d in DEPTHS}
    assert len(set(n.values())) == 1


def test_baseline_small_param_count_reproduced():
    """Accounting reproduces DML-0001's recorded 41,527,680 parameters."""
    cfg = load_config(REPO / "configs" / "baseline_small.yaml")
    assert architecture_accounting(cfg.model)["unique_params"] == 41_527_680
    assert count_parameters(build_model(cfg.model))["applied_per_token"] == 41_527_680


def test_kv_cache_accounting_scales_with_depth(tiny_model_cfg_dict):
    kv = [architecture_accounting(_cfg(tiny_model_cfg_dict, depth=d).model)
          ["kv_cache_bytes_per_token"] for d in DEPTHS]
    assert kv == [kv[0] * d for d in DEPTHS]


# ------------------------------------------------- baseline equivalence ---
def test_depth1_is_exactly_the_baseline(tiny_model_cfg_dict):
    """Same seed -> identical init; baseline weights load strictly and give
    bit-identical logits.  So any A/B gap comes from sharing alone."""
    torch.manual_seed(5)
    base = build_model(config_from_dict(tiny_model_cfg_dict).model)
    rec = _model(tiny_model_cfg_dict, seed=5, depth=1)
    sb, sr = base.state_dict(), rec.state_dict()
    assert list(sb) == list(sr) and all(torch.equal(sb[k], sr[k]) for k in sb)
    rec.load_state_dict(sb, strict=True)
    x = torch.randint(0, 256, (2, 20))
    assert torch.equal(base(x).logits, rec(x).logits)


@pytest.mark.parametrize("order", ["cycle", "sequence"])
def test_shared_equals_unrolled_copies(tiny_model_cfg_dict, order):
    """A shared model == a baseline with n_layers*depth independent blocks
    whose weights are copies of the shared ones.  Forward must match, and the
    shared gradient must equal the SUM of the copies' gradients."""
    depth = 3
    rec = _model(tiny_model_cfg_dict, depth=depth, order=order)
    d = copy.deepcopy(tiny_model_cfg_dict)
    d["model"]["n_layers"] = 2 * depth
    unrolled = build_model(config_from_dict(d).model)
    schedule = rec.application_schedule()
    with torch.no_grad():
        for name in ("wte", "wpe", "ln_f"):
            getattr(unrolled, name).load_state_dict(getattr(rec, name).state_dict())
        for pos, b in enumerate(schedule):
            unrolled.blocks[pos].load_state_dict(rec.blocks[b].state_dict())
    x = torch.randint(0, 256, (2, 12))
    y = torch.randint(0, 256, (2, 12))
    lr, lu = rec(x, targets=y).loss, unrolled(x, targets=y).loss
    assert torch.allclose(lr, lu, atol=1e-6)
    lr.backward()
    lu.backward()
    for b in range(2):
        for (name, p) in rec.blocks[b].named_parameters():
            summed = sum(dict(unrolled.blocks[pos].named_parameters())[name].grad
                         for pos, sb in enumerate(schedule) if sb == b)
            assert torch.allclose(p.grad, summed, atol=1e-5), name


# ------------------------------------------- forward / backward / gradients ---
@pytest.mark.parametrize("depth", DEPTHS)
def test_forward_shapes_and_loss(tiny_model_cfg_dict, depth):
    model = _model(tiny_model_cfg_dict, depth=depth)
    x = torch.randint(0, 256, (3, 17))
    out = model(x, targets=x)
    assert out.logits.shape == (3, 17, 256)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    assert out.caches is None


@pytest.mark.parametrize("depth", DEPTHS)
def test_gradients_reach_every_application(tiny_model_cfg_dict, depth):
    """No detached recurrence: every application's output state receives a
    nonzero gradient, and every stored parameter gets a finite gradient."""
    model = _model(tiny_model_cfg_dict, depth=depth)
    x = torch.randint(0, 256, (2, 10))
    states = model.application_hidden_states(x)
    for s in states:
        s.retain_grad()
    logits = model.lm_head(model.ln_f(states[-1]))
    torch.nn.functional.cross_entropy(logits.view(-1, 256), x.view(-1)).backward()
    assert len(states) == 2 * depth
    for i, s in enumerate(states):
        assert s.grad is not None and s.grad.abs().sum() > 0, f"application {i}"
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_hidden_states_match_forward(tiny_model_cfg_dict):
    model = _model(tiny_model_cfg_dict, depth=4).eval()
    x = torch.randint(0, 256, (1, 9))
    with torch.no_grad():
        last = model.application_hidden_states(x)[-1]
        assert torch.equal(model.lm_head(model.ln_f(last)), model(x).logits)
    with pytest.raises(ValueError, match="exceeds"):
        model.application_hidden_states(torch.zeros(1, 65, dtype=torch.long))


def test_deeper_recurrence_changes_the_function(tiny_model_cfg_dict):
    """Same weights, different depth -> different output (recurrence is not a no-op)."""
    m2 = _model(tiny_model_cfg_dict, seed=1, depth=2)
    m4 = _model(tiny_model_cfg_dict, seed=1, depth=4)
    m4.load_state_dict(m2.state_dict())
    x = torch.randint(0, 256, (1, 8))
    assert not torch.allclose(m2(x).logits, m4(x).logits)


def test_training_step_reduces_loss(tiny_model_cfg_dict):
    model = _model(tiny_model_cfg_dict, depth=4)
    x = torch.randint(0, 256, (1, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        loss = model(x, targets=x).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


# ---------------------------------------------------------------- causality ---
@pytest.mark.parametrize("depth", [1, 4])
def test_causal_no_future_leak(tiny_model_cfg_dict, depth):
    """Changing token t must not change logits at positions < t, at any depth."""
    model = _model(tiny_model_cfg_dict, depth=depth).eval()
    x = torch.randint(0, 256, (1, 20))
    x2 = x.clone()
    x2[0, 12:] = (x2[0, 12:] + 7) % 256
    with torch.no_grad():
        a, b = model(x).logits, model(x2).logits
    assert torch.allclose(a[0, :12], b[0, :12], atol=1e-6)
    assert not torch.allclose(a[0, 12:], b[0, 12:])


# ----------------------------------------------------------------- KV cache ---
@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("order", ["cycle", "sequence"])
def test_cached_matches_uncached_logits(tiny_model_cfg_dict, depth, order):
    """Prefill + multi-token extend + single-token steps reproduce the full
    uncached forward at every position."""
    model = _model(tiny_model_cfg_dict, depth=depth, order=order).eval()
    x = torch.randint(0, 256, (2, 24))
    with torch.no_grad():
        full = model(x).logits
        out = model(x[:, :10], use_cache=True)
        pieces = [out.logits]
        out = model(x[:, 10:16], use_cache=True, caches=out.caches)   # extend, T>1
        pieces.append(out.logits)
        caches = out.caches
        for t in range(16, 24):                                         # T == 1 steps
            out = model(x[:, t:t + 1], use_cache=True, caches=caches)
            caches = out.caches
            pieces.append(out.logits)
    assert torch.allclose(torch.cat(pieces, dim=1), full, atol=1e-5)


def test_cache_has_one_entry_per_application(tiny_model_cfg_dict):
    """Shared weights do NOT mean shared KV: each application of the same
    block caches its own, different keys/values."""
    model = _model(tiny_model_cfg_dict, depth=4).eval()
    with torch.no_grad():
        caches = model(torch.randint(0, 256, (1, 6)), use_cache=True).caches
    assert len(caches) == 2 * 4
    assert all(k.shape == (1, 2, 6, 16) for k, _ in caches)
    # applications 0 and 2 both use block 0 (cycle order) yet differ
    assert not torch.allclose(caches[0][0], caches[2][0])


def test_wrong_cache_length_rejected(tiny_model_cfg_dict):
    m2 = _model(tiny_model_cfg_dict, depth=2).eval()
    m4 = _model(tiny_model_cfg_dict, depth=4).eval()
    with torch.no_grad():
        caches = m2(torch.randint(0, 256, (1, 4)), use_cache=True).caches
        with pytest.raises(ValueError, match="cache entries"):
            m4(torch.randint(0, 256, (1, 1)), use_cache=True, caches=caches)


@pytest.mark.parametrize("depth", [2, 8])
def test_generate_cached_equals_uncached_greedy(tiny_model_cfg_dict, depth):
    model = _model(tiny_model_cfg_dict, depth=depth).eval()
    tok, prompt, n = _Tok(), "shared core", 12
    res = generate(model, prompt, tok, max_new_tokens=n, temperature=0.0)
    ids = tok.encode(prompt)
    with torch.no_grad():
        for _ in range(n):
            ids.append(int(model(torch.tensor([ids])).logits[0, -1].argmax()))
    assert tok.decode(ids[len(tok.encode(prompt)):]) == res["new_text"]


def test_generate_respects_context(tiny_model_cfg_dict):
    model = _model(tiny_model_cfg_dict, depth=2).eval()
    r = generate(model, "aaaaa", _Tok(), max_new_tokens=200, temperature=0.0)
    assert r["n_prompt_tokens"] + r["n_new_tokens"] <= model.cfg.context_length


# ------------------------------------------------ determinism & stability ---
def test_deterministic_init_and_forward(tiny_model_cfg_dict):
    a = _model(tiny_model_cfg_dict, seed=11, depth=4).eval()
    b = _model(tiny_model_cfg_dict, seed=11, depth=4).eval()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    x = torch.randint(0, 256, (1, 30))
    with torch.no_grad():
        assert torch.equal(a(x).logits, b(x).logits)


def test_residual_init_scaled_by_effective_depth(tiny_model_cfg_dict):
    d = copy.deepcopy(tiny_model_cfg_dict)
    d["model"].update(d_model=128, n_heads=2, ffn_dim=256)
    stds = {}
    for depth in (1, 8):
        m = _model(d, depth=depth, n_layers=2)
        stds[depth] = m.blocks[0].mlp.proj.weight.std().item()
    expected = 0.02 / (2 * 2) ** 0.5, 0.02 / (2 * 16) ** 0.5
    assert abs(stds[1] - expected[0]) / expected[0] < 0.05
    assert abs(stds[8] - expected[1]) / expected[1] < 0.05


def test_numerically_stable_at_depth_8(tiny_model_cfg_dict):
    """Deepest planned setting: finite activations, loss, grads; residual
    stream norm does not explode at init; bf16 autocast stays finite."""
    model = _model(tiny_model_cfg_dict, depth=8)
    x = torch.randint(0, 256, (2, 64))
    states = model.application_hidden_states(x)
    rms = [s.pow(2).mean().sqrt().item() for s in states]
    assert all(torch.isfinite(torch.tensor(rms)))
    assert rms[-1] < 10 * rms[0]
    loss = model(x, targets=x).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(model(x).logits.float()).all()


# -------------------------------------------------------------- checkpoints ---
def test_checkpoint_roundtrip(tiny_model_cfg_dict, tmp_path):
    cfg = _cfg(tiny_model_cfg_dict, depth=4, order="sequence")
    torch.manual_seed(0)
    model = build_model(cfg.model)
    opt = build_optimizer(model, 1e-3, 0.1, (0.9, 0.95))
    sched = build_scheduler(opt, 1, 10, 0.1)
    model(torch.randint(0, 256, (1, 8)), targets=torch.randint(0, 256, (1, 8))).loss.backward()
    opt.step()
    save_checkpoint(tmp_path / "ck", model, opt, sched, step=1, tokens_seen=8, config=cfg)
    meta = json.loads((tmp_path / "ck" / "checkpoint.json").read_text(encoding="utf-8"))
    assert meta["config"]["model"]["recurrence_depth"] == 4
    model2, cfg2, _ = build_model_from_checkpoint(tmp_path / "ck")
    assert isinstance(model2, DomovoiRecurrentTransformer)
    assert cfg2.model.recurrence_order == "sequence"
    assert model2.application_schedule() == model.application_schedule()
    model.eval(), model2.eval()
    x = torch.randint(0, 256, (1, 20))
    with torch.no_grad():
        assert torch.equal(model(x).logits, model2(x).logits)


def test_checkpoint_size_independent_of_depth(tiny_model_cfg_dict, tmp_path):
    from safetensors.torch import load_file
    sizes, keys = {}, {}
    for depth in (1, 8):
        cfg = _cfg(tiny_model_cfg_dict, depth=depth)
        model = build_model(cfg.model)
        opt = build_optimizer(model, 1e-3, 0.1, (0.9, 0.95))
        path = tmp_path / f"d{depth}"
        save_checkpoint(path, model, opt, build_scheduler(opt, 1, 10, 0.1),
                        step=0, tokens_seen=0, config=cfg)
        sizes[depth] = (path / "model.safetensors").stat().st_size
        keys[depth] = set(load_file(str(path / "model.safetensors")))
    assert keys[1] == keys[8]
    assert abs(sizes[1] - sizes[8]) < 64        # header text differs only


# ------------------------------------------------------ pipeline integration ---
def test_profile_inference_cpu_honest(tiny_model_cfg_dict):
    model = _model(tiny_model_cfg_dict, depth=4).eval()
    prof = profile_inference(model, _Tok(), torch.device("cpu"),
                             max_new_tokens=6, n_repeats=2)
    assert prof["decode_tok_per_s_median"] > 0
    assert prof["peak_vram_allocated_mb"] is None     # CPU: never estimated


def test_trainer_end_to_end_cpu(smoke_dataset, tmp_path):
    """The unmodified Trainer trains, checkpoints, resumes and generates with
    the recurrent architecture (CPU, synthetic data, tmp dir only)."""
    from domovoi.training import Trainer
    d = copy.deepcopy(smoke_dataset["config_dict"])
    d["model"].update(architecture="domovoi_recurrent_v1", recurrence_depth=2)
    d["output"]["experiments_dir"] = str(tmp_path / "experiments")
    cfg = config_from_dict(d)
    cfg.training.steps, cfg.training.checkpoint_interval = 2, 2
    exp_dir = tmp_path / "experiments" / "RUN"
    cpu = torch.device("cpu")
    trainer = Trainer(cfg, experiment_dir=exp_dir, device=cpu)
    assert isinstance(trainer.model, DomovoiRecurrentTransformer)
    summary = trainer.train()
    assert summary["status"] == "COMPLETED"
    model2, _, _ = build_model_from_checkpoint(exp_dir / "checkpoints" / "step_00000002")
    trainer.model.eval(), model2.eval()
    x = torch.tensor([trainer.tokenizer.encode("alpha beta")[:8]])
    with torch.no_grad():
        assert torch.allclose(trainer.model(x).logits, model2(x).logits, atol=1e-6)
    cfg.training.steps = 3
    resumed = Trainer(cfg, experiment_dir=exp_dir, resume=True, device=cpu,
                      extend_steps=True)      # longer schedule: explicit opt-in
    assert resumed.start_step == 2
    assert resumed.train()["steps_completed"] == 3
    r = generate(resumed.model, "alpha", resumed.tokenizer, max_new_tokens=4, temperature=0.0)
    assert r["n_new_tokens"] == 4
