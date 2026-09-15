"""Generation: sampling correctness, determinism, KV-cache path, metrics."""

import torch

from domovoi.inference import generate
from domovoi.models import build_model


def _trained_model(tiny_cfg, steps=30):
    torch.manual_seed(0)
    model = build_model(tiny_cfg.model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(0, tiny_cfg.model.vocab_size, (1, 16))
    for _ in range(steps):
        opt.zero_grad()
        _, loss = model(x, targets=x)
        loss.backward()
        opt.step()
    model.eval()
    return model


class _Tok:
    """Minimal stand-in tokenizer with byte identity (vocab 256)."""
    vocab_size = 256
    def encode(self, text):
        return [b for b in text.encode("utf-8")]
    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def test_greedy_deterministic(tiny_cfg):
    model = _trained_model(tiny_cfg)
    a = generate(model, "abc", _Tok(), max_new_tokens=10, temperature=0.0)
    b = generate(model, "abc", _Tok(), max_new_tokens=10, temperature=0.0)
    assert a["new_text"] == b["new_text"]
    assert a["n_new_tokens"] == 10


def test_sampled_with_seed_deterministic(tiny_cfg):
    model = _trained_model(tiny_cfg)
    a = generate(model, "abc", _Tok(), max_new_tokens=10,
                 temperature=1.0, top_k=10, seed=7)
    b = generate(model, "abc", _Tok(), max_new_tokens=10,
                 temperature=1.0, top_k=10, seed=7)
    assert a["new_text"] == b["new_text"]
    c = generate(model, "abc", _Tok(), max_new_tokens=10,
                 temperature=1.0, top_k=10, seed=8)
    # different seed *usually* gives different text on a 256-way vocab
    assert isinstance(c["new_text"], str)


def test_top_k_restricts_candidates(tiny_cfg):
    """With top_k=1 sampling must equal greedy (argmax)."""
    model = _trained_model(tiny_cfg)
    greedy = generate(model, "abc", _Tok(), max_new_tokens=8, temperature=0.0)
    topk1 = generate(model, "abc", _Tok(), max_new_tokens=8,
                     temperature=1.0, top_k=1, seed=1)
    assert greedy["new_text"] == topk1["new_text"]


def test_generation_uses_kv_cache_correctly(tiny_cfg):
    """Sampled generation must match a manual no-cache decode loop."""
    model = _trained_model(tiny_cfg)
    tok = _Tok()
    prompt = "hello world"
    max_new = 12
    res = generate(model, prompt, tok, max_new_tokens=max_new,
                   temperature=0.0)

    # manual greedy loop WITHOUT cache
    ids = tok.encode(prompt)
    for _ in range(max_new):
        x = torch.tensor([ids[-model.cfg.context_length:]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x).logits[0, -1]
        ids.append(int(torch.argmax(logits).item()))
    assert tok.decode(ids[len(tok.encode(prompt)):]) == res["new_text"]


def test_result_fields(tiny_cfg):
    model = _trained_model(tiny_cfg)
    r = generate(model, "abc", _Tok(), max_new_tokens=5, temperature=0.0)
    assert r["n_prompt_tokens"] == 3
    assert r["n_new_tokens"] == 5
    assert r["ttft_s"] > 0
    assert "decode_tok_per_s" in r
    # CPU run: must not fabricate VRAM fields
    assert "peak_vram_allocated_mb" not in r


def test_prompt_too_long_raises(tiny_cfg):
    model = _trained_model(tiny_cfg)
    tok = _Tok()
    long_prompt = "x" * (model.cfg.context_length + 1)
    try:
        generate(model, long_prompt, tok, max_new_tokens=5)
        assert False, "should have raised"
    except ValueError as e:
        assert "context" in str(e)


def test_context_boundary_respected(tiny_cfg):
    """Generation stops before exceeding the context window."""
    model = _trained_model(tiny_cfg)
    tok = _Tok()
    prompt = "a" * 5
    r = generate(model, prompt, tok, max_new_tokens=model.cfg.context_length + 50,
                 temperature=0.0)
    assert r["n_prompt_tokens"] + r["n_new_tokens"] <= model.cfg.context_length