"""Causal masking correctness.

Core property: position i's logits may depend only on tokens 0..i.  We
test this two ways:

1. prefix invariance — logits for a prefix are unchanged when the suffix is
   modified (strict causality);
2. KV-cache equivalence — incremental decoding with a cache must produce
   exactly the same logits as a full forward over the same sequence, both
   in variable-size chunks and single-token steps.
"""

import torch

from domovoi.models import build_model


def _tiny_model(tiny_cfg):
    torch.manual_seed(11)
    model = build_model(tiny_cfg.model)
    model.eval()
    return model


def test_prefix_invariance(tiny_cfg):
    model = _tiny_model(tiny_cfg)
    V = tiny_cfg.model.vocab_size
    x = torch.randint(0, V, (1, 32))
    x_mod = x.clone()
    x_mod[0, 20:] = (x_mod[0, 20:] + 7) % V   # change only the suffix

    with torch.no_grad():
        logits = model(x).logits
        logits_mod = model(x_mod).logits

    # logits at position i see only tokens 0..i — so logits [0..19] must be
    # identical (tokens 0..19 are unchanged)
    assert torch.allclose(logits[0, :20], logits_mod[0, :20], atol=1e-5), \
        "logits at position <20 changed when only the suffix changed: mask leak"
    assert not torch.allclose(logits[0, 25:], logits_mod[0, 25:], atol=1e-5), \
        "suffix logits unexpectedly identical after modifying input"


def test_kv_cache_matches_full_forward(tiny_cfg):
    model = _tiny_model(tiny_cfg)
    V = tiny_cfg.model.vocab_size
    x = torch.randint(0, V, (1, 24))

    with torch.no_grad():
        full = model(x).logits                      # (1, 24, V)

        # extend the sequence in variable-size chunks through the cache
        out = model(x[:, :5], use_cache=True)
        caches = out.caches
        chunks = [out.logits]
        pos = 5
        while pos < 24:
            take = min(3, 24 - pos)                # variable-size extensions
            out = model(x[:, pos:pos + take], use_cache=True, caches=caches)
            caches = out.caches
            chunks.append(out.logits)
            pos += take
    incremental = torch.cat(chunks, dim=1)

    assert incremental.shape == full.shape
    assert torch.allclose(full, incremental, atol=1e-4), \
        "KV-cache incremental logits differ from full forward"


def test_decode_loop_matches_full(tiny_cfg):
    """The exact generation shape: prefill, then single-token steps, feeding
    the true continuation — must match a full forward of the same tokens."""
    model = _tiny_model(tiny_cfg)
    V = tiny_cfg.model.vocab_size
    x = torch.randint(0, V, (1, 24))
    with torch.no_grad():
        ext = model(x).logits
        out = model(x[:, :16], use_cache=True)
        caches = out.caches
        step_logits = [out.logits[0, -1]]
        for t in range(16, 24):
            out = model(x[:, t:t + 1], use_cache=True, caches=caches)
            caches = out.caches
            step_logits.append(out.logits[0, -1])
    for t, sl in enumerate(step_logits):
        assert torch.allclose(sl, ext[0, 15 + t], atol=1e-4), \
            f"decode step {t} logits differ from full forward"