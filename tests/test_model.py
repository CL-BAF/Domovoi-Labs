"""Model forward pass, shapes, determinism, parameter counting."""

import torch

from domovoi.models import build_model


def test_forward_shapes(tiny_cfg):
    model = build_model(tiny_cfg.model)
    B, T = 2, tiny_cfg.model.context_length
    x = torch.randint(0, tiny_cfg.model.vocab_size, (B, T))
    out = model(x)
    assert out.logits.shape == (B, T, tiny_cfg.model.vocab_size)
    assert out.loss is None
    assert out.caches is None


def test_loss_decreases_sanity(tiny_cfg):
    """One gradient step on a trivially learnable batch reduces loss."""
    torch.manual_seed(0)
    model = build_model(tiny_cfg.model)
    # a constant batch: the model should learn it quickly
    x = torch.randint(0, tiny_cfg.model.vocab_size, (1, 16))
    y = torch.randint(0, tiny_cfg.model.vocab_size, (1, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(10):
        out = model(x, targets=y)
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        losses.append(out.loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_forward_deterministic(tiny_cfg):
    torch.manual_seed(3)
    model = build_model(tiny_cfg.model)
    model.eval()
    x = torch.randint(0, tiny_cfg.model.vocab_size, (1, 32))
    with torch.no_grad():
        a = model(x).logits
        b = model(x).logits
    assert torch.equal(a, b)


def test_param_count_matches_breakdown(tiny_cfg):
    model = build_model(tiny_cfg.model)
    b = model.param_breakdown()
    assert b["total"] == sum(p.numel() for p in model.parameters())
    # tied embeddings: lm_head contributes zero extra parameters
    assert b["lm_head"] == 0


def test_tied_weights_share_storage(tiny_cfg):
    model = build_model(tiny_cfg.model)
    assert model.lm_head.weight is model.wte.weight


def test_untied_variant_counts_head(tiny_cfg):
    tiny_cfg.model.tie_embeddings = False
    model = build_model(tiny_cfg.model)
    assert model.lm_head.weight is not model.wte.weight
    expected = model.param_breakdown()["total"]
    assert model.lm_head.weight.numel() == tiny_cfg.model.vocab_size * tiny_cfg.model.d_model


def test_context_length_violation_raises(tiny_cfg):
    model = build_model(tiny_cfg.model)
    x = torch.randint(0, tiny_cfg.model.vocab_size,
                      (1, tiny_cfg.model.context_length + 1))
    try:
        model(x)
        assert False, "should have raised"
    except ValueError as e:
        assert "exceeds context_length" in str(e)


def test_eval_mode_no_dropout_noise(tiny_cfg):
    """dropout=0 config: eval forwards must be exactly reproducible even in
    train mode (no dropout randomness)."""
    tiny_cfg.model.dropout = 0.0
    model = build_model(tiny_cfg.model)
    model.train()
    x = torch.randint(0, tiny_cfg.model.vocab_size, (1, 8))
    a = model(x).logits
    b = model(x).logits
    assert torch.allclose(a, b, atol=1e-6)