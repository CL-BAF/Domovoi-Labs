"""Profiling: parameter counts, checkpoint size, inference speed fields,
VRAM honesty on CPU."""

import torch

from domovoi.inference import generate
from domovoi.models import build_model
from domovoi.profiling import (count_parameters, profile_inference,
                               vram_stats)


class _Tok:
    vocab_size = 256
    def encode(self, text):
        return [b for b in text.encode("utf-8")]
    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def test_count_parameters(tiny_cfg):
    model = build_model(tiny_cfg.model)
    p = count_parameters(model)
    assert p["total"] == model.num_params(non_embedding=False)
    assert p["trainable"] == p["total"]       # all trainable by default
    assert p["active_per_token"] == p["total"]  # dense baseline


def test_frozen_params_reported(tiny_cfg):
    model = build_model(tiny_cfg.model)
    model.wpe.weight.requires_grad_(False)
    p = count_parameters(model)
    assert p["trainable"] < p["total"]


def test_vram_stats_cpu_honest():
    stats = vram_stats(torch.device("cpu"))
    assert stats["cuda"] is False
    assert stats["peak_allocated_mb"] is None
    assert stats["peak_reserved_mb"] is None


def test_profile_inference_fields(tiny_cfg):
    torch.manual_seed(0)
    model = build_model(tiny_cfg.model)
    model.eval()
    prof = profile_inference(model, _Tok(), torch.device("cpu"),
                              max_new_tokens=8, n_repeats=2)
    for key in ("prefill_tok_per_s_median", "decode_tok_per_s_median",
                "ttft_s_median", "n_prompt_tokens", "max_new_tokens"):
        assert key in prof
    assert prof["decode_tok_per_s_median"] > 0
    # CPU: VRAM must be null, never guessed
    assert prof["peak_vram_allocated_mb"] is None
    assert prof["batch_size"] == 1


def test_generate_reports_speeds(tiny_cfg):
    model = build_model(tiny_cfg.model)
    model.eval()
    r = generate(model, "hello", _Tok(), max_new_tokens=5, temperature=0.0)
    assert r["ttft_s"] > 0
    assert r["decode_tok_per_s"] > 0