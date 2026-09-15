"""Shared fixtures: a tiny model config and a synthetic smoke dataset.

All tests run on CPU and are fast (< a few seconds each).
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from domovoi.config import config_from_dict, load_config


@pytest.fixture()
def tiny_model_cfg_dict() -> dict:
    """A complete config dict for a ~100k-parameter model, CPU friendly."""
    return {
        "experiment": {"name": "test", "notes": "", "status": "BASELINE"},
        "model": {
            "architecture": "baseline_transformer",
            "vocab_size": 256,
            "context_length": 64,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 2,
            "ffn_dim": 64,
            "dropout": 0.0,
            "tie_embeddings": True,
        },
        "tokenizer": {"type": "byte_bpe", "path": "", "version": "byte-bpe-v1"},
        "dataset": {
            "name": "smoke",
            "source_url": "synthetic",
            "version": "smoke-v1",
            "train_fraction": 0.9,
            "processed_dir": "",
        },
        "training": {
            "steps": 3,
            "batch_size": 2,
            "gradient_accumulation": 1,
            "context_length": 32,
            "learning_rate": 1e-3,
            "min_lr_ratio": 0.1,
            "weight_decay": 0.1,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "warmup_steps": 1,
            "precision": "fp32",
            "seed": 42,
            "eval_interval": 3,
            "eval_batches": 2,
            "checkpoint_interval": 2,
            "log_interval": 1,
        },
        "output": {"experiments_dir": "experiments", "experiment_id": None},
    }


@pytest.fixture()
def tiny_cfg(tiny_model_cfg_dict):
    return config_from_dict(tiny_model_cfg_dict)


@pytest.fixture()
def smoke_dataset(tmp_path, tiny_model_cfg_dict):
    """Fully prepared synthetic dataset in a tmp dir + patched config dict
    with the correct processed_dir / tokenizer path."""
    from domovoi.datasets import prepare_dataset

    processed_dir = tmp_path / "data" / "processed" / "smoke"
    tok_path = tmp_path / "data" / "tokenizers" / "smoke_bpe.json"
    cfg_dict = dict(tiny_model_cfg_dict)
    cfg_dict["dataset"]["processed_dir"] = str(processed_dir)
    cfg_dict["tokenizer"]["path"] = str(tok_path)
    meta = prepare_dataset(
        name="smoke", source_url="synthetic", version="smoke-v1",
        train_fraction=0.9, processed_dir=processed_dir,
        tokenizer_path=tok_path, vocab_size=300,
    )
    # The trained vocab can be smaller than the requested 300 (tiny corpus
    # runs out of merge candidates).  The model must be built with the
    # actual size or verify_vocab in the Trainer fails.
    cfg_dict["model"]["vocab_size"] = meta["tokens"]["vocab_size"]
    return {"meta": meta, "config_dict": cfg_dict, "tmp_path": tmp_path}