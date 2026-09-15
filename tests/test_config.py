"""Config loading and validation."""

import pytest

from domovoi.config import (ConfigError, clone_config, config_from_dict,
                            load_config)


def _base_dict():
    return {
        "experiment": {"name": "t"},
        "model": {
            "architecture": "baseline_transformer", "vocab_size": 256,
            "context_length": 64, "d_model": 32, "n_layers": 2, "n_heads": 2,
            "ffn_dim": 64, "dropout": 0.0, "tie_embeddings": True,
        },
        "dataset": {"name": "smoke", "train_fraction": 0.9},
        "training": {"steps": 3, "batch_size": 2, "context_length": 32,
                     "precision": "fp32", "learning_rate": 1e-3},
        "output": {"experiments_dir": "experiments"},
    }


def test_valid_dict_loads():
    cfg = config_from_dict(_base_dict())
    assert cfg.model.d_model == 32
    assert cfg.training.betas == (0.9, 0.95)


def test_missing_section_raises():
    d = _base_dict()
    del d["training"]
    with pytest.raises(ConfigError, match="training"):
        config_from_dict(d)


def test_unknown_key_rejected():
    d = _base_dict()
    d["model"]["layers"] = 3
    with pytest.raises(ConfigError, match="unknown keys"):
        config_from_dict(d)


@pytest.mark.parametrize("field,value,pat", [
    ("d_model", 33, "divisible"),          # 33 % 2 != 0
    ("vocab_size", 128, "256"),
    ("context_length", 9999, "context_length"),
])
def test_model_validation(field, value, pat):
    d = _base_dict()
    d["model"][field] = value
    with pytest.raises(ConfigError, match=pat):
        config_from_dict(d)


def test_training_validation():
    d = _base_dict()
    d["training"]["precision"] = "int8"
    with pytest.raises(ConfigError, match="precision"):
        config_from_dict(d)
    d["training"]["precision"] = "fp32"
    d["training"]["steps"] = -1
    with pytest.raises(ConfigError, match="steps"):
        config_from_dict(d)


def test_yaml_load_and_copy(tmp_path):
    import yaml
    src = tmp_path / "c.yaml"
    raw = _base_dict()
    raw["dataset"]["train_fraction"] = 0.9
    src.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(src)
    assert cfg.source_path == str(src)
    cfg.save_copy(tmp_path / "copy.yaml")
    assert yaml.safe_load((tmp_path / "copy.yaml").read_text(encoding="utf-8")) == raw


def test_clone_config_override():
    cfg = config_from_dict(_base_dict())
    cfg2 = clone_config(cfg, model={"n_layers": 4})
    assert cfg2.model.n_layers == 4
    assert cfg.model.n_layers == 2  # original untouched
    assert cfg2.model.d_model == 32


def test_status_validation():
    d = _base_dict()
    d["experiment"]["status"] = "WONDERFUL"
    with pytest.raises(ConfigError, match="status"):
        config_from_dict(d)