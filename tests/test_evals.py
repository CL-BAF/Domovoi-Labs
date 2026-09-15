"""Evaluation harness: MC scoring, output format, GSM8K extraction, IF
checkers, val perplexity."""

import json

import pytest
import torch

from domovoi.evals import MultipleChoiceEval, build_task, run_evals
from domovoi.evals.gsm8k import answers_match, extract_answer, gold_answer
from domovoi.models import build_model


def _write_mc_data(tmp_path, records):
    task_dir = tmp_path / "toy_mc"
    task_dir.mkdir(parents=True)
    task_dir.joinpath("data.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    task_dir.joinpath("meta.json").write_text(json.dumps({
        "source": "toy", "sha256": "0" * 64}), encoding="utf-8")
    return tmp_path


class _ByteTok:
    vocab_size = 256
    def encode(self, text):
        return [b for b in text.encode("utf-8")]
    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def test_mc_eval_perfect_score_when_choice_is_forced(tiny_cfg, tmp_path):
    """Model trained to copy: context 'A:' + choice letter tokens.
    A deliberately learnable toy check that scoring picks the right choice."""
    torch.manual_seed(0)
    model = build_model(tiny_cfg.model)
    model.eval()
    # monkeypatch-free shortcut: a model that assigns higher logprob to the
    # choice containing 'z' — we fake via deterministic bias toward 'z' by
    # training briefly on the pattern "the z"
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    xs = torch.tensor([_ByteTok().encode("the z")], dtype=torch.long)
    ys = torch.roll(xs, -1, dims=1)
    for _ in range(40):
        opt.zero_grad()
        out = model(xs, targets=ys)
        out.loss.backward()
        opt.step()
    model.eval()

    records = [
        {"id": "1", "context": "the", "choices": ["z", "q", "r"], "label": 0},
        {"id": "2", "context": "the", "choices": ["a", "z", "r"], "label": 1},
        {"id": "3", "context": "the", "choices": ["a", "b", "z"], "label": 2},
    ]
    _write_mc_data(tmp_path, records)
    task = MultipleChoiceEval("toy_mc", "Toy MC")
    result = task.run(model, _ByteTok(), torch.device("cpu"),
                      data_root=tmp_path)
    assert result["acc"] == 1.0
    assert result["chance"] == round(1 / 3, 4)
    assert result["n"] == 3
    assert result["data_sha256"] == "0" * 64


def test_mc_eval_records_and_formats(tiny_cfg, tmp_path):
    torch.manual_seed(0)
    model = build_model(tiny_cfg.model)
    model.eval()
    records = [
        {"id": "1", "context": "the", "choices": ["a", "b"], "label": 0},
    ]
    _write_mc_data(tmp_path, records)
    result = MultipleChoiceEval("toy_mc", "Toy MC").run(
        model, _ByteTok(), torch.device("cpu"), data_root=tmp_path)
    for key in ("task", "acc", "acc_norm", "chance", "n", "data_sha256"):
        assert key in result


def test_run_evals_missing_data_raises(tiny_cfg, tmp_path):
    model = build_model(tiny_cfg.model)
    with pytest.raises(FileNotFoundError):
        MultipleChoiceEval("nonexistent", "X").run(
            model, _ByteTok(), torch.device("cpu"), data_root=tmp_path)


def test_build_task_registry():
    for name in ("val_ppl", "hellaswag", "arc_challenge", "piqa",
                 "winogrande", "gsm8k", "if_mini"):
        task = build_task(name)
        assert task.name == name
    with pytest.raises(ValueError):
        build_task("not_a_task")


def test_gsm8k_answer_extraction():
    assert extract_answer("blah 12 blah #### 42") == "42"
    assert extract_answer("the answer is 7.") == "7"
    assert extract_answer("no numbers here") is None
    assert extract_answer("so 1,234 people") == "1234"
    assert gold_answer("reasoning...\n#### 42") == "42"
    assert gold_answer("#### 3.14") == "3.14"
    assert answers_match("72.0", "72") is True
    assert answers_match("71", "72") is False
    assert answers_match(None, "72") is False


def test_if_mini_checkers_are_valid():
    # the checker set must be non-empty and versioned
    from domovoi.evals.if_mini import _CONSTRAINTS, IF_MINI_VERSION
    assert len(_CONSTRAINTS) >= 15
    assert IF_MINI_VERSION == "if-mini-v1"
    # sanity: the checkers actually validate their target behavior
    three = [c for p, c, d in _CONSTRAINTS if d == "exactly_3_words"][0]
    assert three("one two three") and not three("one two")


def test_val_ppl_runs(smoke_dataset, tmp_path):
    from domovoi.config import config_from_dict
    cfg = config_from_dict(smoke_dataset["config_dict"])
    torch.manual_seed(0)
    model = build_model(cfg.model)
    model.eval()
    task = build_task("val_ppl")
    result = task.run(model, _ByteTok(), torch.device("cpu"),
                     training_config=cfg, n_batches=2)
    assert result["val_loss"] > 0
    assert result["val_ppl"] > 1.0
    assert result["dataset"] == "smoke"