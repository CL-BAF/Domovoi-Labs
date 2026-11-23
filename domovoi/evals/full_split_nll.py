"""Token-weighted NLL on a versioned, document-packed validation or test split."""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..datasets import PackedTokenData, load_processed_meta
from ..datasets.tokenizer import sha256_file


def _document_windows(data, ends, context):
    start = 0
    for document, end in enumerate(ends):
        for left in range(start, end - 1, context):
            right = min(left + context + 1, end)
            yield document, torch.from_numpy(data[left:right].astype(np.int64))
        start = end


def _bootstrap_interval(losses, counts):
    generator = np.random.default_rng(0)
    draws = []
    for _ in range(1000):
        docs = generator.integers(len(counts), size=len(counts))
        tokens = counts[docs].sum()
        if tokens:
            draws.append(losses[docs].sum() / tokens)
    return np.quantile(draws, [0.025, 0.975]).tolist()


class FullSplitNLLEval:
    def __init__(self, split: str, early_exits: bool = False):
        self.split = split
        self.early_exits = early_exits

    @torch.inference_mode()
    def run(self, model, tokenizer, device, training_config, checkpoint_meta=None):
        cfg = training_config
        split = self.split
        meta = load_processed_meta(cfg.dataset.processed_dir)
        if meta.get("tokenizer", {}).get("sha256") != tokenizer.sha256:
            raise ValueError("Held-out split tokenizer hash differs from the evaluation tokenizer.")
        files = meta["files"]
        try:
            bin_path = Path(files[f"{split}_bin"])
            ends_path = Path(files[f"{split}_document_ends"])
            count = meta["tokens"][split]
        except KeyError as exc:
            raise ValueError(f"Missing packed {split} split or document boundaries: {exc}") from exc
        data = PackedTokenData(bin_path)
        boundaries = ends_path.read_bytes()
        ends = json.loads(boundaries)
        if (not isinstance(ends, list) or not ends
                or any(type(end) is not int for end in ends)
                or any(lo >= hi for lo, hi in zip([0, *ends[:-1]], ends))
                or ends[-1] != count or count != len(data)):
            raise ValueError(f"{split} document ends must be strictly increasing token offsets "
                             "ending at the packed split size.")
        bin_hash = sha256_file(bin_path)
        provenance = (checkpoint_meta or {}).get("data_provenance") or {}
        if split == "val" and provenance.get("val_bin_sha256", bin_hash) != bin_hash:
            raise ValueError("Validation packed data differs from the checkpoint.")
        if not all(p.dtype == torch.float32 for p in model.parameters()):
            raise ValueError("Full-split NLL requires FP32 model weights.")
        if self.early_exits and not hasattr(model, "application_hidden_states"):
            raise ValueError("Early-exit NLL requires a recurrent checkpoint.")
        context = cfg.training.context_length
        batch_size = cfg.training.batch_size
        counts = np.zeros(len(ends), dtype=np.int64)
        losses = np.zeros(len(ends), dtype=np.float64)
        exit_losses = None
        exit_norms = None
        batch = []
        was_training = model.training
        model.eval()
        try:
            with torch.autocast(device_type=device.type, enabled=False):
                for window in _document_windows(data.data, ends, context):
                    batch.append(window)
                    if len(batch) == batch_size:
                        self._score(model, device, batch, losses, counts)
                        if self.early_exits:
                            exit_losses, exit_norms = self._score_exits(
                                model, device, batch, exit_losses, exit_norms)
                        batch.clear()
                if batch:
                    self._score(model, device, batch, losses, counts)
                    if self.early_exits:
                        exit_losses, exit_norms = self._score_exits(
                            model, device, batch, exit_losses, exit_norms)
        finally:
            model.train(was_training)
        n_tokens = int(counts.sum())
        if not n_tokens:
            raise ValueError(f"{split} contains no within-document next-token targets.")
        nll = float(losses.sum() / n_tokens)
        if not math.isfinite(nll):
            raise ValueError(f"Non-finite full-split NLL on {split}.")
        result = {
            "task": f"{split}_nll",
            "display_name": f"Full {split} split NLL",
            "metric": "nats_per_token",
            "nll": nll,
            "n": n_tokens,
            "n_documents": len(ends),
            "context_length": context,
            "precision": "fp32",
            "bootstrap_95_interval": _bootstrap_interval(losses, counts),
            "dataset": meta["dataset"]["name"],
            "dataset_version": meta["dataset"]["version"],
            "packed_sha256": bin_hash,
            "document_ends_sha256": sha256_file(ends_path),
            "tokenizer_sha256": tokenizer.sha256,
        }
        if exit_losses is not None:
            result["application_nll"] = (exit_losses / n_tokens).tolist()
            result["application_mean_residual_l2"] = (exit_norms / n_tokens).tolist()
        return result

    @staticmethod
    def _inputs(batch, device):
        x = torch.nn.utils.rnn.pad_sequence([ids[:-1] for _, ids in batch], batch_first=True)
        y = torch.nn.utils.rnn.pad_sequence(
            [ids[1:] for _, ids in batch], batch_first=True, padding_value=-100)
        return x.to(device), y.to(device)

    @staticmethod
    def _loss(logits, y):
        return F.cross_entropy(logits.transpose(1, 2), y, reduction="none")

    def _score(self, model, device, batch, losses, counts):
        x, y = self._inputs(batch, device)
        values = self._loss(model(x).logits, y).sum(dim=1, dtype=torch.float64).tolist()
        for (doc, ids), loss in zip(batch, values):
            losses[doc] += loss
            counts[doc] += len(ids) - 1

    def _score_exits(self, model, device, batch, losses, norms):
        x, y = self._inputs(batch, device)
        states = model.application_hidden_states(x)
        if losses is None:
            losses = np.zeros(len(states), dtype=np.float64)
            norms = np.zeros(len(states), dtype=np.float64)
        for i, state in enumerate(states):
            losses[i] += self._loss(model.lm_head(model.ln_f(state)), y).sum(
                dtype=torch.float64).item()
            norms[i] += state.norm(dim=-1).masked_fill(y == -100, 0).sum(
                dtype=torch.float64).item()
        return losses, norms
