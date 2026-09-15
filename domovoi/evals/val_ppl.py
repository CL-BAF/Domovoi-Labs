"""Validation cross-entropy / perplexity on the fixed held-out corpus.

Uses the same fixed deterministic batches as the trainer's periodic
validation, so eval numbers and training-time numbers agree exactly.
"""

from __future__ import annotations

import torch

from ..datasets import PackedTokenData, load_processed_meta


class ValPPLEval:
    name = "val_ppl"

    def run(self, model, tokenizer, device, training_config=None,
            n_batches: int = 50, **_) -> dict:
        if training_config is None:
            raise ValueError("val_ppl requires the training config.")
        cfg = training_config
        meta = load_processed_meta(cfg.dataset.processed_dir)
        val_data = PackedTokenData(meta["files"]["val_bin"])
        model.eval()
        gen = torch.Generator()
        gen.manual_seed(0)
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = val_data.get_batch(cfg.training.batch_size,
                                          cfg.training.context_length,
                                          device, gen)
                _, loss = model(x, targets=y)
                losses.append(loss.item())
        mean = sum(losses) / len(losses)
        ppl = float(torch.exp(torch.tensor(mean)).item())
        return {
            "task": "val_ppl",
            "display_name": "Validation perplexity",
            "metric": "cross_entropy",
            "val_loss": round(mean, 6),
            "val_ppl": round(ppl, 4),
            "n_batches": n_batches,
            "dataset": meta["dataset"]["name"],
            "dataset_version": meta["dataset"]["version"],
        }