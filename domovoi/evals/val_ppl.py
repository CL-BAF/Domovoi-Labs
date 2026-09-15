"""Validation cross-entropy / perplexity on the fixed held-out corpus.

Batches come from the same fixed-seed generator as the trainer's periodic
validation (`fixed_val_generator`), and `n_batches` defaults to the training
config's `eval_batches` — so on the default paths the eval scores exactly
the same batches the trainer validated on, in fp32.

Honest caveat: the trainer's periodic validation runs under autocast at the
training precision.  Exact numeric agreement with training-time val loss
therefore holds for fp32 training; under bf16/fp16 the trainer's number is
computed at lower precision and will differ slightly from this eval even on
identical batches.  The eval result records `precision: "fp32"` and its
`n_batches` so the two numbers can never be mistaken for one measurement.
"""

from __future__ import annotations

from typing import Optional

import torch

from ..datasets import PackedTokenData, fixed_val_generator, load_processed_meta


class ValPPLEval:
    name = "val_ppl"

    def run(self, model, tokenizer, device, training_config=None,
            n_batches: Optional[int] = None, **_) -> dict:
        if training_config is None:
            raise ValueError("val_ppl requires the training config.")
        cfg = training_config
        # default to the trainer's periodic-validation batch count so the
        # stored numbers are directly comparable with training-time val loss
        n_batches = n_batches if n_batches is not None \
            else cfg.training.eval_batches
        meta = load_processed_meta(cfg.dataset.processed_dir)
        val_data = PackedTokenData(meta["files"]["val_bin"])
        model.eval()
        gen = fixed_val_generator(device)
        losses = []
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = val_data.get_batch(cfg.training.batch_size,
                                          cfg.training.context_length,
                                          device, gen)
                out = model(x, targets=y)
                losses.append(out.loss.item())
        mean = sum(losses) / len(losses)
        ppl = float(torch.exp(torch.tensor(mean)).item())
        return {
            "task": "val_ppl",
            "display_name": "Validation perplexity",
            "metric": "cross_entropy",
            "val_loss": round(mean, 6),
            "val_ppl": round(ppl, 4),
            "n_batches": n_batches,
            "precision": "fp32",
            "dataset": meta["dataset"]["name"],
            "dataset_version": meta["dataset"]["version"],
        }