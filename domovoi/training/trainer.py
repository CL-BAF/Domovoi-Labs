"""Training loop.

The Trainer owns one experiment run end to end:

    build model → resume if asked → train with grad accumulation, clipping,
    mixed precision, periodic validation and checkpointing → write metrics
    and a final summary.json.

Everything is driven by a domovoi.config.Config; the trainer takes no
experiment-specific constants.

Resume is bit-exact: model + optimizer + scheduler + GradScaler + step + all
RNG states (including the data-batch generator) are restored, so continuing
from a checkpoint produces the same trajectory as an uninterrupted run.
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Optional

import torch

from ..config import Config
from ..datasets import (DomovoiTokenizer, PackedTokenData, fixed_val_generator,
                       load_processed_meta, verify_vocab)
from ..models import build_model
from ..utils import (detect_device, resolve_precision, set_seed)
from ..utils.seeding import capture_rng_state, restore_rng_state
from .checkpoint import CheckpointManager, load_checkpoint
from .optim import build_optimizer, build_scheduler, current_lr


class Trainer:
    def __init__(
        self,
        config: Config,
        experiment_dir: str | pathlib.Path | None = None,
        device: Optional[torch.device] = None,
        resume: bool = False,
    ):
        self.config = config
        t = config.training
        self.device = device if device is not None else detect_device()
        self.experiment_dir = pathlib.Path(experiment_dir) if experiment_dir else None

        # -- seeding before anything else -------------------------------
        set_seed(t.seed)

        # -- data --------------------------------------------------------
        self.data_meta = load_processed_meta(config.dataset.processed_dir)
        self.tokenizer = DomovoiTokenizer(config.tokenizer.path)
        verify_vocab(self.tokenizer, config.model.vocab_size)
        # The packed bins were tokenized by whatever tokenizer file existed at
        # prepare time; meta.json records its sha256.  If the file at
        # tokenizer.path has since been retrained or replaced, every token id
        # in train.bin/val.bin would be misinterpreted — refuse to train.
        packed_sha = (self.data_meta.get("tokenizer") or {}).get("sha256")
        if packed_sha and self.tokenizer.sha256 != packed_sha:
            raise RuntimeError(
                f"Tokenizer at {config.tokenizer.path} (sha256 "
                f"{self.tokenizer.sha256}) is not the tokenizer that packed "
                f"{config.dataset.processed_dir} (sha256 {packed_sha}). "
                f"Training on these bins would corrupt the token-id mapping. "
                f"Re-run scripts/prepare_data.py --force (or point "
                f"tokenizer.path at the original file).")
        self.train_data = PackedTokenData(self.data_meta["files"]["train_bin"])
        self.val_data = PackedTokenData(self.data_meta["files"]["val_bin"])
        # the batch-offset generator is a *separate* deterministic stream,
        # saved and restored across checkpoints
        self.data_generator = torch.Generator()
        self.data_generator.manual_seed(t.seed + 1)

        # -- model --------------------------------------------------------
        self.model = build_model(config.model).to(self.device)
        self.n_params = self.model.num_params(non_embedding=False)
        self.n_params_non_embedding = self.model.num_params(non_embedding=True)

        # -- optimization --------------------------------------------------
        self.optimizer = build_optimizer(
            self.model, t.learning_rate, t.weight_decay, t.betas)
        self.scheduler = build_scheduler(
            self.optimizer, t.warmup_steps, t.steps, t.min_lr_ratio)

        # -- precision ----------------------------------------------------
        amp_dtype = resolve_precision(t.precision, self.device)
        self.amp_dtype = amp_dtype
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp_dtype == "float16"))
        self._autocast = lambda: torch.autocast(
            device_type=self.device.type,
            dtype=getattr(torch, amp_dtype) if amp_dtype else torch.float32,
            enabled=amp_dtype is not None,
        )

        # -- checkpoints / metrics ------------------------------------------
        self.ckpt = CheckpointManager(self.experiment_dir / "checkpoints") \
            if self.experiment_dir else None

        # -- resume ----------------------------------------------------------
        self.start_step = 0
        self.tokens_seen = 0
        if resume and self.ckpt is not None and self.ckpt.latest() is not None:
            meta, rng = load_checkpoint(
                self.ckpt.latest(), self.model, self.optimizer, self.scheduler,
                scaler=self.scaler, restore_rng=False)
            self.start_step = meta["step"]
            self.tokens_seen = meta["tokens_seen"]
            ckpt_steps = meta.get("config", {}).get("training", {}).get("steps")
            if ckpt_steps is not None and ckpt_steps != t.steps:
                print(f"[trainer] WARNING: resuming at step {self.start_step} "
                      f"with training.steps={t.steps}, but the checkpoint was "
                      f"saved from a run configured for steps={ckpt_steps}. "
                      f"The LR schedule differs from the original run; the "
                      f"trajectory will NOT match an uninterrupted run.")
            gen_state = rng.get("extra", {}).get("data_generator")
            if gen_state is not None:
                self.data_generator.set_state(gen_state.to(torch.uint8))
            restore_rng_state(rng.get("standard") or {})
        elif resume:
            # nothing to resume from: say so loudly instead of silently
            # training from step 0 (hours of compute restarted without a trace)
            where = (f"under {self.ckpt.root}" if self.ckpt is not None
                     else "(no experiment dir given)")
            print(f"[trainer] WARNING: resume requested but no checkpoint "
                  f"found {where}; training starts from step 0.", flush=True)

        # -- VRAM tracking ---------------------------------------------------
        self._vram_supported = self.device.type == "cuda"
        if self._vram_supported:
            torch.cuda.reset_peak_memory_stats(self.device)

    # ------------------------------------------------------------------
    def tokens_per_step(self) -> int:
        t = self.config.training
        return t.batch_size * t.context_length * t.gradient_accumulation

    # ------------------------------------------------------------------
    def get_train_batch(self):
        t = self.config.training
        return self.train_data.get_batch(
            t.batch_size, t.context_length, self.device, self.data_generator)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, n_batches: int) -> dict:
        """Average validation loss over `n_batches` fixed deterministic
        batches (same offsets every call, so val loss is comparable across
        steps/checkpoints/runs)."""
        t = self.config.training
        self.model.eval()
        gen = fixed_val_generator(self.device)
        losses = []
        with self._autocast():
            for _ in range(n_batches):
                x, y = self.val_data.get_batch(
                    t.batch_size, t.context_length, self.device, gen)
                out = self.model(x, targets=y)
                losses.append(out.loss.item())
        self.model.train()
        val_loss = sum(losses) / len(losses)
        return {
            "val_loss": val_loss,
            "val_ppl": float(torch.exp(torch.tensor(val_loss)).item()),
            "n_batches": n_batches,
        }

    # ------------------------------------------------------------------
    def _write_metrics(self, name: str, record: dict) -> None:
        if self.experiment_dir is None:
            return
        path = self.experiment_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _save_checkpoint(self, step: int, extra: dict | None = None) -> None:
        if self.ckpt is None:
            return
        self.ckpt.save(
            step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            tokens_seen=self.tokens_seen,
            config=self.config,
            extra=extra,
            rng_extra={"data_generator": self.data_generator.get_state()},
        )

    def vram_stats(self) -> dict:
        """Real CUDA memory statistics, not guesses.  Values are None when
        CUDA is unavailable so downstream reports never fabricate numbers."""
        if not self._vram_supported:
            return {"peak_allocated_mb": None, "peak_reserved_mb": None,
                    "cuda": False}
        return {
            "peak_allocated_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 2),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(self.device) / (1024 * 1024), 2),
            "cuda": True,
        }

    # ------------------------------------------------------------------
    def train(self) -> dict:
        """Run to completion.  Returns the summary dict (also written to
        summary.json).  On Ctrl+C, saves a rescue checkpoint and exits with
        status INTERRUPTED."""
        t = self.config.training
        started = time.time()
        # tokens trained in THIS session (tokens_seen is cumulative across
        # resumes; wall clock is not, so throughput must compare like with like)
        session_start_tokens = self.tokens_seen
        self.model.train()
        log_start_time = time.time()
        log_start_tokens = self.tokens_seen
        running_loss = 0.0
        running_batches = 0
        status = "COMPLETED"
        last_step = self.start_step  # last fully completed optimizer step

        # RNG state to resume from `last_step`: captured before the loop and
        # refreshed the moment a step completes.  An interrupt at any point
        # (even mid-step, after the batch draw consumed generator state)
        # rolls back to the last clean boundary so resume replays the
        # interrupted step exactly.
        resume_rng = (capture_rng_state(),
                      self.data_generator.get_state().clone())

        try:
            for step in range(self.start_step, t.steps):
                lr = current_lr(self.optimizer)

                # -- gradient accumulation ---------------------------------
                self.optimizer.zero_grad(set_to_none=True)
                micro_losses = []
                for _ in range(t.gradient_accumulation):
                    x, y = self.get_train_batch()
                    with self._autocast():
                        out = self.model(x, targets=y)
                    scaled = out.loss / t.gradient_accumulation
                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                    micro_losses.append(out.loss.item())

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), t.grad_clip)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()

                self.tokens_seen += self.tokens_per_step()

                # The optimizer update for this step is complete and the
                # weights/accounting/RNG are mutually consistent RIGHT NOW.
                # Everything below (logging, validation, checkpoint save)
                # neither changes weights nor advances the batch stream, so
                # the rescue checkpoint may safely reference this step from
                # here on.  An interrupt anywhere below — including inside
                # periodic validation, the widest window in the loop — replays
                # from a consistent boundary.
                last_step = step + 1
                resume_rng = (capture_rng_state(),
                              self.data_generator.get_state().clone())

                batch_loss = sum(micro_losses) / len(micro_losses)
                running_loss += batch_loss
                running_batches += 1
                now = time.time()

                # -- periodic logging ----------------------------------------
                if (step + 1) % t.log_interval == 0 or step == t.steps - 1:
                    window_tokens = self.tokens_seen - log_start_tokens
                    window_s = max(1e-9, now - log_start_time)
                    record = {
                        "step": step + 1,
                        "train_loss": round(running_loss / max(1, running_batches), 6),
                        "lr": round(lr, 8),
                        "grad_norm": round(float(grad_norm), 4),
                        "tokens_seen": self.tokens_seen,
                        "tok_per_s": round(window_tokens / window_s, 1),
                        "elapsed_s": round(now - started, 1),
                    }
                    self._write_metrics("metrics.jsonl", record)
                    print(
                        f"step {step + 1:6d}/{t.steps} | loss {record['train_loss']:.4f} "
                        f"| lr {lr:.2e} | tok/s {record['tok_per_s']:.0f} "
                        f"| {record['elapsed_s']:.0f}s", flush=True)
                    running_loss, running_batches = 0.0, 0
                    log_start_time, log_start_tokens = now, self.tokens_seen

                # -- periodic validation -------------------------------------
                if (step + 1) % t.eval_interval == 0 or step == t.steps - 1:
                    val = self.evaluate(t.eval_batches)
                    val_rec = {"step": step + 1, **val}
                    self._write_metrics("val_metrics.jsonl", val_rec)
                    print(f"           val loss {val['val_loss']:.4f} "
                          f"(ppl {val['val_ppl']:.2f})", flush=True)

                # -- periodic checkpoints ---------------------------------------
                if (step + 1) % t.checkpoint_interval == 0:
                    self._save_checkpoint(step + 1)

        except KeyboardInterrupt:
            print("\n[trainer] Interrupted — saving rescue checkpoint.", flush=True)
            status = "INTERRUPTED"
            if last_step > self.start_step:
                # roll RNG back to the boundary of the last completed step;
                # the interrupted step replays exactly on resume
                restore_rng_state(resume_rng[0])
                self.data_generator.set_state(resume_rng[1])
                self._save_checkpoint(last_step, extra={"interrupted": True})

        # -- final checkpoint + summary ------------------------------------
        final_step = last_step
        summary_path = self.experiment_dir / "summary.json" \
            if self.experiment_dir is not None else None
        if (final_step == self.start_step and summary_path is not None
                and summary_path.is_file()):
            # No steps completed this session (e.g. --resume on an
            # already-complete run, or an interrupt before the first step
            # finished).  A fresh summary would record session-only wall
            # clock, ~0 tok/s and an eval-only VRAM peak, destroying the
            # original end-to-end measurement.  Nothing about the run
            # changed, so the stored summary remains the honest record.
            print("[trainer] no new steps this session; keeping the existing "
                  "summary.json (nothing about the run changed)", flush=True)
            return json.loads(summary_path.read_text(encoding="utf-8"))
        if status == "COMPLETED" and self.ckpt is not None:
            self._save_checkpoint(t.steps)
        final = self.evaluate(t.eval_batches)
        wall = time.time() - started

        summary = {
            "status": status,
            "steps_completed": final_step,
            "tokens_trained": self.tokens_seen,
            "final_train_loss": None,   # filled from last metrics line below
            "final_val_loss": final["val_loss"],
            "final_val_ppl": final["val_ppl"],
            "wall_clock_s": round(wall, 1),
            "avg_tok_per_s": round(
                (self.tokens_seen - session_start_tokens) / max(1e-9, wall), 1),
            "params_total": self.n_params,
            "params_non_embedding": self.n_params_non_embedding,
            "active_params_per_token": self.n_params,  # dense model: active == total
            "precision": t.precision,
            "seed": t.seed,
            "vram": self.vram_stats(),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        # last logged train loss, if any
        last = self._last_metric("metrics.jsonl")
        if last:
            summary["final_train_loss"] = last["train_loss"]

        if self.experiment_dir is not None:
            (self.experiment_dir / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # ------------------------------------------------------------------
    def _last_metric(self, name: str) -> Optional[dict]:
        if self.experiment_dir is None:
            return None
        path = self.experiment_dir / name
        if not path.is_file():
            return None
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return json.loads(lines[-1]) if lines else None