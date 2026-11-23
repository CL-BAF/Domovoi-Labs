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

import dataclasses
import json
import math
import os
import pathlib
import socket
import time
import uuid
from typing import Optional

import torch

from ..config import Config, ModelSection, TrainingSection
from ..datasets import (DomovoiTokenizer, PackedTokenData, fixed_val_generator,
                       load_processed_meta, verify_vocab)
from ..datasets.loader import DisjointSegmentSampler
from ..datasets.tokenizer import sha256_file
from ..experiments.metrics_log import read_metrics, start_session
from ..models import build_model, signature_differences
from ..utils import (detect_device, resolve_precision, set_seed)
from ..utils.seeding import capture_rng_state, restore_rng_state
from .checkpoint import (CheckpointError, CheckpointManager, LatestCheckpoint,
                         load_checkpoint, read_checkpoint_meta,
                         recover_checkpoint, scan_checkpoints,
                         verify_checkpoint_architecture)
from .optim import build_optimizer, build_scheduler, current_lr
from .pilot_limits import PilotLimits, PilotStopError


def _step(path: pathlib.Path) -> int:
    """Step number of a step_XXXXXXXX[.bak|.tmp] directory."""
    return int(path.name.split("_")[1].split(".")[0])


# training.* settings that define the optimisation trajectory or the
# measurement protocol of a run.  Resume refuses any change to them (Codex
# round 4): a run continued with another batch size, LR, schedule length,
# clip, precision, seed or eval-batch count is a different experiment and
# needs a new experiment ID.  log/eval/checkpoint INTERVALS are not listed:
# they change only when rows are written, not what is trained or measured.
TRAJECTORY_FIELDS = ("steps", "batch_size", "gradient_accumulation",
                     "context_length", "learning_rate", "min_lr_ratio",
                     "weight_decay", "betas", "grad_clip", "warmup_steps",
                     "precision", "seed", "eval_batches", "sampler",
                     "pilot_stop_checks", "pilot_calibrated_seconds", "deterministic_cuda",
                     "verify_checkpoint_writes")


class NonFiniteTrainingError(RuntimeError):
    """A non-finite loss, gradient norm or validation loss: the run is FAILED
    (Codex round 4: a NaN run used to finish as COMPLETED)."""


class Trainer:
    def __init__(
        self,
        config: Config,
        experiment_dir: str | pathlib.Path | None = None,
        device: Optional[torch.device] = None,
        resume: bool = False,
        extend_steps: bool = False,
        pilot_deadline: float | None = None,
        stop_file: str | pathlib.Path | None = None,
    ):
        self.config = config
        self.stop_file = pathlib.Path(stop_file) if stop_file is not None else None
        t = config.training
        if t.deterministic_cuda:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        self.device = device if device is not None else detect_device()
        if t.deterministic_cuda and self.device.type == "cuda":
            torch.use_deterministic_algorithms(True)
        self.experiment_dir = pathlib.Path(experiment_dir) if experiment_dir else None
        # opt-in: a resume may only lengthen training.steps (recorded as a
        # protocol deviation); every other trajectory change is refused
        self.extend_steps = extend_steps
        self._protocol_deviations: list = []

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
        self.segment_sampler = None
        if t.sampler == "disjoint_segments":
            self.segment_sampler = DisjointSegmentSampler(
                len(self.train_data), t.context_length, t.seed + 1,
                t.steps * t.batch_size * t.gradient_accumulation)
        self.pilot_limits = PilotLimits(
            t.warmup_steps, t.steps, t.pilot_calibrated_seconds, pilot_deadline
        ) if t.pilot_stop_checks else None

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

        # -- data provenance: hashes of the packed files actually opened ----
        self._provenance = self._data_provenance()

        # -- writer lock, then resume (Codex round 3) -----------------------
        # The lock is taken FIRST, so recovery, loading, the CLI's metadata
        # and status writes and training all happen under it.  A refused lock
        # writes nothing.  Everything below either succeeds or releases the
        # lock and leaves the experiment as it was.
        self._lock_token: Optional[str] = None
        self.ckpt = None
        self.start_step = 0
        self.tokens_seen = 0
        self._acquire_writer_lock(create_dir=not resume)
        try:
            # read-only validation (CheckpointManager's constructor creates
            # directories, so it is built only afterwards)
            found = self._resolve_resume_checkpoint(resume)
            if found is not None:
                # load weights/optimizer/scheduler/RNG from where the files
                # ARE (a crashed swap's *.bak is read in place) ...
                self._load_resume(found, t)
                # ... and only after that load succeeded, put it back in place
                recover_checkpoint(found)
            self.ckpt = CheckpointManager(self.experiment_dir / "checkpoints") \
                if self.experiment_dir else None
        except BaseException:
            self.release_lock()
            raise

        # -- VRAM tracking ---------------------------------------------------
        self._vram_supported = self.device.type == "cuda"
        if self._vram_supported:
            torch.cuda.reset_peak_memory_stats(self.device)

    # ------------------------------------------------------------------
    WRITER_LOCK = ".writer.lock"

    def _acquire_writer_lock(self, create_dir: bool) -> None:
        """Create <experiment>/.writer.lock atomically (O_EXCL).  If another
        writer holds it, raise CheckpointError having written nothing.  For a
        resume of a directory that does not exist nothing is created (the
        preflight then reports the missing checkpoint)."""
        if self.experiment_dir is None:
            return
        if not self.experiment_dir.is_dir():
            if not create_dir:
                return
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
        lock = self.experiment_dir / self.WRITER_LOCK
        token = f"pid={os.getpid()} host={socket.gethostname()} id={uuid.uuid4().hex} " \
                f"since={time.strftime('%Y-%m-%dT%H:%M:%S')}"
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = lock.read_text(encoding="utf-8", errors="replace").strip()
            raise CheckpointError(
                f"{self.experiment_dir} is locked by another writer ({holder}). "
                f"If no process is training this experiment, delete {lock} "
                f"and retry.") from None
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token)
        self._lock_token = token

    def release_lock(self) -> None:
        """Remove the writer lock if THIS Trainer holds it (the stored token
        must match, so another writer's lock is never deleted)."""
        if self._lock_token is None or self.experiment_dir is None:
            return
        lock = self.experiment_dir / self.WRITER_LOCK
        try:
            if lock.read_text(encoding="utf-8", errors="replace") == self._lock_token:
                lock.unlink()
        except FileNotFoundError:
            pass
        self._lock_token = None

    close = release_lock

    def __del__(self):
        # best effort for Trainers that are built but never trained/closed
        try:
            self.release_lock()
        except Exception:
            pass

    def _load_resume(self, found: LatestCheckpoint, t) -> None:
        """Restore model/optimizer/scheduler/scaler/RNG from the validated
        checkpoint, reading it where it currently is (found.read_from)."""
        meta, rng = load_checkpoint(
            found.read_from, self.model, self.optimizer, self.scheduler,
            scaler=self.scaler, restore_rng=False)
        self.start_step = meta["step"]
        self.tokens_seen = meta["tokens_seen"]
        # (a changed training.steps used to be a warning here; since round 4
        # _resolve_resume_checkpoint refuses it with the other trajectory fields)
        gen_state = rng.get("extra", {}).get("data_generator")
        if gen_state is not None:
            self.data_generator.set_state(gen_state.to(torch.uint8))
        if self.segment_sampler is not None:
            position = rng.get("extra", {}).get("segment_position")
            if position != self.start_step * t.batch_size * t.gradient_accumulation:
                raise CheckpointError("Checkpoint has no valid committed segment sampler position.")
            self.segment_sampler.restore(position)
        if self.pilot_limits is not None:
            state = rng.get("extra", {}).get("pilot_limits")
            if state is None:
                raise CheckpointError("Checkpoint is missing pilot stop-check state.")
            self.pilot_limits.load_state_dict(state)
        restore_rng_state(rng.get("standard") or {})

    # ------------------------------------------------------------------
    def _resolve_resume_checkpoint(self, resume: bool) -> Optional[LatestCheckpoint]:
        """Which checkpoint to resume from, or None for a fresh run.  Strictly
        READ-ONLY: the directory is inventoried with scan_checkpoints() and a
        crashed swap's *.bak is read in place; the caller renames it back only
        after every check here has passed.  Raises CheckpointError when:
          - a fresh run targets a directory with ANY checkpoint trace (step
            dirs, *.bak/*.tmp leftovers or latest.json), found independently
            of latest.json (two histories would be mixed in one experiment);
          - resume is requested but latest.json names no readable checkpoint
            (previously a warning and a silent restart from step 0);
          - complete step dirs newer than the indexed checkpoint exist
            (inconsistent directory; refuse to guess);
          - the checkpoint's model config differs from this run's in ANY
            field, dropout included, or its graph signature differs;
          - its recorded data provenance (tokenizer / dataset hashes,
            token counts) differs from this run's data."""
        t_cfg = self.config.training
        root = self.experiment_dir / "checkpoints" if self.experiment_dir else None
        inv = scan_checkpoints(root)          # None (no experiment dir) => empty
        if not resume:
            if inv["any"]:
                traces = [p.name for p in inv["complete"] + inv["bak"] + inv["tmp"] + inv["other"]]
                raise CheckpointError(
                    f"{self.experiment_dir} already holds checkpoints or their leftovers "
                    f"({traces + (['latest.json'] if inv['index'] else [])}). A fresh "
                    f"run here would mix two training histories in one experiment. "
                    f"Resume it (resume=True / --resume) or train into a new "
                    f"experiment directory.")
            return None
        found = inv["latest"]
        if found is None:
            extra = (f" Step directories exist ({[p.name for p in inv['complete'] + inv['bak']]}) "
                     f"but latest.json is missing or names no readable checkpoint; "
                     f"refusing to guess which one is current." if inv["any"] else "")
            raise CheckpointError(
                f"Resume requested but no checkpoint can be resumed under "
                f"{root if root is not None else '(no experiment dir given)'}; "
                f"refusing to silently restart from step 0.{extra}")
        newer = [p.name for p in inv["complete"] if _step(p) > found.step]
        if newer:
            raise CheckpointError(
                f"latest.json names {found.path.name}, but newer complete "
                f"checkpoints exist: {newer}. The directory is inconsistent; "
                f"inspect it by hand before resuming.")
        meta = read_checkpoint_meta(found.read_from)
        saved = (meta.get("config") or {}).get("model")
        if saved is None:
            raise CheckpointError(f"Checkpoint {found.read_from} records no model config; cannot verify resume.")
        diffs = signature_differences(dataclasses.asdict(ModelSection(**saved)),
                                      dataclasses.asdict(self.config.model),
                                      "checkpoint", "this run's config")
        if diffs:
            raise CheckpointError(
                f"Cannot resume {found.read_from}: this run's model config differs from "
                f"the one the checkpoint was trained with:\n  " + "\n  ".join(diffs)
                + "\nResume with the original model section (copy config.model "
                  "from checkpoint.json), or start a new experiment for the "
                  "changed architecture.")
        saved_t = (meta.get("config") or {}).get("training")
        if saved_t is None:
            raise CheckpointError(f"Checkpoint {found.read_from} records no training config; cannot verify resume.")
        # normalise both sides (stored betas are a JSON list, live ones a tuple)
        old_t, new_t = dataclasses.asdict(TrainingSection(**saved_t)), dataclasses.asdict(t_cfg)
        old_t["betas"], new_t["betas"] = list(old_t["betas"]), list(new_t["betas"])
        fields = [k for k in TRAJECTORY_FIELDS if old_t[k] != new_t[k]]
        deviations = list(meta.get("protocol_deviations") or [])
        if (self.extend_steps and "steps" in fields
                and new_t["steps"] > old_t["steps"]):
            # explicit opt-in: ONLY a longer schedule, recorded permanently so
            # the run can be excluded from controlled comparisons
            fields.remove("steps")
            deviations.append({"field": "steps", "from": old_t["steps"],
                               "to": new_t["steps"], "at_step": meta["step"]})
            print(f"[trainer] WARNING: --extend-steps: training.steps "
                  f"{old_t['steps']} -> {new_t['steps']} at step {meta['step']}. "
                  f"The LR schedule now differs from the original run; this is "
                  f"recorded as a protocol deviation.", flush=True)
        self._protocol_deviations = deviations
        diffs = signature_differences({k: old_t[k] for k in fields},
                                      {k: new_t[k] for k in fields},
                                      "checkpoint", "this run's config")
        if diffs:
            raise CheckpointError(
                f"Cannot resume {found.read_from}: this run's training protocol "
                f"differs from the one the checkpoint was trained with:\n  "
                + "\n  ".join(diffs)
                + "\nA resumed run must continue the same trajectory.  Resume with "
                  "the checkpoint's training settings (config.training in "
                  "checkpoint.json), or start a new experiment (new ID) for the "
                  "changed protocol.  Only log/eval/checkpoint intervals may change "
                  "(and training.steps may grow with --extend-steps, recorded as a "
                  "protocol deviation).")
        # the graph check load_checkpoint repeats later, done here before any write
        verify_checkpoint_architecture(meta, self.model, found.read_from)
        saved_data = meta.get("data_provenance")
        if saved_data is None:
            print(f"[trainer] WARNING: {found.read_from} predates data-provenance "
                  f"records; tokenizer/dataset identity cannot be verified for "
                  f"this resume.", flush=True)
        else:
            # every field the checkpoint recorded must match; fields it did
            # not record (older provenance format) are reported, not assumed
            diffs = signature_differences(
                saved_data, {k: self._provenance.get(k) for k in saved_data},
                "checkpoint", "this run")
            if diffs:
                raise CheckpointError(
                    f"Cannot resume {found.read_from}: the training data or "
                    f"tokenizer differs from the run that wrote it:\n  "
                    + "\n  ".join(diffs))
            unrecorded = [k for k in self._provenance if k not in saved_data]
            if unrecorded:
                print(f"[trainer] WARNING: {found.read_from} does not record "
                      f"{unrecorded}; those cannot be verified for this resume.",
                      flush=True)
        return found

    def _data_provenance(self) -> dict:
        """Identity of the data this run ACTUALLY trains on, recorded in every
        checkpoint and compared on resume.  Codex round 3: the manifest's
        text hashes and token counts do not prove what the loader reads, so
        the packed .bin files the loader opened are hashed and their token
        counts taken from the opened files themselves (and must agree with
        the manifest)."""
        ds = self.data_meta.get("dataset") or {}
        tok = self.data_meta.get("tokens") or {}
        actual = {"train": self.train_data.n_tokens, "val": self.val_data.n_tokens}
        for split in ("train", "val"):
            if tok.get(split) is not None and tok[split] != actual[split]:
                raise RuntimeError(
                    f"Packed {split} file {self.data_meta['files'][split + '_bin']} "
                    f"holds {actual[split]} tokens but meta.json records "
                    f"{tok[split]}; the processed dataset is inconsistent.")
        return {
            "tokenizer_sha256": self.tokenizer.sha256,
            "dataset_name": ds.get("name"),
            "dataset_version": ds.get("version"),
            "raw_sha256": ds.get("raw_sha256"),           # source text (manifest)
            "train_sha256": ds.get("train_sha256"),       # split text (manifest)
            "val_sha256": ds.get("val_sha256"),
            "train_bin_sha256": sha256_file(self.train_data.path),   # bytes the loader reads
            "val_bin_sha256": sha256_file(self.val_data.path),
            "train_tokens": actual["train"],                         # from the opened files
            "val_tokens": actual["val"],
        }

    # ------------------------------------------------------------------
    def tokens_per_step(self) -> int:
        t = self.config.training
        return t.batch_size * t.context_length * t.gradient_accumulation

    # ------------------------------------------------------------------
    def get_train_batch(self):
        t = self.config.training
        if self.segment_sampler is not None:
            return self.train_data.batch_at(
                self.segment_sampler.take(t.batch_size), t.context_length, self.device)
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
        if self.config.training.synchronize_cuda_timing:
            latest = scan_checkpoints(self.ckpt.root)["latest"]
            if latest is not None and latest.step == step:
                return  # The periodic save already committed this exact step.
        rng_extra = {"data_generator": self.data_generator.get_state()}
        if self.segment_sampler is not None:
            rng_extra["segment_position"] = self.segment_sampler.position
        if self.pilot_limits is not None:
            rng_extra["pilot_limits"] = self.pilot_limits.state_dict()
        self.ckpt.save(
            step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            tokens_seen=self.tokens_seen,
            config=self.config,
            extra={**(extra or {}), "data_provenance": self._provenance,
                   "protocol_deviations": self._protocol_deviations},
            rng_extra=rng_extra,
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
    def train(self, release_lock: bool = True) -> dict:
        """Run the training loop under the writer lock taken in __init__
        (re-acquired if this Trainer released it).  The lock is released
        afterwards, including on Ctrl+C or an exception, unless
        release_lock=False: scripts/train.py keeps it through its final
        metadata and status writes and then calls close().  A lock left by a
        killed process must be deleted by hand after confirming nothing is
        training."""
        if self.experiment_dir is not None and self._lock_token is None:
            self._acquire_writer_lock(create_dir=True)
        try:
            return self._train_unlocked()
        finally:
            if release_lock:
                self.release_lock()

    def _train_unlocked(self) -> dict:
        """Run to completion.  Returns the summary dict (also written to
        summary.json).  On Ctrl+C, saves a rescue checkpoint and exits with
        status INTERRUPTED -- unless the interrupt hit an optimizer update, in
        which case nothing is saved and the last committed checkpoint is the
        recovery point.  A non-finite loss, gradient norm or validation loss
        ends the run as FAILED without a checkpoint of that step."""
        t = self.config.training
        started = time.time()
        # tokens trained in THIS session (tokens_seen is cumulative across
        # resumes; wall clock is not, so throughput must compare like with like)
        session_start_tokens = self.tokens_seen
        self.model.train()
        log_start_time = time.time()
        log_start_tokens = self.tokens_seen
        running_loss = 0.0
        measured_step_seconds = 0.0
        running_batches = 0
        last_validation = None
        status = "COMPLETED"
        failure = None               # reason when status becomes FAILED
        failure_code = None
        last_step = self.start_step  # last fully completed optimizer step
        committed_tokens = self.tokens_seen   # tokens_seen as of last_step
        # True only while optimizer/scheduler/scaler state is being mutated.
        # An interrupt there leaves weights and AdamW moments/counters for a
        # step that last_step does not include, so no rescue checkpoint may be
        # written from that state (Codex round 4).
        in_update = False
        torn = False
        # this session's attempt number; every metric row carries it so
        # read_metrics() can drop rows a later resume replayed (Codex round 4)
        attempt = start_session(self.experiment_dir, self.start_step) \
            if self.experiment_dir is not None else 0

        # RNG state to resume from `last_step`: captured before the loop and
        # refreshed the moment a step completes.  An interrupt at any point
        # (even mid-step, after the batch draw consumed generator state)
        # rolls back to the last clean boundary so resume replays the
        # interrupted step exactly.
        resume_rng = (capture_rng_state(),
                      self.data_generator.get_state().clone())
        limits = self.pilot_limits
        committed_limits = limits.state_dict() if limits is not None else None
        prior_elapsed = limits.elapsed_seconds if limits is not None else 0.0
        limit_started = time.perf_counter()
        extra_checkpoint_steps = set(t.extra_checkpoint_steps)

        try:
            for step in range(self.start_step, t.steps):
                if self.stop_file is not None and self.stop_file.exists():
                    raise KeyboardInterrupt
                if t.synchronize_cuda_timing and self._vram_supported:
                    torch.cuda.synchronize(self.device)
                step_started = time.perf_counter()
                lr = current_lr(self.optimizer)

                # -- gradient accumulation ---------------------------------
                self.optimizer.zero_grad(set_to_none=True)
                micro_losses = []
                for _ in range(t.gradient_accumulation):
                    x, y = self.get_train_batch()
                    with self._autocast():
                        out = self.model(x, targets=y)
                    loss_value = out.loss.item()
                    # stop BEFORE a non-finite loss can reach the weights
                    if not math.isfinite(loss_value):
                        raise NonFiniteTrainingError(
                            f"non-finite training loss {loss_value} at step {step + 1}")
                    scaled = out.loss / t.gradient_accumulation
                    if self.scaler.is_enabled():
                        self.scaler.scale(scaled).backward()
                    else:
                        scaled.backward()
                    micro_losses.append(loss_value)

                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), t.grad_clip)
                # fp16 GradScaler legitimately sees inf grads and skips that
                # update itself; without a scaler a non-finite norm means the
                # update would write NaN/inf into the weights
                if (limits is not None or not self.scaler.is_enabled()) and not torch.isfinite(grad_norm):
                    raise NonFiniteTrainingError(
                        f"non-finite gradient norm {float(grad_norm)} at step {step + 1}")

                batch_loss = sum(micro_losses) / len(micro_losses)
                if limits is not None:
                    limits.check_update(step + 1, batch_loss, float(grad_norm))
                    self._check_pilot_resources(step + 1, prior_elapsed, limit_started)

                # ---- update window: state is mutated until last_step moves ----
                in_update = True
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                if limits is not None:
                    limits.elapsed_seconds = self._check_pilot_resources(
                        step + 1, prior_elapsed, limit_started, after_update=True)
                    committed_limits = limits.state_dict()

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
                committed_tokens = self.tokens_seen
                in_update = False            # committed: consistent at last_step
                if t.synchronize_cuda_timing and self._vram_supported:
                    torch.cuda.synchronize(self.device)

                running_loss += batch_loss
                running_batches += 1
                measured_step_seconds += time.perf_counter() - step_started
                now = time.time()

                # -- periodic logging ----------------------------------------
                if (step + 1) % t.log_interval == 0 or step == t.steps - 1:
                    window_tokens = self.tokens_seen - log_start_tokens
                    window_s = max(1e-9, now - log_start_time)
                    record = {
                        "step": step + 1,
                        "attempt": attempt,
                        "train_loss": round(running_loss / max(1, running_batches), 6),
                        "lr": round(lr, 8),
                        "grad_norm": round(float(grad_norm), 4),
                        "tokens_seen": self.tokens_seen,
                        "tok_per_s": round(window_tokens / window_s, 1),
                        "elapsed_s": round(now - started, 1),
                    }
                    if t.synchronize_cuda_timing:
                        record["mean_step_s"] = round(measured_step_seconds / running_batches, 6)
                        record["step_tokens_per_s"] = round(
                            running_batches * self.tokens_per_step() / max(measured_step_seconds, 1e-9), 1)
                    self._write_metrics("metrics.jsonl", record)
                    print(
                        f"step {step + 1:6d}/{t.steps} | loss {record['train_loss']:.4f} "
                        f"| lr {lr:.2e} | tok/s {record['tok_per_s']:.0f} "
                        f"| {record['elapsed_s']:.0f}s", flush=True)
                    running_loss, running_batches = 0.0, 0
                    measured_step_seconds = 0.0
                    log_start_time, log_start_tokens = now, self.tokens_seen

                # -- periodic validation -------------------------------------
                if (step + 1) % t.eval_interval == 0 or step == t.steps - 1:
                    val = self.evaluate(t.eval_batches)
                    last_validation = val
                    val_rec = {"step": step + 1, "attempt": attempt, **val}
                    self._write_metrics("val_metrics.jsonl", val_rec)
                    print(f"           val loss {val['val_loss']:.4f} "
                          f"(ppl {val['val_ppl']:.2f})", flush=True)
                    if not math.isfinite(val["val_loss"]):
                        # the row above records it; the weights at this step
                        # are not checkpointed and the run is FAILED
                        raise NonFiniteTrainingError(
                            f"non-finite validation loss at step {step + 1}")
                    if limits is not None:
                        limits.elapsed_seconds = self._check_pilot_resources(
                            step + 1, prior_elapsed, limit_started)
                        committed_limits = limits.state_dict()

                # -- periodic checkpoints ---------------------------------------
                if (step + 1) % t.checkpoint_interval == 0 or step + 1 in extra_checkpoint_steps:
                    self._save_checkpoint(step + 1)

        except KeyboardInterrupt:
            status = "INTERRUPTED"
            if in_update:
                # weights/AdamW state may already include (part of) the next
                # update: saving them under last_step would let a resume apply
                # that batch twice.  The last COMMITTED checkpoint stays the
                # recovery point; nothing is written from this state.
                torn = True
                print("\n[trainer] Interrupted DURING an optimizer update — no "
                      "rescue checkpoint (the in-memory state is between steps); "
                      "resume will use the last committed checkpoint.", flush=True)
            elif last_step > self.start_step:
                print("\n[trainer] Interrupted — saving rescue checkpoint.", flush=True)
                # roll RNG back to the boundary of the last completed step;
                # the interrupted step replays exactly on resume
                restore_rng_state(resume_rng[0])
                self.data_generator.set_state(resume_rng[1])
                if self.segment_sampler is not None:
                    self.segment_sampler.restore(committed_tokens // t.context_length)
                if limits is not None:
                    limits.load_state_dict(committed_limits)
                self._save_checkpoint(last_step, extra={"interrupted": True})
        except (NonFiniteTrainingError, PilotStopError) as e:
            # raised before the optimizer update (loss/grad) or after a
            # committed step (validation): no checkpoint is written from it
            status, failure = "FAILED", str(e)
            failure_code = e.code if isinstance(e, PilotStopError) else "nonfinite"
            print(f"\n[trainer] FAILED: {failure}; no checkpoint written for this "
                  f"step.", flush=True)

        # -- final checkpoint + summary ------------------------------------
        final_step = last_step
        summary_path = self.experiment_dir / "summary.json" \
            if self.experiment_dir is not None else None
        if (status == "COMPLETED" and final_step == self.start_step and summary_path is not None
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
        if status == "FAILED" or torn:
            # no evaluation of a NaN or between-steps model
            final = {"val_loss": None, "val_ppl": None}
        else:
            final = last_validation if limits is not None and status == "COMPLETED" \
                and last_validation is not None else self.evaluate(t.eval_batches)
            if not math.isfinite(final["val_loss"]):
                status, failure = "FAILED", "non-finite final validation loss"
                failure_code = "nonfinite"
            elif status == "COMPLETED" and self.ckpt is not None:
                # final checkpoint only for a finite, completed run
                self._save_checkpoint(t.steps)
        wall = time.time() - started

        summary = {
            "status": status,
            "failure": failure,
            "failure_code": failure_code,
            "attempt": attempt,
            # non-empty => not a single controlled trajectory (e.g. --extend-steps)
            "protocol_deviations": self._protocol_deviations,
            # torn interrupt: the in-memory model is between steps; the
            # recovery point is the last committed checkpoint on disk
            "recovery_checkpoint": self._committed_checkpoint_name() if torn else None,
            "steps_completed": final_step,
            # committed accounting (a torn update may have counted its tokens)
            "tokens_trained": committed_tokens,
            "final_train_loss": None,   # filled from last metrics line below
            "final_val_loss": final["val_loss"],
            "final_val_ppl": final["val_ppl"],
            "wall_clock_s": round(wall, 1),
            "avg_tok_per_s": round(
                (committed_tokens - session_start_tokens) / max(1e-9, wall), 1),
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

    def _check_pilot_resources(self, step: int, prior_elapsed: float, started: float,
                               *, after_update: bool = False) -> float:
        check_projection = after_update and step == min(200, self.pilot_limits.total_steps)
        peak = None
        if self._vram_supported:
            if check_projection:
                torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_allocated(self.device)
        elapsed = prior_elapsed + time.perf_counter() - started
        self.pilot_limits.check_resources(step, elapsed, peak, check_projection=check_projection)
        return elapsed

    # ------------------------------------------------------------------
    def _committed_checkpoint_name(self) -> Optional[str]:
        """Name of the newest checkpoint on disk (read-only lookup)."""
        found = scan_checkpoints(self.ckpt.root if self.ckpt is not None else None)["latest"]
        return found.path.name if found is not None else None

    def _last_metric(self, name: str) -> Optional[dict]:
        if self.experiment_dir is None:
            return None
        # reconciled view: rows a later resume replayed are not "last"
        rows = read_metrics(self.experiment_dir, name)
        return rows[-1] if rows else None
