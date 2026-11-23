"""Per-run limits from the preregistered Domovoi pilot."""

from collections import deque
import time


class PilotStopError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class PilotLimits:
    def __init__(self, warmup_steps: int, total_steps: int,
                 calibrated_seconds: float | None, deadline: float | None = None):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.calibrated_seconds = calibrated_seconds
        self.deadline = deadline
        self.losses = deque(maxlen=50)
        self.best_loss = None
        self.divergent_steps = 0
        self.high_grad_steps = 0
        self.elapsed_seconds = 0.0

    def check_update(self, step: int, loss: float, grad_norm: float) -> None:
        self.high_grad_steps = self.high_grad_steps + 1 if grad_norm > 100 else 0
        if self.high_grad_steps >= 50:
            raise PilotStopError("gradient_norm", f"pre-clip gradient norm above 100 "
                                 f"for 50 consecutive steps at step {step}")
        if step <= self.warmup_steps:
            return
        self.losses.append(loss)
        if len(self.losses) < self.losses.maxlen:
            return
        mean = sum(self.losses) / len(self.losses)
        self.best_loss = mean if self.best_loss is None else min(self.best_loss, mean)
        self.divergent_steps = self.divergent_steps + 1 if mean > 1.5 * self.best_loss else 0
        if self.divergent_steps >= 200:
            raise PilotStopError("divergence", f"50-step mean loss exceeded 1.5 times "
                                 f"its minimum for 200 consecutive steps at step {step}")

    def check_resources(self, step: int, elapsed: float, peak_bytes: int | None,
                        *, check_projection: bool = False) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise PilotStopError("pilot_time_limit", "aggregate pilot time budget exhausted")
        if peak_bytes is not None and peak_bytes > 12 * 2**30:
            raise PilotStopError("memory_limit", f"peak allocated memory exceeded 12 GiB "
                                 f"at step {step}")
        if (check_projection and self.calibrated_seconds is not None
                and step == min(200, self.total_steps)):
            projected = elapsed * self.total_steps / step
            if projected > 3 * self.calibrated_seconds:
                raise PilotStopError("time_projection", f"step {step} projects a runtime "
                                     f"of {projected:.1f}s, above three times calibration")

    def state_dict(self) -> dict:
        return {"losses": list(self.losses), "best_loss": self.best_loss,
                "divergent_steps": self.divergent_steps, "high_grad_steps": self.high_grad_steps,
                "elapsed_seconds": self.elapsed_seconds}

    def load_state_dict(self, state: dict) -> None:
        self.losses.clear()
        self.losses.extend(state["losses"])
        self.best_loss = state["best_loss"]
        self.divergent_steps = state["divergent_steps"]
        self.high_grad_steps = state["high_grad_steps"]
        self.elapsed_seconds = state["elapsed_seconds"]
