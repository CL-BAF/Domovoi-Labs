"""Environment and version recording for reproducibility.

Every experiment records the output of `collect_env_info()` so a result can
be traced back to the exact Python, PyTorch, CUDA, GPU and git commit that
produced it.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from pathlib import Path
from typing import Optional

# Packages whose versions materially affect reproducibility.
TRACKED_PACKAGES = ("torch", "numpy", "tokenizers", "safetensors", "PyYAML",
                    "tqdm", "requests", "datasets")


def get_git_commit(repo_dir: str | Path = ".") -> str:
    """Current git commit hash, or 'unknown' if git is unavailable or the
    directory is not a repository.  Never raises."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def get_git_dirty(repo_dir: str | Path = ".") -> bool:
    """True if the working tree has uncommitted changes ('dirty')."""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_dir), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return False


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_env_info(repo_dir: str | Path = ".") -> dict:
    """Machine-readable snapshot of the software/hardware environment."""
    import torch

    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": str(torch.backends.cudnn.version()),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "gpu_total_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 1),
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    info["packages"] = {p: v for p in TRACKED_PACKAGES
                        if (v := _package_version(p)) is not None}
    info["git_commit"] = get_git_commit(repo_dir)
    info["git_dirty"] = get_git_dirty(repo_dir)
    return info


def detect_device() -> torch.device:
    """CUDA when available, CPU otherwise.  Kept as a single place so a
    future distributed/accelerator backend can be swapped in here."""
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_precision(precision: str, device: torch.device) -> Optional[str]:
    """Map a config precision string to a torch autocast dtype.

    Returns None for fp32 (no autocast).  Raises with a clean message if the
    requested precision is unsupported on this hardware (e.g. bf16 on an
    old GPU) so training never silently falls back and lies about it.
    """
    import torch

    if precision == "fp32":
        return None
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError(
                "Config requests bf16 but this GPU does not support it. "
                "Change training.precision to 'fp16' or 'fp32'.")
        return "bfloat16"
    if precision == "fp16":
        if device.type != "cuda":
            raise RuntimeError(
                "fp16 mixed precision is only supported on CUDA in this pipeline; "
                "use 'fp32' on CPU.")
        return "float16"
    raise RuntimeError(f"Unknown precision {precision!r}")