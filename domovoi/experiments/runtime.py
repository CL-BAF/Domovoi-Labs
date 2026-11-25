"""Preflight and Python I/O restrictions for a pinned research workload."""

from __future__ import annotations

import ast
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

from scripts.experiment_protocol import plain
from .protocol import read_json, require


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_fingerprint(root: Path) -> str:
    """Hash all dependency bytes and paths, including unlisted wheels' support files."""
    require(root.is_dir() and plain(root), "Provide a plain dependency tree.")
    digest = hashlib.sha256(b"domovoi/dependencies/v1\0")
    for current, dirs, files in os.walk(root):
        dirs.sort()
        path = Path(current)
        for name in sorted(dirs + files):
            child = path / name
            require(plain(child), f"Dependency link or junction refused: {child.relative_to(root)}")
            is_file = child.is_file()
            require(is_file or child.is_dir(), "Special files are forbidden in dependencies.")
            key = child.relative_to(root).as_posix().encode("utf-8")
            digest.update(b"F" if is_file else b"D")
            digest.update(len(key).to_bytes(8, "big") + key)
            if is_file:
                digest.update(bytes.fromhex(hash_file(child)))
    return digest.hexdigest()


def runtime_identity(dependencies: Path) -> dict:
    return {
        "python": platform.python_version(), "python_sha256": hash_file(Path(sys.executable)),
        "base_python_sha256": hash_file(Path(sys._base_executable)),
        "platform": sys.platform, "machine": platform.machine(),
        "dependencies_sha256": tree_fingerprint(dependencies),
    }


def _cuda_package_version(dependencies):
    path = dependencies / "torch/version.py"
    for line in ast.parse(path.read_text(encoding="utf-8"), filename="torch/version.py").body:
        if isinstance(line, ast.AnnAssign) and getattr(line.target, "id", None) == "cuda":
            return ast.literal_eval(line.value)
        if isinstance(line, ast.Assign) and any(getattr(t, "id", None) == "cuda" for t in line.targets):
            return ast.literal_eval(line.value)
    return None


def _within(path, root):
    return path == root or root in path.parents


@dataclass(frozen=True)
class Runtime:
    source: Path
    workspace: Path
    dependencies: Path
    read_files: frozenset[Path]
    output_dirs: tuple[Path, ...]
    temporary_dirs: tuple[Path, ...]
    tools: dict[str, Path]
    gpu_device: str
    lock: dict


def prepare(source: Path, workspace: Path, dependencies: Path, plan: dict,
            *, worker: bool = False) -> Runtime:
    # Check the input path before resolve, which would hide a reparse point.
    require(all(p.is_dir() and plain(p) for p in (source, workspace, dependencies)),
            "Source, workspace and dependencies must be existing plain directories.")
    source, workspace, dependencies = (p.resolve() for p in (source, workspace, dependencies))
    roots = (source, workspace, dependencies)
    require(not any(_within(a, b) for a in roots for b in roots if a is not b),
            "Source, workspace and dependencies must be disjoint.")
    execution = plan["execution"]
    lock = read_json((source / execution["runtime_lock"]).read_bytes(), "runtime lock")
    identity = runtime_identity(dependencies)
    require(set(lock) == {*identity, "gpu", "tools", "environment"},
            "Runtime lock has missing or unexpected fields.")
    require(all(lock[k] == v for k, v in identity.items()),
            "Interpreter, platform or dependency tree differs from the lock.")
    tool_pins = lock["tools"]
    require(isinstance(tool_pins, dict), "Expected locked tool digests.")
    tools = {}
    for name, digest in tool_pins.items():
        require(name in {"nvidia-smi"}, f"Unsupported workload executable: {name}")
        path = shutil.which(name)
        require(path and hash_file(Path(path)) == digest, f"External tool changed: {name}")
        tools[name] = Path(path).resolve()
    env = lock["environment"]
    require(isinstance(env, dict) and all(key in {
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG", "CUDA_LAUNCH_BLOCKING",
    } and isinstance(value, str)
        and "\x00" not in value for key, value in env.items()),
        "Runtime environment contains an unsupported setting.")
    gpu = lock["gpu"]
    device = ""
    if gpu is not None:
        require(isinstance(gpu, dict) and set(gpu) == {"uuid", "name", "driver", "cuda_runtime"}
                and all(isinstance(v, str) and v.strip() for v in gpu.values())
                and "nvidia-smi" in tools, "GPU runtime must pin device, driver, CUDA and nvidia-smi.")
        text = subprocess.check_output(
            [str(tools["nvidia-smi"]), "--query-gpu=uuid,name,driver_version",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
        devices = {tuple(part.strip() for part in line.split(",")) for line in text.splitlines()}
        require((gpu["uuid"], gpu["name"], gpu["driver"]) in devices and
                _cuda_package_version(dependencies) == gpu["cuda_runtime"],
                "GPU identity, driver or locked CUDA package version differs.")
        device = gpu["uuid"]
    outputs, temps = (tuple(workspace / path for path in execution[key])
                      for key in ("output_dirs", "temporary_dirs"))
    writable = outputs + temps
    assets = {workspace / path: digest for path, digest in execution["assets"].items()}
    require(not any(_within(a, b) or _within(b, a) for i, a in enumerate(writable)
                    for b in writable[i + 1:]), "Writable directories collide on this filesystem.")
    require(len(assets) == len(execution["assets"]), "Asset paths collide on this filesystem.")
    for file in assets:
        require(not any(_within(file, folder) for folder in writable),
                "Inputs may not overlap writable directories.")
    seen = set()
    for current, dirs, files in os.walk(workspace):
        here = Path(current)
        for name in dirs:
            child = here / name
            require(plain(child) and (any(_within(child, p) or _within(p, child) for p in writable)
                                      or any(child in f.parents for f in assets)),
                    f"Undeclared workspace directory: {child.relative_to(workspace)}")
        for name in files:
            file = here / name
            require(file.is_file() and plain(file),
                    f"Special file refused in workspace: {file.relative_to(workspace)}")
            if worker and any(_within(file, folder) for folder in writable):
                continue
            require(file in assets and hash_file(file) == assets[file],
                    f"Unapproved or modified workspace input: {file.relative_to(workspace)}")
            seen.add(file)
    require(seen == set(assets), "A pinned workspace input is missing.")
    return Runtime(source, workspace, dependencies, frozenset(assets), outputs, temps,
                   tools, device, lock)


_runtime: Runtime | None = None
_supervising: ContextVar[bool] = ContextVar("domovoi_runtime_supervisor", default=False)


def active_runtime() -> Runtime:
    runtime = _runtime
    require(runtime is not None, "There is no active locked workload.")
    return runtime


def _audit(event, args):
    runtime = _runtime
    if runtime is None or _supervising.get():
        return
    writable = runtime.output_dirs + runtime.temporary_dirs

    def check(value, writing=False, directory=False):
        if value is None or isinstance(value, int):
            require(not writing, "Unidentified write path in research workload.")
            return
        path = Path(os.fsdecode(value)).resolve()
        allowed = any(_within(path, folder) for folder in writable)
        if not writing:
            standard = (_within(path, _STDLIB) and
                        not any(_within(path, _STDLIB / site) for site in ("site-packages", "dist-packages")))
            source = _within(path, runtime.source) and not _within(path, runtime.source / ".git")
            allowed = (allowed or path in runtime.read_files or source or standard or
                       _within(path, runtime.dependencies) or _within(path, _DLLS))
            allowed |= directory and (path == runtime.workspace or
                         any(path in f.parents for f in runtime.read_files))
        require(allowed, f"Research workload path refused: {path.name}")

    if event == "open":
        flags = args[2]
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        check(args[0], bool(writing))
    elif event in {"os.listdir", "os.scandir"}:
        check(args[0], directory=True)
    elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.truncate", "os.utime"}:
        require(len(args) < 2 or event == "os.truncate" or args[-1] in (-1, None),
                "Directory-descriptor writes are refused.")
        check(args[0], True)
    elif event in {"os.rename", "os.replace"}:
        require(all(fd in (-1, None) for fd in args[2:]), "Directory-descriptor rename refused.")
        check(args[0], True)
        check(args[1], True)
    elif event in {"os.symlink", "os.link", "os.chdir", "os.system", "os.exec", "os.fork",
                   "os.posix_spawn", "socket.connect", "socket.bind", "socket.getaddrinfo"}:
        raise PermissionError(f"Research workload operation refused: {event}")
    elif event == "subprocess.Popen":
        executable = args[0]
        require(isinstance(executable, (str, bytes, os.PathLike)) and
                any(str(tool) == os.fsdecode(executable) for tool in runtime.tools.values())
                and not args[3],
                "Use a pinned in-process adapter or an explicitly locked tool.")


_STDLIB = Path(os.__file__).resolve().parent
_DLLS = Path(sys.base_prefix).resolve() / "DLLs"
sys.addaudithook(_audit)


@contextmanager
def supervisor():
    """Trusted launcher bookkeeping. Approved payloads must not call this API."""
    token = _supervising.set(True)
    try:
        yield
    finally:
        _supervising.reset(token)


@contextmanager
def workload_environment(runtime: Runtime, source_commit: str):
    global _runtime
    require(_runtime is None, "Only one research workload can run in this process.")
    here, env, search, old_temp = Path.cwd(), dict(os.environ), sys.path[:], tempfile.tempdir
    os.chdir(runtime.workspace)
    try:
        # Retain Windows OS bootstrap variables only. Never propagate API keys or PYTHONPATH.
        keep = {key: value for key, value in env.items() if key.upper() in {
            "SYSTEMROOT", "WINDIR", "SYSTEMDRIVE", "LANG", "LC_ALL"}}
        keep.update(runtime.lock["environment"])
        tmp = str(runtime.temporary_dirs[0])
        keep.update({"TEMP": tmp, "TMP": tmp, "TMPDIR": tmp, "CUDA_VISIBLE_DEVICES": runtime.gpu_device,
                     "DOMOVOI_SOURCE_COMMIT": source_commit, "HF_HUB_OFFLINE": "1",
                     "TRANSFORMERS_OFFLINE": "1"})
        os.environ.clear()
        os.environ.update(keep)
        tempfile.tempdir = tmp
        sys.path.append(str(runtime.dependencies))  # -S keeps .pth and user-site hooks disabled.
        _runtime = runtime
        try:
            yield
        finally:
            _runtime = None
    finally:
        os.chdir(here)
        os.environ.clear()
        os.environ.update(env)
        sys.path[:] = search
        tempfile.tempdir = old_temp
