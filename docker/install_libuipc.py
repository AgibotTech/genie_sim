# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

"""Fetch and install a pinned native libuipc wheel for SM 89 or 120, without Newton."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import tempfile
import time
from email.parser import Parser
from contextlib import contextmanager
from importlib import metadata as packages
from pathlib import Path, PurePosixPath
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen
from zipfile import ZipFile

ARTIFACTS = json.loads(Path(__file__).with_name("libuipc-artifacts.json").read_text())
WHEELS = {int(sm): (row["filename"], row["sha256"]) for sm, row in ARTIFACTS["artifacts"].items()}
DOWNLOAD_TIMEOUT = 240


def validate_platform() -> None:
    """Both supplied native extensions require this ABI, regardless of wheel tag."""
    if (
        sys.platform != "linux"
        or platform.machine() != "x86_64"
        or platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError("The libuipc wheels require Linux x86_64 CPython 3.12")


def detect_sm(gpu: int) -> int:
    """Read the selected nvidia-smi GPU's compute capability without CUDA imports."""
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=compute_cap", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    capability = result.stdout.strip()
    match = re.fullmatch(r"(\d+)\.(\d)", capability)
    if not match:
        raise RuntimeError(f"Cannot read GPU {gpu} compute capability: {capability!r}")
    return int(match[1]) * 10 + int(match[2])


def select_wheel(sm: int) -> Path:
    """Acquire only the selected architecture, publishing verified bytes to a durable cache."""
    if sm not in WHEELS:
        raise RuntimeError(f"Unsupported GPU sm_{sm}; only sm_89 and sm_120 are supported")
    name, expected = WHEELS[sm]
    row = ARTIFACTS["artifacts"][str(sm)]
    path = PurePosixPath(row["path"])
    if (
        ARTIFACTS.get("schema_version") != 1
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in row["path"]
        or str(path) != row["path"]
        or path.name != name
        or not re.fullmatch(r"[a-f0-9]{64}", expected)
        or type(row["size"]) is not int
        or row["size"] < 0
    ):
        raise RuntimeError("Invalid libuipc artifact manifest")
    cache = (
        Path(
            os.environ.get("GENIESIM_LIBUIPC_CACHE")
            or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "geniesim" / "libuipc"
        )
        .expanduser()
        .absolute()
    )
    directory = cache / expected
    if any(p.is_symlink() for p in (directory, *directory.parents)):
        raise RuntimeError("Symlink in libuipc cache path")
    wheel = directory / name
    if wheel.is_symlink():
        raise RuntimeError("Symlink at libuipc wheel cache")
    if _valid_artifact(wheel, row["size"], expected):
        return wheel
    directory.mkdir(parents=True, exist_ok=True)
    with _download_lock(directory):
        if wheel.is_symlink():
            raise RuntimeError("Symlink at libuipc wheel cache")
        if _valid_artifact(wheel, row["size"], expected):
            return wheel
        partial = wheel.with_name(wheel.name + ".part")
        if partial.is_symlink():
            raise RuntimeError("Symlink at libuipc partial download")
        local_root = os.environ.get("GENIESIM_ASSETS_SRC")
        if local_root:
            local = Path(local_root).expanduser() / path
            if local.exists():
                if not _valid_artifact(local, row["size"], expected):
                    raise RuntimeError(f"Asset wheel size or SHA256 mismatch: {local}")
                shutil.copyfile(local, partial)
                if not _valid_artifact(partial, row["size"], expected):
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(f"Asset wheel size or SHA256 mismatch: {local}")
                partial.replace(wheel)
                print(f"[libuipc] Cached verified asset wheel: {local}", flush=True)
                return wheel
        _fetch_wheel(row, partial, expected)
        partial.replace(wheel)
        return wheel


class DownloadDeadlineExceeded(RuntimeError):
    """Stop a stalled provider without suppressing fallback to another host."""


def _fetch_wheel(row: dict, partial: Path, expected: str) -> None:
    """Try ModelScope, then the independent HF revision, with one bounded total budget."""
    primary = os.environ.get("GENIESIM_LIBUIPC_SOURCE") or ARTIFACTS["default_source"]
    providers = [primary]
    if primary == "modelscope" and "huggingface" in ARTIFACTS["sources"]:
        providers.append("huggingface")
    started = time.monotonic()
    deadline = started + DOWNLOAD_TIMEOUT
    errors = []
    for position, provider in enumerate(providers):
        source_deadline = started + DOWNLOAD_TIMEOUT / 2 if position == 0 and len(providers) > 1 else deadline
        url = None
        failure = None
        for attempt in range(3):
            try:
                with _download_deadline(source_deadline):
                    if url is None:
                        url = _artifact_url(row, source_name=provider)
                        print(f"[libuipc] Downloading {row['filename']} ({row['size'] / 1024**2:.1f} MiB)", flush=True)
                    _transfer_artifact(url, partial, row["size"], expected, deadline=source_deadline)
                return
            except (OSError, ValueError, HTTPException, DownloadDeadlineExceeded) as error:
                failure = str(error)
                if isinstance(error, HTTPError) and error.code == 404:
                    failure = "wheel is not published at this source revision"
                    break
                if time.monotonic() >= source_deadline:
                    failure = "download deadline exceeded"
                    break
                if attempt < 2:
                    time.sleep(min(attempt + 1, max(0, source_deadline - time.monotonic())))
        errors.append(f"{provider}: {failure}")
        if position + 1 < len(providers):
            print(f"[libuipc] ModelScope unavailable ({failure}); switching to HuggingFace", flush=True)
    raise RuntimeError(f"Download failed for {row['filename']}: " + "; ".join(errors) + "; rerun to resume")


@contextmanager
def _download_deadline(deadline: float):
    """Bound connection, redirects, headers and body in this main-thread POSIX installer."""

    def expired(signum, frame):
        raise DownloadDeadlineExceeded("Download deadline exceeded; rerun to resume")

    started = time.monotonic()
    remaining = deadline - started
    if remaining <= 0:
        raise DownloadDeadlineExceeded("Download deadline exceeded; rerun to resume")
    previous = signal.signal(signal.SIGALRM, expired)
    timer = signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.000001, timer[0] - (time.monotonic() - started)), timer[1])


def _valid_artifact(path: Path, size: int, expected: str) -> bool:
    """Compare both size and SHA256 without trusting a cache entry's filename."""
    if not path.is_file() or path.stat().st_size != size:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected


@contextmanager
def _download_lock(directory: Path):
    """Serialize writers to one artifact while allowing different architectures in parallel."""
    import fcntl

    path = directory / ".download.lock"
    if path.is_symlink():
        raise RuntimeError("Symlink at libuipc download lock")
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another libuipc download is using this cache; retry later") from error
        yield


def _artifact_url(row: dict, *, source_name: str | None = None) -> str:
    """Resolve a real hosting commit, then fetch only the fixed size/hash artifact."""
    primary = os.environ.get("GENIESIM_LIBUIPC_SOURCE") or ARTIFACTS["default_source"]
    name = source_name or primary
    if name not in {"modelscope", "huggingface"} or name not in ARTIFACTS["sources"]:
        raise RuntimeError(f"Unknown libuipc source: {name}")
    source = ARTIFACTS["sources"][name]
    endpoint = source["endpoint"].rstrip("/")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Invalid libuipc source endpoint")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source["dataset"]):
        raise RuntimeError("Invalid libuipc dataset identity")
    override = os.environ.get("GENIESIM_LIBUIPC_REVISION") if name == primary else None
    if override and not re.fullmatch(r"[a-f0-9]{40}", override):
        raise RuntimeError("GENIESIM_LIBUIPC_REVISION must be a pinned 40-character commit")
    revision = override or source.get("revision")
    if not revision:
        if name == "modelscope":
            query = urlencode({"Ref": source.get("ref", "master"), "PageNumber": 1, "PageSize": 1})
            metadata_url = f"{endpoint}/api/v1/datasets/{source['dataset']}/commits?{query}"
        else:
            metadata_url = f"{endpoint}/api/datasets/{source['dataset']}"
        with urlopen(metadata_url, timeout=30) as response:
            metadata = json.loads(response.read(8 * 1024 * 1024 + 1))
        if not isinstance(metadata, dict):
            raise ValueError("Invalid libuipc source metadata")
        if name == "modelscope":
            data = metadata.get("Data")
            commits = data.get("Commit") if isinstance(data, dict) else None
            if (
                metadata.get("Code") != 200
                or not isinstance(commits, list)
                or not commits
                or not isinstance(commits[0], dict)
            ):
                raise ValueError("Invalid ModelScope commit metadata")
            revision = commits[0].get("Id")
        else:
            revision = metadata.get("sha")
    if not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("libuipc source must resolve to a pinned 40-character commit")
    print(f"[libuipc] Source: {name}; revision: {revision}", flush=True)
    if name == "modelscope":
        query = urlencode({"Source": "SDK", "Revision": revision, "FilePath": row["path"], "View": False})
        return f"{endpoint}/api/v1/datasets/{source['dataset']}/repo?{query}"
    path = "/".join(quote(segment, safe="") for segment in row["path"].split("/"))
    return f"{endpoint}/datasets/{source['dataset']}/resolve/{revision}/{path}"


def _transfer_artifact(url: str, partial: Path, size: int, expected: str, *, deadline: float) -> None:
    """Resume a compatible HTTP range, verify the complete file, and reject bad bytes."""
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= size and partial.exists():
        if _valid_artifact(partial, size, expected):
            return
        partial.unlink()
        offset = 0
    request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Download deadline exceeded")
    with urlopen(request, timeout=min(90, remaining)) as response:
        if offset and response.status == 206:
            if not re.fullmatch(rf"bytes {offset}-\d+/{size}", response.headers.get("Content-Range", "")):
                raise ValueError("Incompatible Content-Range")
        elif response.status == 200:
            offset = 0
        else:
            raise ValueError(f"Unexpected HTTP status: {response.status}")
        with partial.open("ab" if offset else "wb") as output:
            received = offset
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Download deadline exceeded")
                block = response.read1(1024 * 1024)
                if time.monotonic() >= deadline:
                    raise TimeoutError("Download deadline exceeded")
                if not block:
                    break
                received += len(block)
                if received > size:
                    partial.unlink(missing_ok=True)
                    raise ValueError("Remote wheel exceeds recorded size")
                output.write(block)
    if partial.stat().st_size < size:
        raise ValueError("Incomplete wheel; preserving partial bytes for resume")
    if not _valid_artifact(partial, size, expected):
        partial.unlink(missing_ok=True)
        raise ValueError("Wheel size or SHA256 mismatch")


def _installation_site() -> tuple[Path, Path | None]:
    """Find the existing wheel's site, refusing editable or unmanaged replacements."""
    try:
        package = packages.distribution("pyuipc")
    except packages.PackageNotFoundError:
        site = Path(sysconfig.get_path("purelib"))
        if (site / "uipc").exists():
            raise RuntimeError("An unmanaged uipc directory exists; use a clean Python environment")
        return site, None
    metadata_file = next((path for path in package.files or [] if str(path).endswith(".dist-info/METADATA")), None)
    if metadata_file is None or any(
        Path(path).parts[0] not in {"uipc", Path(metadata_file).parts[0]} for path in package.files or []
    ):
        raise RuntimeError("Cannot replace an editable or nonstandard pyuipc installation automatically")
    info = Path(package.locate_file(metadata_file)).parent
    if info.parent.resolve() not in {
        Path(sysconfig.get_path("purelib")).resolve(),
        Path(sysconfig.get_path("platlib")).resolve(),
    }:
        raise RuntimeError(
            "Existing pyuipc is outside the current Python environment; use a clean evaluation environment"
        )
    return info.parent, info


@contextmanager
def _installation_lock():
    """Reject concurrent changes to this interpreter's native runtime without waiting."""
    import fcntl

    lock_path = Path(sysconfig.get_path("purelib")) / ".libuipc-install.lock"
    # flock does not require a writable descriptor. Reuse the root-created
    # lock for read-only checks from the remapped container user as well.
    try:
        lock = lock_path.open("r")
    except FileNotFoundError:
        lock = lock_path.open("a")
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another libuipc installation is modifying this Python environment; retry later"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def _defer_interrupts():
    """Keep each filesystem change and its rollback record indivisible to signals."""
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def install(wheel: Path, *, no_deps: bool = False) -> None:
    """Serialize native replacement, including old-state inspection and rollback."""
    with _installation_lock():
        _install(wheel, no_deps=no_deps)


def _dependencies_satisfied(requirements: list[str]) -> bool:
    """Check declared dependencies, including transitive requirements and extras."""
    try:
        from packaging.requirements import Requirement

        pending = [(raw, "") for raw in requirements]
        checked = set()
        while pending:
            raw, extra = pending.pop()
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": extra}):
                continue
            distribution = packages.distribution(requirement.name)
            if requirement.url or not requirement.specifier.contains(distribution.version, prereleases=True):
                return False
            key = (re.sub(r"[-_.]+", "-", requirement.name).lower(), tuple(sorted(requirement.extras)))
            if key in checked:
                continue
            checked.add(key)
            pending.extend((raw, extra) for raw in distribution.requires or [] for extra in {""} | requirement.extras)
        return True
    except (ImportError, packages.PackageNotFoundError, ValueError):
        return False


def _install_dependencies(requirements: list[str], directory: Path) -> None:
    """Fill missing Python dependencies without changing installed versions."""
    if _dependencies_satisfied(requirements):
        print("[libuipc] Python dependencies are already satisfied; skipping pip dependency installation", flush=True)
        return
    constraints = directory / "installed.txt"
    # apt and pip may both provide metadata for the same distribution.
    # Pin the interpreter-selected version, not every shadowed version.
    names = {
        re.sub(r"[-_.]+", "-", name).lower() for dist in packages.distributions() if (name := dist.metadata["Name"])
    } - {"pyuipc"}
    constraints.write_text("".join(f"{name}=={packages.version(name)}\n" for name in sorted(names)))
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-c", str(constraints), *requirements],
        check=True,
        timeout=180,
    )


def install_python_dependencies() -> None:
    """Prepare both wheels' Python requirements during a GPU-free image build."""
    requirements = set()
    for sm in WHEELS:
        with ZipFile(select_wheel(sm)) as archive:
            path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            metadata = Parser().parsestr(archive.read(path).decode())
            requirements.update(metadata.get_all("Requires-Dist") or [])
    with _installation_lock(), tempfile.TemporaryDirectory(prefix="libuipc-deps-") as directory:
        _install_dependencies(sorted(requirements), Path(directory))
    print("[libuipc] Python dependencies installed; GPU wheel selection is deferred to runtime", flush=True)


def _remove_installation_path(path: Path) -> None:
    """Remove a package directory or its link without following external targets."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _install(wheel: Path, *, no_deps: bool = False) -> None:
    """Validate a staged wheel before replacing pyuipc; restore old directories on failure."""
    with ZipFile(wheel) as archive:
        metadata_path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_path).decode())
        wheel_metadata = Parser().parsestr(archive.read(metadata_path.replace("METADATA", "WHEEL")).decode())
    filename = f"{metadata['Name']}-{metadata['Version']}-{wheel_metadata['Tag']}.whl"
    site, old_info = _installation_site()
    if (
        sys.prefix == sys.base_prefix
        and site == Path(sysconfig.get_path("purelib"))
        and (Path(sysconfig.get_path("stdlib")) / "EXTERNALLY-MANAGED").exists()
        and os.environ.get("PIP_BREAK_SYSTEM_PACKAGES", "").lower() not in {"1", "true", "yes"}
    ):
        raise RuntimeError("Python is externally managed; use the evaluation virtual environment")
    site.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=".libuipc-install-", dir=site))
    retain_backup = False
    try:
        staged = directory / filename
        shutil.copyfile(wheel, staged)
        command = [sys.executable, "-m", "pip", "install"]
        if not no_deps:
            _install_dependencies(metadata.get_all("Requires-Dist") or [], Path(directory))
        payload = directory / "payload"
        subprocess.run(
            [*command, "--no-deps", "--no-compile", "--target", str(payload), str(staged)],
            check=True,
            timeout=180,
        )
        new_info = next(payload.glob("pyuipc-*.dist-info"))
        if {path.name for path in payload.iterdir()} != {"uipc", new_info.name}:
            raise RuntimeError("Unexpected native wheel layout; refusing to modify the environment")
        _link_cuda_runtime(payload, wheel)
        check_native(payload)
        backup = directory / "backup"
        backup.mkdir()
        old_paths = [
            old for old in [site / "uipc", *([old_info] if old_info else [])] if old.exists() or old.is_symlink()
        ]
        # OverlayFS can reject renaming a directory inherited from an image.
        # Finish every backup before deleting anything from the live installation.
        for old in old_paths:
            if old.is_symlink():
                (backup / old.name).symlink_to(old.readlink())
            else:
                shutil.copytree(old, backup / old.name, symlinks=True)
        saved = []
        published = []
        try:
            for old in old_paths:
                with _defer_interrupts():
                    saved.append(old)
                    _remove_installation_path(old)
            for path in [payload / "uipc", new_info]:
                target = site / path.name
                with _defer_interrupts():
                    path.rename(target)
                    published.append(target)
            check_native()
        except BaseException as error:
            rollback_errors = []
            with _defer_interrupts():
                for target in reversed(published):
                    if target in saved:
                        continue  # Remove this path as part of restoring its backup.
                    try:
                        _remove_installation_path(target)
                    except BaseException as rollback_error:
                        rollback_errors.append(f"{target}: {rollback_error}")
                for old in reversed(saved):
                    try:
                        _remove_installation_path(old)
                        (backup / old.name).rename(old)
                    except BaseException as rollback_error:
                        rollback_errors.append(f"{old}: {rollback_error}")
                if rollback_errors:
                    retain_backup = True
                    print(f"[libuipc] Rollback incomplete; recovery files retained at {backup}", flush=True)
            if rollback_errors:
                raise RuntimeError(
                    f"Native installation failed and rollback was incomplete; recovery files: {backup}; "
                    + "; ".join(rollback_errors)
                ) from error
            print("[libuipc] Restored the previous native installation", flush=True)
            raise
    finally:
        if not retain_backup:
            shutil.rmtree(directory)


def _cuda_runtime_roots(major: int) -> list[Path]:
    """Keep each CUDA provider together: cuSPARSE has the same SONAME in 12/13."""
    return [
        Path(f"/opt/geniesim/cuda{major}/nvidia"),
        Path("/isaac-sim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"),
        Path("/isaac-sim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle/nvidia"),
        Path("/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia"),
        *[Path(path) / "nvidia" / f"cu{major}" for path in sys.path if path],
        *[Path(path) / "nvidia" for path in sys.path if path],
        *sorted(Path("/usr/local").glob(f"cuda-{major}.*")),
        Path("/usr/local/cuda"),
    ]


def _link_cuda_runtime(payload: Path, wheel: Path) -> None:
    """Use the backend's $ORIGIN RUNPATH without changing the shared SDK loader."""
    major, solver = {"pyuipc-sm89.whl": (12, 11), "pyuipc-sm120.whl": (12, 11)}[wheel.name]
    components = {
        f"libcudart.so.{major}": "cuda_runtime",
        f"libcublas.so.{major}": "cublas",
        f"libcublasLt.so.{major}": "cublas",
        "libcusparse.so.12": "cusparse",
        f"libcusolver.so.{solver}": "cusolver",
        f"libnvJitLink.so.{major}": "nvjitlink",
    }
    for root in _cuda_runtime_roots(major):
        layouts = [
            {name: root / directory / name for name in components}
            for directory in ("", "lib", "lib64", "targets/x86_64-linux/lib")
        ]
        layouts.append({name: root / component / "lib" / name for name, component in components.items()})
        for libraries in layouts:
            if not all(path.is_file() for path in libraries.values()):
                continue
            native = payload / "uipc" / "_native"
            record = next(payload.glob("pyuipc-*.dist-info/RECORD"))
            with record.open("a", newline="") as stream:
                writer = csv.writer(stream)
                for name, source in libraries.items():
                    target = native / name
                    if target.exists() or target.is_symlink():
                        raise RuntimeError(f"Native wheel already owns CUDA library {name}")
                    target.symlink_to(source.resolve())
                    writer.writerow([f"uipc/_native/{name}", "", ""])
            print(f"[libuipc] Linked private CUDA {major} runtime from {root}", flush=True)
            return
    # A configured system loader may already expose the complete runtime.
    try:
        check_native(payload)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"CUDA {major} runtime unavailable; use the matching GenieSim image "
            "or install its CUDA runtime libraries before retrying"
        ) from error


def installed_matches(wheel: Path) -> bool:
    """Only skip a package with the selected version and original archive hash."""
    try:
        package = packages.distribution("pyuipc")
        direct_url = json.loads(package.read_text("direct_url.json") or "{}")
        with ZipFile(wheel) as archive:
            path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            version = Parser().parsestr(archive.read(path).decode())["Version"]
        digest = next(digest for name, digest in WHEELS.values() if name == wheel.name)
        return (
            package.version == version and direct_url.get("archive_info", {}).get("hashes", {}).get("sha256") == digest
        )
    except (packages.PackageNotFoundError, ValueError, TypeError):
        return False


def check_native(root: Path | None = None) -> None:
    """Check native linking in a child process, without running a GPU solve."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import ctypes, pathlib, sys; "
            "sys.path.insert(0, sys.argv[1]) if len(sys.argv) > 1 else None; "
            "import uipc; "
            "assert len(sys.argv) == 1 or pathlib.Path(uipc.__file__).resolve().is_relative_to(sys.argv[1]); "
            "ctypes.CDLL(str(pathlib.Path(uipc.__file__).parent / '_native' / 'libuipc_backend_cuda.so'))",
            *([str(root)] if root is not None else []),
        ],
        check=True,
        timeout=30,
    )


def install_image(sm: int, wheel: Path) -> None:
    """Bake one architecture into an image without requiring a build-time GPU."""
    marker = Path(sysconfig.get_path("purelib")) / ".libuipc-image.json"
    with _installation_lock():
        _install(wheel)
        marker.write_text(json.dumps({"sm": sm, "sha256": WHEELS[sm][1]}) + "\n")
    print(f"[libuipc] Image preinstalled for sm_{sm}; runtime installation disabled", flush=True)


def verify_image(sm: int, wheel: Path) -> bool:
    """Check a baked artifact without pip or automatic runtime replacement."""
    marker = Path(sysconfig.get_path("purelib")) / ".libuipc-image.json"
    if not marker.exists():
        return False
    with _installation_lock():
        try:
            baked = json.loads(marker.read_text())
            if baked != {"sm": sm, "sha256": WHEELS[sm][1]} or not installed_matches(wheel):
                raise RuntimeError(f"Image artifact does not match GPU sm_{sm}")
            with ZipFile(wheel) as archive:
                path = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
                requirements = Parser().parsestr(archive.read(path).decode()).get_all("Requires-Dist") or []
            if not _dependencies_satisfied(requirements):
                raise RuntimeError("Image Python dependencies are missing or incompatible")
            check_native()
        except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                f"{error}; rebuild the image for this GPU with geniesim docker build "
                "(or docker6.2 build for the dedicated 6.2 image)"
            ) from error
    print("[libuipc] Preinstalled image artifact verified; no installation required", flush=True)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index to select (default: 0)")
    parser.add_argument(
        "--dry-run", action="store_true", help="detect GPU and verify the selected wheel without installing"
    )
    parser.add_argument(
        "--no-deps", action="store_true", help="install only the wheel into an already prepared environment"
    )
    parser.add_argument(
        "--if-needed", action="store_true", help="skip a matching installed artifact and verify native loading"
    )
    parser.add_argument(
        "--dependencies-only", action="store_true", help="install both wheels' Python dependencies without a GPU"
    )
    parser.add_argument(
        "--build-sm", type=int, choices=sorted(WHEELS), help="preinstall an image for this SM without a GPU"
    )
    parser.add_argument(
        "--download-sm",
        type=int,
        choices=sorted(WHEELS),
        help="cache and verify this SM's wheel without GPU or runtime ABI checks",
    )
    args = parser.parse_args(argv)
    if args.dependencies_only and (args.dry_run or args.no_deps or args.if_needed):
        parser.error("--dependencies-only cannot be combined with --dry-run, --no-deps or --if-needed")
    if args.build_sm is not None and (args.dependencies_only or args.dry_run or args.no_deps or args.if_needed):
        parser.error("--build-sm cannot be combined with other installation modes")
    if args.download_sm is not None and (
        args.build_sm is not None or args.dependencies_only or args.dry_run or args.no_deps or args.if_needed
    ):
        parser.error("--download-sm cannot be combined with other installation modes")

    def cancelled(signum, frame):
        raise InterruptedError("libuipc installation cancelled; rolling back native files")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        if args.download_sm is not None:
            print(f"[libuipc] Verified wheel: {select_wheel(args.download_sm)}", flush=True)
            return
        validate_platform()
        if args.dependencies_only:
            install_python_dependencies()
            return
        if args.build_sm is not None:
            install_image(args.build_sm, select_wheel(args.build_sm))
            return
        sm = detect_sm(args.gpu)
        wheel = select_wheel(sm)
        print(f"[libuipc] GPU {args.gpu}: sm_{sm} -> {wheel.name}", flush=True)
        if not args.dry_run:
            if verify_image(sm, wheel):
                return
            matched = False
            if args.if_needed:
                with _installation_lock():
                    matched = installed_matches(wheel)
                    if matched:
                        try:
                            check_native()
                        except (OSError, subprocess.SubprocessError):
                            print("[libuipc] Repairing the installed artifact's native runtime", flush=True)
                            matched = False
                        else:
                            print("[libuipc] Selected artifact is already installed", flush=True)
            if not matched:
                install(wheel, no_deps=args.no_deps)
            if args.if_needed:
                print("[libuipc] Native library loaded successfully", flush=True)
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, f"[libuipc] {error}\n")
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
