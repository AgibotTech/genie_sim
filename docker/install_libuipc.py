# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

"""Install the bundled native libuipc wheel for SM 89 or 120, without Newton."""

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
from email.parser import Parser
from contextlib import contextmanager
from importlib import metadata as packages
from pathlib import Path
from zipfile import ZipFile

WHEELS = {
    89: ("pyuipc-sm89.whl", "ff838f588f84c695a7fd41dd8769d8068f42d02987896e89882494e4a17e7c6d"),
    120: ("pyuipc-sm120.whl", "3a7d0ea16b2221797d78b14de964042222569cd29f77e64e5d83654ff0e467f3"),
}


def validate_platform() -> None:
    """Both supplied native extensions require this ABI, regardless of wheel tag."""
    if (
        sys.platform != "linux"
        or platform.machine() != "x86_64"
        or platform.python_implementation() != "CPython"
        or sys.version_info[:2] != (3, 12)
    ):
        raise RuntimeError("The bundled libuipc wheels require Linux x86_64 CPython 3.12")


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
    """Reject unsupported GPUs and changed or missing artifacts before pip runs."""
    if sm not in WHEELS:
        raise RuntimeError(f"Unsupported GPU sm_{sm}; only sm_89 and sm_120 are supported")
    name, expected = WHEELS[sm]
    wheel = Path(__file__).resolve().parents[1] / "3rdparty" / name
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"SHA256 mismatch for {wheel}")
    return wheel


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
        Path("/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia"),
        *[Path(path) / "nvidia" / f"cu{major}" for path in sys.path if path],
        *[Path(path) / "nvidia" for path in sys.path if path],
        *sorted(Path("/usr/local").glob(f"cuda-{major}.*")),
        Path("/usr/local/cuda"),
    ]


def _link_cuda_runtime(payload: Path, wheel: Path) -> None:
    """Use the backend's $ORIGIN RUNPATH without changing the shared SDK loader."""
    major, solver = {"pyuipc-sm89.whl": (13, 12), "pyuipc-sm120.whl": (13, 12)}[wheel.name]
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
    args = parser.parse_args(argv)
    if args.dependencies_only and (args.dry_run or args.no_deps or args.if_needed):
        parser.error("--dependencies-only cannot be combined with --dry-run, --no-deps or --if-needed")
    if args.build_sm is not None and (args.dependencies_only or args.dry_run or args.no_deps or args.if_needed):
        parser.error("--build-sm cannot be combined with other installation modes")

    def cancelled(signum, frame):
        raise InterruptedError("libuipc installation cancelled; rolling back native files")

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
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
