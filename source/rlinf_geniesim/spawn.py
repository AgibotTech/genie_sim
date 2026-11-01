# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Launch a private local server in a prepared Linux simulation runtime."""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from rlinf_geniesim.comm.zmq.client import ZmqClient
from rlinf_geniesim.owner_watchdog import ENV_OWNER_PID, ENV_OWNER_START, process_start_time

# Separate from the manual launcher's 0..101, outside Linux's default ephemeral ports.
DOMAIN_FIRST, DOMAIN_LAST = 215, 232
_DROPPED_ENV = {
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONNOUSERSITE",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "AMENT_PREFIX_PATH",
    "CMAKE_PREFIX_PATH",
    "COLCON_PREFIX_PATH",
    "ROS_PACKAGE_PATH",
    "BASH_ENV",
    "ENV",
}
_LAUNCH = """
set -e
source "$1"
source "$2"
export PYTHONPATH="$3/source:$3/source/geniesim_benchmark/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$4" "$3/source/rlinf_geniesim/scripts/sim_server.py" --comm-config "$5"
"""


class SpawnError(RuntimeError):
    """A private server could not acquire its resources or become ready."""


def _child_environment(code_dir: Path, assets_dir: Path, ws_build: Path, tag: str, domain: int) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _DROPPED_ENV and not k.startswith(("CONDA_", "VIRTUAL_ENV", "RAY_"))
    }
    env.update(
        PATH=os.defpath,
        SIM_REPO_ROOT=str(code_dir),
        GENIESIM_ASSETS_PATH=str(assets_dir),
        RLINF_WS_BUILD=str(ws_build),
        RLINF_SERVER_TAG=tag,
        ROS_DOMAIN_ID=str(domain),
    )
    env[ENV_OWNER_PID] = str(os.getpid())
    start = process_start_time(os.getpid())
    if start is None:
        raise SpawnError("Cannot read owner identity from /proc")
    env[ENV_OWNER_START] = start
    for key, value in {
        "ACCEPT_EULA": "Y",
        "OMNI_KIT_ACCEPT_EULA": "Y",
        "PRIVACY_CONSENT": "Y",
        "RLINF_ISAAC_PYTHON": "/isaac-sim/python.sh",
        "RLINF_TILED_HEADLESS": env.get("HEADLESS", "1"),
        "RLINF_TILED_FABRIC": "1",
        "RLINF_TILED_READY_TIMEOUT": env.get("READY_TIMEOUT", "600"),
        "RLINF_FUSED_READY_TIMEOUT": env.get("READY_TIMEOUT", "600"),
    }.items():
        env.setdefault(key, value)
    return env


class SpawnedServer:
    """Own one server process. stop() closes its client and reaps the process.

    The client is idle on return: use set_backends/init before reset/step.
    Keep client calls and stop on the creating thread, as required by ZMQ.
    """

    def __init__(
        self,
        process: subprocess.Popen,
        *,
        address: str,
        run_dir: Path,
        log_path: Path,
        tag: str,
        ros_domain: int,
        stop_timeout_s: float,
        slot_library: Path,
    ) -> None:
        self.process = process
        self.address = address
        self.run_dir = run_dir
        self.log_path = log_path
        self.tag = tag
        self.ros_domain = ros_domain
        self.client: ZmqClient | None = None
        self._stop_timeout_s = stop_timeout_s
        self._slot_library = slot_library
        self._stopped = False

    def alive(self) -> bool:
        """Return whether the server process is still running."""
        return self.process.poll() is None

    def _error(self, reason: str) -> SpawnError:
        with self.log_path.open("rb") as log:
            log.seek(0, os.SEEK_END)
            log.seek(max(0, log.tell() - 4096))
            tail = log.read().decode(errors="replace")
        return SpawnError(f"{reason} (tag={self.tag}, log={self.log_path}):\n{tail}")

    def _group_members(self) -> dict[int, str]:
        members = {}
        identity = f"RLINF_SERVER_TAG={self.tag}".encode()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[-1].split()
                if fields[0] in ("Z", "X") or fields[2:4] != [str(self.process.pid)] * 2:
                    continue
                if identity in (entry / "environ").read_bytes().split(b"\0"):
                    members[int(entry.name)] = fields[19]
            except (OSError, IndexError):
                continue
        return members

    def _stop_group(self) -> None:
        # A sourced setup script may leave helpers after the group leader exits.
        members = self._group_members()
        for signum, grace in ((signal.SIGTERM, self._stop_timeout_s), (signal.SIGKILL, 5.0)):
            deadline = time.monotonic() + grace
            signaled = set()
            while members:
                for pid, start in members.items():
                    if (pid, start) in signaled:
                        continue
                    try:
                        fd = os.pidfd_open(pid)
                        try:
                            if process_start_time(pid) == start:
                                signal.pidfd_send_signal(fd, signum)
                        finally:
                            os.close(fd)
                    except ProcessLookupError:
                        pass
                    signaled.add((pid, start))
                self.process.poll()
                time.sleep(0.02)
                members = self._group_members()
                if time.monotonic() >= deadline:
                    break
            if not members:
                self.process.wait(timeout=5)
                return
        raise self._error(f"Server processes did not exit: {sorted(members)}")

    def stop(self) -> None:
        """Stop this server and setup helpers, escalate if needed, and retain logs."""
        if self._stopped:
            return
        try:
            self._stop_group()
            # Reuse the launcher's PID-scoped cleanup, including reload generations.
            subprocess.run(
                [
                    "/bin/bash",
                    "-c",
                    'source "$1"; slot_cleanup_shm "$2"',
                    "_",
                    str(self._slot_library),
                    str(self.process.pid),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                timeout=10,
                env={"PATH": os.defpath},
            )
            self._stopped = True
        finally:
            if self.client is not None:
                self.client.close()
                self.client = None

    def __enter__(self) -> SpawnedServer:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


def spawn_server(
    tag: str,
    *,
    ros_domain: int,
    code_dir: str | Path | None = None,
    assets_dir: str | Path | None = None,
    run_root: str | Path | None = None,
    log_dir: str | Path | None = None,
    python: str | Path | None = None,
    ready_timeout_s: float = 900.0,
    request_timeout_s: float = 60.0,
    stop_timeout_s: float = 15.0,
    jpeg_quality: int = 85,
) -> SpawnedServer:
    """Start an idle server with its own ROS context and a local ZMQ client.

    Requires a prepared ROS workspace, simulator runtime and asset checkout.
    ros_domain must be unique on the ROS network and in 215..232. Callers on
    one host must share run_root for domain locking. CUDA_VISIBLE_DEVICES is
    inherited; python selects the server interpreter, independently of the caller.
    Startup failure or interruption stops the child before propagating the error.
    """
    if not sys.platform.startswith("linux"):
        raise RuntimeError("spawn_server requires Linux")
    import fcntl

    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", tag):
        raise ValueError("tag must contain 1..32 letters, digits, underscores or hyphens")
    if type(ros_domain) is not int or not DOMAIN_FIRST <= ros_domain <= DOMAIN_LAST:
        raise ValueError(f"ros_domain must be in {DOMAIN_FIRST}..{DOMAIN_LAST}, got {ros_domain}")
    for name, value in (("ready", ready_timeout_s), ("request", request_timeout_s), ("stop", stop_timeout_s)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} timeout must be finite and positive, got {value}")
    if type(jpeg_quality) is not int or not 0 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer in 0..100")
    code_dir = Path(code_dir or os.environ.get("SIM_REPO_ROOT") or Path(__file__).resolve().parents[2]).resolve()
    assets_dir = Path(
        assets_dir or os.environ.get("GENIESIM_ASSETS_PATH") or os.environ.get("SIM_ASSETS_ROOT") or "/geniesim_assets"
    ).resolve()
    root = Path(run_root or os.environ.get("RLINF_RUN_ROOT") or tempfile.gettempdir()).resolve()
    ws = Path(os.environ.get("RLINF_WS_BUILD") or os.environ.get("GENIESIM_ENGINE_WS") or "/tmp/ros_ws_build").resolve()
    ros = Path(
        os.environ.get("RLINF_ROS_SETUP") or f"/opt/ros/{os.environ.get('ROS_DISTRO', 'jazzy')}/setup.bash"
    ).resolve()
    python = Path(python or os.environ.get("RLINF_SERVER_PYTHON") or "/usr/bin/python3")
    if not python.is_absolute():
        raise ValueError("python must be an absolute interpreter path")
    scripts = code_dir / "source/rlinf_geniesim/scripts"
    for path in (scripts / "sim_server.py", scripts / "_server_slot.sh", ros, ws / "install/setup.bash", python):
        if not path.is_file():
            raise FileNotFoundError(f"Required server runtime file not found: {path}")
    with python.open("rb") as executable:
        if executable.read(4) != b"\x7fELF":
            raise ValueError("python must be an executable interpreter, not a script wrapper")
    if not assets_dir.is_dir():
        raise FileNotFoundError(f"Asset directory not found: {assets_dir}")
    root.mkdir(parents=True, exist_ok=True)
    # The child inherits this lock so owner death does not release a live server's domain.
    with (root / f"rlinf_spawn_domain_{ros_domain}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SpawnError(f"ROS domain {ros_domain} is already owned under {root}") from exc
        run_dir = Path(tempfile.mkdtemp(prefix=f"rlinf_spawn_{tag}_", dir=root))
        tag = run_dir.name
        log_path = Path(log_dir).resolve() / f"{tag}.log" if log_dir else run_dir / "sim.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Linux abstract sockets disappear with their process and avoid pathname limits.
        address = f"ipc://@{tag}"
        config = run_dir / "comm.json"
        config.write_text(
            json.dumps({"comm": {"backend": "zmq", "bind": address, "jpeg_quality": jpeg_quality, "bootstrap_out": ""}})
        )
        env = _child_environment(code_dir, assets_dir, ws, tag, ros_domain)
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [
                    "/bin/bash",
                    "-c",
                    _LAUNCH,
                    "_",
                    str(ros),
                    str(ws / "install/setup.bash"),
                    str(code_dir),
                    str(python),
                    str(config),
                ],
                env=env,
                cwd=code_dir,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                pass_fds=(lock.fileno(),),
            )
        server = SpawnedServer(
            process,
            address=address,
            run_dir=run_dir,
            log_path=log_path,
            tag=tag,
            ros_domain=ros_domain,
            stop_timeout_s=stop_timeout_s,
            slot_library=scripts / "_server_slot.sh",
        )
        try:
            _wait_ready(server, ready_timeout_s)
            server.client = ZmqClient(address=address, timeout_s=request_timeout_s)
        except BaseException:
            server.stop()
            raise
    return server


def _wait_ready(server: SpawnedServer, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not server.alive():
            raise server._error(f"Server exited with {server.process.returncode} before becoming ready")
        client = ZmqClient(address=server.address, timeout_s=min(0.2, max(0.001, (deadline - time.monotonic()) / 2)))
        try:
            # Readback works while idle without loading the task catalog.
            client.randomization_readback()
            return
        except TimeoutError:
            pass
        except RuntimeError as exc:
            raise server._error(f"Server readiness probe failed: {exc}") from exc
        finally:
            client.close()
    raise server._error(f"Server not ready after {timeout_s:g}s")
