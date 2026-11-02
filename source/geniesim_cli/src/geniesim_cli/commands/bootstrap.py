# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim bootstrap`` — install / re-initialize the geniesim stack.

This is a separate verb from ``status`` even though they share the same
inspection layer (``commands.status.DISTRIBUTIONS`` and ``inspect``):
``status`` reports, ``bootstrap`` fixes. Keeping them as two
modules makes the read-only path completely side-effect-free, so users
can always run ``geniesim status`` on a partially broken install
without triggering a pip resolver.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW
from geniesim_cli.commands.status import DISTRIBUTIONS, inspect


# Topological order (leaves first). The umbrella ``geniesim`` MUST be
# last: its pyproject.toml has bare-name deps on every peer below, and
# those peers are not on PyPI. Installing them first puts them in the
# local site-packages so pip's resolver sees the requirements as
# already-satisfied and never queries the index.
#
# The tier-1 list is **derived** from `source/geniesim/pyproject.toml`'s
# `[project].dependencies` via ``geniesim_cli._tiers``. Tier-2 peers
# (`geniesim_teleop`, `geniesim_scene_agent`, `geniesim_world`) are
# opt-in via ``pip install -e "source/geniesim/[<extra>]"`` and
# deliberately excluded from the default bootstrap flow.
#
# To re-tier a peer, edit `source/geniesim/pyproject.toml` only —
# this list updates automatically.
def _build_targets() -> tuple[str, ...]:
    from geniesim_cli._tiers import tier1

    # `geniesim_cli` is deliberately excluded — re-installing the
    # running CLI mid-execution is a footgun. The umbrella (`geniesim`)
    # comes last so its dep resolution sees all peers already present.
    leaves = [p for p in tier1() if p != "geniesim_cli"]
    return tuple(leaves + ["geniesim"])


_TARGETS: tuple[str, ...] = _build_targets()
_LIBUIPC_TIMEOUT = 300
_LIBUIPC_STOP_TIMEOUT = 15

_DOWNLOAD_SOURCES: dict[str, tuple[tuple[str, str], ...]] = {
    "geniesim_assets": (
        ("HuggingFace", "https://huggingface.co/datasets/agibot-world/GenieSimAssets"),
        ("ModelScope", "https://modelscope.cn/datasets/agibot_world/GenieSimAssets"),
    ),
}


def _install_command(dist_name: str) -> list[str] | None:
    """Return the pip command that would install ``dist_name`` from a local checkout.

    Returns ``None`` when no local source tree can be located. PyPI is
    *never* a fallback: ``geniesim`` / ``geniesim_assets`` are private
    distributions and must be installed from a local checkout. Callers are
    expected to surface a friendly error and ask the user to clone /
    `pip install -e` the source tree manually.

    Resolution order:
      1. If ``dist_name`` is already importable, use its on-disk source root
         (the editable checkout it was installed from).
      2. Otherwise, look in the CLI's checkout or the current checkout,
         then the local asset-source locations.
    """
    from pathlib import Path

    try:
        spec = importlib.util.find_spec(dist_name)
    except ModuleNotFoundError:
        spec = None
    if spec is not None and spec.origin is not None:
        pkg_dir = Path(spec.origin).resolve().parent
        # Locate the source root the package was installed from. The
        # *standard* layout (`<root>/src/<pkg>/__init__.py`) means the
        # source root is two ``.parent`` hops up. But some peers ship
        # as a *flat* install (``<root>/<pkg>/__init__.py`` — no ``src/``
        # shim), and a blind ``.parent.parent`` there walks two levels
        # above the package and lands on a meaningless directory, which
        # then breaks the ``pip install -e <…>`` command.
        # Walk outward and pick an ancestor with packaging metadata.
        # A wheel's site-packages directory is not an editable source tree.
        source_root: Path | None = None
        for candidate in (pkg_dir.parent.parent, pkg_dir.parent, pkg_dir):
            if (candidate / "pyproject.toml").is_file() or (candidate / "setup.py").is_file():
                source_root = candidate
                break
        if source_root is not None:
            return [sys.executable, "-m", "pip", "install", "-e", str(source_root)]

    from geniesim_cli._tiers import _source_pyproject

    umbrella = _source_pyproject()
    if umbrella is not None:
        candidate = umbrella.parent.parent / dist_name
        if (candidate / "pyproject.toml").is_file():
            return [sys.executable, "-m", "pip", "install", "-e", str(candidate.resolve())]

    cli_spec = importlib.util.find_spec("geniesim_cli")
    if cli_spec is not None and cli_spec.origin is not None:
        cli_source_root = Path(cli_spec.origin).resolve().parent.parent.parent
        candidates = [
            cli_source_root.parent / dist_name,
            cli_source_root.parent.parent / "assets" if dist_name == "geniesim_assets" else None,
            Path.home() / "assets" if dist_name == "geniesim_assets" else None,
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if (candidate / "pyproject.toml").is_file():
                return [sys.executable, "-m", "pip", "install", "-e", str(candidate.resolve())]
    return None


def _prompt(question: str, default: bool) -> bool:
    """Y/N prompt; default is the value returned on bare Enter / EOF.

    ``default=True`` renders ``[Y/n]``; ``default=False`` renders ``[y/N]``.
    """
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"   {BOLD}{question} {suffix}: {RST}").strip().lower()
    except EOFError:
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def _run_libuipc(command: list[str]) -> int:
    """Stream progress and bound the optional installer and its entire process group."""
    import os
    import signal
    import subprocess

    def cancelled(signum, frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, cancelled)
    try:
        process = subprocess.Popen(command, start_new_session=os.name == "posix")
        try:
            return process.wait(timeout=_LIBUIPC_TIMEOUT)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"libuipc installation timed out after {_LIBUIPC_TIMEOUT} seconds") from error
        finally:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            elif process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=_LIBUIPC_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                pass
            finally:
                if os.name == "posix":
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                elif process.poll() is None:
                    process.kill()
                process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)


def _try_install_libuipc() -> bool:
    """Attempt the default native runtime in this interpreter; failures are warnings."""
    from geniesim_cli import _env

    try:
        override = _env.repo_root() or _env.repo_path()
        roots = (
            [Path(override).expanduser().resolve()]
            if override
            else [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
        )
        script = next(
            (root / "docker/install_libuipc.py" for root in roots if (root / "docker/install_libuipc.py").is_file()),
            None,
        )
        if script is None:
            raise FileNotFoundError("Cannot find docker/install_libuipc.py; set GENIESIM_REPO_ROOT to the checkout")
        print(f"   {YELLOW}📡 Checking/installing default libuipc runtime (optional)...{RST}", flush=True)
        code = _run_libuipc([sys.executable, str(script), "--if-needed"])
        if code:
            raise RuntimeError(f"native installer exited with code {code}")
    except (OSError, RuntimeError) as error:
        print(f"   {YELLOW}⚠️  libuipc unavailable: {error}. Continuing without it.{RST}")
        print(f"   {DIM}Retry in the evaluation Python 3.12 environment: geniesim bootstrap --libuipc-only{RST}")
        return False
    print(f"   {GREEN}✅ libuipc installed and native library loaded{RST}")
    return True


def _report_completion(libuipc_ok: bool, *, native_only: bool = False) -> None:
    """Distinguish optional native warnings from required-peer failures."""
    label = "libuipc bootstrap" if native_only else "geniesim bootstrap"
    if libuipc_ok:
        print(f"{BOLD}{GREEN}🎉 {label} complete!{RST}")
    else:
        print(f"{BOLD}{YELLOW}⚠️  {label} complete; libuipc unavailable (optional).{RST}")
    print(f"   {DIM}Verify required modules with:{RST} {CYAN}geniesim status{RST}")


def run(args: list[str]) -> None:
    """Bootstrap / re-initialize the geniesim stack.

    Installs every peer distribution from its local source tree in
    topological order (leaves first), then the ``geniesim`` umbrella last.
    Peers are not on PyPI, so install order matters: each peer must be
    in ``site-packages`` before pip evaluates the umbrella's bare-name
    dependency on it, otherwise pip queries the index and fails with
    ``No matching distribution found for geniesim_benchmark``.
    """
    import subprocess

    if "--help" in args or "-h" in args:
        print(f"{BOLD}Usage: geniesim bootstrap{RST}")
        print("Install or repair required peers from local source trees; optional peers are skipped.")
        print("Optional native libuipc is checked by default; --libuipc-only skips peers and prompts.")
        return

    import argparse

    parser = argparse.ArgumentParser(prog="geniesim bootstrap")
    parser.add_argument("--libuipc-only", action="store_true", help="check/install only the optional native runtime")
    options = parser.parse_args(args)

    print(f"{BOLD}{MAGENTA}🧞 geniesim bootstrap{RST} {DIM}— bootstrap the geniesim stack{RST}")
    print()

    if options.libuipc_only:
        _report_completion(_try_install_libuipc(), native_only=True)
        return

    specs_by_dist = {s["dist"]: s for s in DISTRIBUTIONS}
    targets: list[tuple[str, dict]] = []
    for dist_name in _TARGETS:
        spec = specs_by_dist[dist_name]
        info = inspect(spec)
        targets.append((dist_name, info))

    fully_healthy = all(
        info["installed"]
        and info["import_ok"]
        and all(ok for _, _, ok, _ in info["deps"])
        and all(ok for _, ok, _ in info["submodules"])
        for _, info in targets
    )

    if fully_healthy:
        print(f"   {GREEN}✅ All targets are healthy.{RST}")
        for dist_name, info in targets:
            mode = "editable" if info["editable"] else "wheel" if info["editable"] is False else "unknown"
            print(f"      {WHITE}{dist_name}{RST} {GREEN}{info['version']}{RST} {DIM}({mode}){RST}")
        print()
        if _prompt("Re-initialize anyway?", default=False):
            to_install = list(_TARGETS)
        else:
            print(f"   {DIM}Nothing to do.{RST}")
            _report_completion(_try_install_libuipc())
            return
    else:
        print(f"   {YELLOW}⚠️  Stack is not fully initialized.{RST}")
        print()
        to_install = []
        for dist_name, info in targets:
            if not info["installed"]:
                print(f"   {RED}❌ {dist_name}{RST} {DIM}— not installed{RST}")
                to_install.append(dist_name)
            elif not info["import_ok"]:
                print(f"   {RED}❌ {dist_name}{RST} {DIM}— installed but import failed{RST}")
                if info["import_err"]:
                    print(f"      {RED}{info['import_err']}{RST}")
                to_install.append(dist_name)
            else:
                bad_deps = [name for name, _, ok, _ in info["deps"] if not ok]
                bad_subs = [name for name, ok, _ in info["submodules"] if not ok]
                if bad_deps or bad_subs:
                    print(f"   {YELLOW}⚠️  {dist_name}{RST} {DIM}— installed, with issues:{RST}")
                    for name in bad_deps:
                        print(f"      {RED}❌ runtime dep:{RST} {WHITE}{name}{RST}")
                    for name in bad_subs:
                        print(f"      {RED}❌ submodule:{RST}   {WHITE}{name}{RST}")
                    to_install.append(dist_name)
                else:
                    print(f"   {GREEN}✅ {dist_name}{RST} {GREEN}{info['version']}{RST} {DIM}(ok){RST}")
        print()

    if not to_install:
        print(f"   {DIM}Nothing to install.{RST}")
        return

    print(f"   {BOLD}🚀 Install plan:{RST}")
    plan: list[tuple[str, list[str]]] = []
    unresolved: list[str] = []
    for dist_name in to_install:
        cmd = _install_command(dist_name)
        if cmd is None:
            unresolved.append(dist_name)
            print(f"      {RED}❌ {dist_name}{RST} {DIM}— no local source tree found on this machine{RST}")
        else:
            plan.append((dist_name, cmd))
            print(f"      {WHITE}{dist_name}{RST}")
            print(f"        {CYAN}{' '.join(cmd)}{RST}")
    print()

    if unresolved:
        print(f"   {YELLOW}⚠️  These distributions must be installed from a local checkout:{RST}")
        for dist_name in unresolved:
            print()
            print(f"      {WHITE}{dist_name}{RST}")
            sources = _DOWNLOAD_SOURCES.get(dist_name)
            if sources:
                print(f"        {DIM}1. Download the source from one of:{RST}")
                for label, url in sources:
                    print(f"           {DIM}- {label}:{RST} {CYAN}{url}{RST}")
                print(f"        {DIM}   (see{RST} {BOLD}README.md{RST}{DIM} for the latest URLs / instructions){RST}")
                print(f"        {DIM}2. Install it editable, then re-run{RST} {CYAN}geniesim bootstrap{RST}{DIM}:{RST}")
                print(f"           {CYAN}pip install -e <path-to-{dist_name}>{RST}")
            else:
                print(f"        {DIM}1. Clone or check out the source repo for{RST} {WHITE}{dist_name}{RST}{DIM}.{RST}")
                print(f"        {DIM}   (see{RST} {BOLD}README.md{RST}{DIM} for the canonical source URL){RST}")
                print(f"        {DIM}2. Install it editable, then re-run{RST} {CYAN}geniesim bootstrap{RST}{DIM}:{RST}")
                print(f"           {CYAN}pip install -e <path-to-{dist_name}>{RST}")
        print()
        print(f"   {DIM}(PyPI installs are not supported for these private distributions.){RST}")
        print()

    if not plan:
        print(f"   {DIM}Nothing to do automatically. Resolve the unresolved targets above, then re-run.{RST}")
        sys.exit(1)

    if not _prompt("Proceed with the install plan above?", default=True):
        print(f"   {DIM}Cancelled. You can run the commands manually whenever you like.{RST}")
        return

    print()
    failures: list[str] = []
    failures.extend(unresolved)
    for dist_name, cmd in plan:
        print(f"   {YELLOW}📡 Installing {BOLD}{dist_name}{RST}{YELLOW} ...{RST}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"   {RED}❌ {dist_name} install failed (exit {result.returncode}){RST}")
            failures.append(dist_name)
        else:
            print(f"   {GREEN}✅ {dist_name} installed{RST}")
        print()

    libuipc_ok = _try_install_libuipc()
    if failures:
        print(f"{BOLD}{RED}❌ Init finished with errors: {', '.join(failures)}{RST}")
        print(
            f"   {DIM}Re-run{RST} {CYAN}geniesim status{RST} {DIM}to inspect, then{RST} {CYAN}geniesim bootstrap{RST} {DIM}again.{RST}"
        )
        sys.exit(1)

    _report_completion(libuipc_ok)
