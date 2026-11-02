# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim version`` — show versions; ``--bump`` / ``--check`` to manage them.

The repo-root ``VERSION`` file is the single source of truth. Each first-class
distribution has a local ``VERSION`` symlink used by its build metadata. ROS
``package.xml`` files are kept in sync by this command's ``--bump`` /
``--sync`` flow, since colcon / ament read ``<version>`` directly from
the file (no dynamic mechanism on that side).

Usage:

  geniesim version               # show installed distribution versions
  geniesim version --bump X.Y.Z  # write VERSION + sync release metadata
  geniesim version --sync        # sync package links and ROS versions
  geniesim version --check       # exit 1 on release-version drift
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

_FIRST_CLASS_PACKAGES: tuple[str, ...] = (
    "geniesim_cli",
    "geniesim",
    "geniesim_benchmark",
    "geniesim_ros",
    "geniesim_teleop",
    "geniesim_scene_agent",
    "geniesim_dream",
    "geniesim_world",
)
_DISTRIBUTIONS = (*_FIRST_CLASS_PACKAGES, "geniesim_assets")

# package.xml files that should track the unified VERSION. Paths are
# repo-relative; independent modules and third-party packages are excluded.
_ROS_PACKAGE_XMLS: tuple[str, ...] = (
    "source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_client/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_common/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_engine_core/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_engine_isaacsim/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_engine_libuipc/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_engine_newton/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_moveit/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_moveit_plugins/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_planning/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_render/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_robot_model/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_ros_control/genie_sim_control/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_ros_control/genie_sim_controllers/package.xml",
    "source/geniesim_ros/src/ros_ws/src/genie_sim_rviz_plugins/package.xml",
)

# Permissive PEP 440 / ROS-style version literal. Allows e.g.
# "3.2.0", "3.2.0rc1", "3.2.0.post3+g1a2b3c4d".
_VERSION_RE = re.compile(r"^[0-9][0-9A-Za-z.+\-]*$")
_TAG_RE = re.compile(r"(<version>)([^<]*)(</version>)")


def _find_repo_root() -> Path:
    """Walk up looking for the repo-root ``VERSION`` file.

    Honours ``$GENIESIM_REPO_ROOT`` first (matches the docker subcommand).
    """
    override = os.environ.get("GENIESIM_REPO_ROOT")
    if override:
        return Path(override).resolve()

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "VERSION").is_file():
            return candidate

    # Editable-install fallback: the file lives at
    # <repo>/source/geniesim_cli/src/geniesim_cli/commands/version.py.
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "VERSION").is_file():
            return ancestor

    print(
        f"{RED}❌ Could not locate the repo root (no VERSION file found).{RST}\n"
        f"   {DIM}Set {BOLD}$GENIESIM_REPO_ROOT{RST}{DIM} or run from inside the repo.{RST}"
    )
    sys.exit(1)


def _read_version(repo: Path) -> str:
    text = (repo / "VERSION").read_text().strip()
    if not _VERSION_RE.match(text):
        print(f"{RED}❌ {repo / 'VERSION'} contains invalid version: {text!r}{RST}")
        sys.exit(1)
    return text


def _write_version(repo: Path, new: str) -> None:
    (repo / "VERSION").write_text(new + "\n")


def _show_installed() -> None:
    from importlib.metadata import PackageNotFoundError, version as pkg_version

    print(f"{BOLD}{MAGENTA}🧞 geniesim{RST}")
    for dist in _DISTRIBUTIONS:
        try:
            ver = pkg_version(dist)
            print(f"   {WHITE}{dist}{RST}        {GREEN}{ver}{RST}")
        except PackageNotFoundError:
            print(f"   {WHITE}{dist}{RST}        {DIM}(not installed){RST}")
    print(f"   {DIM}Python{RST}              {sys.version}")


def _scan_package_xmls(repo: Path, expected: str) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Return (in-sync, out-of-sync) splits.

    Out-of-sync entries pair the path with the version found in the file,
    so callers can produce a useful diff message.
    """
    in_sync: list[Path] = []
    out_of_sync: list[tuple[Path, str]] = []
    for rel in _ROS_PACKAGE_XMLS:
        path = repo / rel
        if not path.is_file():
            continue
        match = _TAG_RE.search(path.read_text())
        if match is None:
            print(f"{YELLOW}⚠️  {path}: no <version> tag, skipping{RST}")
            continue
        current = match.group(2).strip()
        if current == expected:
            in_sync.append(path)
        else:
            out_of_sync.append((path, current))
    return in_sync, out_of_sync


def _rewrite_package_xml(path: Path, new: str) -> None:
    """Replace exactly the first ``<version>…</version>`` occurrence."""
    text = path.read_text()

    def _sub(m: re.Match[str]) -> str:
        return f"{m.group(1)}{new}{m.group(3)}"

    new_text, n = _TAG_RE.subn(_sub, text, count=1)
    if n != 1:
        # Caller already filtered to files with a tag; fail loud rather
        # than silently corrupting.
        raise RuntimeError(f"failed to rewrite <version> in {path}")
    if new_text != text:
        path.write_text(new_text)


def _scan_package_versions(repo: Path, expected: str) -> list[tuple[Path, str]]:
    """Check release links only for first-class packages present in a checkout."""
    out_of_sync = []
    for name in _FIRST_CLASS_PACKAGES:
        package = repo / "source" / name
        if not (package / "pyproject.toml").is_file():
            continue
        path = package / "VERSION"
        current = path.read_text().strip() if path.is_file() else "(missing)"
        if not path.is_symlink() or path.resolve() != (repo / "VERSION").resolve():
            out_of_sync.append((path, f"{current}; not linked to root VERSION"))
        elif current != expected:
            out_of_sync.append((path, current))
    return out_of_sync


def _do_sync(repo: Path, expected: str, *, dry_run: bool) -> int:
    """Link first-class package versions and synchronize owned ROS manifests.

    Returns the number of files changed. ``dry_run=True`` only reports.
    """
    _, out_of_sync = _scan_package_xmls(repo, expected)
    out_of_sync = _scan_package_versions(repo, expected) + out_of_sync
    if not out_of_sync:
        print(f"{GREEN}✅ Release versions already in sync at {expected}.{RST}")
        return 0

    label = "would update" if dry_run else "updated"
    for path, current in out_of_sync:
        rel = path.relative_to(repo)
        print(f"   {CYAN}{rel}{RST}  {DIM}{current}{RST} → {GREEN}{expected}{RST}")
        if not dry_run:
            if path.name == "VERSION":
                path.unlink(missing_ok=True)
                path.symlink_to(os.path.relpath(repo / "VERSION", path.parent))
            else:
                _rewrite_package_xml(path, expected)
    print(f"{GREEN}✅ {label} {len(out_of_sync)} release metadata file(s).{RST}")
    return len(out_of_sync)


def _do_check(repo: Path, expected: str) -> None:
    """Pre-commit-friendly check: exit 1 if any drift exists."""
    _, out_of_sync = _scan_package_xmls(repo, expected)
    out_of_sync = _scan_package_versions(repo, expected) + out_of_sync
    if not out_of_sync:
        print(f"{GREEN}✅ Versions in sync at {expected}.{RST}")
        sys.exit(0)
    print(f"{RED}❌ Out-of-sync release metadata (expected {expected}):{RST}")
    for path, current in out_of_sync:
        rel = path.relative_to(repo)
        print(f"   {CYAN}{rel}{RST}  {RED}{current}{RST}")
    print(f"   {DIM}Run{RST} {CYAN}geniesim version --sync{RST} {DIM}to fix.{RST}")
    sys.exit(1)


def _do_bump(repo: Path, new: str) -> None:
    if not _VERSION_RE.match(new):
        print(f"{RED}❌ Invalid version: {new!r}{RST}")
        sys.exit(1)
    current = _read_version(repo)
    if new == current:
        print(f"{DIM}VERSION already at {current}; checking release metadata.{RST}")
        _do_sync(repo, new, dry_run=False)
        return
    print(f"{BOLD}{MAGENTA}🔖 geniesim version --bump{RST}")
    print(f"   {DIM}VERSION:{RST} {DIM}{current}{RST} → {GREEN}{new}{RST}")
    _write_version(repo, new)
    _do_sync(repo, new, dry_run=False)
    print(f"   {DIM}First-class package builds read their VERSION link; dependency versions are unchanged.{RST}")


def _print_usage() -> None:
    print(f"{BOLD}{MAGENTA}🧞 geniesim version{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim version [{CYAN}--bump VERSION{RST} | {CYAN}--sync{RST} | {CYAN}--check{RST}]")
    print()
    print(f"{BOLD}Modes:{RST}")
    print(f"  {DIM}(no flags){RST}              📋 Show installed dist versions")
    print(
        f"  {CYAN}--bump VERSION{RST}         🔖 Write {WHITE}VERSION{RST} and sync package links and owned ROS versions"
    )
    print(f"  {CYAN}--sync{RST}                 🔁 Sync package links and owned ROS versions (no bump)")
    print(f"  {CYAN}--check{RST}                ✅ Exit 1 if release metadata drifts from {WHITE}VERSION{RST}")


def run(args: list[str]) -> None:
    if not args:
        _show_installed()
        sys.exit(0)

    if args[0] in ("-h", "--help", "help"):
        _print_usage()
        sys.exit(0)

    if args[0] == "--bump":
        if len(args) != 2:
            print(f"{RED}❌ Usage: geniesim version --bump <NEW_VERSION>{RST}")
            sys.exit(1)
        _do_bump(_find_repo_root(), args[1])
        sys.exit(0)

    if args[0] == "--sync":
        repo = _find_repo_root()
        _do_sync(repo, _read_version(repo), dry_run=False)
        sys.exit(0)

    if args[0] == "--check":
        repo = _find_repo_root()
        _do_check(repo, _read_version(repo))

    print(f"{RED}❌ Unknown flag: {args[0]}{RST}")
    print()
    _print_usage()
    sys.exit(1)
