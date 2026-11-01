# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``geniesim doctor`` — diagnose and (with consent) repair the local install.

Modeled after ``brew doctor``: every check returns a structured result
(``ok`` / ``warn`` / ``fail``) plus an optional ``fix`` callable. The
runner prints a per-check report, then offers to apply each available
fix interactively.

The MVP ships two checks:

* **``stack``**     — re-runs the same inspection that powers
  ``geniesim status`` and flags any distribution that's missing or
  fails to import. The fix is to run ``geniesim bootstrap``.
* **``rosdep``**    — runs ``rosdep check --from-paths <ws>/src
  --ignore-src --rosdistro=$ROS_DISTRO`` against the colcon workspace
  resolved by :func:`geniesim_cli._workspace.ros_workspace_root`. When
  ``rosdep`` reports missing system dependencies, the fix invokes
  ``sudo rosdep install --from-paths ... -y``.

Adding new checks is intentionally one-liner: append a new dict to
:data:`_CHECKS`. Each check is independent — a failure in one does
not skip the others.
"""

from __future__ import annotations

import sys
from typing import Callable, Optional

from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

# ---------------------------------------------------------------------------
# Check result protocol
# ---------------------------------------------------------------------------

# A check function returns a tuple:
#     (status, summary_lines, fix_label, fix_callable)
# where:
#   status:        "ok" | "warn" | "fail"
#   summary_lines: list[str]   already coloured / formatted
#   fix_label:     str | None  short description (None ⇒ no fix offered)
#   fix_callable:  Callable[[], int] | None
#                              returns 0 on success, non-zero on failure
CheckResult = tuple[str, list[str], Optional[str], Optional[Callable[[], int]]]


# ---------------------------------------------------------------------------
# Check: stack health (re-uses commands.status engine)
# ---------------------------------------------------------------------------


def _check_stack() -> CheckResult:
    """Re-run ``geniesim status``-style inspection; flag missing/broken peers."""
    from geniesim_cli.commands.status import DISTRIBUTIONS, health_issues, inspect

    rows = [(spec, inspect(spec)) for spec in DISTRIBUTIONS]

    missing: list[str] = []
    broken: list[str] = []
    for spec, info in rows:
        issues = health_issues(spec, info)
        if not issues:
            continue
        if not info["installed"]:
            missing.append(spec["dist"])
        else:
            broken.append(f"{spec['dist']} ({', '.join(issues)})")

    if not missing and not broken:
        lines = [f"   {GREEN}✅ Required distributions and installed optional peers are healthy.{RST}"]
        for spec, info in rows:
            if not info["installed"]:
                lines.append(f"      {DIM}⏭️  {spec['dist']} (optional, not installed){RST}")
                continue
            mode = "editable" if info["editable"] else "wheel" if info["editable"] is False else "?"
            lines.append(f"      {WHITE}{spec['dist']}{RST} {GREEN}{info['version']}{RST} {DIM}({mode}){RST}")
        return "ok", lines, None, None

    lines: list[str] = []
    if missing:
        lines.append(f"   {RED}❌ Missing distributions:{RST} {WHITE}{', '.join(missing)}{RST}")
    if broken:
        lines.append(f"   {RED}❌ Unhealthy distributions:{RST} {WHITE}{', '.join(broken)}{RST}")
    lines.append(f"   {DIM}Run{RST} {CYAN}geniesim status{RST} {DIM}for the full report.{RST}")

    def _fix() -> int:
        import subprocess

        return subprocess.run([sys.executable, "-m", "geniesim_cli", "bootstrap"]).returncode

    return "fail", lines, "run geniesim bootstrap", _fix


# ---------------------------------------------------------------------------
# Check: rosdep
# ---------------------------------------------------------------------------


def _ros_distro_or_none() -> str | None:
    """Return ``$ROS_DISTRO`` or ``None`` if ROS 2 is not sourced."""
    import os

    return os.environ.get("ROS_DISTRO") or None


def _check_rosdep() -> CheckResult:
    """``rosdep check`` over the colcon workspace; offer ``rosdep install`` on miss.

    Resolution path mirrors :mod:`geniesim_cli.commands.ros`: the
    workspace is whatever
    :func:`geniesim_cli._workspace.ros_workspace_root` returns. We do
    *not* fail the doctor outright when the workspace cannot be found
    — it's a warning, because a Python-only user may legitimately not
    have ``geniesim_ros`` installed.
    """
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("rosdep") is None:
        return (
            "warn",
            [
                f"   {YELLOW}⚠️  rosdep not on PATH.{RST}",
                f"   {DIM}Install with:{RST} {CYAN}sudo apt install python3-rosdep{RST}{DIM};{RST} "
                f"{CYAN}sudo rosdep init && rosdep update{RST}",
            ],
            None,
            None,
        )

    ros_distro = _ros_distro_or_none()
    if ros_distro is None:
        return (
            "warn",
            [
                f"   {YELLOW}⚠️  ROS 2 is not sourced ($ROS_DISTRO unset).{RST}",
                f"   {DIM}Source it first:{RST} {CYAN}source /opt/ros/<distro>/setup.bash{RST}",
            ],
            None,
            None,
        )

    try:
        from geniesim_cli._workspace import ros_workspace_root

        ws = Path(ros_workspace_root())
    except SystemExit:
        return (
            "warn",
            [
                f"   {YELLOW}⚠️  No colcon workspace found; skipping rosdep check.{RST}",
                f"   {DIM}Install{RST} {BOLD}geniesim_ros{RST} {DIM}or set{RST} "
                f"{CYAN}GENIESIM_WORKSPACE{RST}{DIM}.{RST}",
            ],
            None,
            None,
        )

    src = ws / "src"

    cmd = [
        "rosdep",
        "check",
        "--from-paths",
        str(src),
        "--ignore-src",
        f"--rosdistro={ros_distro}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0 and "All system dependencies have been satisfied" in out:
        return (
            "ok",
            [
                f"   {GREEN}✅ rosdep: all system dependencies satisfied.{RST}",
                f"      {DIM}workspace:{RST} {WHITE}{ws}{RST}",
                f"      {DIM}rosdistro:{RST} {WHITE}{ros_distro}{RST}",
            ],
            None,
            None,
        )

    missing_keys: list[str] = []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("apt\t") or s.startswith("apt "):
            parts = s.split(None, 1)
            if len(parts) == 2:
                missing_keys.append(parts[1])
        elif s.startswith("#") and "rosdep key" in s:
            pass

    lines = [
        f"   {RED}❌ rosdep reports missing dependencies.{RST}",
        f"      {DIM}workspace:{RST} {WHITE}{ws}{RST}",
        f"      {DIM}rosdistro:{RST} {WHITE}{ros_distro}{RST}",
    ]
    if missing_keys:
        lines.append(f"   {DIM}Missing keys:{RST}")
        for key in missing_keys[:10]:
            lines.append(f"      {WHITE}{key}{RST}")
        if len(missing_keys) > 10:
            lines.append(f"      {DIM}... and {len(missing_keys) - 10} more{RST}")
    else:
        snippet = out.strip().splitlines()
        for ln in snippet[:6]:
            lines.append(f"      {DIM}{ln}{RST}")
        if len(snippet) > 6:
            lines.append(f"      {DIM}... ({len(snippet) - 6} more lines){RST}")

    install_cmd = [
        "sudo",
        "rosdep",
        "install",
        "--from-paths",
        str(src),
        "--ignore-src",
        f"--rosdistro={ros_distro}",
        "-y",
    ]
    lines.append(f"   {DIM}Fix command:{RST} {CYAN}{' '.join(install_cmd)}{RST}")

    def _fix() -> int:
        return subprocess.run(install_cmd).returncode

    return "fail", lines, "run rosdep install -y", _fix


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Each entry: (label, emoji, check_callable). Order is the order printed.
# Add new checks here; no other code needs to change.
_CHECKS: tuple[tuple[str, str, Callable[[], CheckResult]], ...] = (
    ("Stack health", "📦", _check_stack),
    ("ROS rosdep", "🤖", _check_rosdep),
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _prompt_yes(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        answer = input(f"   {BOLD}{question} {suffix}: {RST}").strip().lower()
    except EOFError:
        print()
        return default
    if not answer:
        return default
    return answer in ("y", "yes")


def run(args: list[str]) -> None:
    """Run every registered check, print a report, offer fixes interactively."""
    print(f"{BOLD}{MAGENTA}🩺 geniesim doctor{RST} {DIM}— diagnose your geniesim install{RST}")
    print()

    results: list[tuple[str, str, str, list[str], Optional[str], Optional[Callable[[], int]]]] = []
    for idx, (label, emoji, check) in enumerate(_CHECKS):
        print(f"{BOLD}[{idx + 1}/{len(_CHECKS)}] {emoji} {label}{RST}")
        try:
            status, lines, fix_label, fix_callable = check()
        except Exception as exc:
            status = "fail"
            lines = [f"   {RED}❌ Check raised {type(exc).__name__}: {exc}{RST}"]
            fix_label = None
            fix_callable = None

        for ln in lines:
            print(ln)
        print()
        results.append((label, emoji, status, lines, fix_label, fix_callable))

    n_ok = sum(1 for _, _, s, *_ in results if s == "ok")
    n_warn = sum(1 for _, _, s, *_ in results if s == "warn")
    n_fail = sum(1 for _, _, s, *_ in results if s == "fail")

    print(f"{BOLD}Summary{RST}")
    print(f"   {GREEN}✅ ok: {n_ok}{RST}   " f"{YELLOW}⚠️  warn: {n_warn}{RST}   " f"{RED}❌ fail: {n_fail}{RST}")
    print()

    fixable = [
        (label, emoji, fix_label, fix_callable)
        for label, emoji, status, _, fix_label, fix_callable in results
        if fix_callable is not None
    ]

    if not fixable:
        if n_fail == 0 and n_warn == 0:
            print(f"{BOLD}{GREEN}🎉 Your system is ready to brew.{RST}")
        else:
            print(f"{BOLD}{YELLOW}⚠️  No automatic fixes available; resolve the warnings above manually.{RST}")
        sys.exit(0 if n_fail == 0 else 1)

    print(f"{BOLD}🔧 Available fixes{RST}")
    for label, emoji, fix_label, _ in fixable:
        print(f"   {emoji} {WHITE}{label}{RST} — {CYAN}{fix_label}{RST}")
    print()

    if not _prompt_yes("Apply the fixes above?", default=True):
        print(f"   {DIM}Cancelled. Re-run{RST} {CYAN}geniesim doctor{RST} {DIM}whenever you're ready.{RST}")
        sys.exit(1)

    print()
    failures: list[str] = []
    for label, emoji, fix_label, fix_callable in fixable:
        print(f"{YELLOW}🔧 Fixing {emoji} {label} — {fix_label}{RST}")
        rc = fix_callable() if fix_callable else 1
        if rc != 0:
            print(f"   {RED}❌ Fix for {label} exited with code {rc}{RST}")
            failures.append(label)
        else:
            print(f"   {GREEN}✅ {label} fixed{RST}")
        print()

    if failures:
        print(f"{BOLD}{RED}❌ Some fixes failed: {', '.join(failures)}{RST}")
        print(f"   {DIM}Re-run{RST} {CYAN}geniesim doctor{RST} {DIM}after resolving the errors above.{RST}")
        sys.exit(1)

    print(f"{BOLD}{GREEN}🎉 All fixes applied. Re-run{RST} {CYAN}geniesim doctor{RST} {GREEN}to verify.{RST}")
