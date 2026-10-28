# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Fail the build if our dependency layer installed a MuJoCo that Isaac Sim's
Newton does not accept.

This is the guard for ``docker/GENIESIM4_MIGRATION.md`` §4 R14. Every
Dockerfile installs the collected geniesim deps with ``pip install
--ignore-installed`` *after* the Isaac Sim wheel, and ``collect_deps.py``'s
``SKIP_EXACT`` does not list ``mujoco`` -- so ``mujoco~=3.12.0`` in
``source/geniesim_ros/pyproject.toml`` overwrites whatever MuJoCo Isaac's
Newton was built against. ``--ignore-installed`` means pip neither conflicts
nor warns, which is what made the failure silent.

Why a script and not a pip constraint
-------------------------------------
The ``mujoco`` line in ``pyproject.toml`` is a **third coupled pin**: it must
move whenever ``ISAACSIM_TAG`` / ``ISAACSIM_PIP_VERSION`` move, because Isaac
chooses which Newton ships and Newton chooses which MuJoCo it needs. Those
first two pins are now resolved from a series spec by
``geniesim_cli._isaacsim``, but the third cannot be: the ``isaacsim``
meta-wheel's ``Requires-Dist`` names only ``isaacsim-kernel`` plus
``isaacsim-*`` extras -- newton appears nowhere, and the subpackage that does
declare it is a multi-GB wheel, so there is nothing cheap to read on the host.

Inside the image there is. Isaac Sim 6.2's pip install puts ``newton`` on
disk as an ordinary distribution, so its own requirement is readable from
installed metadata::

    newton 1.6.1 -> mujoco~=3.12.0 ; extra == "sim"
                    mujoco-warp~=3.12.0 ; extra == "sim"

So the check runs where the information actually exists, and turns a silent
mismatch into a failed build.

Not a substitute for the pin: ``pyproject.toml`` still needs a correct
``mujoco`` line for non-docker installs. What this removes is the *silence*
when it drifts.

Exit codes: ``0`` pass (or nothing to check), ``1`` mismatch.
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

# Newton's own requirements we hold the image to. Both sit behind its ``sim``
# extra; we check them unconditionally because the simulation stack is the
# whole reason Newton is here.
_GUARDED = ("mujoco", "mujoco-warp")


def _installed(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> int:
    newton = _installed("newton")
    if newton is None:
        # Not every variant pip-installs Isaac Sim's Newton into the
        # interpreter this runs in (Isaac Sim 5.1 keeps it in ``extscache``,
        # invisible to importlib.metadata). Nothing to assert -- the check is
        # a no-op rather than a failure, so the same line is safe in every
        # Dockerfile.
        print("[check-newton-pins] newton is not pip-visible here; nothing to check")
        return 0

    declared = {}
    for raw in requires("newton") or []:
        requirement = Requirement(raw)
        canonical = requirement.name.lower().replace("_", "-")
        if canonical in _GUARDED:
            declared[canonical] = requirement

    if not declared:
        print(f"[check-newton-pins] newton {newton} declares no {'/'.join(_GUARDED)} requirement")
        return 0

    failures = []
    for name, requirement in sorted(declared.items()):
        have = _installed(name)
        if have is None:
            failures.append(f"{name} is not installed, but newton {newton} requires {requirement.specifier}")
            continue
        try:
            Version(have)
        except InvalidVersion:
            print(f"[check-newton-pins] {name} {have} is not a PEP 440 version; skipping")
            continue
        # ``prereleases=True``: Isaac ships release candidates often enough
        # (e.g. opencv ``...rc3`` in this very image) that excluding them
        # would fail builds that are actually consistent.
        if requirement.specifier.contains(have, prereleases=True):
            print(f"[check-newton-pins] ok: {name} {have} satisfies newton {newton}'s {requirement.specifier}")
        else:
            failures.append(f"{name} {have} does NOT satisfy newton {newton}'s requirement {requirement.specifier}")

    if not failures:
        return 0

    print()
    print("=" * 72, file=sys.stderr)
    print("[check-newton-pins] MuJoCo / Newton version mismatch (GENIESIM4_MIGRATION §4 R14)", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    print(file=sys.stderr)
    print("Isaac Sim owns these versions: it chooses which Newton ships, and Newton", file=sys.stderr)
    print("declares which MuJoCo it needs. Fix by copying Newton's specifier verbatim", file=sys.stderr)
    print("into `mujoco` in source/geniesim_ros/pyproject.toml, then rebuild.", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
