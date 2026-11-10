# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Preserve installed Isaac Sim and colcon dependency contracts during image builds."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def collect_constraints():
    """Read SDK requirements and dependency extras using this interpreter's markers."""
    sdk_names = {
        canonicalize_name(dist.metadata["Name"])
        for dist in metadata.distributions()
        if canonicalize_name(dist.metadata["Name"]).startswith("isaacsim")
    }
    if "isaacsim" not in sdk_names:
        raise RuntimeError("Isaac Sim must be installed before collecting runtime constraints")
    bundled = metadata.distribution("isaacsim").read_text("geniesim-bundled.json")
    bundled_distributions = {}
    if bundled:
        for path in json.loads(bundled)["dependency_paths"]:
            for dist in metadata.distributions(path=[path]):
                bundled_distributions.setdefault(canonicalize_name(dist.metadata["Name"]), dist)
    metadata.distribution("colcon-core")
    selected = {name: set() for name in sdk_names | {"colcon-core"}}
    selected["isaacsim"] = {"all", "extscache"}
    pending = list(sorted(selected))
    constraints = set()
    while pending:
        name = pending.pop()
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            if name not in bundled_distributions:
                raise
            dist = bundled_distributions[name]
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not any(
                requirement.marker.evaluate({"extra": extra}) for extra in {""} | selected[name]
            ):
                continue
            if requirement.url:
                raise RuntimeError(f"Cannot constrain direct URL requirement from {name}: {raw}")
            dependency = canonicalize_name(requirement.name)
            constraints.add(f"{dependency}{requirement.specifier}")
            if requirement.extras:
                extras = selected.setdefault(dependency, set())
                if not requirement.extras <= extras:
                    extras.update(requirement.extras)
                    pending.append(dependency)
    return sorted(constraints)


def repair_requirements(constraints):
    """Select only missing or incompatible packages, preserving all their constraints."""
    grouped = {}
    for raw in constraints:
        requirement = Requirement(raw)
        grouped.setdefault(requirement.name, []).append(requirement)
    repairs = []
    for name, requirements in sorted(grouped.items()):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        if version is None or any(not req.specifier.contains(version, prereleases=True) for req in requirements):
            specifiers = sorted({str(req.specifier) for req in requirements if req.specifier})
            repairs.append(name + ",".join(specifiers))
    return repairs


def main():
    """Write pip constraints plus the packages requiring explicit repair."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--constraints", required=True, type=Path)
    parser.add_argument("--repairs", required=True, type=Path)
    args = parser.parse_args()
    constraints = collect_constraints()
    repairs = repair_requirements(constraints)
    args.constraints.write_text("\n".join(constraints) + "\n")
    args.repairs.write_text("\n".join(repairs) + ("\n" if repairs else ""))
    print(f"[runtime-constraints] {len(constraints)} SDK/colcon constraints; {len(repairs)} repairs: {repairs}")


if __name__ == "__main__":
    main()
