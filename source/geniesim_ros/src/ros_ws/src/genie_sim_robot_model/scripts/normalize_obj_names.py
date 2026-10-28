#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Normalize OBJ internal object names to defeat 6.x converter de-duplication.

Why this exists
---------------
The Isaac Sim 6.x URDF→USD converter (``urdf_usd_converter._impl.mesh``)
authors each mesh prim under ``Geometries/<name>`` where ``<name>`` is the
.obj file's ``o <name>`` directive. When multiple .obj files share generic
default object names (3ds Max / Blender / Maya commonly emit
``Cylinder001``, ``Box001``, ``s``, ``2``, ``1``), the converter
de-duplicates by name across files: first occurrence wins, subsequent files
get a ``_N`` suffix or are silently dropped — and the per-mesh references
in ``instances.usda`` end up pointing at the wrong (or no) geometry. The
visible symptom is *missing or swapped link visuals* in the assembled robot
USD.

ARX acone is the canonical case: 14+ visual meshes share four distinct
internal names (``Cylinder001``, ``Cylinder002``, ``Box001``, ``s``), and
without normalization the assembled USD is missing ``arm_r_*`` and half the
gripper visuals.

UR5 doesn't trip this because its ``.dae`` meshes carry per-link names that
the DCC tool baked in. .obj/.fbx files often don't.

What this tool does
-------------------
For each ``.obj`` file under the given paths (or the package tree by default),
rewrite the file in place with all ``o`` and ``g`` directives stripped and a
single ``o <file_stem>`` directive prepended. The result is one unique
object name per file, derived deterministically from the filename → one
unique ``Geometries/<file_stem>`` prim in the converted USD, no
cross-file collisions.

Vertex (``v``), normal (``vn``), texture (``vt``), face (``f``), material
(``mtllib`` / ``usemtl``), comment, and smoothing-group (``s``) lines are
preserved unmodified. Only ``o``/``g`` directives are replaced.

The tool is idempotent: an already-normalized .obj is left untouched on a
second run. Running it on a clean tree reports "no changes needed".

Usage
-----
::

    # Diagnose only — list every .obj under the current directory tree, show
    # which ones have name collisions across files and which would be
    # rewritten. No file is modified.
    python3 normalize_obj_names.py --dry-run [PATH ...]

    # Auto-fix — rewrite every .obj in place. Re-run after each batch of
    # mesh updates from the DCC tool.
    python3 normalize_obj_names.py [PATH ...]

If no PATH is given, the tool scans the current working directory recursively.

Mesh-developer workflow
-----------------------
1. Export new .obj meshes from the DCC tool (3ds Max / Blender / Maya / …).
2. Drop them under ``robots/<vendor>/<robot>/meshes/``.
3. Run ``python3 scripts/normalize_obj_names.py --dry-run robots/<vendor>``
   to see what's about to change.
4. Run ``python3 scripts/normalize_obj_names.py robots/<vendor>`` to apply
   the rewrite.
5. Commit the normalized .obj files.
6. Delete any stale ``assets/scenes/<scene_stem>/`` cache so the next launch
   re-runs the urdf_to_usd assemble stage against the fixed meshes.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple


def _find_obj_files(paths: List[Path]) -> List[Path]:
    """Collect every ``.obj`` under the given paths (recursive for directories)."""
    if not paths:
        paths = [Path.cwd()]
    seen: set[Path] = set()
    results: List[Path] = []
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            print(f"WARNING: path does not exist: {path}", file=sys.stderr)
            continue
        if path.is_file() and path.suffix.lower() == ".obj":
            if path not in seen:
                seen.add(path)
                results.append(path)
        elif path.is_dir():
            for f in path.rglob("*.obj"):
                if f not in seen:
                    seen.add(f)
                    results.append(f)
    return sorted(results)


def _parse_object_directives(path: Path) -> Tuple[List[str], List[str]]:
    """Return ``(o_names, g_names)`` from a .obj file, in encounter order."""
    o_names: List[str] = []
    g_names: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                stripped = line.lstrip()
                if stripped.startswith("o "):
                    o_names.append(stripped[2:].strip())
                elif stripped.startswith("g "):
                    g_names.append(stripped[2:].strip())
    except OSError as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
    return o_names, g_names


def _is_already_normalized(stem: str, o_names: List[str], g_names: List[str]) -> bool:
    """Return True if the .obj already carries exactly one ``o <stem>`` and no ``g``."""
    if g_names:
        return False
    if len(o_names) != 1:
        return False
    return o_names[0] == stem


def _rewrite_obj_in_place(path: Path, target_stem: str) -> bool:
    """Strip every ``o``/``g`` directive and prepend ``o <target_stem>``."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        return False
    out: List[str] = [f"o {target_stem}\n"]
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("o ") or stripped.startswith("g "):
            continue
        out.append(line)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(out)
    except OSError as exc:
        print(f"ERROR: cannot write {path}: {exc}", file=sys.stderr)
        return False
    return True


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Normalize OBJ object names so the URDF→USD converter "
        "doesn't drop meshes via name-based de-duplication.",
        epilog="See module docstring for the why and the mesh-developer workflow.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Files or directories to scan. Default: current directory (recursive).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Diagnose collisions and list pending changes; do not modify any file.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    files = _find_obj_files(args.paths)
    if not files:
        print("No .obj files found.")
        return 0

    file_info: List[Tuple[Path, str, List[str], List[str]]] = []
    name_to_files: dict[str, List[Tuple[Path, str]]] = defaultdict(list)

    for path in files:
        o_names, g_names = _parse_object_directives(path)
        stem = path.stem
        file_info.append((path, stem, o_names, g_names))
        for name in o_names:
            name_to_files[name].append((path, stem))

    print(f"Scanned {len(files)} .obj file(s).")
    print()

    collisions = {n: owners for n, owners in name_to_files.items() if len(owners) > 1}
    if collisions:
        print(f"Cross-file name collisions ({len(collisions)} distinct internal name(s)):")
        for name, owners in sorted(collisions.items()):
            print(f"  '{name}' appears in {len(owners)} file(s):")
            for path, stem in owners:
                print(f"    {path}  (file stem: '{stem}')")
        print()
    else:
        print("No cross-file name collisions.")
        print()

    pending: List[Tuple[Path, str, List[str], List[str]]] = []
    already_ok: List[Path] = []
    for entry in file_info:
        path, stem, o_names, g_names = entry
        if _is_already_normalized(stem, o_names, g_names):
            already_ok.append(path)
        else:
            pending.append(entry)

    print(f"{len(already_ok)} file(s) already normalized.")
    print(f"{len(pending)} file(s) need normalization:")
    for path, stem, o_names, g_names in pending:
        o_view = ", ".join(o_names) if o_names else "<none>"
        g_view = ", ".join(g_names) if g_names else "<none>"
        print(f"  {path}")
        print(f"    o=[{o_view}]  g=[{g_view}]  ->  o=[{stem}]  (drop g)")

    if args.dry_run:
        print()
        print("--dry-run: no files modified.")
        return 1 if pending else 0

    if not pending:
        print()
        print("Nothing to do.")
        return 0

    print()
    print(f"Rewriting {len(pending)} file(s) in place...")
    failures = 0
    for path, stem, _, _ in pending:
        if _rewrite_obj_in_place(path, stem):
            print(f"  fixed: {path}")
        else:
            failures += 1
    print()
    if failures:
        print(f"{failures} file(s) failed to write.")
        return 2
    print(f"Done. {len(pending)} file(s) normalized.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
