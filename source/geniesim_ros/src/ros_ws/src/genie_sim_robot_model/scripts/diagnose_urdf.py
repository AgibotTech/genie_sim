#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Diagnose URDF/xacro authoring problems without modifying input files.

What it checks
--------------
For each ``<link>`` in the input xacro/URDF file(s):

* **Missing ``<inertial>``** (detect-only) — flags non-root links
  without inertial or collision, except frame partners with an ``_inertia``
  sibling. On Isaac Sim 6.2 an empty frame can remain kinematic; it must
  not receive placeholder mass. A link intended to be dynamic needs real
  mass and inertia. The author must distinguish these cases.

* **URDF ghost root** (detect-only) — when the root link has no
  ``<visual>`` / ``<collision>`` / ``<inertial>``, the Isaac Sim 6.x
  URDF→USD converter takes its ``is_ghost_link`` branch and authors
  ``body0 = default_prim`` for every joint, including fixed joints.
  PhysX cannot construct an articulation from such USD. The tool can't
  auto-fix this — KDL forbids inertia on the root, so the fix must
  restructure the URDF (add visual/collision to the root, or start the
  kinematic chain at a non-ghost link).

* **Wheel without ``<cylinder>`` collision** (detect-only) — wheel-like
  links (name contains ``wheel`` or ``tire``, case-insensitive) benefit
  from a primitive ``<cylinder>`` collider rather than a triangle-mesh
  collider. Cylinder collision against a ground plane is ~10× faster
  than mesh-vs-mesh and numerically much more stable for sustained
  contact (the dominant workload on a mobile platform). The 6.x
  default-collision policy converts authored mesh colliders to convex
  hull, which is a workable but inferior alternative. The
  diagnose tool flags wheel-like links missing a cylinder collider so
  the operator can author one explicitly. This check is **detect-only**:
  the tool can't infer wheel radius / length from URDF text alone, so
  the fix must be authored by hand.

* **Mesh exported in non-metre units** (detect-only) — URDF / SDF /
  Newton / mjwarp / PhysX / RViz all assume mesh vertex data is in
  METRES.  COLLADA (``.dae``) lets the exporter declare a different
  unit via ``<unit meter="X"/>``, but no robotics loader honours
  the declaration — they read raw vertex values and treat them as
  metres.  When a CAD export with ``meter="0.001"`` (mm) leaks
  through, the loaded mesh is 1000× too large: inertia from
  uniform density is 10⁹× off, collision dispatch sees a giant
  hovering ghost-mesh, RViz shows the link blown up.

  This rule loads each ``<mesh filename>`` reachable from the
  URDF / xacro through ``trimesh`` and flags any whose vertex
  bounds extend past 5 m from origin — well past the spec of any
  human-scale robot link.  The reported file is the one to fix.
  Auto-fix is **not** in this tool; use the dedicated
  ``scripts/fix_dae_units.py`` (it rewrites the ``.dae`` vertices
  in place, with backup, structure-verified) — kept separate
  because the fix is mesh-data surgery, not text-substitution
  in the URDF.

* **Non-diagonal ``<inertia>`` tensor** (detect-only) — URDF inertia
  authored as a full 3×3 tensor with non-zero ``ixy`` / ``ixz`` / ``iyz``
  represents the same physical inertia as a diagonal tensor rotated by
  an ``<origin rpy>``, but the diagonal-plus-rpy form is:

    - what USD's ``PhysicsMassAPI`` actually stores
      (``diagonalInertia`` + ``principalAxes``) — the URDF→USD
      converter diagonalises either way, so the diagonal form removes
      one transformation;
    - what UR5's reference xacros do (``physical_parameters.yaml``
      lists ``ixy: 0, ixz: 0, iyz: 0`` for every link with the
      principal-axis rotation in ``inertia['rotation']``);
    - much easier to read — readers don't have to mentally
      diagonalise to see whether the tensor is well-conditioned;
    - what avoids the recurring "the inertia gizmo is rotated 45°
      from the visible mesh" GitHub issue (the
      ``recompute_inertia.py`` docstring's swiftpicker case).

  The rule flags any ``<inertia>`` whose
  ``max(|ixy|,|ixz|,|iyz|) / max(|ixx|,|iyy|,|izz|) > 1e-3`` — i.e.
  off-diagonals that aren't numerical noise.  Auto-fix is **not** in
  this tool; the fix needs the mesh (to recompute under uniform
  density) or a manual diagonalisation, both of which live in
  ``scripts/recompute_inertia.py`` and the per-package mesh-based
  recompute scripts.

* **Near-degenerate diagonal inertia with rpy=0** (detect-only) —
  when ``<inertia>`` is diagonal (``ixy=ixz=iyz=0``), ``<origin
  rpy="0 0 0">``, AND two of (``ixx``, ``iyy``, ``izz``) are within
  5% relative of each other, the link can trip an Isaac Sim ↔ Newton
  USD↔MJCF convention mismatch.  Isaac sorts ``diagonalInertia``
  ascending and emits a ``principalAxes`` quaternion; for certain
  sort permutations this is the 3-cycle quat ``(0.5, 0.5, 0.5,
  0.5)``.  Newton's ``SolverMuJoCo.save_to_mjcf`` then reads USD
  using the convention ``I_body = R · diag · Rᵀ`` while USD spec
  requires ``I_body = Rᵀ · diag · R`` — the two only differ for
  3-cycle R, which is exactly what Isaac wrote.  Result: MuJoCo
  shows the link's inertia tensor permuted by a cyclic shift (90°
  rotated ellipsoid) relative to RViz / URDF.  See
  ``newton_quirks.md`` for the full trace.

  The fix is **NOT** to snap the pair to exact equality — when
  ``ixx ≡ iyy`` Isaac still picks the cyclic permutation in the
  degenerate subspace, and the bug persists.  The actual workaround
  is two-step: (a) spread the near-degenerate pair to at least 6%
  relative gap, and (b) if the resulting sorted-ascending order is
  a 3-cycle of link axes (eg. ``izz<ixx<iyy``), swap two of D's
  entries so the order becomes a transposition.  Transpositions
  are invariant under the convention mismatch, so Newton's bug
  becomes invisible.  ``scripts/recompute_g2_inertia.py``
  implements both steps; this rule's purpose is to catch
  hand-authored URDFs that would land in the buggy regime.

What it does
------------
Both modes report findings without modifying files. ``--dry-run`` exits 1
for any finding and can be used as a strict CI gate. The default mode
exits 1 for root, wheel, tensor, or mesh-unit findings; missing inertials
alone are advisory because empty kinematic frames may be intentional.

Usage
-----
::

    python3 diagnose_urdf.py --dry-run [PATH ...]
    python3 diagnose_urdf.py [PATH ...]

PATH may name files or directories. Default: current directory recursively.
Pass ``--mesh-root`` to resolve ``${mesh_dir}`` references in xacro inputs.
Correct findings in the source xacro, regenerate URDFs, and validate the
composed model before converting robot assets.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

# File extensions accepted by the diagnostic scanner.
_TARGET_SUFFIXES = (".xacro", ".urdf")

# Wheel-link name pattern. Substring match for ``wheel`` / ``tire``,
# case-insensitive. No word boundaries because URDF link names use
# underscores universally (``chassis_rwheel_link``, ``left_wheel_link``)
# and Python's ``\b`` doesn't fire between ``_`` and ``\w``. Castors are
# deliberately excluded — castor mounts are typically box-shaped, and
# castor wheels often live under generic names like ``link1`` that we
# can't safely flag without false positives.
_WHEEL_RE = re.compile(r"(?i)wheel|tire")


def _find_target_files(paths: List[Path]) -> List[Path]:
    """Collect every ``.xacro``/``.urdf`` file under the given paths."""
    if not paths:
        paths = [Path.cwd()]
    seen: set[Path] = set()
    results: List[Path] = []
    for p in paths:
        path = Path(p).resolve()
        if not path.exists():
            print(f"WARNING: path does not exist: {path}", file=sys.stderr)
            continue
        if path.is_file():
            if any(path.name.endswith(s) for s in _TARGET_SUFFIXES):
                if path not in seen:
                    seen.add(path)
                    results.append(path)
            else:
                print(
                    f"WARNING: skipping {path} (not a .xacro/.urdf file)",
                    file=sys.stderr,
                )
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and any(f.name.endswith(s) for s in _TARGET_SUFFIXES):
                    if f not in seen:
                        seen.add(f)
                        results.append(f)
    return sorted(results)


def _links_missing_inertial(text: str) -> Tuple[List[str], List[str]]:
    """Return missing-inertial candidates and empty-root warnings.

    Non-root links without inertial or collision are candidates for author
    review; a frame partner with a named ``_inertia`` sibling is excluded.
    On Isaac Sim 6.2, an empty link can remain a kinematic frame, so a
    candidate is not automatically a defect and no mass is injected.

    Root links stay massless for KDL. A root with no visual or collision is
    reported separately for review of the converter's ghost-root branch.
    """
    body = re.sub(r"^\s*<\?xml[^?]*\?>\s*", "", text, count=1)
    wrapped = '<__diagnose_root xmlns:xacro="http://ros.org/wiki/xacro">\n' f"{body}\n" "</__diagnose_root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return [], []

    # Map: child link name → True (it's a joint child, hence has a parent)
    child_link_names: set[str] = set()
    for joint in root.iter("joint"):
        child = joint.find("child")
        if child is not None:
            link_name = child.attrib.get("link")
            if link_name:
                child_link_names.add(link_name)

    # All declared link names (for the _inertia sibling lookup)
    all_link_names: set[str] = {link.attrib["name"] for link in root.iter("link") if "name" in link.attrib}

    needs_injection: List[str] = []
    ghost_root_warnings: List[str] = []
    for link in root.iter("link"):
        name = link.attrib.get("name")
        if not name:
            continue
        if link.find("inertial") is not None:
            continue

        has_visual = link.find("visual") is not None
        has_collision = link.find("collision") is not None
        is_urdf_root = name not in child_link_names
        has_inertia_sibling = (name + "_inertia") in all_link_names

        if is_urdf_root:
            # KDL requires the root to be massless. We don't inject.
            # But warn if it's also ghost — the 6.x converter breaks on this.
            if not has_visual and not has_collision:
                ghost_root_warnings.append(name)
            continue

        if has_inertia_sibling:
            # UR / vendor convention — the *_inertia sibling carries
            # the mass. Skip silently.
            continue

        if has_collision:
            # PhysX auto-computes mass from the collision shape's volume ×
            # density, so a missing <inertial> is not a problem here.
            continue

        # An empty frame may be intentional. Report it for author review
        # without creating a near-massless rigid body via placeholder inertia.
        needs_injection.append(name)

    return needs_injection, ghost_root_warnings


def _wheels_without_cylinder_collision(text: str) -> List[str]:
    """Return wheel-like links that lack a ``<cylinder>`` collision shape.

    Detect-only — there is no auto-fix. Wheel radius and length depend on
    the visual mesh, which the URDF text doesn't expose to a static
    analyzer.

    A link is "wheel-like" iff its ``name`` matches the wheel regex AND it
    is the child of a ``continuous`` joint (spinning wheel rim). Links that
    match the name pattern but are children of ``revolute`` joints are
    steering brackets/pivots — they use non-cylinder collision and are
    excluded from this check.

    A wheel passes if it has at least one ``<collision>`` whose direct
    ``<geometry>`` child is ``<cylinder>``. Wheels with mesh collisions
    (or no collision at all) are flagged.
    """
    body = re.sub(r"^\s*<\?xml[^?]*\?>\s*", "", text, count=1)
    wrapped = '<__diagnose_root xmlns:xacro="http://ros.org/wiki/xacro">\n' f"{body}\n" "</__diagnose_root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []

    # Only flag links that spin (child of continuous joint) — not steering pivots.
    spinning_links: set[str] = set()
    for joint in root.iter("joint"):
        if joint.attrib.get("type") == "continuous":
            child = joint.find("child")
            if child is not None:
                link_name = child.attrib.get("link")
                if link_name:
                    spinning_links.add(link_name)

    out: List[str] = []
    for link in root.iter("link"):
        name = link.attrib.get("name")
        if not name or not _WHEEL_RE.search(name):
            continue
        if name not in spinning_links:
            continue
        has_cyl = False
        for coll in link.findall("collision"):
            geom = coll.find("geometry")
            if geom is not None and geom.find("cylinder") is not None:
                has_cyl = True
                break
        if not has_cyl:
            out.append(name)
    return out


# Off-diagonals smaller than this fraction of the max diagonal are
# treated as numerical noise (e.g. an authored 1e-12 left from a CAD
# export rounding to zero on the diag).  Above this they're real
# rotated-frame components and the link should be re-authored as
# diagonal-plus-rpy.  Matches the threshold the per-package mesh-
# recompute scripts use to decide whether to emit ``rpy="0 0 0"``.
_INERTIA_OFF_DIAG_TOL = 1e-3


def _links_with_off_diagonal_inertia(text: str) -> List[Tuple[str, float]]:
    """Return ``[(link_name, off_diag_ratio), ...]`` for every link whose
    ``<inertia>`` has non-trivial off-diagonal terms.

    ``off_diag_ratio`` = ``max(|ixy|,|ixz|,|iyz|) / max(|ixx|,|iyy|,|izz|)``.
    Links above the tolerance are listed in decreasing order of ratio so
    the worst offenders surface first.

    Detect-only — the fix needs mesh data (``recompute_inertia.py`` or a
    per-package mesh-based recompute) so it can't live in this tool.
    """
    body = re.sub(r"^\s*<\?xml[^?]*\?>\s*", "", text, count=1)
    wrapped = '<__diagnose_root xmlns:xacro="http://ros.org/wiki/xacro">\n' f"{body}\n" "</__diagnose_root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []

    out: List[Tuple[str, float]] = []
    for link in root.iter("link"):
        name = link.attrib.get("name")
        if not name:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia = inertial.find("inertia")
        if inertia is None:
            continue

        def _f(attr: str) -> float:
            v = inertia.attrib.get(attr, "0")
            try:
                return float(v)
            except ValueError:
                # xacro expression that didn't expand (``${...}``) — skip,
                # we can't evaluate without the xacro processor.
                return 0.0

        # Mark unevaluated xacro expressions so we don't false-flag.
        if any("${" in inertia.attrib.get(k, "") for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")):
            continue

        ixx, iyy, izz = _f("ixx"), _f("iyy"), _f("izz")
        ixy, ixz, iyz = _f("ixy"), _f("ixz"), _f("iyz")
        diag_max = max(abs(ixx), abs(iyy), abs(izz))
        off_max = max(abs(ixy), abs(ixz), abs(iyz))
        if diag_max <= 0:
            continue
        ratio = off_max / diag_max
        if ratio > _INERTIA_OFF_DIAG_TOL:
            out.append((name, ratio))
    out.sort(key=lambda kv: -kv[1])
    return out


# Two diagonal entries within this fraction trigger Isaac's URDF→USD
# degenerate-shortcut bug. 5% chosen by inspection of the observed
# failures (body_link5 at 1.8%, body_link1 at 4.7%, head_link1 ~0%).
_INERTIA_DEGEN_TOL = 5e-2

# Pairs whose relative gap is below this floor are treated as INTENTIONALLY
# degenerate (placeholder inertias, snapped-to-equal recompute output,
# mesh-symmetric values that match to ~10 significant digits). Floor at 1e-6
# — million times tighter than the bug threshold; the bug needs values
# *close* but not *identical*. Floating-point comparisons aren't equal at
# 0.0 unless the author or tooling deliberately made them equal.
_INERTIA_DEGEN_MIN_GAP = 1e-6


def _links_with_degenerate_diagonal(text: str) -> List[Tuple[str, float, str]]:
    """Return ``[(link_name, gap, pair_label), ...]`` for every link whose
    ``<inertia>`` is diagonal (off-diagonals are zero), ``<origin rpy>``
    is zero, AND two of (ixx, iyy, izz) are within
    ``_INERTIA_DEGEN_TOL`` of each other.

    These specifically trip the Isaac/Newton USD↔MJCF convention
    mismatch — see the module docstring entry for the failure mode.
    Detect-only — the fix is to spread the near-degenerate pair and
    swap the diagonal entries so the sorted-ascending order is a
    transposition rather than a 3-cycle (see
    ``scripts/recompute_g2_inertia.py``).
    """
    body = re.sub(r"^\s*<\?xml[^?]*\?>\s*", "", text, count=1)
    wrapped = '<__diagnose_root xmlns:xacro="http://ros.org/wiki/xacro">\n' f"{body}\n" "</__diagnose_root>"
    try:
        root = ET.fromstring(wrapped)
    except ET.ParseError:
        return []

    out: List[Tuple[str, float, str]] = []
    for link in root.iter("link"):
        name = link.attrib.get("name")
        if not name:
            continue
        inertial = link.find("inertial")
        if inertial is None:
            continue
        inertia = inertial.find("inertia")
        if inertia is None:
            continue

        # Need numeric attrs throughout. Bail on any xacro expression.
        attrs = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        if any("${" in inertia.attrib.get(k, "") for k in attrs):
            continue

        def _f(attr: str) -> float:
            try:
                return float(inertia.attrib.get(attr, "0"))
            except ValueError:
                return 0.0

        # Skip links whose <inertia> already isn't diagonal — the off-
        # diagonal check covers them separately, and the degeneracy
        # bug specifically requires the diagonal-shortcut shape.
        ixy, ixz, iyz = _f("ixy"), _f("ixz"), _f("iyz")
        ixx, iyy, izz = _f("ixx"), _f("iyy"), _f("izz")
        diag_max = max(abs(ixx), abs(iyy), abs(izz))
        if diag_max <= 0:
            continue
        if max(abs(ixy), abs(ixz), abs(iyz)) / diag_max >= _INERTIA_OFF_DIAG_TOL:
            continue

        # The bug specifically triggers for rpy="0 0 0" on the inertial
        # origin (or absent rpy → defaults to zero). Non-zero rpy pushes
        # the converter into a different path that doesn't have this bug.
        origin = inertial.find("origin")
        rpy_str = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"
        if "${" in rpy_str:
            continue
        try:
            rpy = [float(v) for v in rpy_str.split()]
        except ValueError:
            continue
        if any(abs(r) > 1e-9 for r in rpy):
            continue

        # Find the closest pair.
        pairs = [("ixx,iyy", ixx, iyy), ("ixx,izz", ixx, izz), ("iyy,izz", iyy, izz)]
        best_label = None
        best_gap = 1.0
        for label, a, b in pairs:
            m = max(abs(a), abs(b))
            if m == 0:
                continue
            gap = abs(a - b) / m
            if gap < best_gap:
                best_gap, best_label = gap, label
        if best_label is not None and _INERTIA_DEGEN_MIN_GAP < best_gap < _INERTIA_DEGEN_TOL:
            out.append((name, best_gap, best_label))
    out.sort(key=lambda kv: kv[1])
    return out


# ---------------------------------------------------------------------------
# Mesh unit (mm-scale export) detection
# ---------------------------------------------------------------------------

# Matches the inner attribute string of any <mesh filename="..."> in
# the URDF text.  Used to enumerate every mesh path declared without
# parsing the full XML (so we work for xacros too, where the filename
# can include ``${mesh_dir}/...`` placeholders).
_MESH_FILENAME_RE = re.compile(r'<mesh\b[^/>]*\bfilename\s*=\s*"([^"]+)"')

# Bounds threshold past which a mesh is almost certainly a non-metre
# export.  Human-scale robots have body links well under 1.5 m; 5 m
# is conservative enough to exclude any plausible robot link and
# loose enough to avoid false positives on whole-robot assemblies
# someone might have authored into a single mesh.
_MESH_BOUNDS_MAX_METRES = 5.0


def _mesh_bad_units(text: str, *, file_path: Path, mesh_root: Optional[Path]) -> List[Tuple[str, str, float]]:
    """Return ``(filename, resolved_path, max_extent_metres)`` for
    every mesh referenced in ``text`` whose vertex bounds extend
    past ``_MESH_BOUNDS_MAX_METRES``.

    Loading is best-effort; meshes that fail to load (path missing,
    bad geometry) are silently skipped — they're a separate class
    of problem.  The check uses ``trimesh`` if available; without
    it the rule is a no-op (returns empty list with a single
    one-time warning on stderr).
    """
    try:
        import trimesh  # noqa: PLC0415
    except ImportError:
        if not getattr(_mesh_bad_units, "_trimesh_warned", False):
            print(
                "  WARN: trimesh not available — skipping mesh-unit check.  "
                "Install with `pip install trimesh` to enable.",
                file=sys.stderr,
            )
            _mesh_bad_units._trimesh_warned = True  # type: ignore[attr-defined]
        return []

    out: List[Tuple[str, str, float]] = []
    seen: set = set()
    for m in _MESH_FILENAME_RE.finditer(text):
        raw = m.group(1)
        if raw in seen:  # same mesh referenced from visual + collision
            continue
        seen.add(raw)
        # Skip paths with unresolved xacro substitutions unless we can
        # rewrite via ``mesh_root``.  Without resolution we can't load.
        resolved = _resolve_mesh_path(raw, file_path=file_path, mesh_root=mesh_root)
        if resolved is None or not resolved.exists():
            continue
        try:
            mesh = trimesh.load(str(resolved), force="mesh")
        except Exception:  # noqa: BLE001
            continue
        if isinstance(mesh, trimesh.Scene):
            try:
                mesh = mesh.to_geometry() if hasattr(mesh, "to_geometry") else mesh.dump(concatenate=True)
            except Exception:  # noqa: BLE001
                continue
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        try:
            bmin, bmax = mesh.bounds
            extent = float(max(abs(bmin).max(), abs(bmax).max()))
        except Exception:  # noqa: BLE001
            continue
        if extent > _MESH_BOUNDS_MAX_METRES:
            out.append((raw, str(resolved), extent))
    return out


def _resolve_mesh_path(raw: str, *, file_path: Path, mesh_root: Optional[Path]) -> Optional[Path]:
    """Resolve a URDF / xacro mesh filename to a filesystem path.

    Mirrors the resolver in ``recompute_inertia.py``; kept inline
    here so ``diagnose_urdf`` stays a single-file tool.  Handles
    ``${mesh_dir}`` placeholder (substituted via ``mesh_root``),
    ``package://<pkg>/...`` (walks up from ``file_path``), absolute,
    and relative paths.
    """
    s = raw.strip()
    if "${mesh_dir}" in s and mesh_root is not None:
        cand = Path(s.replace("${mesh_dir}", str(mesh_root)))
        if cand.exists():
            return cand
        if not cand.is_absolute():
            cand = (file_path.parent / cand).resolve()
            if cand.exists():
                return cand
        return None
    if s.startswith("package://"):
        rest = s[len("package://") :]
        slash = rest.find("/")
        if slash < 0:
            return None
        pkg = rest[:slash]
        sub = rest[slash + 1 :]
        cur = file_path.parent
        for _ in range(8):
            for d in (cur / pkg, cur / "share" / pkg):
                if d.is_dir():
                    cand = d / sub
                    if cand.exists():
                        return cand
            cur = cur.parent
        if mesh_root is not None:
            cand = mesh_root / sub
            if cand.exists():
                return cand
        return None
    p = Path(s)
    if p.is_absolute():
        return p if p.exists() else None
    cand = (file_path.parent / p).resolve()
    return cand if cand.exists() else None


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Report URDF/xacro authoring problems without modifying files.",
        epilog="See module docstring for the why and the developer workflow.",
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
        help="Exit non-zero for every finding, including missing inertials on frames.",
    )
    parser.add_argument(
        "--mesh-root",
        type=Path,
        default=None,
        help="Substitute ${mesh_dir} placeholders against this directory "
        "(for xacro inputs).  Required for the mm-scale mesh-units check "
        "when meshes are referenced through ${mesh_dir}; without it those "
        "meshes can't be resolved and the check silently skips them.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    files = _find_target_files(args.paths)
    if not files:
        print("No .xacro/.urdf files found.")
        return 0

    print(f"Scanned {len(files)} file(s).")
    print()

    # Scan and report without modifying input files.
    #   * inertial    — advisory in default mode
    #   * ghost_root  — detect-only (URDF root with no visual/collision/inertial)
    #   * wheels      — detect-only
    inertial_per_file: List[Tuple[Path, List[str]]] = []
    ghost_root_per_file: List[Tuple[Path, List[str]]] = []
    wheel_per_file: List[Tuple[Path, List[str]]] = []
    # Off-diagonal inertia: per file, list of (link_name, ratio) tuples.
    off_diag_per_file: List[Tuple[Path, List[Tuple[str, float]]]] = []
    # Degenerate-diagonal inertia (trips Isaac's URDF→USD shortcut bug):
    # per file, list of (link_name, gap_ratio, pair_label) tuples.
    degen_per_file: List[Tuple[Path, List[Tuple[str, float, str]]]] = []
    # Mesh-unit check: per file, list of (filename, resolved_path,
    # bounds_extent_metres) tuples.  Detect-only; auto-fix is in the
    # sibling ``fix_dae_units.py`` tool.
    mesh_unit_per_file: List[Tuple[Path, List[Tuple[str, str, float]]]] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"  ERROR    cannot read {path}: {exc}", file=sys.stderr)
            continue
        missing, ghost_roots = _links_missing_inertial(text)
        if missing:
            inertial_per_file.append((path, missing))
        if ghost_roots:
            ghost_root_per_file.append((path, ghost_roots))
        wheels = _wheels_without_cylinder_collision(text)
        if wheels:
            wheel_per_file.append((path, wheels))
        off_diag = _links_with_off_diagonal_inertia(text)
        if off_diag:
            off_diag_per_file.append((path, off_diag))
        degen = _links_with_degenerate_diagonal(text)
        if degen:
            degen_per_file.append((path, degen))
        bad_units = _mesh_bad_units(text, file_path=path, mesh_root=args.mesh_root)
        if bad_units:
            mesh_unit_per_file.append((path, bad_units))

    total_inertial = sum(len(n) for _, n in inertial_per_file)
    total_ghost_root = sum(len(n) for _, n in ghost_root_per_file)
    total_wheels = sum(len(n) for _, n in wheel_per_file)
    total_off_diag = sum(len(n) for _, n in off_diag_per_file)
    total_degen = sum(len(n) for _, n in degen_per_file)
    total_mesh_units = sum(len(n) for _, n in mesh_unit_per_file)

    if (
        not inertial_per_file
        and not ghost_root_per_file
        and not wheel_per_file
        and not off_diag_per_file
        and not degen_per_file
        and not mesh_unit_per_file
    ):
        print("No problems detected.")
        return 0

    if inertial_per_file:
        print(f"[inertial] Links missing <inertial>: {total_inertial}")
        print(
            "           (non-root links with no <inertial> and no <collision>.\n"
            "           DETECT-ONLY since Isaac Sim 6.2: such a link is left a pure\n"
            "           kinematic frame rather than promoted to a massless rigid\n"
            "           body. For a mount\n"
            "           or sensor frame that is the desired outcome. It is still a\n"
            "           real defect for a link that must be DYNAMIC -- without an\n"
            "           <inertial> it silently never becomes a body at all, so give\n"
            "           those links real mass rather than a placeholder.)"
        )
        for path, names in inertial_per_file:
            print(f"  {path}")
            for n in names:
                print(f"    - {n}")
        print()

    if ghost_root_per_file:
        print(
            f"[ghost-root] URDF root link with no visual/collision/inertial: {total_ghost_root}\n"
            f"             (detect-only — KDL forbids inertia on the root, so the\n"
            f"             tool can't auto-fix. The 6.x URDF→USD converter takes\n"
            f"             its ``is_ghost_link`` branch and authors broken body0\n"
            f"             refs for every joint. Fix by giving the root a visual\n"
            f"             or starting the kinematic chain at a non-ghost link.)"
        )
        for path, names in ghost_root_per_file:
            print(f"  {path}")
            for n in names:
                print(f"    - {n}")
        print()

    if wheel_per_file:
        print(
            f"[wheel] Wheel-like links lacking <cylinder> collision: {total_wheels}\n"
            f"        (detect-only — author <cylinder> by hand; radius/length\n"
            f"        depend on the visual mesh and can't be inferred from\n"
            f"        URDF text.)"
        )
        for path, names in wheel_per_file:
            print(f"  {path}")
            for n in names:
                print(f"    - {n}")
        print()

    if off_diag_per_file:
        print(
            f"[inertia-form] Links with non-diagonal <inertia>: {total_off_diag}\n"
            f"               (detect-only — author as diagonal tensor plus\n"
            f"               <origin rpy>; matches USD PhysicsMassAPI form\n"
            f"               and is much easier to read. Recompute via\n"
            f"               `scripts/recompute_inertia.py` (CAD-tilted-frame\n"
            f"               cases) or a mesh-based diagonalising recompute\n"
            f"               for the link's own values. Threshold:\n"
            f"               max|off-diag| / max|diag| > {_INERTIA_OFF_DIAG_TOL:g}.)"
        )
        for path, entries in off_diag_per_file:
            print(f"  {path}")
            for name, ratio in entries:
                print(f"    - {name}  (off/diag={ratio:.3f})")
        print()

    if degen_per_file:
        print(
            f"[inertia-degenerate] Diagonal <inertia> with rpy=0 and two nearly-equal\n"
            f"                     diagonal entries: {total_degen}\n"
            f"                     (detect-only — this shape can trip the Isaac/Newton\n"
            f"                     USD↔MJCF convention mismatch, which only manifests\n"
            f"                     for 3-cycle principalAxes rotations and produces a\n"
            f"                     90°-rotated inertia ellipsoid in MuJoCo. Fix: spread\n"
            f"                     the near-equal pair to >=6%% relative gap AND swap\n"
            f"                     entries so the sort order is a transposition, not a\n"
            f"                     3-cycle (see recompute_g2_inertia.py). Threshold:\n"
            f"                     |a-b| / max(|a|,|b|) < {_INERTIA_DEGEN_TOL:g}.)"
        )
        for path, entries in degen_per_file:
            print(f"  {path}")
            for name, gap, pair_label in entries:
                print(f"    - {name}  ({pair_label}, gap={gap:.3f})")
        print()

    if mesh_unit_per_file:
        print(
            f"[mesh-units] Meshes with bounds > {_MESH_BOUNDS_MAX_METRES} m "
            f"(mm-scale exports?): {total_mesh_units}\n"
            f"             (detect-only — fix with `scripts/fix_dae_units.py "
            f"<mesh-or-dir>`.\n"
            f"             That tool rewrites the .dae vertex coordinates,\n"
            f"             scales <translate> elements, and updates the\n"
            f"             <unit> declaration in place with a .bak backup.)"
        )
        for path, entries in mesh_unit_per_file:
            print(f"  {path}")
            for filename, resolved, extent in entries:
                print(f"    - {filename}  →  bounds ±{extent:.1f} m  ({resolved})")
        print()

    if args.dry_run:
        print("--dry-run: no files modified.")
        return 1

    if total_inertial:
        # No longer auto-fixed. Injecting a 1e-6 placeholder was correct when the
        # importer promoted every link to a rigid body; on 6.2 it is the thing
        # that *creates* a near-massless body out of a kinematic frame. Report
        # and let the author decide: real mass, or leave it a frame.
        print(
            f"[inertial] {total_inertial} link(s) reported above are NOT auto-fixed.\n"
            "           Empty kinematic frames need no placeholder mass; dynamic\n"
            "           bodies need real mass and inertia -- see the note above."
        )
        print()
    if total_wheels or total_ghost_root or total_off_diag or total_degen or total_mesh_units:
        print()
        if total_wheels:
            print(
                f"NOTE: {total_wheels} wheel link(s) listed above lack <cylinder>\n"
                f"collision. The tool can't auto-fix these — author the cylinder by\n"
                f"hand."
            )
        if total_ghost_root:
            print(
                f"NOTE: {total_ghost_root} ghost-root link(s) listed above. The tool\n"
                f"can't auto-fix these — KDL forbids inertia on the URDF root, so\n"
                f"the fix must restructure the URDF (add visual/collision to the root\n"
                f"or start the kinematic chain at a non-ghost link)."
            )
        if total_off_diag:
            print(
                f"NOTE: {total_off_diag} link(s) listed above have a non-diagonal\n"
                f"<inertia>. Recompute the link's tensor (under uniform density from\n"
                f"the visual mesh, or via the existing operator-calibrated values)\n"
                f"and re-author as diagonal ixx/iyy/izz with the principal-axis\n"
                f"rotation moved into <origin rpy>. See `scripts/recompute_inertia.py`."
            )
        if total_degen:
            print(
                f"NOTE: {total_degen} link(s) listed above have a near-degenerate\n"
                f"diagonal <inertia> with rpy=0 — this shape can trip the Isaac/Newton\n"
                f"USD↔MJCF convention mismatch (3-cycle principalAxes quaternion vs\n"
                f"Newton's transposed reading), causing a 90° rotated inertia tensor\n"
                f"in MuJoCo. Fix by spreading the near-equal pair to >=6%% gap AND\n"
                f"swapping diagonal entries so the sort order is a transposition\n"
                f"rather than a 3-cycle. See scripts/recompute_g2_inertia.py."
            )
        if total_mesh_units:
            print(
                f"NOTE: {total_mesh_units} mesh file(s) listed above appear to be "
                f"mm-scale\nexports (bounds > {_MESH_BOUNDS_MAX_METRES} m).  Auto-fix lives in\n"
                f"`scripts/fix_dae_units.py` — run it on the meshes to rewrite the\n"
                f".dae vertex coordinates + <unit> declaration in place."
            )
        print("Exit code is non-zero so these stay visible in CI.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
