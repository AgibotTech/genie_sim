#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Recompute G2 inertias from visual meshes and emit diagonal+rpy form.

What this is
------------

A G2-specific cousin of ``recompute_inertia.py``. ``recompute_inertia.py``
fixes CAD-supplied inertia tensors whose authored *frame* was wrong (the
swiftpicker-fingers case in its docstring). This script does a stronger
thing: it discards the URDF's authored inertia tensor entirely and
**re-derives it from the visual mesh** under uniform density, then
writes the result back in the project's canonical diagonal-plus-``rpy``
form.

Per link the pipeline is:

  1. Read the URDF/xacro ``<inertial>``:
       * mass — preserved verbatim (CAD-calibrated, hand-tuned)
       * CoM xyz — preserved verbatim (CAD-calibrated)
  2. Resolve every ``<visual><mesh>`` for the link; load through
     ``trimesh`` (multi-part DAE scenes auto-merge with node
     transforms applied); apply each visual's own ``<origin xyz rpy>``
     to put the mesh into link frame; sum the parts at uniform density
     ``rho = mass / sum(V_i)``.
  3. Translate the combined inertia from the mesh CoM to the
     URDF-authored CoM via the parallel-axis theorem.
  4. Diagonalise. Several post-processing steps fight known
     downstream-converter bugs (see "Why all the cleanup steps" below).
  5. Rewrite the ``<inertial>`` block in place with:

         <inertial>
           <origin xyz="..." rpy="r p y"/>
           <mass value="m"/>
           <inertia ixx="I1" iyy="I2" izz="I3" ixy="0" ixz="0" iyz="0"/>
         </inertial>

Mass and CoM xyz are byte-identical to the input. Only the inertia
tensor (and its rotation) are rewritten.

Why all the cleanup steps
-------------------------

Three post-diagonalisation rewrites in ``compute_link_inertia``, each
working around a specific downstream issue. Removing any of them
re-introduces a known failure mode — keep them all.

* **SVD degenerate-subspace snap (in ``diagonalize``)**: ``np.linalg.eigh``
  returns an arbitrary orthonormal basis in any 2D degenerate eigenvalue
  subspace, which then gets stamped into the URDF ``<origin rpy>`` and
  shows up in MuJoCo as a spurious ~45° rotation on a visually-symmetric
  link (head/torso stalk). The snap picks the in-plane rotation closest
  to identity (orthogonal Procrustes / SVD), so symmetric meshes get
  ``rpy=0`` instead of arbitrary garbage.

* **Snap small post-diagonalize ``rpy`` to zero**: even after the SVD
  snap, tilts ≤ 5.7° often survive from minor CAD asymmetries (mounting
  holes, wire routing). For a robot whose links should be mirror-symmetric
  L/R, those small tilts add asymmetric reaction forces during settling.
  Below 0.1 rad we use the raw link-frame diagonal of ``I_total`` (which
  discards the small off-diagonals) and emit ``rpy="0 0 0"``.

* **Eigenvalue spread to 6%**: Isaac Sim 6.x's URDF→USD converter takes
  a shortcut for diagonal ``<inertia>`` with ``rpy=0``. When the
  sorted-ascending order of (ixx, iyy, izz) forms a 3-cycle of link
  axes — eg. ``izz<ixx<iyy`` — Isaac picks a 3-cycle ``principalAxes``
  quaternion. Newton's ``SolverMuJoCo.save_to_mjcf`` then converts that
  USD to MJCF using the OPPOSITE convention from USD spec (``R · diag · Rᵀ``
  vs USD's ``Rᵀ · diag · R``), and the two only differ for 3-cycle R.
  The result: MuJoCo shows the inertia ellipsoid permuted by a cyclic
  shift — 90° off from what RViz shows. See ``newton_quirks.md`` for
  the upstream trace.

* **3-cycle-permutation avoidance**: if after spreading the sort order
  is still a 3-cycle, swap two diagonal entries so the order becomes a
  transposition. Transpositions are invariant under the convention
  mismatch (``R · diag · Rᵀ == Rᵀ · diag · R`` for transposition R), so
  Newton's bug becomes invisible. The two diagonal entries we swap are
  near-equal after spreading, so the physical impact is at the noise
  floor of the uniform-density assumption.

CoM policy
----------

CoM is NOT recomputed. The URDF's authored ``<origin xyz>`` is preserved
verbatim. URDFs often have CoMs hand-tuned for balance or measured
against hardware; replacing them with the mesh's volumetric CoM
(reasonable only for uniform-density solid links) would silently change
the physics. The mesh CoM is used internally for the parallel-axis shift
to wherever the URDF CoM sits.

Mass policy
-----------

Same as ``recompute_inertia.py``: mass is preserved verbatim. The mass
defines the density (``rho = m / V``), which scales the recomputed
inertia, but the authored mass value flows through.

Idempotence
-----------

After one full pass the script is idempotent: rerunning produces no
further edits because the post-processing pinned every output (rpy and
sort permutation) to a deterministic shape.

Usage
-----

::

    python3 scripts/recompute_g2_inertia.py

No arguments — the script operates on the G2 xacros in this package.
``diagnose_urdf.py --dry-run robots/genie/g2`` should pass cleanly
afterwards; if a new check fires (``[inertia-form]``,
``[inertia-degenerate]``), the pipeline has regressed and the new entry
needs investigation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

try:
    import trimesh
except ImportError as exc:
    print(
        f"[recompute_g2_inertia] trimesh is required: {exc}. " f"Install with `pip install trimesh`.",
        file=sys.stderr,
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Paths and tunables
# ---------------------------------------------------------------------------

# Resolve the package root relative to this script's location so the tool
# is portable across checkouts. ``__file__`` sits in ``scripts/``; the
# package root is its parent.
_PKG = Path(__file__).resolve().parent.parent

_MESH_ROOT = _PKG / "robots/genie/g2/meshes"
_XACROS = [
    _PKG / "robots/genie/g2/xacro/G2.body.urdf.xacro",
    _PKG / "robots/genie/g2/xacro/G2.chassis.urdf.xacro",
    _PKG / "robots/genie/g2/xacro/G2.arm.crs.urdf.xacro",
    _PKG / "robots/genie/g2/xacro/G2.arm.crsB.urdf.xacro",
    _PKG / "robots/genie/g2/xacro/gripper.omnipicker.urdf.xacro",
    _PKG / "robots/genie/g2/xacro/gripper.swiftpicker.urdf.xacro",
]

# Below this relative off-diagonal magnitude the input tensor is treated
# as already axis-aligned and ``diagonalize`` emits ``rpy=0`` directly,
# skipping the eigendecomposition path. 3% chosen empirically: anything
# below is dominated by mesh-noise off-diagonals; above is a real CAD
# rotation.
_OFF_DIAG_REL_TOL = 3e-2

# After diagonalisation, any ``rpy`` whose largest component is below
# this magnitude (rad) is replaced with ``(0, 0, 0)`` and the URDF
# inertia is taken as the raw link-frame diagonal. 0.1 rad = 5.7°.
_SMALL_RPY_THRESHOLD = 0.1

# Two principal moments closer than this fraction get pushed apart to
# exactly this fraction, preserving their average. Anti-Isaac shortcut
# (see module docstring).
_SPREAD_TARGET = 6e-2

# When ``np.linalg.eigh`` returns an arbitrary basis in a near-degenerate
# eigvalue subspace, snap the in-plane rotation to identity if the gap
# is below this fraction.
_DEGEN_SUBSPACE_TOL = 5e-3

# Links with mass below this value are placeholder kinematic frames
# (mass=0.001, inertia=1e-6); skipped entirely.
_PLACEHOLDER_MASS = 1.1e-3


# ---------------------------------------------------------------------------
# Regexes — URDF/xacro text extraction
# ---------------------------------------------------------------------------

_LINK_RE = re.compile(r"(<link\b[^>]*?>)(.*?)(</link>)", re.DOTALL)
_LINK_NAME_RE = re.compile(r'\bname\s*=\s*"([^"]+)"')
_INERTIAL_RE = re.compile(r"<inertial\b[^>]*>(.*?)</inertial>", re.DOTALL)
_MASS_RE = re.compile(r'<mass\b[^/>]*\bvalue\s*=\s*"([^"]+)"')
_ORIGIN_RE = re.compile(r"<origin\b([^/>]*)/?>")
_VISUAL_RE = re.compile(r"<visual\b[^>]*>(.*?)</visual>", re.DOTALL)
_MESH_RE = re.compile(r'<mesh\b[^/>]*\bfilename\s*=\s*"([^"]+)"')


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _rpy_to_R(rpy):
    """URDF rpy → rotation matrix. ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _R_to_rpy(R):
    """Rotation matrix → URDF rpy. Handles gimbal-lock at pitch=±π/2."""
    sy = -R[2, 0]
    cy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    pitch = np.arctan2(sy, cy)
    if cy > 1e-9:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return roll, pitch, yaw


def _parse_origin(attr):
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    m = re.search(r'\bxyz\s*=\s*"([^"]+)"', attr)
    if m:
        xyz = np.array([float(v) for v in m.group(1).split()])
    m = re.search(r'\brpy\s*=\s*"([^"]+)"', attr)
    if m:
        rpy = np.array([float(v) for v in m.group(1).split()])
    return xyz, rpy


# ---------------------------------------------------------------------------
# Diagonalisation + Newton-quirk workarounds
# ---------------------------------------------------------------------------


def _snap_degen_subspace(eigvecs, *, unique_col):
    """Within the 2D subspace spanned by the two non-unique eigenvector
    columns, rotate so those columns sit closest to the corresponding pair
    of basis axes (orthogonal Procrustes via SVD). Avoids spurious ~45°
    ``rpy`` values that ``eigh`` would otherwise emit on rotationally-
    symmetric links."""
    others = [i for i in range(3) if i != unique_col]
    U_sub = eigvecs[:, others]
    u = eigvecs[:, unique_col]
    u = u / np.linalg.norm(u)
    proj = np.abs(u)
    keep_basis = sorted([(proj[k], k) for k in range(3)])[:2]
    keep_basis.sort(key=lambda kv: kv[1])
    target = np.zeros((3, 2))
    target[keep_basis[0][1], 0] = 1.0
    target[keep_basis[1][1], 1] = 1.0
    M2 = U_sub.T @ target
    Uv, _, Vt = np.linalg.svd(M2)
    R2 = Uv @ Vt
    new_pair = U_sub @ R2
    out = eigvecs.copy()
    out[:, others[0]] = new_pair[:, 0]
    out[:, others[1]] = new_pair[:, 1]
    return out


def _diagonalize(M):
    """Diagonalise a 3×3 symmetric inertia tensor in link frame. Returns
    ``(D, rpy)`` such that ``R(rpy) · diag(D) · R(rpy)ᵀ = M``.

    Fast-path returns ``rpy=(0,0,0)`` for near-diagonal input. Otherwise
    uses ``eigh`` with the degenerate-subspace SVD snap to avoid spurious
    rotations on rotationally-symmetric inputs.
    """
    diag_max = max(abs(M[0, 0]), abs(M[1, 1]), abs(M[2, 2]))
    off_max = max(abs(M[0, 1]), abs(M[0, 2]), abs(M[1, 2]))
    if diag_max > 0 and off_max / diag_max < _OFF_DIAG_REL_TOL:
        return np.array([M[0, 0], M[1, 1], M[2, 2]]), (0.0, 0.0, 0.0)
    eigvals, eigvecs = np.linalg.eigh(M)
    # Degenerate-subspace cleanup
    max_eig = float(abs(eigvals).max())
    if max_eig > 0:
        gap01 = abs(eigvals[1] - eigvals[0]) / max_eig
        gap12 = abs(eigvals[2] - eigvals[1]) / max_eig
        if gap01 < _DEGEN_SUBSPACE_TOL and gap12 < _DEGEN_SUBSPACE_TOL:
            eigvecs = np.eye(3)
        elif gap01 < _DEGEN_SUBSPACE_TOL:
            eigvecs = _snap_degen_subspace(eigvecs, unique_col=2)
        elif gap12 < _DEGEN_SUBSPACE_TOL:
            eigvecs = _snap_degen_subspace(eigvecs, unique_col=0)
    # Greedy permutation to keep R close to identity
    perm = [-1, -1, -1]
    used = set()
    abs_v = np.abs(eigvecs)
    pairs = sorted([(abs_v[i, j], i, j) for i in range(3) for j in range(3)], reverse=True)
    for _, i, j in pairs:
        if perm[i] == -1 and j not in used:
            perm[i] = j
            used.add(j)
    R = np.column_stack([eigvecs[:, perm[i]] for i in range(3)])
    D = np.array([eigvals[perm[i]] for i in range(3)])
    for i in range(3):
        if R[i, i] < 0:
            R[:, i] *= -1
    if np.linalg.det(R) < 0:
        worst = int(np.argmin([abs(R[i, i]) for i in range(3)]))
        R[:, worst] *= -1
    return D, _R_to_rpy(R)


def _spread_near_degenerate(D, target_gap):
    """Push the worst near-degenerate pair apart to ``target_gap``
    relative gap (preserving the pair's average). Iterates because one
    push can elevate a different pair to the worst slot."""
    D = D.copy()
    for _ in range(3):
        worst = None
        for i, j in ((0, 1), (0, 2), (1, 2)):
            m = max(abs(D[i]), abs(D[j]))
            if m == 0:
                continue
            gap = abs(D[i] - D[j]) / m
            if gap < target_gap and (worst is None or gap < worst[0]):
                worst = (gap, i, j)
        if worst is None:
            break
        _, i, j = worst
        avg = (D[i] + D[j]) / 2.0
        half = avg * target_gap / 2.0
        if D[i] < D[j]:
            D[i], D[j] = avg - half, avg + half
        else:
            D[i], D[j] = avg + half, avg - half
    return D


def _avoid_3cycle_sort(D):
    """Newton's ``save_to_mjcf`` uses ``I_body = R · diag · Rᵀ``, but USD's
    ``principalAxes`` convention requires ``I_body = Rᵀ · diag · R``. The
    two coincide for transposition rotations and differ for 3-cycle
    rotations. Isaac picks a 3-cycle quaternion exactly when the
    sorted-ascending order of (ixx, iyy, izz) cyclically permutes the
    link axes. If so, swap two diagonal entries so the sort order becomes
    a transposition; Newton's bug then becomes invisible.

    See ``newton_quirks.md`` for the full trace.
    """
    D = D.copy()
    order = sorted(range(3), key=lambda k: D[k])
    if order == [1, 2, 0]:
        D[2], D[0] = D[0], D[2]
    elif order == [2, 0, 1]:
        D[0], D[1] = D[1], D[0]
    return D


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------


def _resolve_mesh(filename):
    """Resolve a ``${mesh_dir}/...`` path against ``_MESH_ROOT``."""
    raw = filename.replace("${mesh_dir}", str(_MESH_ROOT))
    p = Path(raw)
    return p if p.exists() else None


def _load_visual_parts(link_body):
    """Load every ``<visual><mesh>`` for a link, applying the visual's
    ``<origin xyz rpy>`` to put each mesh into link frame. Returns a list
    of ``{"volume": ..., "mesh": Trimesh}`` per visual, or ``None`` if
    any referenced mesh is unresolvable or has bad geometry."""
    parts = []
    for v_match in _VISUAL_RE.finditer(link_body):
        vbody = v_match.group(1)
        mm = _MESH_RE.search(vbody)
        if not mm:
            continue
        mesh_path = _resolve_mesh(mm.group(1))
        if mesh_path is None:
            return None
        origin_m = _ORIGIN_RE.search(vbody)
        v_xyz, v_rpy = _parse_origin(origin_m.group(1) if origin_m else "")
        mesh = trimesh.load(mesh_path, force="mesh")
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        if not isinstance(mesh, trimesh.Trimesh) or mesh.volume is None or mesh.volume <= 0:
            return None
        T = np.eye(4)
        T[:3, :3] = _rpy_to_R(v_rpy)
        T[:3, 3] = v_xyz
        mesh = mesh.copy()
        mesh.apply_transform(T)
        parts.append({"volume": float(mesh.volume), "mesh": mesh})
    return parts


def _compute_link_inertia(mass, com_xyz, parts):
    """Sum per-part inertias under uniform link density, translated to
    the URDF-authored CoM via parallel-axis, then diagonalise + apply
    the cleanup steps documented at the top of this file."""
    V_tot = sum(p["volume"] for p in parts)
    if V_tot <= 0:
        return None, None
    rho = mass / V_tot
    I_total = np.zeros((3, 3))
    com_xyz = np.asarray(com_xyz, dtype=float)
    for p in parts:
        m_p = rho * p["volume"]
        mesh = p["mesh"]
        mesh.density = rho
        com_p = np.asarray(mesh.center_mass, dtype=float)
        I_part_com = np.asarray(mesh.moment_inertia, dtype=float)
        d = com_xyz - com_p
        I_part_at_urdf = I_part_com + m_p * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        I_total += I_part_at_urdf

    D, rpy = _diagonalize(I_total)
    # Drop sub-5.7° tilts (CAD-asymmetry noise).
    if max(abs(r) for r in rpy) < _SMALL_RPY_THRESHOLD:
        D = np.array([I_total[0, 0], I_total[1, 1], I_total[2, 2]])
        rpy = (0.0, 0.0, 0.0)
    # Newton-quirk workarounds.
    D = _spread_near_degenerate(D, _SPREAD_TARGET)
    D = _avoid_3cycle_sort(D)
    return D, rpy


# ---------------------------------------------------------------------------
# Block emit + file rewrite
# ---------------------------------------------------------------------------


def _fmt(x):
    if abs(x) < 1e-12:
        return "0"
    return f"{x:.8g}"


def _build_inertial_block(indent, com_xyz, rpy, mass, D):
    pad = " " * indent
    sub = " " * (indent + 2)
    return (
        f"{pad}<inertial>\n"
        f'{sub}<origin xyz="{com_xyz[0]} {com_xyz[1]} {com_xyz[2]}" '
        f'rpy="{_fmt(rpy[0])} {_fmt(rpy[1])} {_fmt(rpy[2])}" />\n'
        f'{sub}<mass value="{mass}" />\n'
        f'{sub}<inertia ixx="{_fmt(D[0])}" iyy="{_fmt(D[1])}" izz="{_fmt(D[2])}" '
        f'ixy="0" ixz="0" iyz="0" />\n'
        f"{pad}</inertial>"
    )


def _process_file(path):
    text = path.read_text(encoding="utf-8")
    edits = []
    for link_match in _LINK_RE.finditer(text):
        link_open = link_match.group(1)
        body = link_match.group(2)
        name_m = _LINK_NAME_RE.search(link_open)
        if not name_m:
            continue
        link_name = name_m.group(1)

        inertial_m = _INERTIAL_RE.search(body)
        if not inertial_m:
            continue
        inertial_body = inertial_m.group(1)

        mass_m = _MASS_RE.search(inertial_body)
        if not mass_m:
            continue
        mass = float(mass_m.group(1))
        if mass < _PLACEHOLDER_MASS:
            print(f"  [{link_name}] skip placeholder (mass={mass})")
            continue

        origin_m = _ORIGIN_RE.search(inertial_body)
        com_xyz, _ = _parse_origin(origin_m.group(1) if origin_m else "")

        parts = _load_visual_parts(body)
        if parts is None or len(parts) == 0:
            print(f"  [{link_name}] skip: no resolvable visual meshes")
            continue

        try:
            D, rpy = _compute_link_inertia(mass, com_xyz, parts)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{link_name}] skip: compute failed: {exc!r}")
            continue
        if D is None:
            print(f"  [{link_name}] skip: bad geometry")
            continue

        link_start = link_match.start()
        link_body_start = link_start + len(link_open)
        block_start = link_body_start + inertial_m.start()
        block_end = link_body_start + inertial_m.end()
        # Snap the replacement to start at the beginning of the line
        # containing <inertial>, so existing leading whitespace is
        # consumed by the rewrite (otherwise the new block's own indent
        # accumulates on top of the pre-existing indent every run, and
        # the file grows kilobytes of whitespace after a few passes).
        # The replacement's own indent comes from the parent <link>'s
        # column plus 2 — a consistent, file-state-independent choice.
        line_start = text.rfind("\n", 0, block_start) + 1
        link_line_start = text.rfind("\n", 0, link_start) + 1
        link_indent = link_start - link_line_start
        indent = link_indent + 2
        block_start = line_start  # consume existing leading whitespace
        new_block = _build_inertial_block(indent, com_xyz.tolist(), rpy, mass, D)
        n_visuals = len(parts)
        edits.append((block_start, block_end, new_block, link_name, D, rpy, n_visuals))

    new_text = text
    for s, e, repl, name, D, rpy, n_v in sorted(edits, key=lambda x: -x[0]):
        new_text = new_text[:s] + repl + new_text[e:]
        nv_tag = f" [{n_v} visuals]" if n_v > 1 else ""
        print(
            f"  [{name}] ixx={_fmt(D[0])} iyy={_fmt(D[1])} izz={_fmt(D[2])} "
            f"rpy=({_fmt(rpy[0])}, {_fmt(rpy[1])}, {_fmt(rpy[2])}){nv_tag}"
        )

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        print(f"  → wrote {len(edits)} block(s)")
    else:
        print("  → no changes (already matches recompute)")


def main():
    for x in _XACROS:
        print(f"\n=== {x.name} ===")
        _process_file(x)
    return 0


if __name__ == "__main__":
    sys.exit(main())
