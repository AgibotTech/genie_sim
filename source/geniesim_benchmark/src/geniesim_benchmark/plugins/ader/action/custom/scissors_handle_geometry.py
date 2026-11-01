# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Asset-agnostic handle-hole geometry for scissors-like articulated tools.

The module answers ONE question: given the LOCAL mesh of a rigid link that
carries a finger-grip ring (a scissors half), where is the hole, and does a
finger segment pass through it?

Design contract (no per-asset calibration):

* The object's id comes from ``problems.json`` ("object"); everything else is
  derived from the composed USD at load time.
* Each rigid link of the articulation is treated as an approximately planar
  tool half.  The thin axis is recovered with a vertex PCA (verified thin
  ratios 0.02-0.11 across benchmark_scissors_001..006), the mesh triangles are
  projected along it and rasterised, and enclosed empty regions are extracted
  with a border flood fill.  This works on OPEN meshes too — the six shipped
  scissors have open boundaries (Euler chi != 2, boundary edges > 0), so
  watertight-only topology methods are not an option.
* Only the DOMINANT enclosed hole per link is kept (benchmark_scissors_002
  carries a small decorative hole ~20x smaller); a significance gate rejects
  ambiguous assets instead of guessing (fail-closed).

Pure-numpy geometry (importable without Isaac):
    fit_thin_axis / build_hole_regions / hole_contains_segment

pxr-dependent USD reading (lazy):
    read_scissors_geometry  — unique RevoluteJoint + body0/body1 + local meshes

RLinf note: ApiCoreShim (source/rlinf_geniesim/eval/api_core_shim.py) exposes
body poses/AABBs/joints only, NOT composed USD mesh topology.  The checkers
built on this module are benchmark-only for now.  Porting them to RLinf later
means carrying the PARSED per-link hole regions in SceneSchema and forwarding
them through ApiCoreShim — NOT duplicating this USD parsing on the RLinf side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:  # cv2 is already an implicit runtime dep of this package (4 import sites)
    import cv2
except Exception:  # pragma: no cover - hosts without cv2 cannot run the resolver
    cv2 = None

# --- tunables (dimensionless / metric-generic, shared by every asset) -------
_RASTER_PX = 512            # projection raster long-edge resolution
_BORDER_PX = 20             # raster border so the outside region is connected
_MIN_HOLE_AREA_PX = 200     # dust filter, ~0.1% of the raster at 512px
_DOMINANCE_RATIO = 0.25     # runner-up hole must be < 25% of the dominant one
_THIN_AXIS_MAX_RATIO = 0.35  # link must be plate-like else fail-closed
_DEFAULT_CLEARANCE = 0.004  # m inward from the hole boundary for the segment


@dataclass
class HoleRegion:
    """One finger hole, expressed in the owning rigid link's local frame."""

    plane_origin: np.ndarray          # (3,) local hole centroid
    plane_normal: np.ndarray          # (3,) local thin axis (unit)
    inplane_axes: np.ndarray          # (2, 3) orthonormal plane basis
    contour: np.ndarray               # (N, 2) hole boundary in plane coords [m]
    axial_half_span: float            # m, half thickness of the link along normal
    area: float                        # m^2, enclosed contour area

    def world_from_local(self, pose: np.ndarray) -> "HoleRegion":
        """Re-express this hole under a 4x4 world<-local ``pose``."""
        R = pose[:3, :3]
        return HoleRegion(
            plane_origin=(R @ self.plane_origin) + pose[:3, 3],
            plane_normal=R @ self.plane_normal,
            inplane_axes=self.inplane_axes @ R.T,  # keep (2,3) row layout
            contour=self.contour,
            axial_half_span=self.axial_half_span,
            area=self.area,
        )


@dataclass
class LinkGeometry:
    """Parsed local geometry of one rigid link of the articulation."""

    link_path: str                    # absolute prim path of the rigid body
    hole: HoleRegion | None = None    # dominant enclosed hole (None if absent)
    reject_reason: str = ""           # non-empty when no hole could be certified


@dataclass
class ScissorsGeometry:
    """Everything the checkers need, resolved once per episode and cached."""

    object_path: str                  # resolved prim path of the articulation
    joint_name: str                   # scene dof name of the unique revolute joint
    joint_lower: float                # rad, joint limit
    joint_upper: float                # rad, joint limit
    links: dict = field(default_factory=dict)  # link_path -> LinkGeometry

    @property
    def ok(self) -> bool:
        """Two parseable links, each with a certified dominant hole."""
        return len(self.links) == 2 and all(lg.hole is not None for lg in self.links.values())


# ---------------------------------------------------------------------------
# pure geometry (numpy only; unit-testable without Isaac)
# ---------------------------------------------------------------------------

def fit_thin_axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """PCA of ``vertices`` (N,3); returns (centroid, axes_rows(3,3), thin_ratio).

    ``axes_rows[2]`` is the least-variance direction (the tool's thin axis).
    ``thin_ratio`` = sqrt(smallest eigenvalue / largest eigenvalue).
    """
    pts = np.asarray(vertices, dtype=float)
    centroid = pts.mean(axis=0)
    _, sv, vt = np.linalg.svd(pts - centroid, full_matrices=False)
    var = (sv**2) / max(len(pts) - 1, 1)
    ratio = float(np.sqrt(max(var.min(), 0.0) / max(var.max(), 1e-12)))
    return centroid, vt, ratio


def _rasterize_projection(uv: np.ndarray, counts, indices) -> np.ndarray:
    """Fill every polygon face of the projected mesh into a uint8 mask."""
    mask = np.zeros((_RASTER_PX, _RASTER_PX), dtype=np.uint8)
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("cv2 is required to rasterise handle-hole projections")
    k = 0
    for n in counts:
        face = np.asarray(indices[k : k + n], dtype=np.int64)
        k += n
        if len(face) >= 3:
            cv2.fillConvexPoly(mask, uv[face], 255)
    return mask


def _interior_holes(labels, stats, cents, num):
    """Enclosed empty components as [(area_px, (cx, cy), label_id)]."""
    holes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        touches_border = x == 0 or y == 0 or x + w >= _RASTER_PX or y + h >= _RASTER_PX
        if touches_border or area < _MIN_HOLE_AREA_PX:
            continue
        holes.append((int(area), (float(cents[i][0]), float(cents[i][1])), int(i)))
    holes.sort(key=lambda t: t[0], reverse=True)
    return holes


def build_hole_regions(
    vertices: np.ndarray,
    face_vertex_counts,
    face_vertex_indices,
    link_path: str = "",
) -> LinkGeometry:
    """Extract the DOMINANT enclosed hole of one planar-ish link mesh.

    Fail-closed: a link that is not plate-like, has no enclosed hole, or whose
    runner-up hole is not clearly smaller yields ``hole=None`` with a reason.
    """
    pts = np.asarray(vertices, dtype=float)
    if len(pts) < 8:
        return LinkGeometry(link_path=link_path, reject_reason="too_few_vertices")

    centroid, axes, thin_ratio = fit_thin_axis(pts)
    if thin_ratio > _THIN_AXIS_MAX_RATIO:
        return LinkGeometry(
            link_path=link_path,
            reject_reason=f"not_plate_like(thin_ratio={thin_ratio:.3f})",
        )

    normal = axes[2]
    uv = (pts - centroid) @ axes[:2].T
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    span = hi - lo
    if min(span) <= 0:
        return LinkGeometry(link_path=link_path, reject_reason="degenerate_projection")
    scale = (_RASTER_PX - 2 * _BORDER_PX) / float(span.max())
    pix = np.rint((uv - lo) * scale + _BORDER_PX).astype(np.int64)
    pix[:, 0] = np.clip(pix[:, 0], 0, _RASTER_PX - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, _RASTER_PX - 1)

    mask = _rasterize_projection(pix, face_vertex_counts, face_vertex_indices)
    if cv2 is not None:
        # Close sub-pixel raster seams between abutting faces (blade neck vs
        # ring wall): without this, a 1-px crack splits the true hole into two
        # equal components and the dominance gate rejects a valid tool.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    empty = (mask == 0).astype(np.uint8)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(empty, connectivity=8)

    holes = _interior_holes(labels, stats, cents, num)
    if not holes:
        return LinkGeometry(link_path=link_path, reject_reason="no_enclosed_hole")
    if len(holes) >= 2 and holes[1][0] > _DOMINANCE_RATIO * holes[0][0]:
        return LinkGeometry(
            link_path=link_path,
            reject_reason=f"ambiguous_holes({holes[0][0]}px_vs_{holes[1][0]}px)",
        )

    best_area, _, best_label = holes[0]

    # Refined contour of the accepted component -> polygon in plane coords [m].
    sel = (labels == best_label).astype(np.uint8)
    contours, _ = cv2.findContours(sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return LinkGeometry(link_path=link_path, reject_reason="no_contour")
    contour_px = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    contour = ((contour_px - _BORDER_PX) / scale + lo).astype(np.float32)
    area_m2 = float(cv2.contourArea(contour))  # plane coords are metres

    # Axial half span of the SOLID region along the thin axis (all mesh
    # vertices belong to the solid; rasterisation only affects plane coords).
    axial = (pts - centroid) @ normal
    axial_half = float(max(abs(axial.min()), abs(axial.max())))

    # Hole-centric frame: the plane origin AND the contour are both expressed
    # relative to the hole centroid.  ``hole_contains_segment`` projects world
    # points against the origin, so leaving the contour in the old
    # PCA-centroid frame made even a segment through the exact hole centre
    # fail the point-in-polygon test on every real asset (verified: 2/12
    # links passed before, 12/12 after).
    c2 = contour.mean(axis=0)
    origin = centroid + axes[0] * c2[0] + axes[1] * c2[1]
    contour = (contour - c2).astype(np.float32)
    hole = HoleRegion(
        plane_origin=origin,
        plane_normal=normal,
        inplane_axes=axes[:2].copy(),
        contour=contour,
        axial_half_span=axial_half,
        area=area_m2,
    )
    return LinkGeometry(link_path=link_path, hole=hole)


def hole_contains_segment(
    hole: HoleRegion,
    p0_world: np.ndarray,
    p1_world: np.ndarray,
    clearance: float = _DEFAULT_CLEARANCE,
    deep_insert_extra: float = 0.01,
) -> bool:
    """True when the segment p0->p1 passes through the hole's opening.

    Two accepted configurations (either proves insertion):
      1. CROSS: the segment straddles the hole plane and the intersection
         point lands inside the contour shrunk by ``clearance``.
      2. DEEP: p1 (the fingertip end) sits within the hole's axial slab
         (link half thickness + ``deep_insert_extra``) and its projection is
         inside the shrunk contour — a finger pushed fully through so both
         joints are on the far side still counts.
    A finger hovering beside or above the ring projects inside the contour
    but fails the axial tests, so it is rejected.
    """
    if len(hole.contour) < 3:
        return False
    o, n = hole.plane_origin, hole.plane_normal
    p0 = np.asarray(p0_world, dtype=float)
    p1 = np.asarray(p1_world, dtype=float)
    d0 = float(np.dot(p0 - o, n))
    d1 = float(np.dot(p1 - o, n))
    if abs(d0) < 1e-9 and abs(d1) < 1e-9:
        return False  # segment lies in the plane — no through insertion

    def _inside(pt_world: np.ndarray) -> bool:
        rel = pt_world - o
        q = (float(np.dot(rel, hole.inplane_axes[0])),
             float(np.dot(rel, hole.inplane_axes[1])))
        c = hole.contour
        shrink = 1.0 - 2.0 * clearance / max(np.abs(c).max(), 1e-6)
        shrunk = (c - c.mean(axis=0)) * shrink + c.mean(axis=0)
        return bool(cv2.pointPolygonTest(
            shrunk.astype(np.float32), q, False) >= 0)

    if d0 * d1 < 0.0:  # straddles the plane
        t = d0 / (d0 - d1)
        hit = p0 + t * (p1 - p0)
        return _inside(hit)

    # DEEP insertion: fingertip inside the axial slab and inside the ring.
    slab = hole.axial_half_span + deep_insert_extra
    return abs(d1) <= slab and _inside(p1)


# ---------------------------------------------------------------------------
# pxr-dependent USD reading (lazy; benchmark APICore only)
# ---------------------------------------------------------------------------

def read_scissors_geometry(stage, object_path: str) -> ScissorsGeometry:
    """Parse the composed USD once: unique RevoluteJoint + body meshes.

    ``stage`` is a live Usd.Stage (Isaac kit).  Mesh vertices are transformed
    into each rigid link's LOCAL frame so the cached HoleRegion follows the
    link with a single world-pose read per tick.
    """
    from pxr import Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    root = stage.GetPrimAtPath(object_path)
    if not root or not root.IsValid():
        raise ValueError(f"object prim not found: {object_path}")

    joints = [p for p in Usd.PrimRange(root) if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(joints) != 1:
        raise ValueError(
            f"expected exactly one RevoluteJoint under {object_path}, found {len(joints)}")
    joint = UsdPhysics.RevoluteJoint(joints[0])
    body0 = [str(t) for t in joint.GetBody0Rel().GetTargets()]
    body1 = [str(t) for t in joint.GetBody1Rel().GetTargets()]
    if not body0 or not body1:
        raise ValueError(f"joint {joints[0].GetPath()} lacks body0/body1")
    lo = joint.GetLowerLimitAttr().Get()
    up = joint.GetUpperLimitAttr().Get()
    if lo is None or up is None:
        lo, up = 0.0, 0.0

    # joint dof name as the runtime reports it: trailing prim name
    joint_name = str(joints[0].GetPath().name)

    geom = ScissorsGeometry(
        object_path=object_path,
        joint_name=joint_name,
        joint_lower=float(lo),
        joint_upper=float(up),
    )
    for body_path in (body0[0], body1[0]):
        link = stage.GetPrimAtPath(body_path)
        if not link or not link.IsValid():
            geom.links[body_path] = LinkGeometry(body_path, reject_reason="missing_body_prim")
            continue
        verts, counts, inds = [], [], []
        link_xf_inv = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()).GetInverse()
        for prim in Usd.PrimRange(link):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            if not pts:
                continue
            xf = link_xf_inv * UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            base = len(verts)
            verts.extend(xf.Transform(p) for p in pts)
            counts.extend(mesh.GetFaceVertexCountsAttr().Get() or [])
            inds.extend(int(v) + base
                        for v in (mesh.GetFaceVertexIndicesAttr().Get() or []))
        vertices = np.array(verts, dtype=float).reshape(-1, 3) if verts else np.zeros((0, 3))
        geom.links[body_path] = build_hole_regions(
            vertices, counts, inds, link_path=body_path)
    return geom


def resolve_handle_pose(api_core, prim_path: str) -> np.ndarray:
    """World pose matrix of a rigid link via the checker dialect of api_core."""
    pose = np.eye(4)
    pos, quat = api_core.get_obj_world_pose(prim_path)
    pose[:3, 3] = pos
    pose[:3, :3] = _quat_wxyz_to_matrix(np.asarray(quat, dtype=float))
    return pose


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
