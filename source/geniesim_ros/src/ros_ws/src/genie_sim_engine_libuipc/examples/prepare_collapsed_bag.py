# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Author a slack bag with an open mouth and unchanged elastic reference geometry.

CPU asset preparation only. Native startup/reset qualification is separate.
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom, Gf, Vt, Sdf
from scipy.optimize import brentq
from scipy.integrate import cumulative_trapezoid
from scipy.special import ellipe

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source_usd", type=Path)
parser.add_argument("output_usd", type=Path)
parser.add_argument("--report", type=Path)
args = parser.parse_args()
source = args.source_usd.resolve()
output = args.output_usd.resolve()
if output.exists():
    raise FileExistsError(output)
s = Usd.Stage.Open(str(source))
shell = UsdGeom.Mesh(s.GetPrimAtPath("/Object/shell"))
p = np.asarray(shell.GetPointsAttr().Get(), dtype=float)
faces = np.asarray(shell.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
center = 0.12688859
half_height = 0.020
floor = 0.008


# A constant-arclength stadium folds the circular side walls into two layers.
# Its positive rounding radius keeps the rim and volumetric cords distinct.
def _fold_flat(points):
    x, y, z = points.T
    r = np.hypot(y, z - center)
    ref = np.interp(x, [0.0685, 0.08016667, 0.09183333, 0.1035], [0.12, 0.10444444, 0.07555556, 0.06])
    L = np.pi * (ref - half_height)
    u = np.mod(np.arctan2(z - center, y), 2 * np.pi) * ref
    Y = np.empty(len(x))
    Z = np.empty(len(x))
    ny = np.empty(len(x))
    nz = np.empty(len(x))
    cut = np.pi * half_height / 2
    masks = [
        u < cut,
        (u >= cut) & (u < cut + L),
        (u >= cut + L) & (u < 3 * cut + L),
        (u >= 3 * cut + L) & (u < 3 * cut + 2 * L),
        u >= 3 * cut + 2 * L,
    ]
    for i, m in enumerate(masks):
        if i == 0:
            a = u[m] / half_height
            cy = L[m] / 2
        elif i == 1:
            Y[m] = L[m] / 2 - (u[m] - cut)
            Z[m] = half_height
            ny[m] = 0
            nz[m] = 1
            continue
        elif i == 2:
            a = np.pi / 2 + (u[m] - (cut + L[m])) / half_height
            cy = -L[m] / 2
        elif i == 3:
            Y[m] = -L[m] / 2 + (u[m] - (3 * cut + L[m]))
            Z[m] = -half_height
            ny[m] = 0
            nz[m] = -1
            continue
        else:
            a = 3 * np.pi / 2 + (u[m] - (3 * cut + 2 * L[m])) / half_height
            cy = L[m] / 2
        Y[m] = cy + half_height * np.cos(a)
        Z[m] = half_height * np.sin(a)
        ny[m] = np.cos(a)
        nz[m] = np.sin(a)
    # Offset from the circular wall is carried along its folded normal.
    d = r - ref
    Y += d * ny
    Z += d * nz
    result = np.c_[x, Y, Z + half_height + floor]
    # The one closed-end fan centre folds outward; never collapse a cap to zero area.
    result[r < 1e-5] = [0.1035, 0, half_height + floor]
    return result


# The open material mouth keeps its circumference but presents an oval aperture
# toward the head camera. Source dimensions are pinned by the retained bag mesh.
mouth_half_height = 0.060
mouth_half_width = brentq(
    lambda width: 4 * width * ellipe(1 - (mouth_half_height / width) ** 2) - 2 * np.pi * 0.12,
    mouth_half_height,
    0.25,
)
ellipse_angles = np.linspace(0, 2 * np.pi, 8193)
ellipse_lengths = cumulative_trapezoid(
    np.hypot(mouth_half_width * np.sin(ellipse_angles), mouth_half_height * np.cos(ellipse_angles)),
    ellipse_angles,
    initial=0,
)


def fold(points):
    """Keep the body slack while opening and gently tilting the material mouth."""
    result = _fold_flat(points)
    x, y, z = points.T
    weight = np.clip((0.045 - x) / 0.125, 0.0, 1.0)
    weight = weight * weight * (3 - 2 * weight)
    arc = np.mod(np.arctan2(z - center, y), 2 * np.pi) * 0.12
    angle = np.interp(arc, ellipse_lengths, ellipse_angles)
    normal = np.c_[np.cos(angle) / mouth_half_width, np.sin(angle) / mouth_half_height]
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    offset = np.hypot(y, z - center) - 0.12
    oval_y = mouth_half_width * np.cos(angle) + offset * normal[:, 0]
    oval_z = mouth_half_height * (1 + np.sin(angle)) + floor + offset * normal[:, 1]
    result[:, 1] += weight * (oval_y - result[:, 1])
    result[:, 2] += weight * (oval_z - result[:, 2])
    section_center = (1 - weight) * (half_height + floor) + weight * (mouth_half_height + floor)
    result[:, 0] += 0.65 * weight * (result[:, 2] - section_center)
    # A smooth, shared displacement wrinkles both layers without random vertex
    # noise or coincident lips. The same map transports the casing and cords.
    amplitude = 0.005 * (1 - 0.55 * weight)
    result[:, 2] += amplitude * (1 + 0.65 * np.sin(70 * x + 18 * y) + 0.35 * np.sin(47 * x - 40 * y + 1))
    return result


def area(v):
    return np.linalg.norm(np.cross(v[faces[:, 1]] - v[faces[:, 0]], v[faces[:, 2]] - v[faces[:, 0]]), axis=1) / 2


folded = fold(p)
cap = np.flatnonzero(np.hypot(p[:, 1], p[:, 2] - center) < 1e-5)
assert len(cap) == 1
capfaces = np.any(faces == cap[0], axis=1)
oldarea = area(p)


def cap_area(offset):
    folded[cap[0], 0] = 0.1035 + offset
    return area(folded)[capfaces].sum()


offset = brentq(lambda v: cap_area(v) - oldarea[capfaces].sum(), 0, 0.15)
cap_area(offset)
stage = Usd.Stage.CreateNew(str(output))
root = stage.DefinePrim("/Object", "Xform")
root.GetReferences().AddReference(os.path.relpath(source, output.parent), "/Object")
stage.SetDefaultPrim(root)
report = {
    "method": "slack folded body, tilted open oval mouth, smooth shared wrinkles",
    "mouth_half_height_m": mouth_half_height,
    "mouth_half_width_m": float(mouth_half_width),
    "half_height_m": half_height,
    "cap_unfold_m": float(offset),
    "meshes": {},
}
for prim in s.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    m = UsdGeom.Mesh(prim)
    v = np.asarray(m.GetPointsAttr().Get(), dtype=float)
    t = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    vobj = np.c_[v, np.ones(len(v))] @ t[:, :3]
    new = fold(vobj)
    if prim.GetName().startswith("cord_"):
        # Each retained tube section consists of its centre and six perimeter points.
        sections = vobj.reshape(-1, 7, 3)
        centres = sections[:, 0]
        target = fold(centres)
        eps = 1e-6
        jac = np.stack(
            [(fold(centres + np.eye(3)[i] * eps) - fold(centres - np.eye(3)[i] * eps)) / (2 * eps) for i in range(3)],
            axis=2,
        )
        u, sv, vt = np.linalg.svd(jac)
        rot = u @ vt
        assert np.all(np.linalg.det(rot) > 0)
        new = (np.einsum("nij,nkj->nki", rot, sections - centres[:, None]) + target[:, None]).reshape(-1, 3)
    if prim.GetName() == "shell":
        new = folded
    newlocal = np.c_[new, np.ones(len(new))] @ np.linalg.inv(t)[:, :3]
    o = UsdGeom.Mesh(stage.OverridePrim(prim.GetPath()))
    o.GetPrim().CreateAttribute("geniesim:initialPoints", Sdf.ValueTypeNames.Point3fArray).Set(
        Vt.Vec3fArray.FromNumpy(newlocal.astype(np.float32))
    )
    o.GetExtentAttr().Set([Gf.Vec3f(*newlocal.min(0)), Gf.Vec3f(*newlocal.max(0))])
    row = {"count": len(v), "min": new.min(0).tolist(), "max": new.max(0).tolist()}
    tet = prim.GetAttribute("tetVertexIndices").Get()
    if tet is not None:
        tet = np.asarray(tet)
        oldvol = np.linalg.det(vobj[tet[:, 1:]] - vobj[tet[:, 0, None]]) / 6
        newvol = np.linalg.det(new[tet[:, 1:]] - new[tet[:, 0, None]]) / 6
        if not np.isfinite(newvol).all() or np.any(newvol <= 0):
            raise ValueError(f"{prim.GetName()} initial configuration has invalid tetrahedra")
        row.update(
            tet_min_volume=float(newvol.min()),
            tet_volume_ratio=[float((newvol / oldvol).min()), float((newvol / oldvol).max())],
            total_volume_ratio=float(newvol.sum() / oldvol.sum()),
        )
    if prim.GetName() == "shell":
        row.update(
            area_ratio=float(area(new).sum() / oldarea.sum()),
            min_triangle_area=float(area(new).min()),
            per_triangle_area_ratio=[float((area(new) / oldarea).min()), float((area(new) / oldarea).max())],
        )
    report["meshes"][prim.GetName()] = row
stage.GetRootLayer().Save()
if args.report is not None:
    args.report.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
