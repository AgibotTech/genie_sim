#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit GPU acceptance probe for native libuipc.

Run this file directly in the audited CPython 3.12/native/GPU environment.
It is deliberately not named test_*.py and is not part of the CPU pytest suite.
The probe imports neither sibling engine backend and does not require ROS.
Cloth, rigid ABD and mixed cases exercise World.init/advance/retrieve/recover(0),
finite state and gravity motion. ABD affine residuals are reported for readback
analysis. Passing does not certify full task contacts or renderer synchronization.

Use --help without importing uipc or initializing a GPU. Artifact identity,
installation and platform requirements are documented in the package README.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

DT = 1.0 / 60.0
GRAVITY = (0.0, 0.0, -9.81)
STEPS = 30


def _load_uipc():
    try:
        import uipc
    except ImportError as exc:  # pragma: no cover - environment failure
        raise SystemExit(
            "could not import pyuipc.\n"
            "Run `geniesim bootstrap --libuipc-only` in the evaluation environment.\n"
            "using Linux x86_64 CPython 3.12, then inspect the chained "
            "native-library error below.\n"
            f"original error: {exc!r}"
        ) from exc
    return uipc


def _grid_cloth(uipc, objs, shell, bending, *, n=3, cell=0.1, z=1.0):
    """Add a square cloth grid as a shell-FEM object. Returns (vertex_count, mean_z).

    ``shell``/``bending`` must be the same instances registered in the scene's
    constitution tabular (solver_ipc.py applies the registered objects,
    not fresh ones).
    """
    verts = np.array(
        [[i * cell, j * cell, z] for i in range(n) for j in range(n)],
        dtype=np.float64,
    )
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            tris.append([a, a + n, a + 1])
            tris.append([a + 1, a + n, a + n + 1])
    tris = np.asarray(tris, dtype=np.int32)

    mesh = uipc.geometry.trimesh(verts, tris)
    uipc.geometry.label_surface(mesh)
    # E=1, nu=1 as at solver_ipc.py. The magnitude is irrelevant here:
    # this test only needs the shell path to exist and integrate.
    lame = uipc.constitution.ElasticModuli2D.lame(1.0, 1.0)
    shell.apply_to(mesh, moduli=lame, mass_density=1.0, thickness=0.01)
    bending.apply_to(mesh, bending_stiffness=1.0)
    uipc.view(mesh.meta().find("self_collision"))[0] = 0

    objs.create("cloth").geometries().create(mesh)
    return len(verts), float(verts[:, 2].mean())


# solver_ipc.py default ABD stiffness; :1595 pairs it with 1000 kg/m^3.
ABD_KAPPA = 1.0e8
ABD_MASS_DENSITY = 1000.0


def _box_body(uipc, objs, abd, *, h=0.05, x=0.5, z=1.0):
    """Add a free box as an ABD object. Returns its initial z."""
    verts = np.array(
        [[sx * h, sy * h, sz * h] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int32,
    )
    mesh = uipc.geometry.trimesh(verts, faces)
    uipc.geometry.label_surface(mesh)
    # kappa is positional: AffineBodyConstitution.apply_to(sc, kappa, mass_density=...)
    abd.apply_to(mesh, ABD_KAPPA, mass_density=ABD_MASS_DENSITY)

    xform = np.eye(4)
    xform[:3, 3] = [x, 0.0, z]
    uipc.view(mesh.transforms())[0] = xform

    objs.create("body_box").geometries().create(mesh)
    return z


def _run_case(uipc, *, name, with_cloth, with_body):
    """init -> advance/retrieve xN -> recover(0) -> retrieve, asserting each stage."""
    workspace = Path(tempfile.mkdtemp(prefix=f"p0_{name}_"))

    cfg = uipc.core.Scene.default_config()
    cfg["dt"] = DT
    cfg["gravity"] = [[g] for g in GRAVITY]

    scene = uipc.core.Scene(cfg)
    # Keep the registered instances: apply_to must be called on the same objects
    # inserted into the tabular (solver_ipc.py), not on fresh ones.
    abd = uipc.constitution.AffineBodyConstitution()
    shell = uipc.constitution.NeoHookeanShell()
    bending = uipc.constitution.DiscreteShellBending()
    tabular = scene.constitution_tabular()
    tabular.insert(abd)
    tabular.insert(shell)
    tabular.insert(bending)

    objs = scene.objects()
    # Ground well below both objects: libuipc requires strictly positive initial
    # clearance at construction (solver_ipc.py), or world.init() silently
    # leaves an invalid world.
    objs.create("ground").geometries().create(uipc.geometry.ground(-1.0, np.array([0.0, 0.0, 1.0])))

    cloth_n = cloth_z0 = body_z0 = None
    if with_cloth:
        cloth_n, cloth_z0 = _grid_cloth(uipc, objs, shell, bending)
    if with_body:
        body_z0 = _box_body(uipc, objs, abd)

    engine = uipc.core.Engine("cuda", str(workspace))
    world = uipc.core.World(engine)
    world.init(scene)
    world.dump()

    features = world.features()

    abd_feature = features.find(uipc.core.AffineBodyStateAccessorFeature)
    fem_feature = features.find(uipc.core.FiniteElementStateAccessorFeature)

    abd_geom = fem_geom = None
    if with_body:
        if abd_feature is None:
            raise AssertionError(f"[{name}] world has no affine-body state accessor")
        abd_geom = abd_feature.create_geometry()
        abd_geom.instances().create(uipc.builtin.transform, np.eye(4, dtype=np.float64))
        abd_geom.instances().create(uipc.builtin.velocity, np.zeros((4, 4), dtype=np.float64))
    if with_cloth:
        if fem_feature is None:
            raise AssertionError(f"[{name}] world has no FEM state accessor")
        fem_geom = fem_feature.create_geometry()
        fem_geom.vertices().create(uipc.builtin.position, np.zeros(3, dtype=np.float64))
        fem_geom.vertices().create(uipc.builtin.velocity, np.zeros(3, dtype=np.float64))
        reported = fem_feature.vertex_count()
        if reported != cloth_n:
            raise AssertionError(f"[{name}] FEM accessor reports {reported} vertices, expected {cloth_n}")

    def read_fem():
        fem_feature.copy_to(fem_geom)
        return fem_geom.vertices().find(uipc.builtin.position).view().reshape(-1, 3).astype(np.float64)

    def read_abd():
        abd_feature.copy_to(abd_geom)
        return np.asarray(abd_geom.instances().find(uipc.builtin.transform).view()).reshape(-1, 4, 4).astype(np.float64)

    report = {"case": name, "steps": STEPS}

    for i in range(STEPS):
        world.advance()
        world.retrieve()
        if with_cloth:
            pos = read_fem()
            if not np.isfinite(pos).all():
                raise AssertionError(f"[{name}] non-finite cloth state after step {i + 1}")
        if with_body:
            if not np.isfinite(read_abd()).all():
                raise AssertionError(f"[{name}] non-finite body state after step {i + 1}")

    if with_cloth:
        pos = read_fem()
        z_end = float(pos[:, 2].mean())
        report.update(cloth_z0=cloth_z0, cloth_z_end=z_end, cloth_vertices=len(pos))
        if not z_end < cloth_z0:
            raise AssertionError(f"[{name}] cloth did not fall: z {cloth_z0:.4f} -> {z_end:.4f}")

    if with_body:
        xf = read_abd()
        z_end = float(xf[0][2, 3])
        report.update(body_z0=body_z0, body_z_end=z_end)
        if not z_end < body_z0:
            raise AssertionError(f"[{name}] body did not fall: z {body_z0:.4f} -> {z_end:.4f}")
        # ABD readback is affine, not (pos, quat). How far the linear block is
        # from a rotation is exactly the residual the rigid readback must account for.
        rot = xf[0][:3, :3]
        report["body_orthogonality_residual"] = float(np.abs(rot.T @ rot - np.eye(3)).max())
        report["body_scale_residual"] = float(np.abs(np.linalg.det(rot) ** (1.0 / 3.0) - 1.0))

    # recover(0) restores the native construction snapshot used by the backend reset.
    world.recover(0)
    world.retrieve()
    if with_cloth:
        pos_r = read_fem()
        report["cloth_z_after_recover"] = float(pos_r[:, 2].mean())
        if not np.isfinite(pos_r).all():
            raise AssertionError(f"[{name}] non-finite cloth state after recover(0)")
        if not np.isclose(pos_r[:, 2].mean(), cloth_z0, atol=1e-3):
            raise AssertionError(
                f"[{name}] recover(0) did not restore cloth height: " f"{pos_r[:, 2].mean():.4f} vs {cloth_z0:.4f}"
            )
    if with_body:
        z_r = float(read_abd()[0][2, 3])
        report["body_z_after_recover"] = z_r
        if not np.isclose(z_r, body_z0, atol=1e-3):
            raise AssertionError(f"[{name}] recover(0) did not restore body height: {z_r:.4f} vs {body_z0:.4f}")

    return report


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    uipc = _load_uipc()
    print(f"pyuipc: {Path(uipc.__file__).resolve()}")
    print(f"dt={DT}  gravity={GRAVITY}  steps={STEPS}")

    for name, cloth, body in (
        ("cloth", True, False),
        ("rigid", False, True),
        ("mixed", True, True),
    ):
        print(f"\n=== case: {name} ===")
        for key, value in _run_case(uipc, name=name, with_cloth=cloth, with_body=body).items():
            print(f"  {key}: {value}")

    print("\nNative libuipc smoke test PASSED")
    print("  World.init / advance / retrieve / recover(0) all executed on this GPU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
