# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Compile the isolated CPU robot/contact model before IPC filters the builder."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np


def validate_dynamic_proxies(objects, config):
    """Resolve the ordered, explicit rigid-body selection shared by scene/export."""
    if config is None:
        return ()
    if not isinstance(config, dict):
        raise ValueError("libuipc.cpu_mujoco must be a mapping")
    names = config.get("dynamic_proxies", ())
    if (
        not isinstance(names, (list, tuple))
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("cpu_mujoco.dynamic_proxies must contain unique nonempty object names")
    if names and config.get("synchronization", "latest") != "barrier":
        raise ValueError("cpu_mujoco.dynamic_proxies requires synchronization: barrier")
    selected = []
    for name in names:
        matches = [obj for obj in objects if obj["name"] == name]
        if len(matches) != 1 or matches[0]["kind"] != "rigid_mesh":
            raise ValueError(f"cpu_mujoco.dynamic_proxies must select declared rigid_mesh objects: {name}")
        selected.append(matches[0])
    return tuple(selected)


def _convex_components(mesh, cpu_collision=None):
    """Split closed surfaces, decomposing concavity only with explicit CPU opt-in."""
    points = np.asarray(mesh.world_points(), dtype=np.float64)
    # USD commonly duplicates vertices at normal/UV seams. Weld exact positions
    # before connectivity analysis so individual faces do not become hulls.
    vertices, inverse = np.unique(points, axis=0, return_inverse=True)
    triangles = inverse[np.asarray(mesh.triangles, dtype=np.int64)]
    parent = np.arange(len(vertices))

    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for triangle in triangles:
        first = root(triangle[0])
        for vertex in triangle[1:]:
            parent[root(vertex)] = first
    groups = {}
    for triangle in triangles:
        groups.setdefault(root(triangle[0]), []).append(triangle)
    for component, faces in enumerate(groups.values()):
        indices, local_faces = np.unique(faces, return_inverse=True)
        local_points = vertices[indices]
        local_faces = local_faces.reshape(-1, 3)
        edges = np.sort(
            np.concatenate((local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]])), axis=1
        )
        _, edge_counts = np.unique(edges, axis=0, return_counts=True)
        if np.any(edge_counts != 2):
            raise ValueError(f"{mesh.path} component {component} must be a closed manifold CPU collider")
        extent = float(np.max(np.ptp(local_points, axis=0)))
        tolerance = max(extent * 1e-6, 1e-9)
        if len(local_points) < 4 or np.linalg.matrix_rank(local_points - local_points[0], tol=tolerance) < 3:
            raise ValueError(
                f"{mesh.path} component {component} is planar; CPU MuJoCo requires a solid convex collider"
            )
        for face in local_faces:
            a, b, c = local_points[face]
            normal = np.cross(b - a, c - a)
            length = np.linalg.norm(normal)
            if length <= tolerance * tolerance:
                continue
            distance = (local_points - a) @ (normal / length)
            if np.min(distance) < -tolerance and np.max(distance) > tolerance:
                if not cpu_collision or cpu_collision.get("method") != "coacd":
                    raise ValueError(
                        f"{mesh.path} component {component} is concave; author separate convex CPU collision "
                        "components or opt in to cpu_collision.method: coacd"
                    )
                from ..mesh import _run_coacd_isolated

                # CoACD can crash in native code. Reuse startup isolation, but
                # never its robot-pad caller's single-hull fallback.
                try:
                    parts = _run_coacd_isolated(
                        local_points, local_faces, threshold=cpu_collision["threshold"], max_convex_hull=-1
                    )
                    if len(parts) < 2:
                        raise ValueError("CoACD returned fewer than two parts for a concave component")
                    # CoACD returns convex parts. MuJoCo constructs each part's
                    # hull, tolerating roundoff on nearly coplanar output faces.
                    for part_points, part_faces in parts:
                        yield np.asarray(part_points), np.asarray(part_faces)
                except Exception as exc:
                    raise RuntimeError(f"{mesh.path} component {component} CPU CoACD failed: {exc}") from exc
                break
        else:
            yield local_points, local_faces


def build_cpu_model(builder, objects, meshes, scene_cfg, robot_prefix, *, source_stage=None) -> bytes:
    """Return a CPU MuJoCo MJB without changing the IPC builder.

    Call before task geometry is appended and before the hand-only IPC filter.
    The copied robot retains its masses, inertias, joints, base transforms and
    complete collision geometry. Scalar robot joints get position actuators;
    the CPU worker supplies their configured drive gains and initial positions.

    Reject imported nonrobot bodies and omit all imported world shapes; only
    static meshes, boxes and explicitly selected rigid proxies enter this model. MuJoCo
    collides mesh convex hulls, so disconnected static components are exported
    separately. Closed concave components require an explicit per-object
    ``cpu_collision`` CoACD request; open, nonmanifold or planar components
    are rejected. Decomposition failure aborts compilation without a single-hull
    fallback. Robot meshes retain Newton's standard
    MuJoCo convex collision semantics. Visual-only shapes and sites are omitted.

    Args:
        builder: Imported Newton robot builder, before IPC scene edits.
        objects: Validated ``libuipc.objects`` entries.
        meshes: Source prim paths mapped to captured IPCMesh instances.
        scene_cfg: Scene YAML containing ``libuipc.cpu_mujoco`` options.
        robot_prefix: Explicit robot namespace used to select scalar drives.
        source_stage: Composed USD stage supplying explicit collision exclusions.

    Returns:
        MuJoCo binary model bytes, loadable through an in-memory assets map.
    """
    import mujoco
    import newton
    import warp as wp
    from newton.solvers import SolverMuJoCo

    prefix = f"/{robot_prefix.strip('/')}/"
    foreign_bodies = [str(label) for label in builder.body_label if not str(label).startswith(prefix)]
    if foreign_bodies:
        raise ValueError(f"CPU MuJoCo requires a robot-only builder; nonrobot bodies: {', '.join(foreign_bodies)}")
    cpu_builder = copy.deepcopy(builder)
    config = (scene_cfg.get("libuipc") or {}).get("cpu_mujoco")
    proxies = validate_dynamic_proxies(objects, config)
    config = {} if config is None else config
    for obj in proxies:
        if obj["name"] == "world" or obj["name"] in cpu_builder.body_label:
            raise ValueError(f"CPU dynamic proxy body name conflicts with robot: {obj['name']}")
    if config.get("robot_self_collision", False):
        from .collision import enable_robot_self_collision

        enable_robot_self_collision(cpu_builder, source_stage, config.get("robot_self_collision_excludes", ()))
    # The main builder adds an implicit ground plane. CPU static collision is
    # defined solely by the explicit YAML objects below.
    for shape, body in enumerate(cpu_builder.shape_body):
        if body < 0:
            cpu_builder.shape_flags[shape] = 0
    joint_names = set()
    for joint, kind in enumerate(cpu_builder.joint_type):
        if kind not in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
            continue
        child = cpu_builder.joint_child[joint]
        if not str(cpu_builder.body_label[child]).startswith(prefix):
            continue
        # Newton sanitizes slashes during export; use the controller's leaf
        # identity explicitly instead of trying to reverse that sanitization.
        name = str(cpu_builder.joint_label[joint]).rsplit("/", 1)[-1]
        if name in joint_names:
            raise ValueError(f"Duplicate CPU robot joint leaf name: {name}")
        joint_names.add(name)
        cpu_builder.joint_label[joint] = name
        dof = cpu_builder.joint_qd_start[joint]
        cpu_builder.joint_target_mode[dof] = newton.JointTargetMode.POSITION
        # The worker overwrites these gains from the scene's drives before any
        # step; positive placeholders ensure Newton emits an actuator.
        cpu_builder.joint_target_ke[dof] = max(cpu_builder.joint_target_ke[dof], 1.0)
        cpu_builder.joint_target_kd[dof] = max(cpu_builder.joint_target_kd[dof], 1.0)

    static_margins = {}
    for obj in objects:
        margin = obj.get("cpu_collision", {}).get("margin", config.get("static_collision_margin", 0.0))
        if obj["kind"] == "box":
            pose = obj.get("pose", [0, 0, 0, 0, 0, 0, 1])
            shape = cpu_builder.add_shape_box(
                body=-1,
                xform=wp.transform(pose[:3], pose[3:]),
                hx=obj["half_extents"][0],
                hy=obj["half_extents"][1],
                hz=obj["half_extents"][2],
                label=obj["name"],
            )
            static_margins[shape] = margin
        elif obj["kind"] == "static_mesh":
            components = _convex_components(meshes[obj["mesh_prim_path"]], obj.get("cpu_collision"))
            for index, (vertices, faces) in enumerate(components):
                with wp.ScopedDevice("cpu"):
                    shape = cpu_builder.add_shape_mesh(
                        body=-1,
                        mesh=newton.Mesh(vertices, faces.flatten()),
                        label=f"{obj['name']}_component_{index}",
                    )
                    static_margins[shape] = margin

    proxy_bodies = []
    for obj in proxies:
        name = obj["name"]
        source = meshes[obj["mesh_prim_path"]]
        pos, rot, scales = source.decompose_axis_scaled()
        local_mesh = replace(source, points=source.points * scales, xform=np.eye(4))
        with wp.ScopedDevice("cpu"):
            rigid_mesh = newton.Mesh(local_mesh.points, local_mesh.triangles.flatten())
        if not rigid_mesh.is_watertight or not math.isfinite(rigid_mesh.mass) or rigid_mesh.mass <= 0:
            raise ValueError(f"{source.path} must be watertight with positive volume for a CPU dynamic proxy")
        density = obj["mass"] / rigid_mesh.mass if "mass" in obj else obj["density"]
        if not math.isfinite(density) or density <= 0:
            raise ValueError(f"{source.path} CPU dynamic proxy mass must be positive and finite")
        mass = float(rigid_mesh.mass * density)
        com = np.asarray(rigid_mesh.com, dtype=np.float64)
        inertia = np.asarray(rigid_mesh.inertia, dtype=np.float64).reshape(3, 3) * density
        if (
            not math.isfinite(mass)
            or mass <= 0
            or not np.isfinite(com).all()
            or not np.isfinite(inertia).all()
            or np.any(np.linalg.eigvalsh(inertia) <= 0)
        ):
            raise ValueError(f"{source.path} CPU dynamic proxy mass, COM and inertia must be finite and physical")
        # Derive inertia from the same scaled full source as IPC. Collision
        # decomposition can overlap and must neither add mass nor move the COM.
        body = cpu_builder.add_body(
            xform=wp.transform(pos, rot),
            label=name,
            mass=mass,
            com=com,
            inertia=inertia,
            lock_inertia=True,
        )
        proxy_bodies.append((name, body))
        for index, (vertices, faces) in enumerate(_convex_components(local_mesh, obj.get("cpu_collision"))):
            with wp.ScopedDevice("cpu"):
                cpu_builder.add_shape_mesh(
                    body=body,
                    mesh=newton.Mesh(vertices, faces.flatten()),
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0),
                    label=f"{name}_component_{index}",
                )

    # Force CPU allocation even when Main is building its IPC model on CUDA.
    with wp.ScopedDevice("cpu"):
        model = cpu_builder.finalize(device="cpu")
        solver = SolverMuJoCo(
            model,
            use_mujoco_cpu=True,
            include_sites=False,
            skip_visual_only_geoms=True,
            iterations=config.get("iterations", 50),
            tolerance=config.get("tolerance", 1e-8),
            integrator="implicitfast",
        )
    mj_model = solver.mj_model
    if proxy_bodies:
        exported_bodies = solver.mjc_body_to_newton.numpy()[0]
        for name, body in proxy_bodies:
            rows = np.flatnonzero(exported_bodies == body)
            if len(rows) != 1 or mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, int(rows[0])) != name:
                raise ValueError(f"CPU dynamic proxy body name conflicts after MuJoCo export: {name}")
    if config.get("robot_self_collision", False):
        # Newton's detection-only gap can emit thousands of inactive robot
        # contacts. Native CPU servos need active margins, not gap contacts.
        mj_model.geom_gap[:] = 0.0
    mj_model.opt.timestep = 1.0 / config.get("physics_hz", 1000.0)
    mj_model.geom_friction[:, 0] = config.get("friction", 0.3)
    # This CPU model omits imported world shapes; only explicit YAML static
    # objects attach to body 0. Fixed robot links retain their own body ids.
    # Set native margins after export, which zeros Newton margins for CCD.
    # CpuStepper uses this MJB directly, without Newton's runtime overrides.
    geom_shapes = solver.mjc_geom_to_newton_shape.numpy()[0]
    mapped = set()
    for geom, shape in enumerate(geom_shapes):
        if int(shape) in static_margins:
            if mj_model.geom_bodyid[geom] != 0:
                raise ValueError("CPU static margin mapped to a robot geom")
            mj_model.geom_margin[geom] = static_margins[int(shape)]
            mapped.add(int(shape))
    if mapped != set(static_margins):
        raise ValueError("CPU static collider missing from exported shape mapping")
    payload = np.empty(mujoco.mj_sizeModel(mj_model), dtype=np.uint8)
    mujoco.mj_saveModel(mj_model, buffer=payload)
    return payload.tobytes()
