# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Build IPC task geometry from the scene YAML using the engine hook lifecycle."""

from __future__ import annotations

import copy
import math
import re

import numpy as np

from .config import IPCConfig
from .geometry import configure_kinematic_robot_collision, read_usd_mesh
from .deformable_publisher import DeformableParticlePublisher, DeformablePublishConfig


def _positive(value, key, *, zero=False):
    """Reject non-finite or non-physical scalar configuration."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a finite number")
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(f"{key} must be {'nonnegative' if zero else 'positive'} and finite")


def _prim_path(value):
    """Require a concrete absolute USD prim path, excluding the pseudo-root."""
    from pxr import Sdf

    if not isinstance(value, str) or not Sdf.Path.IsValidPathString(value):
        raise ValueError(f"Invalid mesh/deactivation prim path: {value!r}")
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or value == "/":
        raise ValueError(f"Expected an absolute prim path: {value!r}")


class IPCScene:
    """Construct static and rigid colliders, cloth, tet volumes and static boxes.

    Object order determines particle order. Recording bindings are checked
    against the constructed ranges without changing the recording protocol.
    """

    def __init__(self, scene_cfg):
        newton_cfg = scene_cfg.get("libuipc") or {}
        if newton_cfg.get("scene_plugin") or newton_cfg.get("entries"):
            raise ValueError("libuipc.objects cannot be combined with scene_plugin or entries")
        if (newton_cfg.get("solver") or {}).get("prefer") != "ipc":
            raise ValueError("libuipc.objects requires libuipc.solver.prefer: ipc")
        cfg = newton_cfg
        self.objects = copy.deepcopy(cfg["objects"])
        if not isinstance(self.objects, list) or not self.objects:
            raise ValueError("libuipc.objects must be a nonempty list")
        names, paths = set(), set()
        cloth_keys = {
            "density",
            "volume_density",
            "tri_ke",
            "tri_ka",
            "edge_ke",
            "expected_particle_count",
        }
        for obj in self.objects:
            if not isinstance(obj, dict):
                raise ValueError("Each libuipc.objects entry must be a mapping")
            name, kind = obj.get("name"), obj.get("kind")
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in names:
                raise ValueError(f"Invalid or duplicate IPC object name: {name!r}")
            names.add(name)
            if kind not in ("static_mesh", "rigid_mesh", "cloth_mesh", "tet_mesh", "box"):
                raise ValueError(f"Unknown IPC object kind: {kind!r}")
            allowed = {"name", "kind", "mesh_prim_path"}
            if kind == "cloth_mesh":
                allowed |= cloth_keys
                if ("density" in obj) == ("volume_density" in obj):
                    raise ValueError(f"IPC cloth {name} requires exactly one density or volume_density")
                for key in ("tri_ke", "tri_ka", "edge_ke"):
                    if key not in obj:
                        raise ValueError(f"IPC cloth {name} requires {key}")
                for key in cloth_keys & obj.keys():
                    _positive(obj[key], f"{name}.{key}", zero=key == "edge_ke")
                count = obj.get("expected_particle_count")
                if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
                    raise ValueError(f"{name}.expected_particle_count must be an integer")
            elif kind == "rigid_mesh":
                allowed |= {"mass", "density", "body_prim_path", "cpu_collision"}
                if "body_prim_path" in obj:
                    _prim_path(obj["body_prim_path"])
                if ("mass" in obj) == ("density" in obj):
                    raise ValueError(f"IPC rigid {name} requires exactly one mass or density")
                key = "mass" if "mass" in obj else "density"
                _positive(obj[key], f"{name}.{key}")
            elif kind == "tet_mesh":
                allowed |= {
                    "density",
                    "youngs_modulus",
                    "poisson_ratio",
                    "damping",
                    "particle_radius",
                    "youngs_modulus_scale_power",
                    "expected_particle_count",
                }
                for key in ("density", "youngs_modulus", "particle_radius"):
                    _positive(obj.get(key), f"{name}.{key}")
                _positive(obj.get("damping", 0.0), f"{name}.damping", zero=True)
                nu = obj.get("poisson_ratio")
                if isinstance(nu, bool) or not isinstance(nu, (int, float)) or not -1 < nu < 0.5:
                    raise ValueError(f"{name}.poisson_ratio must be finite in (-1, 0.5)")
                power = obj.get("youngs_modulus_scale_power", 0.0)
                if isinstance(power, bool) or not isinstance(power, (int, float)) or not math.isfinite(power):
                    raise ValueError(f"{name}.youngs_modulus_scale_power must be finite")
                count = obj.get("expected_particle_count")
                if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count <= 0):
                    raise ValueError(f"{name}.expected_particle_count must be a positive integer")
            elif kind == "static_mesh":
                allowed.add("cpu_collision")
            elif kind == "box":
                allowed = {"name", "kind", "half_extents", "pose", "cpu_collision"}
                extents = obj.get("half_extents")
                if not isinstance(extents, list) or len(extents) != 3:
                    raise ValueError(f"{name}.half_extents must contain three lengths")
                for v in extents:
                    _positive(v, f"{name}.half_extents")
                pose = obj.get("pose", [0, 0, 0, 0, 0, 0, 1])
                if (
                    not isinstance(pose, list)
                    or len(pose) != 7
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in pose)
                ):
                    raise ValueError(f"{name}.pose must be finite [x,y,z,qx,qy,qz,qw]")
                if not np.isclose(np.linalg.norm(pose[3:]), 1.0):
                    raise ValueError(f"{name}.pose quaternion must be normalized")
            if kind in ("static_mesh", "rigid_mesh", "box") and "cpu_collision" in obj:
                cpu = obj["cpu_collision"]
                keys = {"margin"}
                if kind != "box":
                    keys |= {"method", "threshold"}
                if kind == "rigid_mesh":
                    keys.remove("margin")
                if not isinstance(cpu, dict) or set(cpu) - keys:
                    raise ValueError(f"{name}.cpu_collision permits only {sorted(keys)}")
                if "method" in cpu or "threshold" in cpu:
                    if cpu.get("method") != "coacd":
                        raise ValueError(f"{name}.cpu_collision requires method: coacd for decomposition")
                    cpu.setdefault("threshold", 0.01)
                    _positive(cpu["threshold"], f"{name}.cpu_collision.threshold")
                if "margin" in cpu:
                    _positive(cpu["margin"], f"{name}.cpu_collision.margin", zero=True)
            if set(obj) - allowed:
                raise ValueError(f"Unknown IPC object keys for {name}: {sorted(set(obj) - allowed)}")
            if kind != "box":
                path = obj.get("mesh_prim_path")
                _prim_path(path)
                if path in paths:
                    raise ValueError(f"Duplicate IPC mesh_prim_path: {path}")
                paths.add(path)
        from .cpu.model import validate_dynamic_proxies

        validate_dynamic_proxies(self.objects, cfg.get("cpu_mujoco"))
        self._length_scale_mesh = cfg.get("length_scale_mesh")
        if self._length_scale_mesh is not None:
            _prim_path(self._length_scale_mesh)
            deformables = {obj["mesh_prim_path"] for obj in self.objects if obj["kind"] in ("cloth_mesh", "tet_mesh")}
            if self._length_scale_mesh not in deformables:
                raise ValueError("libuipc.length_scale_mesh must name a declared deformable mesh")
        self._length_scale = 1.0
        self._deactivate = cfg.get("deactivate_prims", [])
        if not isinstance(self._deactivate, list):
            raise ValueError("libuipc.deactivate_prims must be a list")
        for path in self._deactivate:
            _prim_path(path)
        self._meshes = {}
        self._bindings = []
        self._ranges = {}
        self._rest_positions = []
        self._viewer_collider_shapes = ()
        self._cpu_model_bytes = None
        self._recording_body_bindings = {}
        self._cpu_source_stage = None
        # Opt-in live particle-state channel, disabled unless the scene yaml
        # declares libuipc.deformable_publish.enabled: true. See
        # deformable_publisher.py's module docstring for the wire format;
        # this is the backend-agnostic deformable frame channel the migration
        # plan's section 5.1.1 identified as missing from every separate-
        # process render backend this repo has.
        self._deformable_publish_cfg = DeformablePublishConfig.from_scene(scene_cfg)
        self._deformable_publisher = None
        self._deformable_tick = 0

    def on_stage(self, stage, ctx):
        """Capture source geometry before disabling its original physics."""
        self._cpu_source_stage = stage
        robot_path = f"/{ctx.robot_prefix.strip('/')}"
        if robot_path == "/":
            raise ValueError("IPC YAML scenes require a nonempty robot namespace")
        paths = [obj["mesh_prim_path"] for obj in self.objects if obj["kind"] != "box"]
        for path in [*paths, *self._deactivate]:
            if path == robot_path or path.startswith(robot_path + "/") or robot_path.startswith(path + "/"):
                raise ValueError(f"IPC source {path} overlaps engine robot {robot_path}")
        self._meshes = {path: read_usd_mesh(stage, path) for path in paths}
        if self._length_scale_mesh is not None:
            self._length_scale = self._meshes[self._length_scale_mesh].decompose()[2]
            for obj in self.objects:
                if obj["kind"] in ("cloth_mesh", "tet_mesh"):
                    scale = self._meshes[obj["mesh_prim_path"]].decompose()[2]
                    if not np.isclose(scale, self._length_scale, rtol=1e-6, atol=1e-9):
                        raise ValueError(
                            f"{obj['name']} scale differs from length_scale_mesh; scale the group together"
                        )
        for obj in self.objects:
            if obj["kind"] == "tet_mesh" and self._meshes[obj["mesh_prim_path"]].tets is None:
                raise ValueError(f"{obj['mesh_prim_path']} requires tetVertexIndices for tet_mesh")
        from .body_bindings import body_bindings

        self._recording_body_bindings = body_bindings(stage, self.objects)
        for path in [*paths, *self._deactivate]:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetActive(False)

    def on_build(self, builder, ctx):
        """Add task objects and retain exact particle-to-authored-mesh bindings."""
        import newton
        import warp as wp

        from genie_sim_engine_libuipc.cpu.worker import CpuMuJoCoConfig

        if CpuMuJoCoConfig.from_scene(ctx.scene_cfg) is not None:
            from .cpu.model import build_cpu_model

            self._cpu_model_bytes = build_cpu_model(
                builder,
                self.objects,
                self._meshes,
                ctx.scene_cfg,
                ctx.robot_prefix,
                source_stage=self._cpu_source_stage,
            )
            ctx.logger.info("[cpu-mujoco] compiled robot + scene colliders and selected rigid proxies (CPU)")
        self._bindings, self._ranges = [], {}
        for obj in self.objects:
            name, kind = obj["name"], obj["kind"]
            if kind == "box":
                pose = obj.get("pose", [0, 0, 0, 0, 0, 0, 1])
                builder.add_shape_box(
                    body=-1,
                    xform=wp.transform(pose[:3], pose[3:]),
                    hx=obj["half_extents"][0],
                    hy=obj["half_extents"][1],
                    hz=obj["half_extents"][2],
                    label=name,
                )
                continue
            mesh = self._meshes[obj["mesh_prim_path"]]
            if kind == "static_mesh":
                builder.add_shape_mesh(
                    body=-1,
                    mesh=newton.Mesh(mesh.world_points(), mesh.triangles.flatten()),
                    label=name,
                )
                continue
            if kind == "rigid_mesh":
                pos, rot, scales = mesh.decompose_axis_scaled()
                rigid_mesh = newton.Mesh(mesh.points * scales, mesh.triangles.flatten())
                if not rigid_mesh.is_watertight or not math.isfinite(rigid_mesh.mass) or rigid_mesh.mass <= 0:
                    raise ValueError(f"{mesh.path} must be watertight with positive volume for SolverIPC")
                density = obj["mass"] / rigid_mesh.mass if "mass" in obj else obj["density"]
                body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat(*rot)), label=name)
                builder.add_shape_mesh(
                    body=body,
                    mesh=rigid_mesh,
                    cfg=newton.ModelBuilder.ShapeConfig(density=density),
                    label=name,
                )
            else:
                pos, rot, scale = mesh.decompose()
                count = len(mesh.points)
                if obj.get("expected_particle_count", count) != count:
                    raise ValueError(f"{name} has {count} vertices; expected {obj['expected_particle_count']}")
                start = builder.particle_count
                reference = mesh.points if mesh.rest_points is None else mesh.rest_points
                common = dict(pos=wp.vec3(*pos), rot=wp.quat(*rot), scale=scale, vel=wp.vec3(), label=name)
                if kind == "tet_mesh":
                    youngs = obj["youngs_modulus"] * scale ** obj.get("youngs_modulus_scale_power", 0.0)
                    nu = obj["poisson_ratio"]
                    mu = youngs / (2 * (1 + nu))
                    lam = youngs * nu / ((1 + nu) * (1 - 2 * nu))
                    builder.add_soft_mesh(
                        **common,
                        mesh=newton.TetMesh(
                            reference, mesh.tets.flatten(), mu, lam, obj.get("damping", 0.0), obj["density"]
                        ),
                        add_surface_mesh_edges=True,
                        particle_radius=obj["particle_radius"] * scale,
                        validate_mesh=True,
                    )
                else:
                    params = {key: obj[key] for key in ("tri_ke", "tri_ka", "edge_ke")}
                    thickness = IPCConfig.from_scene(ctx.scene_cfg).cloth_thickness * self._length_scale
                    density = obj["density"] if "density" in obj else obj["volume_density"] * thickness
                    builder.add_cloth_mesh(
                        **common,
                        vertices=reference,
                        indices=mesh.triangles.flatten(),
                        density=density,
                        tri_kd=0.0,
                        tri_drag=0.0,
                        tri_lift=0.0,
                        edge_kd=0.0,
                        particle_radius=0.5 * thickness,
                        **params,
                    )
                if mesh.rest_points is not None:
                    # Builder computes mass/elastic metrics from the reference first.
                    # Model/state positions, rendering and native frame-zero recovery
                    # must instead start from the authored collapsed configuration.
                    rest_world = np.hstack((reference, np.ones((count, 1)))) @ mesh.xform[:, :3]
                    self._rest_positions.append((start, rest_world))
                    builder.particle_q[start : start + count] = [wp.vec3(*v) for v in mesh.world_points()]
                self._bindings.append(
                    {
                        "mesh_prim_path": mesh.path,
                        "particle_indices": np.arange(start, start + count, dtype=np.uint32),
                    }
                )
                self._ranges[mesh.path] = (start, count)
        recording = (ctx.scene_cfg.get("libuipc") or {}).get("recording") or {}
        instances = (recording.get("instances") or ()) if recording.get("enabled", True) is not False else ()
        for instance in instances:
            path = instance.get("mesh_prim_path")
            actual = self._ranges.get(path)
            if actual != (
                instance.get("particle_start"),
                instance.get("particle_count"),
            ):
                raise ValueError(
                    f"Recording instance {instance.get('name')} does not match constructed cloth range {path}: {actual}"
                )
        # Preserve collider provenance before the hand-only contact filter
        # removes arm collision flags; visual-only shapes never enter this set.
        self._viewer_collider_shapes = tuple(
            shape
            for shape, flags in enumerate(builder.shape_flags)
            if int(flags) & int(newton.ShapeFlags.COLLIDE_SHAPES) and not int(flags) & int(newton.ShapeFlags.SITE)
        )
        ipc = IPCConfig.from_scene(ctx.scene_cfg)
        configure_kinematic_robot_collision(
            builder,
            robot_prefix=ctx.robot_prefix,
            collision_link_names=ipc.collision_link_names,
            logger=ctx.logger,
        )

    def on_model_ready(self, model, ctx):
        """Attach immutable renderer-only collider indices, never a contact mask."""
        model.ipc_viewer_collider_shapes = self._viewer_collider_shapes
        model.ipc_recording_body_bindings = self._recording_body_bindings
        model.ipc_length_scale = self._length_scale
        if self._rest_positions:
            reference = model.particle_q.numpy().astype(np.float64)
            for start, positions in self._rest_positions:
                reference[start : start + len(positions)] = positions
            reference.setflags(write=False)
            model.ipc_rest_particle_q = reference
        if self._cpu_model_bytes is not None:
            model.ipc_cpu_model_bytes = self._cpu_model_bytes
        if self._deformable_publish_cfg is not None:
            deformable_objects = [
                (mesh_id, obj["name"], obj["mesh_prim_path"], self._ranges[obj["mesh_prim_path"]][1])
                for mesh_id, obj in enumerate(self.objects)
                if obj["kind"] in ("cloth_mesh", "tet_mesh")
            ]
            self._deformable_publisher = DeformableParticlePublisher(
                self._deformable_publish_cfg, deformable_objects, ctx.logger
            )

    def start_deformable_publish(self, node) -> None:
        """Attach the live particle-state publisher to the lifecycle node, if configured.

        Called from ``engine_base.start_deformable_publisher()`` — the same
        call-site pattern as ``IPCDriver.start_feedback`` — after
        ``start_listener()`` has brought the shared lifecycle node up. A no-op
        when ``libuipc.deformable_publish`` was absent from the scene yaml.
        """
        if self._deformable_publisher is not None:
            self._deformable_publisher.start(node)

    def on_post_step(self, state, sim_time, dt, ctx) -> None:
        """Publish this tick's solved particle positions, if a publisher is attached."""
        if self._deformable_publisher is None:
            return
        self._deformable_publisher.publish_frame(
            tick=self._deformable_tick,
            sim_time=sim_time,
            particle_q=state.particle_q.numpy(),
            ranges=self._ranges,
        )
        self._deformable_tick += 1

    def get_deformable_render_bindings(self):
        """Return stable authored-mesh mappings for rendering and recording."""
        return self._bindings
