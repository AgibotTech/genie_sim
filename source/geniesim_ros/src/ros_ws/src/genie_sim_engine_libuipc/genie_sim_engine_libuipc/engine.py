# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Independent libuipc PhysicsEngine using Newton SDK model/FK data."""

from __future__ import annotations

import math
import time
import threading
import hashlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from genie_sim_engine_core.common.lifecycle_ros import RosResetCoordinator
from genie_sim_engine_core.engine.base import PhysicsEngine
from genie_sim_engine_core.engine.lifecycle import EpisodeHandle, EpisodeSummary

from .usd import USDPolicies, _strip_disabled_physics_schemas


@PhysicsEngine.register("libuipc")
class LibuipcEngine(USDPolicies, PhysicsEngine):
    """Single-world native IPC simulation; no Newton backend package dependency.

    The public constructor accepts the existing core session ABI. Solver tuning
    belongs exclusively to scene_cfg["libuipc"]; SDK model arrays are a bridge,
    while the native World owns dynamic body and particle evolution.
    """

    def __init__(
        self,
        *,
        scene_usda,
        robot_usda,
        robot_prefix,
        physics_hz,
        logger,
        scene_cfg=None,
        init_joint_pos=None,
        physics_solver="ipc",
        physics_solver_substep=1,
        num_envs=1,
        pin_base_to_world=True,
        pin_base_mode=None,
        convert_joints_to_fixed=None,
        runtime_usd_dump_path="",
        **_unused,
    ):
        if physics_solver != "ipc":
            raise ValueError("libuipc supports physics_solver: ipc only")
        if int(physics_solver_substep) != 1:
            raise ValueError("libuipc requires physics_solver_substep: 1 (one native fixed step)")
        if int(num_envs) != 1:
            raise ValueError("libuipc currently supports exactly one world")
        if not math.isfinite(float(physics_hz)) or float(physics_hz) <= 0:
            raise ValueError("physics_hz must be finite and positive")
        if pin_base_mode not in (None, "", "weld", "free"):
            raise ValueError("libuipc supports weld/free base modes; mocap requires native-world rebuild")
        self._logger = logger
        self._scene_usda = str(scene_usda)
        self._robot_usda = str(robot_usda or "")
        self._robot_prefix_str = str(robot_prefix).strip("/")
        if not self._robot_prefix_str:
            raise ValueError("libuipc requires a nonempty robot_prefix")
        self._physics_hz = float(physics_hz)
        self._dt = 1.0 / self._physics_hz
        self._scene_cfg = dict(scene_cfg or {})
        if "newton" in self._scene_cfg:
            raise ValueError("IPC scenes must migrate the newton block to libuipc; mixed configuration is unsupported")
        self._init_joint_pos = dict(init_joint_pos or {})
        self._pin_base_to_world = pin_base_to_world if not pin_base_mode else pin_base_mode == "weld"
        self._convert_joints_to_fixed = list(convert_joints_to_fixed or [])
        self._static_joint_q = {}
        self._stage = self._model = self._state_0 = self._state_1 = None
        self._driver = self._scene = None
        self._started = False
        self._closed = False
        self._paused = False
        self._tick = self._epoch = self._episode_id = 0
        self._episode_start = 0
        self._sim_time = 0.0
        self._source_id = uuid.uuid4().hex
        self._execution_failure = None
        self._failure_lock = threading.Lock()
        self._failure_finalized = False
        self._failure_version = 0
        self._actions = self._action_source = None
        self._identity_source = None
        self._snapshot_requests = self._snapshots = self._state_recorder = None
        self._lifecycle = RosResetCoordinator(self, logger)
        try:
            self._build()
            self._initialize_state_services()
            if runtime_usd_dump_path:
                # Merge local/session edits while preserving external geometry arcs.
                # Debug output must not inline the entire asset scene or abort startup.
                try:
                    from pxr import UsdUtils

                    Path(runtime_usd_dump_path).parent.mkdir(parents=True, exist_ok=True)
                    UsdUtils.FlattenLayerStack(self._stage).Export(str(runtime_usd_dump_path))
                except Exception as exc:
                    self._logger.warn(f"[libuipc] runtime USD snapshot skipped: {exc}")
        except BaseException:
            self.shutdown()
            raise

    def _build(self):
        import newton
        import warp as wp
        from pxr import Usd, UsdPhysics
        from genie_sim_engine_core.engine._mimic import expand_targets, parse_mimic
        from .driver import IPCDriver
        from .joint_index import JointIndex
        from .scene import IPCScene
        from .assemble.physics_variant import select_physics_variant

        self._stage = Usd.Stage.Open(self._scene_usda)
        if not self._stage:
            raise ValueError(f"Cannot open scene stage: {self._scene_usda}")
        self._stage.SetEditTarget(self._stage.GetSessionLayer())
        if self._robot_usda:
            if not Path(self._robot_usda).is_file():
                raise FileNotFoundError(self._robot_usda)
            robot = self._stage.DefinePrim(f"/{self.robot_prefix}", "Xform")
            robot.GetReferences().AddReference(self._robot_usda)
        select_physics_variant(self._stage, f"/{self.robot_prefix}", logger=self._logger)
        self._apply_init_base_pose()
        self._repair_root_weld_anchor()
        if not self._pin_base_to_world:
            self._deactivate_root_joint()
        tokens = {"base": "chassis", "head": "head_joint", "body": "body_joint", "arm": "arm_", "gripper": "gripper_"}
        for token in self._convert_joints_to_fixed:
            if token == "no_robot":
                self._stage.GetPrimAtPath(f"/{self.robot_prefix}").SetActive(False)
            elif token in tokens:
                self._replace_joints_with_fixed(name_substring=tokens[token], label=token)
            else:
                raise ValueError(f"Unsupported convert_joints_to_fixed token: {token}")
        self._mimic = parse_mimic(self._stage, self._logger)
        self._scene = IPCScene(self._scene_cfg)
        self._ctx = SimpleNamespace(
            scene_cfg=self._scene_cfg,
            robot_prefix=self.robot_prefix,
            logger=self._logger,
            physics_hz=self._physics_hz,
            sim_substeps=1,
        )
        self._scene.on_stage(self._stage, self._ctx)
        _strip_disabled_physics_schemas(self._stage)
        self._driver = IPCDriver(physics_hz=self._physics_hz, scene_cfg=self._scene_cfg)
        builder = newton.ModelBuilder()
        self._driver.register_custom_attributes(builder)
        # Mjc reads SI mjc:damping on Newton 1.6; keep all three resolvers.
        try:
            from newton.usd import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

            resolver_module = "newton.usd"
        except ImportError:
            from newton._src.usd.schemas import SchemaResolverMjc, SchemaResolverNewton, SchemaResolverPhysx

            resolver_module = "newton._src.usd.schemas"
        builder.add_usd(
            source=self._stage,
            verbose=False,
            collapse_fixed_joints=False,
            schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx(), SchemaResolverMjc()],
        )
        self._logger.info(f"[libuipc] schema_resolvers=Newton+Physx+Mjc from {resolver_module}")
        builder.add_ground_plane()
        self._scene.on_build(builder, self._ctx)
        self._model = builder.finalize()
        self._scene.on_model_ready(self._model, self._ctx)
        self._jindex = JointIndex(self._model)
        prefix = f"/{self.robot_prefix}/"
        children = self._model.joint_child.numpy()
        scalar = [
            s
            for s in self._jindex.slices()
            if s.q_count == s.dof_count == 1
            and str(self._model.body_label[int(children[s.joint_index])]).startswith(prefix)
        ]
        if len({s.name for s in scalar}) != len(scalar):
            raise ValueError("Ambiguous robot joint leaf names")
        self._scalar = {s.name: s for s in scalar}
        self._joint_names = [*self._scalar, *self._static_joint_q]
        self._joint_prim_map = {s.name: s.label for s in scalar}
        self._q_rows = np.array([s.q_start for s in scalar], dtype=int)
        self._qd_rows = np.array([s.qd_start for s in scalar], dtype=int)
        q = self._model.joint_q.numpy().copy()
        initial = {}
        for name, raw in self._init_joint_pos.items():
            s = self._scalar.get(name)
            if s:
                value = float(raw.value if hasattr(raw, "value") else raw)
                initial[name] = math.radians(value) if s.joint_type == int(newton.JointType.REVOLUTE) else value
            elif name not in self._static_joint_q:
                self._logger.warn(f"[libuipc] initial joint {name!r} is not a scalar robot joint")
        for name, value in {**initial, **expand_targets(initial, self._mimic)}.items():
            if name in self._scalar:
                q[self._scalar[name].q_start] = value
        if not np.isfinite(q).all():
            raise ValueError("Initial joint state must be finite")
        self._model.joint_q.assign(q)
        self._model.joint_qd.zero_()
        self._state_0, self._state_1 = self._model.state(), self._model.state()
        for state in (self._state_0, self._state_1):
            state.joint_q.assign(q)
            state.joint_qd.zero_()
            newton.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, state)
        # Native geometry is constructed from model.body_q, not state.body_q.
        # Seed that construction/reset reference with the requested FK pose too.
        self._model.body_q.assign(self._state_0.body_q)
        self._control = self._model.control()
        self._driver.prepare_model(self._model, self._logger, robot_prefix=self.robot_prefix)
        with wp.ScopedDevice(self._model.device):
            self._driver.init_target_buffer(self._model, self._logger)
            self._driver.bind_feedback(self._model, self._jindex, self._joint_names)
        self._driver.build_solver(self._model, self._logger)
        self._initial_q = q.copy()
        self._body_paths = []
        self._body_offsets = {}
        for i, label in enumerate(self._model.body_label):
            binding = self._scene._recording_body_bindings.get(str(label))
            if binding:
                path, offset = binding
                self._body_offsets[i] = np.array(offset)
            else:
                path = str(label)
            if not path.startswith("/"):
                raise ValueError(f"Body {label!r} has no authored absolute prim path")
            self._body_paths.append(path)
        self._logger.info(
            f"[libuipc] ready: {self._model.body_count} bodies, "
            f"{len(self._scalar)} scalar joints, {self._model.particle_count} particles, dt={self._dt:g}"
        )

    def _initialize_state_services(self):
        """Create owned services only after FK, native construction and bindings."""
        from .action_sequence import ActionSequence, topic_from_scene
        from .physical_state_recording import PhysicalStateRecorder, PhysicalStateRecordingConfig
        from .observation_snapshot import SnapshotRequests

        self._action_topic = topic_from_scene(self._scene_cfg, "action_sequence")
        self._snapshot_topic = topic_from_scene(self._scene_cfg, "observation_snapshot")
        if self._action_topic is not None:
            cpu = self._driver._cpu_config
            if cpu is None or cpu.synchronization != "barrier":
                raise ValueError("libuipc.action_sequence requires CPU barrier synchronization")
            self._actions = ActionSequence(
                self._driver._cpu_names,
                self._initial_q[self._driver._cpu_q_indices],
                physics_hz=self._physics_hz,
                source_id=self._source_id,
                epoch=self._epoch,
            )
        if self._snapshot_topic is not None:
            self._snapshot_requests = SnapshotRequests(self)
        config = PhysicalStateRecordingConfig.from_scene(self._scene_cfg)
        if config is not None:
            self._state_recorder = PhysicalStateRecorder(config, metadata=self._physical_metadata())
            # on_model_ready precedes initial joint seeding/FK/native construction.
            # Recording there would certify a different initial physical state.
            self._state_recorder.record(self._state_0, tick=0, sim_time=0.0, dt=0.0, initial=True)

    def _physical_metadata(self):
        from .physical_state_recording import STATE_ARRAYS

        provenance = {}
        for key, path in (("scene_usda", self._scene_usda), ("robot_usda", self._robot_usda)):
            source = Path(path)
            provenance[key] = {"path": path}
            if path and source.is_file():
                provenance[key]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        return {
            "source_id": self._source_id,
            "epoch": self._epoch,
            "backend": "libuipc",
            "robot_prefix": self.robot_prefix,
            "physics_hz": self._physics_hz,
            "sim_substeps": 1,
            "scene_provenance": provenance,
            "scene_config": json.loads(json.dumps(self._scene_cfg, default=str, allow_nan=False)),
            "body_labels": [str(label) for label in self._model.body_label],
            "body_paths": list(self._body_paths),
            "joint_labels": [str(label) for label in self._model.joint_label],
            "controlled_joint_names": list(self._driver._cpu_names) if self._actions else list(self._scalar),
            "body_pose_schema": "xyz+xyzw (raw SDK body origin)",
            "snapshot_pose_schema": "xyz+wxyz (authored root)",
            "matrix_conventions": {"usd": "row-vector", "body_root_offsets": "column-vector"},
            "body_bindings": {
                str(label): {"prim_path": path, "body_to_authored_root": np.asarray(offset).tolist()}
                for label, (path, offset) in self._scene._recording_body_bindings.items()
            },
            "objects": [
                dict(
                    obj,
                    particle_start=self._scene._ranges[obj["mesh_prim_path"]][0],
                    particle_count=self._scene._ranges[obj["mesh_prim_path"]][1],
                )
                for obj in self._scene.objects
                if obj["kind"] in ("cloth_mesh", "tet_mesh")
            ],
            "deformable_bindings": [
                {
                    "mesh_prim_path": entry["mesh_prim_path"],
                    "particle_indices": np.asarray(entry["particle_indices"]).tolist(),
                }
                for entry in self._scene.get_deformable_render_bindings()
            ],
            # SDK-provided empty arrays are valid for absent model populations;
            # missing arrays are never synthesized by the producer.
            "zero_count_arrays": [
                name
                for name in STATE_ARRAYS
                if getattr(self._state_0, name, None) is not None and getattr(self._state_0, name).numpy().size == 0
            ],
        }

    @property
    def stage(self):
        return self._stage

    @property
    def robot_prefix(self):
        return self._robot_prefix_str

    @property
    def joint_names(self):
        return self._joint_names

    @property
    def joint_prim_map(self):
        return self._joint_prim_map

    @property
    def body_paths(self):
        return self._body_paths

    @property
    def reset_epoch(self):
        return self._epoch

    @property
    def activity(self):
        return self._lifecycle

    def startup(self, headless=True):
        if self._closed:
            raise RuntimeError("libuipc engine is closed")
        if self._started:
            return
        try:
            self._lifecycle.start_listener()
            self._driver.start_feedback(self._lifecycle._node)
            self._scene.start_deformable_publish(self._lifecycle._node)
            if self._actions is not None:
                from .action_sequence import ActionSequenceSource

                self._action_source = ActionSequenceSource(
                    self._actions,
                    self._lifecycle._node,
                    self._action_topic,
                    on_failure=self.report_transport_failure,
                )
            if self._snapshot_requests is not None:
                from .observation_snapshot import ObservationSnapshotSource

                self._snapshots = ObservationSnapshotSource(
                    self, self._lifecycle._node, self._snapshot_topic, requests=self._snapshot_requests
                )
            prefixes = [
                prefix
                for prefix in (getattr(self, "_action_topic", None), getattr(self, "_snapshot_topic", None))
                if prefix is not None
            ]
            if prefixes:
                from .action_sequence import StateIdentitySource

                self._identity_source = StateIdentitySource(self, self._lifecycle._node, prefixes)
            self._started = True
            self._publish_identity("ready")
        except BaseException:
            self.shutdown()
            raise

    def step(self, dt, step_start):
        if not math.isfinite(dt) or not math.isclose(dt, self._dt, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"libuipc requires fixed dt={self._dt:g}; got {dt}")
        if self._closed:
            raise RuntimeError("libuipc engine is closed")
        self._raise_if_execution_failed()
        if self._paused:
            return 0.0
        start = time.monotonic()
        try:
            recorder = getattr(self, "_state_recorder", None)
            if recorder is not None:
                recorder.require_healthy()
            arguments = dict(completed_sim_time=self._sim_time)
            actions = getattr(self, "_actions", None)
            if actions is not None:
                arguments["targets"] = actions.begin_step()
            self._state_0.clear_forces()
            self._raise_if_execution_failed()
            self._driver.substep(self._model, self._state_0, self._state_1, self._control, dt, **arguments)
            self._raise_if_execution_failed()
            self._state_0, self._state_1 = self._state_1, self._state_0
            self._tick += 1
            self._sim_time += dt
            self._scene.on_post_step(self._state_0, self._sim_time, dt, self._ctx)
            if recorder is not None:
                recorder.record(self._state_0, tick=self._tick, sim_time=self._sim_time, dt=dt)
            self._raise_if_execution_failed()
            snapshots = getattr(self, "_snapshots", None) or getattr(self, "_snapshot_requests", None)
            if snapshots is not None:
                snapshots.on_tick()
            self._raise_if_execution_failed()
            if actions is not None:
                stats = getattr(self._driver, "_cpu_snapshot", {})
                cpu = getattr(self._driver, "_cpu_config", None)
                self._publish_action(
                    actions.complete_step(
                        tick=self._tick,
                        sim_time=self._sim_time,
                        cpu_steps=stats.get("steps"),
                        cpu_batch_steps=cpu.sync_substeps if cpu is not None else None,
                    )
                )
        except BaseException as exc:
            self._latch_execution_failure(exc)
            raise
        return (time.monotonic() - start) * 1000.0

    @property
    def execution_status(self):
        failure = getattr(self, "_execution_failure", None)
        return {
            "status": "failed" if failure else "closed" if self._closed else "running",
            "source_id": getattr(self, "_source_id", None),
            "epoch": self._epoch,
            "tick": self._tick,
            "sim_time": self._sim_time,
            "error": str(failure) if failure else None,
        }

    def _publish_action(self, reply):
        source = getattr(self, "_action_source", None)
        if source is not None and reply is not None:
            source.publish(reply)

    def _publish_identity(self, status):
        source = getattr(self, "_identity_source", None)
        if source is not None:
            source.publish(status)

    def _cleanup_attempt(self, label, operation):
        try:
            operation()
            return True
        except BaseException as exc:
            logger = getattr(self, "_logger", None)
            if logger is not None:
                logger.warn(f"[libuipc] {label}: {exc}")
            return False

    def _failure_guard(self):
        # Constructor-bypassing lightweight owners retain the same internal
        # lifecycle path; a real engine initializes this before exposing ROS.
        if not hasattr(self, "_failure_lock"):
            self._failure_lock = threading.Lock()
        return self._failure_lock

    def report_transport_failure(self, failure):
        """Callback-safe fault flag only: no Warp reads, publication or file IO."""
        with self._failure_guard():
            self._failure_version = getattr(self, "_failure_version", 0) + 1
            if getattr(self, "_execution_failure", None) is None:
                self._execution_failure = failure

    def _raise_if_execution_failed(self):
        failure = getattr(self, "_execution_failure", None)
        if failure is not None:
            self._finalize_execution_failure()
            raise RuntimeError("libuipc execution failed; reset or create a new session") from failure

    def _latch_execution_failure(self, failure):
        self.report_transport_failure(failure)
        self._finalize_execution_failure()

    def _finalize_execution_failure(self):
        """Run once on the physics owner, never from the ROS fault callback."""
        with self._failure_guard():
            failure = getattr(self, "_execution_failure", None)
            if failure is None or getattr(self, "_failure_finalized", False):
                return
            self._failure_finalized = True
        self._cleanup_attempt("publish failed identity", lambda: self._publish_identity("failed"))
        actions = getattr(self, "_actions", None)
        if actions is not None:
            self._cleanup_attempt("cancel actions", lambda: self._publish_action(actions.close(str(failure))))
        snapshots = getattr(self, "_snapshots", None) or getattr(self, "_snapshot_requests", None)
        if snapshots is not None:
            self._cleanup_attempt("cancel snapshots", lambda: snapshots.cancel(str(failure)))
        recorder = getattr(self, "_state_recorder", None)
        if recorder is not None:
            self._cleanup_attempt("close failed archive", lambda: recorder.close(clean=False))

    def tick_extras(self):
        return None

    def apply_commands(self, cmd_positions, cmd_4ws_steer_pos, cmd_4ws_drive_vel, cmd_4ws_stamp):
        from genie_sim_engine_core.engine._mimic import expand_targets

        self._lifecycle.drain_resets()
        if cmd_4ws_drive_vel and any(float(v) != 0 for v in cmd_4ws_drive_vel.values()):
            raise NotImplementedError("libuipc's kinematic command path accepts scalar position targets only")
        commands = {**cmd_positions, **cmd_4ws_steer_pos}
        commands.update(expand_targets(commands, self._mimic))
        values = self._driver.target_buffer().numpy().copy()
        for name, value in commands.items():
            s = self._scalar.get(name)
            if s:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite target for {name}")
                values[s.q_start] = value
        self._driver.target_buffer().assign(values)

    def get_joint_states(self):
        q = self._state_0.joint_q.numpy()[self._q_rows]
        qd = self._state_0.joint_qd.numpy()[self._qd_rows]
        positions = np.concatenate((q, list(self._static_joint_q.values()))).astype(np.float64)
        velocities = np.concatenate((qd, np.zeros(len(self._static_joint_q)))).astype(np.float64)
        return positions, velocities

    def get_body_transforms(self):
        from .solver.solver_ipc import _matrix_to_quat_xyzw, _quat_xyzw_to_matrix

        # Body poses and authored offsets both use column-vector matrices.
        poses = self._state_0.body_q.numpy().astype(np.float64)
        for row, offset in self._body_offsets.items():
            matrix = np.eye(4)
            matrix[:3, :3] = _quat_xyzw_to_matrix(poses[row, 3:])
            matrix[:3, 3] = poses[row, :3]
            matrix = matrix @ offset
            poses[row, :3] = matrix[:3, 3]
            poses[row, 3:] = _matrix_to_quat_xyzw(matrix[:3, :3])
        return poses[:, [0, 1, 2, 6, 3, 4, 5]], list(self._body_paths)

    def get_odom(self, sim_time):
        from .solver.solver_ipc import _quat_xyzw_to_matrix

        row = next((i for i, p in enumerate(self._body_paths) if p.endswith("/base_link")), None)
        if row is None:
            return None
        poses, _ = self.get_body_transforms()
        pose = poses[row]
        rotation = _quat_xyzw_to_matrix(pose[[4, 5, 6, 3]])
        spatial = self._state_0.body_qd.numpy()[row]
        twist = np.concatenate((rotation.T @ spatial[:3], rotation.T @ spatial[3:]))
        return pose, twist

    def episode_reset(self, env_ids=None, seeds=None, instances=None):
        if getattr(self, "_state_recorder", None) is not None:
            raise RuntimeError("physical archive owns one episode; create a fresh session before reset")
        ids = [0] if env_ids is None else list(env_ids)
        if not ids:
            return
        if any(w != 0 for w in ids):
            raise ValueError("libuipc has only world 0")
        if instances and any(v not in (-1, 0) for v in instances.values()):
            raise NotImplementedError("libuipc instance switching requires rebuilding the native world")
        # Discard worker input, restore native state and only then certify completion.
        try:
            with self._failure_guard():
                reset_failure_version = getattr(self, "_failure_version", 0)
            node, was_started = self._driver._feedback_node, self._driver._cpu_started
            actions = getattr(self, "_actions", None)
            if actions is not None:
                self._publish_action(actions.cancel("episode reset", hold=self._initial_q[self._driver._cpu_q_indices]))
            snapshots = getattr(self, "_snapshots", None) or getattr(self, "_snapshot_requests", None)
            if snapshots is not None:
                snapshots.cancel("episode reset")
            self._driver.stop_feedback()
            self._driver._solver.reset(self._state_0)
            for field in ("body_q", "body_qd", "particle_q", "particle_qd"):
                src, dst = getattr(self._state_0, field, None), getattr(self._state_1, field, None)
                if src is not None and dst is not None:
                    dst.assign(src)
            self._model.joint_q.assign(self._initial_q)
            for state in (self._state_0, self._state_1):
                state.joint_q.assign(self._initial_q)
                state.joint_qd.zero_()
            self._driver.reset_inputs()
            if node is not None:
                self._driver.start_feedback(node)
            elif was_started:
                self._driver.start_cpu_worker()
            with self._failure_guard():
                if getattr(self, "_failure_version", 0) != reset_failure_version:
                    raise RuntimeError("transport failed during native reset")
            self._epoch += 1
            if actions is not None:
                self._publish_action(
                    actions.reset(epoch=self._epoch, hold=self._initial_q[self._driver._cpu_q_indices])
                )
            if snapshots is not None:
                snapshots.reset_epoch(self._epoch)
            with self._failure_guard():
                if getattr(self, "_failure_version", 0) != reset_failure_version:
                    raise RuntimeError("transport failed during native reset")
                self._execution_failure = None
                self._failure_finalized = False
            self._publish_identity("ready")
            self._raise_if_execution_failed()
        except BaseException as exc:
            self._latch_execution_failure(exc)
            raise

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def set_seed(self, seed):
        self._seed = int(seed)

    def episode_begin(self, seed):
        self.set_seed(seed)
        self._episode_id += 1
        self._episode_start = self._tick
        self.resume()
        return EpisodeHandle(self._episode_id, int(seed), self._tick, {"backend": "libuipc", "num_envs": 1})

    def episode_end(self, reason):
        self.pause()
        return EpisodeSummary(self._episode_id, str(reason), self._tick, self._tick - self._episode_start)

    def raw_model(self):
        return self._model

    def raw_data(self):
        return self._state_0

    def shutdown(self):
        if self._closed:
            return
        self._closed = True
        clean = getattr(self, "_execution_failure", None) is None
        if self._driver is not None:
            clean = self._cleanup_attempt("stop feedback", self._driver.stop_feedback) and clean
        actions = getattr(self, "_actions", None)
        if actions is not None:
            clean = (
                self._cleanup_attempt("close actions", lambda: self._publish_action(actions.close("engine closed")))
                and clean
            )
        source = getattr(self, "_action_source", None)
        if source is not None:
            clean = self._cleanup_attempt("close action transport", source.close) and clean
        snapshots = getattr(self, "_snapshots", None) or getattr(self, "_snapshot_requests", None)
        if snapshots is not None:
            clean = self._cleanup_attempt("close snapshots", snapshots.close) and clean
        identity = getattr(self, "_identity_source", None)
        if identity is not None:
            clean = self._cleanup_attempt("close state identity", identity.close) and clean
        recorder = getattr(self, "_state_recorder", None)
        if recorder is not None:
            # Transport close serializes with in-flight callbacks; sample the
            # final fault flag only after those callbacks can no longer publish.
            clean = clean and getattr(self, "_execution_failure", None) is None
            self._cleanup_attempt("close physical archive", lambda: recorder.close(clean=clean))
        # Each resource is independent: an exception must not prevent releasing
        # the remaining owned resources or touch the shared rclpy context.
        coordinator = self._lifecycle
        if coordinator._executor is not None:
            self._cleanup_attempt("stop executor", coordinator._executor.shutdown)
        if coordinator._spin_thread is not None:
            self._cleanup_attempt("join executor", lambda: coordinator._spin_thread.join(timeout=5))
        if coordinator._node is not None:
            self._cleanup_attempt("destroy lifecycle node", coordinator._node.destroy_node)
        coordinator._node = None
        self._started = False
        self._driver = self._scene = None
        self._model = self._stage = self._state_0 = self._state_1 = None
