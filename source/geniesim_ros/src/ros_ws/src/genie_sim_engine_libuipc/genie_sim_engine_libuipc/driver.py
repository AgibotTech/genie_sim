# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure SolverIPC adapter for command-driven kinematic robot scenes."""

from __future__ import annotations

from typing import Any, Optional
import math
import time

import numpy as np
import warp as wp


from genie_sim_engine_libuipc.config import IPCConfig
from genie_sim_engine_libuipc.geometry import select_robot_collision_bodies
from genie_sim_engine_libuipc.feedback import JointFeedbackConfig, JointFeedbackSelector
from genie_sim_engine_libuipc.cpu.worker import CpuMuJoCoConfig, CpuJointWorker
from genie_sim_engine_libuipc.action_sequence import topic_from_scene


@wp.kernel
def _write_cpu_body_state(
    rows: wp.array[int],
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    index = wp.tid()
    row = rows[index]
    body_q[row] = poses[index]
    body_qd[row] = velocities[index]


class IPCDriver:
    """Drive kinematic robot links through FK and solve all contact in IPC."""

    def __init__(self, *, physics_hz: float, scene_cfg: Optional[dict] = None):
        self._physics_hz = float(physics_hz)
        self._scene_cfg = dict(scene_cfg or {})
        self._config = IPCConfig.from_scene(self._scene_cfg)
        self._solver = None
        self._target_buffer = None
        self._robot_prefix = ""
        self._feedback_config = JointFeedbackConfig.from_scene(self._scene_cfg)
        self._feedback = None
        self._feedback_node = None
        self._feedback_subscription = None
        self._selected_buffer = None
        self._feedback_log_wall = 0.0
        self._cpu_config = CpuMuJoCoConfig.from_scene(self._scene_cfg)
        action_topic = topic_from_scene(self._scene_cfg, "action_sequence")
        topic_from_scene(self._scene_cfg, "observation_snapshot")
        if action_topic is not None and (self._cpu_config is None or self._cpu_config.synchronization != "barrier"):
            raise ValueError("libuipc.action_sequence requires CPU barrier synchronization")
        if self._cpu_config is not None and self._cpu_config.synchronization == "barrier":
            if not math.isclose(
                self._cpu_config.physics_hz,
                self._physics_hz * self._cpu_config.sync_substeps,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("barrier cpu_mujoco.physics_hz must equal IPC physics_hz * sync_substeps")
        self._cpu_coupling_failure = None
        self._cpu_sync_sequence = 0
        self._cpu_worker = None
        self._cpu_started = False
        self._cpu_failure_logged = False
        self._cpu_snapshot = {}
        self._cpu_needs_restart = False

    def register_custom_attributes(self, builder: Any) -> None:
        """Register the attributes consumed by SolverIPC."""
        from genie_sim_engine_libuipc.solver import SolverIPC

        SolverIPC.register_custom_attributes(builder)

    def prepare_model(self, model: Any, logger: Any, robot_prefix: str = "") -> None:
        """Validate that the scene plugin made the robot links kinematic."""
        import newton

        self._robot_prefix = robot_prefix
        flags = model.body_flags.numpy()
        prefix = f"/{robot_prefix}/" if robot_prefix else ""
        robot_rows = [i for i, label in enumerate(model.body_label) if not prefix or str(label).startswith(prefix)]
        dynamic = [i for i in robot_rows if not int(flags[i]) & int(newton.BodyFlags.KINEMATIC)]
        if dynamic:
            raise RuntimeError(
                "IPC command-only mode requires every robot body to be KINEMATIC; " f"non-kinematic rows: {dynamic[:8]}"
            )
        logger.info(f"[ipc-adapter] validated {len(robot_rows)} kinematic robot bodies")

    def build_solver(self, model: Any, logger: Any) -> Any:
        """Construct SolverIPC with scene-YAML-only tuning."""
        from genie_sim_engine_libuipc.solver import SolverIPC

        step_dt = 1.0 / self._physics_hz
        active_names = self._config.collision_link_names
        prefix = f"/{self._robot_prefix}/" if self._robot_prefix else ""
        options = self._config.solver_options(
            urdf_body_prefix=prefix, length_scale=getattr(model, "ipc_length_scale", 1.0)
        )
        if active_names:
            if not prefix:
                raise ValueError("IPC collision_link_names requires an explicit robot_prefix in prepare_model")
            rows = select_robot_collision_bodies(
                model.body_label, robot_prefix=self._robot_prefix, collision_link_names=active_names
            )
            labels = [str(model.body_label[row]) for row in sorted(rows)]
            missing = set(active_names) - {label.rsplit("/", 1)[-1] for label in labels}
            if missing:
                raise ValueError(f"IPC collision_link_names not found under {prefix}: {sorted(missing)}")
            # SolverIPC matches the entire suffix, not the leaf name. Keep the
            # namespace at the robot root so free props never enter its filter.
            options["gripper_link_names"] = tuple(label[len(prefix) :] for label in labels)
        if not self._config.robot_static_collision:
            import newton

            # Keep geometry collidable with props/cloth and leave the CPU model alone.
            bodies = model.shape_body.numpy()
            flags = model.shape_flags.numpy()
            types = model.shape_type.numpy()
            options["gripper_noncolliding_static_shape_labels"] = tuple(
                str(model.shape_label[i])
                for i in range(model.shape_count)
                if bodies[i] < 0
                and types[i] != int(newton.GeoType.PLANE)
                and int(flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES)
            )
        self._solver = SolverIPC(model, dt=step_dt, **options)
        logger.info(
            f"[ipc-adapter] SolverIPC dt={step_dt:g}s contact_d_hat={options['contact_d_hat']:g} "
            f"cloth_thickness={options['cloth_thickness']:g} gripper_links={len(active_names)} "
            f"robot_static_collision={self._config.robot_static_collision} "
            f"filtered_static_shapes={len(options.get('gripper_noncolliding_static_shape_labels', ()))}"
        )
        return self._solver

    def init_target_buffer(self, model: Any, logger: Any) -> None:
        """Seed the command target buffer from the initialized joint pose."""
        self._target_buffer = wp.empty_like(model.joint_q)
        self._target_buffer.assign(model.joint_q)
        self._initial_q = model.joint_q.numpy().copy()
        self._logger = logger
        logger.info(f"[ipc-adapter] target buffer allocated ({model.joint_coord_count} coordinates)")

    def target_buffer(self) -> Optional[wp.array]:
        """Return the joint-coordinate target buffer."""
        return self._target_buffer

    def bind_feedback(self, model: Any, jindex: Any, joint_names: list[str]) -> None:
        """Bind measured input to the engine's exposed scalar robot joints."""
        if self._feedback_config is None and self._cpu_config is None:
            return
        names = set(joint_names)
        lower = model.joint_limit_lower.numpy()
        upper = model.joint_limit_upper.numpy()
        bindings = {}
        velocity_indices = {}
        for joint in jindex.slices():
            if joint.name not in names or joint.q_count != 1 or joint.dof_count != 1:
                continue
            if joint.name in bindings:
                raise ValueError(f"ambiguous measured joint name: {joint.name}")
            bindings[joint.name] = (joint.q_start, float(lower[joint.qd_start]), float(upper[joint.qd_start]))
            velocity_indices[joint.name] = joint.qd_start
        if not bindings:
            raise ValueError("libuipc.joint_feedback requires exposed scalar robot joints")
        self._selected_buffer = wp.empty_like(model.joint_q)
        self._selected_buffer.assign(model.joint_q)
        if self._cpu_config is not None:
            payload = getattr(model, "ipc_cpu_model_bytes", None)
            if not payload:
                raise ValueError("cpu_mujoco requires a declarative IPC scene with compiled robot/static geometry")
            self._cpu_q_indices = np.asarray([binding[0] for binding in bindings.values()], dtype=int)
            self._cpu_qd_indices = np.asarray([velocity_indices[name] for name in bindings], dtype=int)
            self._cpu_payload = payload
            self._cpu_names = tuple(bindings)
            self._cpu_selected_q = self._initial_q.copy()
            self._cpu_selected_qd = np.zeros(model.joint_qd.size, dtype=np.float32)
            self._selected_velocity_buffer = wp.zeros_like(model.joint_qd)
            if not self._robot_prefix.strip("/"):
                raise ValueError("CPU link input requires an explicit robot prefix")
            prefix = f"/{self._robot_prefix.strip('/')}/"
            self._cpu_body_indices = np.asarray(
                [i for i, label in enumerate(model.body_label) if str(label).startswith(prefix)], dtype=np.int32
            )
            if not self._cpu_body_indices.size:
                raise ValueError("CPU link input requires robot body labels in the configured namespace")
            # Newton's MuJoCo exporter replaces slashes in exact body labels.
            # Ambiguous exported names must fail instead of silently misbinding.
            self._cpu_body_names = tuple(str(model.body_label[i]).replace("/", "_") for i in self._cpu_body_indices)
            if len(set(self._cpu_body_names)) != len(self._cpu_body_names):
                raise ValueError("CPU robot body labels collide after MuJoCo name sanitization")
            self._cpu_body_rows = wp.array(self._cpu_body_indices, dtype=wp.int32, device=model.device)
            self._cpu_body_pose_buffer = wp.empty(len(self._cpu_body_names), dtype=wp.transform, device=model.device)
            self._cpu_body_velocity_buffer = wp.zeros(
                len(self._cpu_body_names), dtype=wp.spatial_vector, device=model.device
            )
            self._cpu_selected_body_q = None
            self._cpu_selected_body_qd = np.zeros((len(self._cpu_body_names), 6), dtype=np.float32)
            self._cpu_proxy_indices = []
            for name in self._cpu_config.dynamic_proxies:
                rows = [i for i, label in enumerate(model.body_label) if str(label) == name]
                if len(rows) != 1 or rows[0] in self._cpu_body_indices:
                    raise ValueError(f"CPU dynamic proxy requires one exact nonrobot body: {name}")
                self._cpu_proxy_indices.append(rows[0])
            self._cpu_proxy_indices = np.asarray(self._cpu_proxy_indices, dtype=np.int32)
            self._cpu_worker = CpuJointWorker(
                payload,
                self._cpu_config,
                tuple(bindings),
                self._initial_q[self._cpu_q_indices],
                body_names=self._cpu_body_names,
            )
        else:
            self._feedback = JointFeedbackSelector(self._feedback_config, bindings, self._initial_q)

    def start_cpu_worker(self) -> None:
        """Start owned dynamics once; usable without ROS by engine benchmarks."""
        if self._cpu_worker is not None and not self._cpu_started:
            if self._cpu_needs_restart:
                self._cpu_worker = CpuJointWorker(
                    self._cpu_payload,
                    self._cpu_config,
                    self._cpu_names,
                    self._cpu_selected_q[self._cpu_q_indices],
                    body_names=self._cpu_body_names,
                )
                self._cpu_snapshot = {}
                self._cpu_failure_logged = False
                self._cpu_needs_restart = False
                self._cpu_coupling_failure = None
                self._cpu_sync_sequence = 0
            self._cpu_worker.start()
            self._cpu_started = True
            self._logger.info(
                f"[cpu-mujoco] started execution={self._cpu_config.execution} "
                f"physics_hz={self._cpu_config.physics_hz:g} joints={len(self._cpu_worker.names)}"
            )

    def start_feedback(self, node: Any) -> None:
        """Attach measured input to the existing lifecycle node and executor."""
        if (self._feedback_config is None and self._cpu_config is None) or self._feedback_subscription is not None:
            return
        if (self._feedback is None and self._cpu_worker is None) or node is None:
            raise RuntimeError("measured joint feedback requires initialized joints and a running lifecycle node")
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import JointState

        topic = "/joint_command" if self._cpu_config is not None else self._feedback_config.topic
        if node.resolve_topic_name(topic) == node.resolve_topic_name("/joint_states"):
            raise ValueError("measured joint feedback input resolves to the engine's joint_states output")
        if self._cpu_config is not None:
            self.start_cpu_worker()
            receiver = self._cpu_worker.receive
        else:
            self._feedback.reset(self._initial_q)
            receiver = self._feedback.receive
        self._feedback_subscription = node.create_subscription(
            JointState,
            topic,
            lambda msg: receiver(msg.name, msg.position),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST),
        )
        self._feedback_node = node
        if self._cpu_config is not None:
            self._logger.info("[cpu-mujoco] subscribed /joint_command on background ROS executor (latest only)")
            return
        self._logger.info(
            f"[joint-feedback] topic={topic} joints={len(self._feedback.bindings)} "
            f"timeout_s={self._feedback_config.timeout_s:g} stale_policy={self._feedback_config.stale_policy}"
        )

    def stop_feedback(self) -> None:
        """Detach only the owned subscription, preserving the shared ROS context."""
        try:
            if self._feedback_node is not None and self._feedback_subscription is not None:
                self._feedback_node.destroy_subscription(self._feedback_subscription)
        finally:
            self._feedback_subscription = None
            self._feedback_node = None
            if self._cpu_worker is not None:
                self._cpu_worker.close()
                self._cpu_started = False
                self._cpu_needs_restart = True

    def reset_inputs(self) -> None:
        """Discard pre-reset targets and measurements after native state recovery."""
        self._target_buffer.assign(self._initial_q)
        if self._feedback is not None:
            self._feedback.reset(self._initial_q)
        if self._cpu_config is not None:
            self._cpu_selected_q = self._initial_q.copy()
            self._cpu_selected_qd.fill(0)
            self._cpu_selected_body_q = None
            self._cpu_selected_body_qd.fill(0)
            self._cpu_snapshot = {}
            self._cpu_failure_logged = False
            self._cpu_coupling_failure = None
            self._cpu_sync_sequence = 0

    def _barrier_snapshot(self, state_in, sim_dt, completed_sim_time, *, targets=None):
        """Accept a complete CPU batch before exposing any state to native IPC."""
        if self._cpu_coupling_failure is not None:
            raise RuntimeError("CPU/IPC coupling failed; reset or create a new owner") from self._cpu_coupling_failure
        if not self._cpu_started:
            raise RuntimeError("CPU barrier worker is not started")
        if not math.isclose(sim_dt, 1.0 / self._physics_hz, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("CPU barrier requires the configured IPC dt")
        if (
            isinstance(completed_sim_time, bool)
            or not isinstance(completed_sim_time, (int, float))
            or not math.isfinite(completed_sim_time)
            or completed_sim_time < 0
        ):
            raise ValueError("CPU barrier requires completed_sim_time from the engine clock")
        try:
            self._cpu_worker.check_health()
            frame = dict(completed_sim_time=completed_sim_time)
            if targets is not None:
                frame["targets"] = np.asarray(targets, dtype=float).copy()
            if len(self._cpu_proxy_indices):
                # Raw completed native body rows already use body origin poses
                # and world COM twists. Public authored-root readback has offsets.
                frame["proxy_poses"] = state_in.body_q.numpy()[self._cpu_proxy_indices].copy()
                frame["proxy_velocities"] = state_in.body_qd.numpy()[self._cpu_proxy_indices].copy()
            snapshot = self._cpu_worker.advance_sync(**frame)
            sequence = self._cpu_sync_sequence + 1
            if (
                not isinstance(snapshot, dict)
                or type(snapshot.get("sync_sequence")) is not int
                or snapshot["sync_sequence"] != sequence
                or type(snapshot.get("steps")) is not int
                or snapshot["steps"] != sequence * self._cpu_config.sync_substeps
            ):
                raise RuntimeError("invalid completed CPU barrier snapshot")
            sim_time = snapshot.get("sim_time")
            if (
                isinstance(sim_time, bool)
                or not isinstance(sim_time, (int, float))
                or not math.isfinite(sim_time)
                or not math.isclose(sim_time, completed_sim_time + sim_dt, rel_tol=0.0, abs_tol=1e-9)
            ):
                raise RuntimeError("invalid completed CPU barrier snapshot time")
            for key, shape in (
                ("positions", (len(self._cpu_names),)),
                ("velocities", (len(self._cpu_names),)),
                ("body_poses", (len(self._cpu_body_names), 7)),
                ("body_velocities", (len(self._cpu_body_names), 6)),
            ):
                value = np.asarray(snapshot.get(key))
                if value.shape != shape or value.dtype.kind not in "fiu" or not np.isfinite(value).all():
                    raise RuntimeError(f"invalid completed CPU barrier snapshot {key}")
            if not np.allclose(np.linalg.norm(snapshot["body_poses"][:, 3:], axis=1), 1.0, rtol=0.0, atol=1e-5):
                raise RuntimeError("invalid completed CPU barrier snapshot quaternion")
            self._cpu_sync_sequence = sequence
            return snapshot
        except BaseException as exc:
            self._cpu_coupling_failure = exc
            raise

    def substep(
        self,
        model: Any,
        state_in: Any,
        state_out: Any,
        control: Any,
        sim_dt: float,
        *,
        completed_sim_time: Optional[float] = None,
        targets=None,
    ) -> None:
        """Advance one coupled tick; any post-ACK failure requires recovery."""
        barrier = self._cpu_config is not None and self._cpu_config.synchronization == "barrier"
        if barrier and self._cpu_coupling_failure is not None:
            raise RuntimeError("CPU/IPC coupling failed; reset or create a new owner") from self._cpu_coupling_failure
        previous_sequence = self._cpu_sync_sequence
        try:
            self._substep(
                model, state_in, state_out, control, sim_dt, completed_sim_time=completed_sim_time, targets=targets
            )
        except BaseException as exc:
            if barrier and self._cpu_sync_sequence != previous_sequence:
                # CPU has advanced: SDK scatter, native advance and output
                # writes must all finish before another coupling transaction.
                self._cpu_coupling_failure = exc
            raise

    def _substep(
        self,
        model: Any,
        state_in: Any,
        state_out: Any,
        control: Any,
        sim_dt: float,
        *,
        completed_sim_time: Optional[float] = None,
        targets=None,
    ) -> None:
        """Use native CPU link targets, or legacy FK, then advance IPC."""
        import newton

        if self._target_buffer is None:
            raise RuntimeError("IPC target buffer was not initialized")
        selected = self._target_buffer
        selected_velocity = None
        barrier = self._cpu_config is not None and self._cpu_config.synchronization == "barrier"
        if barrier and (self._cpu_worker is None or not self._cpu_started):
            raise RuntimeError("CPU barrier worker is not started")
        if self._cpu_worker is not None:
            # Hold positions but report zero velocity when the CPU sample is stale.
            # Preserve every nonrobot coordinate/DOF; CPU owns robot rows only.
            self._cpu_selected_q = model.joint_q.numpy().copy()
            self._cpu_selected_qd = model.joint_qd.numpy().copy()
            self._cpu_selected_qd[self._cpu_qd_indices] = 0.0
            self._cpu_selected_body_qd.fill(0.0)
            if self._cpu_selected_body_q is None:
                # Before the worker starts, hold the engine's initialized pose.
                self._cpu_selected_body_q = state_in.body_q.numpy()[self._cpu_body_indices].copy()
            now = time.monotonic()
            if barrier:
                snapshot = self._barrier_snapshot(state_in, sim_dt, completed_sim_time, targets=targets)
                self._cpu_selected_q[self._cpu_q_indices] = snapshot["positions"]
                self._cpu_selected_qd[self._cpu_qd_indices] = snapshot["velocities"]
                self._cpu_selected_body_q = snapshot["body_poses"]
                self._cpu_selected_body_qd[:] = snapshot["body_velocities"]
                self._cpu_snapshot = snapshot
            if self._cpu_started and not barrier:
                healthy = True
                try:
                    self._cpu_worker.check_health()
                except RuntimeError as exc:
                    healthy = False
                    if not self._cpu_failure_logged:
                        self._logger.error(f"[cpu-mujoco] {exc}; IPC holds last valid joints")
                        self._cpu_failure_logged = True
                snapshot = self._cpu_worker.snapshot() if healthy else None
                if snapshot is not None and now - snapshot["wall_time"] <= self._cpu_config.state_timeout_s:
                    self._cpu_selected_q[self._cpu_q_indices] = snapshot["positions"]
                    self._cpu_selected_qd[self._cpu_qd_indices] = snapshot["velocities"]
                    self._cpu_selected_body_q = snapshot["body_poses"]
                    self._cpu_selected_body_qd[:] = snapshot["body_velocities"]
                    self._cpu_snapshot = snapshot
                if now - self._feedback_log_wall >= 1.0:
                    stats = self._cpu_snapshot
                    age = now - stats.get("wall_time", now)
                    self._logger.info(
                        f"[cpu-mujoco] steps={stats.get('steps', 0)} "
                        f"received={self._cpu_worker.received} rejected={self._cpu_worker.rejected} "
                        f"age_ms={age * 1000:.2f} contacts={stats.get('contacts', 0)}"
                    )
                    self._feedback_log_wall = now
            self._selected_buffer.assign(self._cpu_selected_q)
            self._selected_velocity_buffer.assign(self._cpu_selected_qd)
            selected = self._selected_buffer
            selected_velocity = self._selected_velocity_buffer
            self._cpu_body_pose_buffer.assign(self._cpu_selected_body_q)
            self._cpu_body_velocity_buffer.assign(self._cpu_selected_body_qd)
            wp.launch(
                _write_cpu_body_state,
                dim=len(self._cpu_body_indices),
                inputs=[self._cpu_body_rows, self._cpu_body_pose_buffer, self._cpu_body_velocity_buffer],
                outputs=[state_in.body_q, state_in.body_qd],
                device=model.device,
            )
        elif self._feedback is not None:
            self._selected_buffer.assign(self._feedback.select(self._target_buffer.numpy()))
            selected = self._selected_buffer
            now = time.monotonic()
            if now - self._feedback_log_wall >= 1.0:
                stats = self._feedback.snapshot()
                counts = {
                    kind: list(stats["selection"].values()).count(kind) for kind in ("measured", "hold", "command")
                }
                self._logger.info(
                    f"[joint-feedback] received={stats['received']} mapped={stats['mapped']} "
                    f"rejected={stats['rejected']} ignored={stats['ignored']} stale={stats['stale']} selection={counts} "
                    f"last_rejection={stats['last_rejection']!r}"
                )
                self._feedback_log_wall = now
        model.joint_q.assign(selected)
        if selected_velocity is not None:
            model.joint_qd.assign(selected_velocity)
        if self._cpu_worker is None:
            newton.eval_fk(
                model,
                selected,
                model.joint_qd,
                state_in,
                body_flag_filter=int(newton.BodyFlags.KINEMATIC),
            )
        self._solver.step(state_in, state_out, control, None, sim_dt)
        if state_out.joint_q is not None:
            if self._cpu_worker is not None:
                output_q = state_out.joint_q.numpy()
                output_q[self._cpu_q_indices] = self._cpu_selected_q[self._cpu_q_indices]
                state_out.joint_q.assign(output_q)
            else:
                state_out.joint_q.assign(selected)
        if state_out.joint_qd is not None:
            if self._cpu_worker is not None:
                output_qd = state_out.joint_qd.numpy()
                output_qd[self._cpu_qd_indices] = self._cpu_selected_qd[self._cpu_qd_indices]
                state_out.joint_qd.assign(output_qd)
            else:
                state_out.joint_qd.zero_()


__all__ = ["IPCDriver"]
