# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Independent actuator control, driven by simulation time, never native forward()."""
import math

import numpy as np


def host(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class GripperControlSession:
    """A single explicit command. Successful drive settings persist; failures restore.

    Only the configured actuator is driven. Native follower/mimic joints retain
    their gains, efforts and targets. A stalled result is not a grasp certificate.
    """

    def __init__(self, robot, gripper, physics, action_factory, config, now):
        self.robot, self.physics, self.action_factory = robot, physics, action_factory
        self.config = config
        self.start_time = self.last_time = float(now)
        names = physics.get_dof_names(robot)
        actuator = gripper._joint_control_prim.rsplit('/', 1)[-1]
        if actuator not in names or actuator not in gripper._joint_prim_names:
            raise ValueError('Configured gripper actuator is absent from the articulation')
        if gripper.gripper_type not in ('linear', 'angular'):
            raise ValueError('Only native linear and angular grippers are supported')
        self.units = 'm' if gripper.gripper_type == 'linear' else 'rad'
        self.index = names.index(actuator)
        self.actuator = actuator
        self.indices = np.asarray([self.index], dtype=int)
        finger_index = gripper._joint_prim_names.index(actuator)
        closed = float(gripper.joint_closed_positions[finger_index])
        opened = float(gripper.joint_opened_positions[finger_index])
        self.requested_target = closed + config['opening_ratio'] * (opened - closed)
        lower, upper = physics.get_dof_limits(robot, as_torch=False)
        self.target = float(np.clip(self.requested_target, host(lower)[self.index], host(upper)[self.index]))
        self.initial = self.read_position()
        kp, kd = robot._articulation_view.get_gains()
        self.original_kp = host(kp).reshape(-1)[self.indices].copy()
        self.original_kd = host(kd).reshape(-1)[self.indices].copy()
        self.original_force = host(robot._articulation_view.get_max_efforts()).reshape(-1)[self.indices].copy()
        selected_kp = config['stiffness'] if config['stiffness'] is not None else float(self.original_kp[0])
        selected_kd = config['damping'] if config['damping'] is not None else float(self.original_kd[0])
        if not all(math.isfinite(v) for v in (self.target, self.initial, selected_kp, selected_kd)):
            raise ValueError('Gripper target, position and gains must be finite')
        if config['mode'] == 'position' and selected_kp <= 0:
            raise ValueError('Position control requires positive stiffness; supply it explicitly after native close')
        if config['mode'] == 'velocity' and selected_kd <= 0:
            raise ValueError('Velocity control requires positive damping')
        self.ramp_duration = abs(self.target - self.initial) / config['speed']
        self.previous_position = self.initial
        self.stall_duration = 0.
        self.result = None
        try:
            physics.set_gains(robot, kps=np.asarray([selected_kp if config['mode'] == 'position' else 0.], dtype=self.original_kp.dtype),
                              kds=np.asarray([selected_kd], dtype=self.original_kd.dtype), joint_indices=self.indices)
            physics.set_max_efforts(robot, np.asarray([config['max_effort']], dtype=self.original_force.dtype), joint_indices=self.indices)
        except Exception as exc:
            try:
                self.restore()
            except Exception as cleanup_exc:
                raise RuntimeError(f'Gripper setup failed: {exc}; cleanup failed: {cleanup_exc}') from exc
            raise

    def read_position(self):
        return float(host(self.physics.get_joint_positions(self.robot, as_torch=False)).reshape(-1)[self.index])

    def action(self, position, velocity):
        self.physics.apply_action(self.robot, self.action_factory(
            joint_positions=np.asarray([position]), joint_velocities=np.asarray([velocity]),
            joint_indices=self.indices))

    def restore(self):
        errors = []
        try:
            self.physics.set_gains(self.robot, kps=self.original_kp, kds=self.original_kd, joint_indices=self.indices)
        except Exception as exc:
            errors.append(f'gains: {exc}')
        try:
            self.physics.set_max_efforts(self.robot, self.original_force, joint_indices=self.indices)
        except Exception as exc:
            errors.append(f'effort: {exc}')
        if errors:
            raise RuntimeError('; '.join(errors))

    def abort(self):
        """Stop the actuator before restoring the settings that preceded this command."""
        try:
            self.action(self.read_position(), 0.)
        finally:
            self.restore()

    def step(self, now):
        if self.result is not None:
            return self.result
        now = float(now)
        if not math.isfinite(now) or now < self.last_time:
            raise ValueError('ControlGripper requires monotonic finite simulation time')
        elapsed, dt = now - self.start_time, now - self.last_time
        current = self.read_position()
        if not math.isfinite(current):
            raise ValueError('Nonfinite actual gripper position')
        error = self.target - current
        if abs(current - self.previous_position) <= self.config['tolerance'] * .1:
            self.stall_duration += dt
        else:
            self.stall_duration = 0.
        self.previous_position, self.last_time = current, now
        if self.config['mode'] == 'position':
            distance = min(abs(self.target - self.initial), self.config['speed'] * elapsed)
            position = self.initial + math.copysign(distance, self.target - self.initial)
            velocity = 0.
        else:
            position = current
            velocity = math.copysign(self.config['speed'], error) if abs(error) > self.config['tolerance'] else 0.
        self.action(position, velocity)
        minimum_elapsed = elapsed >= max(self.config['minimum_duration'], self.ramp_duration)
        reached = minimum_elapsed and abs(error) <= self.config['tolerance']
        stalled = (minimum_elapsed and self.config['completion'] == 'settled'
                   and self.stall_duration >= self.config['stall_time'])
        timeout = elapsed >= self.config['timeout']
        if reached or stalled or timeout:
            reason = 'target' if reached else 'stalled' if stalled else 'timeout'
            if reason == 'target':
                self.action(self.target, 0.)
            elif reason == 'timeout':
                self.abort()
            # On a requested contact stop, keep the explicit position/velocity
            # target so the held object's normal force is not silently removed.
            self.result = dict(success=reason != 'timeout', reason=reason,
                arm=self.config['arm'], mode=self.config['mode'], actuator=self.actuator,
                units=self.units, target_position=self.target, requested_target=self.requested_target,
                target_clamped=abs(self.target - self.requested_target) > 1e-12,
                actual_position=current, position_error=abs(error), elapsed_seconds=elapsed,
                max_effort=self.config['max_effort'], settings_restored=reason == 'timeout')
        return self.result
