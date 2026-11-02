# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0

"""Finite-deadline pose control for an acceleration-limited four-steer base.

The optimizer shoots two velocity intervals through the steering response and
asset wheel ramp. Only the first command is executed; the next observation
restarts prediction from measured pose/velocity and the actual filtered targets.
Terminal wheel speeds and kinematic disagreement are penalized, so opposed
rolling wheels cannot masquerade as a stationary body in the preview model.

This is an approximate kinematic predictor, not a substitute for PhysX contact
or a guarantee that an infeasible one-second target can be reached.
"""

import numpy as np


class SwervePosePlanner:
    def __init__(self, limits, positions, radius, steer_rate, steer_tau, dt=1 / 30, max_speed=1.0, max_yaw_rate=1.57):
        self.limits = np.asarray(limits, dtype=float).reshape(4, 2).copy()
        if np.any(np.isnan(self.limits)) or np.any(self.limits[:, 0] > self.limits[:, 1]):
            raise ValueError("Invalid steering limits")
        self._has_narrow_limits = bool(np.any(self.limits[:, 1] - self.limits[:, 0] < np.pi))
        self.dt = float(dt)
        self.max_speed = float(max_speed)
        self.max_yaw_rate = float(max_yaw_rate)
        self.radius = radius
        self.steer_rate = steer_rate
        self.steer_tau = np.broadcast_to(np.asarray(steer_tau, dtype=float), (4,)).copy()
        if not np.all(np.isfinite(self.steer_tau)) or np.any(self.steer_tau <= 0):
            raise ValueError("Steering prediction requires positive live damping/stiffness")
        self.rx, self.ry = np.asarray(positions, dtype=float).reshape(4, 2).T
        self.lever = np.sum(self.rx**2 + self.ry**2)
        self.last_command = np.zeros(3)
        self.yaw_weight = 0.35

    def wheels(self, u, angles):
        u = np.atleast_2d(u)
        wx = u[:, 0, None] - u[:, 2, None] * self.ry
        wy = u[:, 1, None] + u[:, 2, None] * self.rx
        speed = np.hypot(wx, wy)
        heading = np.arctan2(wy, wx)
        first = np.ceil((self.limits[:, 0] - heading) / np.pi)
        last = np.floor((self.limits[:, 1] - heading) / np.pi)
        turns = np.minimum(np.maximum(np.rint((angles - heading) / np.pi), first), last)
        theta = heading + turns * np.pi
        v = speed * np.where(turns % 2 == 0, 1.0, -1.0)
        # Narrow joints may have no exact equivalent direction. Use the legal
        # boundary with least lateral mismatch, matching the legacy allocator.
        if self._has_narrow_limits and np.any(first > last):
            lo, hi = np.where(np.isfinite(self.limits), self.limits, 0.0).T
            lo_error = np.abs(wx * np.sin(lo) - wy * np.cos(lo))
            hi_error = np.abs(wx * np.sin(hi) - wy * np.cos(hi))
            choose_lo = (lo_error < hi_error) | ((lo_error == hi_error) & (np.abs(lo - angles) <= np.abs(hi - angles)))
            boundary = np.where(choose_lo, lo, hi)
            theta = np.where(first > last, boundary, theta)
            v = np.where(first > last, wx * np.cos(boundary) + wy * np.sin(boundary), v)
        theta = np.where(speed < 1e-7, np.minimum(np.maximum(angles, self.limits[:, 0]), self.limits[:, 1]), theta)
        scale = np.maximum(np.max(np.abs(v), axis=1) / self.max_speed, 1.0)
        return theta, v / scale[:, None]

    def steering_target(self, target, theta, speed, accel):
        # Bound lateral wheel-vector acceleration while already rolling.
        rate = np.minimum(self.steer_rate, accel * self.radius / np.maximum(np.abs(speed), 0.02))
        travel = rate * self.steer_tau
        return np.minimum(
            np.maximum(theta + np.minimum(np.maximum(target - theta, -travel), travel), self.limits[:, 0]),
            self.limits[:, 1],
        )

    def forward(self, theta, speed):
        wx = speed * np.cos(theta)
        wy = speed * np.sin(theta)
        return np.stack(
            [np.mean(wx, axis=-1), np.mean(wy, axis=-1), np.sum(-self.ry * wx + self.rx * wy, axis=-1) / self.lever],
            axis=-1,
        )

    def predict(self, commands, state, horizon, accel):
        n = len(commands)
        theta = np.broadcast_to(state["steer"], (n, 4)).copy()
        speed = np.broadcast_to(state["filtered"], (n, 4)).copy() * self.radius
        xy = np.zeros((n, 2))
        yaw = np.zeros(n)
        scrub = np.zeros(n)
        observed = np.asarray(state["twist"])
        mismatch = observed - self.forward(state["steer"], np.asarray(state["drive"]) * self.radius)
        steps = max(1, int(np.ceil(horizon / self.dt)))
        dt = horizon / steps
        steer_fraction = 1 - np.exp(-dt / self.steer_tau)
        for j in range(steps):
            c, s = np.cos(yaw), np.sin(yaw)
            cmd = commands[:, :3] if j < max(1, steps // 2) else commands[:, 3:]
            body = np.stack([c * cmd[:, 0] + s * cmd[:, 1], -s * cmd[:, 0] + c * cmd[:, 1], cmd[:, 2]], axis=-1)
            target, goal = self.wheels(body, theta)
            goal *= np.cos(target - theta)
            target = self.steering_target(target, theta, speed, accel)
            old_theta = theta.copy()
            old_speed = speed.copy()
            theta += np.minimum(
                np.maximum((target - theta) * steer_fraction, -self.steer_rate * dt), self.steer_rate * dt
            )
            speed += np.minimum(np.maximum(goal - speed, -accel * self.radius * dt), accel * self.radius * dt)
            tm = 0.5 * (old_theta + theta)
            vm = 0.5 * (old_speed + speed)
            wx = vm * np.cos(tm)
            wy = vm * np.sin(tm)
            twist = np.stack(
                [
                    np.mean(wx, axis=-1),
                    np.mean(wy, axis=-1),
                    np.sum(-self.ry * wx + self.rx * wy, axis=-1) / self.lever,
                ],
                axis=-1,
            )
            scrub += dt * np.mean(
                (wx - twist[:, 0, None] + twist[:, 2, None] * self.ry) ** 2
                + (wy - twist[:, 1, None] - twist[:, 2, None] * self.rx) ** 2,
                axis=1,
            )
            twist += mismatch * np.exp(-(j + 0.5) * dt / 0.08)
            mid = yaw + 0.5 * twist[:, 2] * dt
            cm, sm = np.cos(mid), np.sin(mid)
            xy[:, 0] += (cm * twist[:, 0] - sm * twist[:, 1]) * dt
            xy[:, 1] += (sm * twist[:, 0] + cm * twist[:, 1]) * dt
            yaw += twist[:, 2] * dt
        return np.column_stack([xy, yaw, speed, np.sqrt(scrub / horizon + 1e-12)])

    def solve(self, error, state, remaining, acceleration):
        """Return legal steer positions and raw wheel angular-velocity goals.

        error and twist are expressed in the current chassis frame. No new
        chunk may reset filtered velocity or assume that the base has stopped.
        The remaining time is the original deadline, never silently extended.
        A negative remainder means a subsequent hold, outside that deadline.
        """
        if not np.isfinite(acceleration) or acceleration <= 0:
            raise ValueError("Pose preview requires a positive asset wheel acceleration")
        if not np.all(np.isfinite(error)) or any(
            not np.all(np.isfinite(state[k])) for k in ("steer", "drive", "filtered", "twist")
        ):
            raise ValueError("Non-finite chassis feedback")
        horizon = max(self.dt, remaining) if remaining > 0 else 0.30
        error = np.array(error, float)
        # Tight pose deadbands apply only to settled axes; braking still runs.
        if np.linalg.norm(error[:2]) < 0.001 and np.linalg.norm(state["twist"][:2]) < 0.005:
            error[:2] = 0
        if abs(error[2]) < np.deg2rad(0.1) and abs(state["twist"][2]) < 0.005:
            error[2] = 0
        if remaining <= 0:
            # After the fixed deadline, use a slower hold servo. Continuing
            # deadline catch-up gains makes long, infeasible moves ring while
            # the wheels are still braking; the deadline score stays unchanged.
            u = error / 0.30
            u[:2] /= max(1.0, np.linalg.norm(u[:2]) / self.max_speed)
            u[2] = np.clip(u[2], -self.max_yaw_rate, self.max_yaw_rate)
            theta, vel = self.wheels(u, state["steer"])
            vel *= np.cos(theta - state["steer"])
            self.last_command = u.copy()
            return (
                self.steering_target(
                    theta[0], np.asarray(state["steer"]), np.asarray(state["filtered"]) * self.radius, acceleration
                ),
                vel[0] / self.radius,
            )
        inactive = (np.abs(error) < np.array([0.0005, 0.0005, np.deg2rad(0.05)])) & (
            np.abs(state["twist"]) < np.array([0.005, 0.005, 0.01])
        )

        def bound(commands):
            commands = np.asarray(commands).reshape(-1, 2, 3).copy()
            commands[:, :, :2] /= np.maximum(
                1.0, np.linalg.norm(commands[:, :, :2], axis=-1, keepdims=True) / self.max_speed
            )
            commands[:, :, 2] = np.clip(commands[:, :, 2], -self.max_yaw_rate, self.max_yaw_rate)
            commands[:, :, inactive] = 0.0
            return commands.reshape(-1, 6)

        # Multiple starts avoid the flat Jacobian of a saturated wheel ramp:
        # accelerate/brake, cruise, and stronger acceleration with reversal.
        seed = error / horizon
        u = bound(np.array([np.r_[2 * seed, np.zeros(3)], np.r_[seed, seed], np.r_[4 * seed, -seed]]))
        target = np.r_[error, np.zeros(5)]
        weights = np.array([1.0, 1.0, self.yaw_weight, 0.06, 0.06, 0.06, 0.06, 0.05])
        eps = np.array([0.01, 0.01, 0.03, 0.01, 0.01, 0.03])
        # Batched finite differences and a bounded line search keep the cost
        # predictable at 30 Hz. The zero-length step preserves the best iterate.
        for _ in range(6):
            previous = u
            candidates = np.repeat(u[:, None, :], 7, axis=1)
            candidates[:, 1:] += np.diag(eps)
            predicted = (self.predict(candidates.reshape(-1, 6), state, horizon, acceleration) * weights).reshape(
                -1, 7, len(weights)
            )
            jac = np.swapaxes((predicted[:, 1:] - predicted[:, 0, None]) / eps[None, :, None], 1, 2)
            residual = target * weights - predicted[:, 0]
            jt = np.swapaxes(jac, 1, 2)
            update = np.linalg.solve(jt @ jac + 0.00001 * np.eye(6), (jt @ residual[:, :, None])).squeeze(-1)
            update = np.clip(update, [-0.5, -0.5, -1.0, -0.5, -0.5, -1.0], [0.5, 0.5, 1.0, 0.5, 0.5, 1.0])
            choices = bound(u[:, None, :] + np.array([1.0, 0.5, 0.25, 0.0])[None, :, None] * update[:, None, :])
            ends = self.predict(choices, state, horizon, acceleration)
            costs = np.sum(((ends - target) * weights) ** 2, axis=1).reshape(-1, 4)
            u = choices.reshape(-1, 4, 6)[np.arange(len(u)), np.argmin(costs, axis=1)]
            # The solve is deterministic. If every start rejects its step,
            # repeating identical rollouts cannot improve this observation.
            if np.array_equal(u, previous):
                break
        predicted = self.predict(u, state, horizon, acceleration)
        u = u[np.argmin(np.sum(((predicted - target) * weights) ** 2, axis=1))]
        self.last_command = u[:3].copy()
        theta, vel = self.wheels(u[:3], state["steer"])
        vel *= np.cos(theta - state["steer"])
        return (
            self.steering_target(
                theta[0], np.asarray(state["steer"]), np.asarray(state["filtered"]) * self.radius, acceleration
            ),
            vel[0] / self.radius,
        )
