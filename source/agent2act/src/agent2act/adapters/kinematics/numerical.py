# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Bounded local damped least-squares IK; no scene geometry or task knowledge."""

import json
import math
import random

from agent2act.adapters.telemetry import record_artifact
from agent2act.domain.errors import LocalIKConvergenceError
from agent2act.domain.models import ContractError

from .urdf import UrdfKinematics, multiply


def pose_error(target: list[float], current: list[float]) -> list[float]:
    q = multiply(target[3:], [-v for v in current[3:6]] + [current[6]])
    if q[3] < 0:
        q = [-v for v in q]
    norm = math.hypot(*q[:3])
    scale = 2 * math.atan2(norm, q[3]) / norm if norm > 1e-12 else 2.0
    return [a - b for a, b in zip(target[:3], current[:3])] + [v * scale for v in q[:3]]


def solve_linear(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    rows = [list(row) + [b] for row, b in zip(matrix, rhs)]
    n = len(rhs)
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(rows[i][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-15:
            raise ContractError("IK linear solve is singular")
        divisor = rows[col][col]
        rows[col] = [v / divisor for v in rows[col]]
        for i in range(n):
            if i != col:
                factor = rows[i][col]
                rows[i] = [a - factor * b for a, b in zip(rows[i], rows[col])]
    return [row[-1] for row in rows]


class NumericalIK:
    position_tolerance_m: float = 1e-5
    rotation_tolerance_rad: float = 1e-4

    def __init__(
        self,
        fk: UrdfKinematics,
        bounds: list[list[float]],
        accept_translation_m: float | None = None,
        accept_rotation_rad: float | None = None,
        waist_control: bool = False,
    ) -> None:
        self.fk, self.bounds = fk, bounds
        # Hints suggest a waist turn only when the planner can command one.
        self.waist_control = waist_control
        # Optional acceptance of a stalled search close to the target; convergence is still attempted.
        self.accept_translation_m, self.accept_rotation_rad = accept_translation_m, accept_rotation_rad
        left = len(fk.spec.left_joints)
        self.indices = {"left": list(range(left)), "right": list(range(left, len(bounds)))}
        for side, other in (("left", fk.spec.right_joints), ("right", fk.spec.left_joints)):
            if any(name in other for name, *_ in fk.chains[side]):
                raise ContractError("local IK requires independent arm joint chains")

    # Deterministic restarts from perturbed seeds when the first local search stalls,
    # typically with one wrist joint saturated on a redundant arm.
    restarts: int = 3
    restart_perturbation_rad: float = 0.6

    def solve(self, target: list[float], joints: list[float], body: list[float], side: str) -> list[float]:
        indices = self.indices[side]
        q, error, iterations = self._search(target, list(joints), body, side)
        if self._converged(error):
            return q
        generator = random.Random(0)
        for _ in range(self.restarts):
            seed = list(joints)
            for i in indices:
                low, high = self.bounds[i]
                seed[i] = min(
                    high, max(low, seed[i] + generator.uniform(-1, 1) * self.restart_perturbation_rad)
                )
            candidate, candidate_error, count = self._search(target, seed, body, side)
            iterations += count
            if self._converged(candidate_error):
                return candidate
            if sum(v * v for v in candidate_error) < sum(v * v for v in error):
                q, error = candidate, candidate_error
        return self._finish(target, q, body, error, iterations, side)

    def _converged(self, error: list[float]) -> bool:
        return (
            math.hypot(*error[:3]) <= self.position_tolerance_m
            and math.hypot(*error[3:]) <= self.rotation_tolerance_rad
        )

    def _search(
        self, target: list[float], q: list[float], body: list[float], side: str
    ) -> tuple[list[float], list[float], int]:
        indices = self.indices[side]
        iteration = 0
        for iteration in range(100):
            current = self.fk.poses(q, body)[side]
            error = pose_error(target, current)
            if self._converged(error):
                return q, error, iteration + 1
            columns = []
            for i in indices:
                perturbed = list(q)
                # One-sided differences remain inside the original joint interval.
                epsilon = min(1e-5, (self.bounds[i][1] - self.bounds[i][0]) / 4)
                if q[i] + epsilon > self.bounds[i][1]:
                    epsilon = -epsilon
                perturbed[i] += epsilon
                columns.append(
                    [v / epsilon for v in pose_error(self.fk.poses(perturbed, body)[side], current)]
                )
            # Re-solve in the feasible tangent directions. Simply clipping an
            # unconstrained step loses the contribution of saturated joints and
            # can stall even when the remaining joints can realize the target.
            blocked: set[int] = set()
            for _ in range(len(indices) + 1):
                free = [c for j, c in enumerate(columns) if j not in blocked]
                normal = [
                    [sum(c[i] * c[j] for c in free) + (1e-4 if i == j else 0) for j in range(6)]
                    for i in range(6)
                ]
                solved = solve_linear(normal, error)
                delta = [
                    0.0 if j in blocked else sum(a * b for a, b in zip(c, solved))
                    for j, c in enumerate(columns)
                ]
                newly_blocked = {
                    j
                    for j, (i, dq) in enumerate(zip(indices, delta))
                    if (q[i] <= self.bounds[i][0] + 1e-10 and dq < 0)
                    or (q[i] >= self.bounds[i][1] - 1e-10 and dq > 0)
                }
                if not newly_blocked - blocked:
                    break
                blocked.update(newly_blocked)
            scale = min(1.0, 0.05 / max(1e-12, max(map(abs, delta))))
            cost = sum(v * v for v in error)
            for _ in range(12):
                candidate = list(q)
                for i, dq in zip(indices, delta):
                    low, high = self.bounds[i]
                    candidate[i] = min(high, max(low, q[i] + scale * dq))
                trial = pose_error(target, self.fk.poses(candidate, body)[side])
                if sum(v * v for v in trial) < cost - 1e-16:
                    q = candidate
                    break
                scale *= 0.5
            else:
                break
        current = self.fk.poses(q, body)[side]
        return q, pose_error(target, current), iteration + 1

    def _finish(
        self,
        target: list[float],
        q: list[float],
        body: list[float],
        error: list[float],
        iterations: int,
        side: str,
    ) -> list[float]:
        indices = self.indices[side]
        current = self.fk.poses(q, body)[side]
        if (
            self.accept_translation_m is not None
            and self.accept_rotation_rad is not None
            and math.hypot(*error[:3]) <= self.accept_translation_m
            and math.hypot(*error[3:]) <= self.accept_rotation_rad
        ):
            return q
        names = self.fk.spec.left_joints + self.fk.spec.right_joints
        margins = [
            {
                "joint": names[i],
                "position": q[i],
                "lower_margin": q[i] - self.bounds[i][0],
                "upper_margin": self.bounds[i][1] - q[i],
            }
            for i in indices
        ]
        diagnostic = {
            "arm": side,
            "frame": self.fk.frame,
            "iterations": iterations,
            "target_pose": target,
            "best_pose": current,
            "remaining_translation_m": error[:3],
            "rotation_error_rad": math.hypot(*error[3:]),
            "saturated_joints": [
                names[i] for i in indices if min(q[i] - self.bounds[i][0], self.bounds[i][1] - q[i]) <= 1e-3
            ],
            "joint_margins": margins,
            "meaning": "Local numerical search failed; not proof of global infeasibility. Best pose is a geometric search result, never an issued command or observed motion.",
            "restarts": self.restarts,
        }
        if diagnostic["saturated_joints"]:
            diagnostic["hint"] = (
                "The best solution has these joints at their limits: "
                + ", ".join(diagnostic["saturated_joints"])
                + ". The requested orientation or position needs more range than this arm posture "
                "allows; change the target orientation (e.g. less tilt), approach from a different "
                "direction, or move the arm to a different posture first"
                + (
                    ", or change the waist angle (waist_yaw) so the same target needs less of these joints."
                    if self.waist_control
                    else "."
                )
            )
        else:
            diagnostic["hint"] = (
                f"No joint is at its limit, but the search stalls {math.hypot(*error[:3]) * 1000:.1f} mm short "
                "of the target: the target is at or beyond this arm's reach from the current torso pose. "
                "A pitched-down approach needs the wrist beyond the object; reaching with the gripper "
                "pointing along the reach direction (fingers ahead, TCP lower) extends reach. Otherwise "
                "choose a target closer to the shoulder"
                + (
                    ", or first bring the torso closer to the target (turn the waist toward it with waist_yaw)."
                    if self.waist_control
                    else "."
                )
            )
        record_artifact("ik-search-failure", diagnostic)
        raise LocalIKConvergenceError(
            f"local IK did not converge within joint bounds for {side}; no motion issued; "
            + json.dumps(diagnostic, separators=(",", ":"))
        )
