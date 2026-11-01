# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read-only solved-state checker for the current passive MuJoCo Rubik cube.

Uses actual cube poses, never robot actions, a solver or a task success flag.
The model must use the cube:middle / cube:cubelet:* topology from cube.py.
"""

from __future__ import annotations

import itertools
import json

import mujoco
import numpy as np


class CubeSolvedChecker:
    """Cache a model binding and scratch kinematics for an evaluation loop.

    ``solved`` describes the current geometry; ``success`` additionally requires
    ``hold_seconds`` of observed solved states. Call on every physics step when
    using the hold timer, and reset() when resetting an episode. The checker
    neither steps physics nor modifies the caller's model, data or callbacks.
    """

    def __init__(
        self, model, *, angle_tolerance_deg=5.0, position_tolerance_m=None, hold_seconds=0.0, cube_module=None
    ):
        if cube_module is None:
            import cube as cube_module
        CubeState, PITCH = cube_module.CubeState, cube_module.PITCH
        self.faces = cube_module.FACES
        if not np.isfinite(angle_tolerance_deg) or not 0 <= angle_tolerance_deg < 45:
            raise ValueError("angle_tolerance_deg must be finite and in [0, 45)")
        if not np.isfinite(hold_seconds) or hold_seconds < 0:
            raise ValueError("hold_seconds must be finite and nonnegative")
        try:
            self.cube = CubeState(model)
        except KeyError as error:
            raise ValueError("Model must contain cube:middle and 26 named cubelets") from error
        expected = {c for c in itertools.product((-1, 0, 1), repeat=3) if any(c)}
        actual = [tuple(c) for _, c in self.cube.items]
        if len(actual) != 26 or set(actual) != expected:
            raise ValueError("Expected exactly 26 uniquely named cubelets")
        self.model = model
        self.bodies = np.array([b for b, _ in self.cube.items], dtype=int)
        coordinates = np.array([c for _, c in self.cube.items], dtype=float)
        self.offsets = model.body_pos[self.bodies].copy()
        pitches = np.sum(self.offsets * coordinates, axis=1) / np.sum(coordinates**2, axis=1)
        self.pitch = float(np.median(pitches))
        if (
            not np.isfinite(self.pitch)
            or self.pitch <= 0
            or not np.allclose(self.offsets, coordinates * self.pitch, rtol=1e-7, atol=1e-10)
        ):
            raise ValueError("Cubelet offsets must form the native uniform 3x3x3 grid")
        tolerance = 0.0015 * self.pitch / PITCH if position_tolerance_m is None else float(position_tolerance_m)
        if not np.isfinite(tolerance) or not 0 <= tolerance < self.pitch / 2:
            raise ValueError("position_tolerance_m must be finite, nonnegative and below half a cubie pitch")
        self.angle_tolerance_deg = float(angle_tolerance_deg)
        self.position_tolerance_m = tolerance
        self.hold_seconds = float(hold_seconds)
        self.scratch = mujoco.MjData(model)
        descendants = {self.cube.root}
        for b in range(self.cube.root + 1, model.nbody):
            if model.body_parentid[b] in descendants:
                descendants.add(b)
        if not set(self.bodies).issubset(descendants):
            raise ValueError("All cubelets must belong to cube:middle")
        self.quaternion_addresses = []
        for j in range(model.njnt):
            if model.jnt_bodyid[j] not in descendants:
                continue
            address = int(model.jnt_qposadr[j])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                self.quaternion_addresses.append(address)
            elif model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                self.quaternion_addresses.append(address + 3)
        self.reset()

    def reset(self):
        """Start a new episode without changing the simulation."""
        self._solved_since = None
        self._last_time = None
        self._last_data = None

    def check(self, data):
        """Inspect current qpos and return a JSON-serializable result dictionary."""
        if data.model is not self.model:
            raise ValueError("MjData must belong to the checker MjModel")
        time = float(data.time)
        result = dict(
            solved=False,
            success=False,
            reason="invalid_state",
            aligned=False,
            assembled=False,
            uniform_faces=0,
            facelets=None,
            faces={},
            max_angle_error_deg=None,
            max_position_error_m=None,
            displaced_cubies=[],
            angle_tolerance_deg=self.angle_tolerance_deg,
            position_tolerance_m=self.position_tolerance_m,
            hold_seconds=self.hold_seconds,
            held_seconds=0.0,
            simulation_time=time if np.isfinite(time) else None,
        )
        if self._last_data is not data or (self._last_time is not None and time < self._last_time):
            self._solved_since = None
        self._last_data = data
        self._last_time = time if np.isfinite(time) else None
        invalid = not np.isfinite(time) or not np.isfinite(data.qpos).all()
        if not invalid:
            invalid = any(abs(np.linalg.norm(data.qpos[q : q + 4]) - 1.0) > 1e-5 for q in self.quaternion_addresses)
        if invalid:
            self._solved_since = None
            return result

        # mj_kinematics refreshes stale FK without forces, callbacks or stepping.
        # Only this private MjData is written, even when the caller has not run FK.
        self.scratch.qpos[:] = data.qpos
        mujoco.mj_kinematics(self.model, self.scratch)
        root = self.cube.root
        root_rotation = self.scratch.xmat[root].reshape(3, 3)
        rotations = root_rotation.T @ self.scratch.xmat[self.bodies].reshape(-1, 3, 3)
        positions = (self.scratch.xpos[self.bodies] - self.scratch.xpos[root]) @ root_rotation
        if not np.isfinite(rotations).all() or not np.isfinite(positions).all():
            self._solved_since = None
            return result
        errors = np.linalg.norm(positions - np.einsum("bij,bj->bi", rotations, self.offsets), axis=1)
        assembled = bool(np.max(errors) <= self.position_tolerance_m)
        facelets, angle = self.cube.decode(self.scratch, self.angle_tolerance_deg)
        faces = {face: facelets[i * 9 : (i + 1) * 9] for i, face in enumerate(self.faces)} if facelets else {}
        uniform = sum(len(set(colors)) == 1 for colors in faces.values())
        aligned = facelets is not None
        solved = bool(assembled and aligned and uniform == 6 and all(facelets.count(face) == 9 for face in self.faces))
        if solved:
            if self._solved_since is None:
                self._solved_since = time
            held = max(0.0, time - self._solved_since)
        else:
            self._solved_since = None
            held = 0.0
        success = bool(solved and held + 1e-12 >= self.hold_seconds)
        reason = (
            "disassembled"
            if not assembled
            else "unaligned" if not aligned else "scrambled" if not solved else "solved" if success else "holding"
        )
        result.update(
            solved=solved,
            success=success,
            reason=reason,
            aligned=aligned,
            assembled=assembled,
            uniform_faces=int(uniform),
            facelets=facelets,
            faces=faces,
            max_angle_error_deg=float(angle),
            max_position_error_m=float(np.max(errors)),
            held_seconds=float(held),
            displaced_cubies=[
                self.model.body(int(b)).name
                for b, error in zip(self.bodies, errors)
                if error > self.position_tolerance_m
            ],
        )
        return result


def check_cube_solved(model, data, *, angle_tolerance_deg=5.0, position_tolerance_m=None):
    """One-shot check; reuse CubeSolvedChecker for repeated evaluation steps."""
    return CubeSolvedChecker(
        model, angle_tolerance_deg=angle_tolerance_deg, position_tolerance_m=position_tolerance_m
    ).check(data)
