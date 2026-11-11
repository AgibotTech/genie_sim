# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Hard IK from the actual robot USD joint frames, with the runtime joint order."""

import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdPhysics
from scipy.spatial.transform import Rotation as R

from geniesim_benchmark.utils.hard_ik import HardIKFKSolver, _transform


def runtime_robot_usd_path(robot_cfg):
    from geniesim_benchmark.utils.system_utils import assets_path
    from geniesim_benchmark.utils.name_utils import robot_type_mapping

    robot_cfg = robot_type_mapping(robot_cfg)
    directory = Path(__file__).resolve().parents[1] / "app/robot_cfg"
    robot = json.loads((directory / f"{robot_cfg}.json").read_text())["robot"]
    return str(Path(assets_path()) / robot["robot_usd"])


class UsdIKFKSolver(HardIKFKSolver):
    """The shared numerical solver, using USD geometry and intersected limits."""

    def __init__(self, robot_cfg, robot_usd_path):
        super().__init__(robot_cfg=robot_cfg)
        stage = Usd.Stage.Open(str(robot_usd_path))
        if stage is None:
            raise ValueError(f"Cannot load IK robot USD: {robot_usd_path}")
        parents, links = {}, {}
        for prim in stage.Traverse():
            links[prim.GetName()] = str(prim.GetPath())
            if prim.IsA(UsdPhysics.Joint):
                child = prim.GetRelationship("physics:body1").GetTargets()
                if len(child) == 1:
                    parents.setdefault(str(child[0]), []).append(prim)
        for side, suffix in (("left", "fl"), ("right", "fr")):
            end = f"arm_{suffix}_end_link" if self.robot_cfg == "agilex" else f"arm_{side[0]}_end_link"
            link, blocks, seen = links[end], [], set()
            start = 0 if side == "left" else self.arm_dof
            names = self.joint_names[start : start + self.arm_dof]
            while Path(link).name != self.frame:
                if link in seen or link not in parents:
                    raise ValueError(f"USD {end} is not a descendant of {self.frame}")
                seen.add(link)
                if len(parents[link]) != 1:
                    raise ValueError(f"Ambiguous USD joint parent for {link}")
                joint = parents[link][0]
                kind = joint.GetTypeName()
                if kind not in ("PhysicsRevoluteJoint", "PhysicsFixedJoint", "PhysicsPrismaticJoint"):
                    raise ValueError(f"Unsupported USD joint: {joint.GetName()}")
                frame0, frame1 = self._joint_frame(joint, 0), self._joint_frame(joint, 1)
                if kind == "PhysicsFixedJoint":
                    block = [(None, "fixed", frame0 @ np.linalg.inv(frame1), np.zeros(3))]
                else:
                    index = names.index(joint.GetName())
                    axis = np.eye(3)["XYZ".index(joint.GetAttribute("physics:axis").Get())]
                    unit = np.pi / 180 if kind == "PhysicsRevoluteJoint" else 1.0
                    low = float(joint.GetAttribute("physics:lowerLimit").Get()) * unit
                    high = float(joint.GetAttribute("physics:upperLimit").Get()) * unit
                    self.lower[start + index] = max(self.lower[start + index], low)
                    self.upper[start + index] = min(self.upper[start + index], high)
                    block = [
                        (index, "revolute" if kind == "PhysicsRevoluteJoint" else "prismatic", frame0, axis),
                        (None, "fixed", np.linalg.inv(frame1), np.zeros(3)),
                    ]
                blocks.append(block)
                parent = joint.GetRelationship("physics:body0").GetTargets()
                if len(parent) != 1:
                    raise ValueError(f"Missing USD parent for {joint.GetName()}")
                link = str(parent[0])
            self._chains[side] = [entry for block in reversed(blocks) for entry in block]
        self._torso_chain = []
        link, seen = links[self.frame], set()
        while Path(link).name != "base_link":
            if link in seen or len(parents.get(link, [])) != 1:
                raise ValueError("USD IK frame has no unique base_link ancestor")
            seen.add(link)
            joint = parents[link][0]
            kind = joint.GetTypeName()
            if kind not in ("PhysicsRevoluteJoint", "PhysicsFixedJoint", "PhysicsPrismaticJoint"):
                raise ValueError(f"Unsupported torso joint: {joint.GetName()}")
            axis = (
                np.zeros(3)
                if kind == "PhysicsFixedJoint"
                else np.eye(3)["XYZ".index(joint.GetAttribute("physics:axis").Get())]
            )
            self._torso_chain.append(
                (joint.GetName(), kind, self._joint_frame(joint, 0), axis, np.linalg.inv(self._joint_frame(joint, 1)))
            )
            link = str(joint.GetRelationship("physics:body0").GetTargets()[0])
        self._torso_chain.reverse()
        if np.any(self.lower >= self.upper):
            raise ValueError("USD and URDF joint bounds have no common interval")

    def base_transform(self, positions):
        result = np.eye(4)
        for name, kind, frame0, axis, inverse1 in self._torso_chain:
            motion = np.eye(4)
            if kind != "PhysicsFixedJoint":
                value = float(positions[name])
                if not np.isfinite(value):
                    raise ValueError(f"Invalid measured torso joint: {name}")
                if kind == "PhysicsRevoluteJoint":
                    motion[:3, :3] = R.from_rotvec(axis * value).as_matrix()
                else:
                    motion[:3, 3] = axis * value
            result = result @ frame0 @ motion @ inverse1
        return result

    @staticmethod
    def _joint_frame(joint, index):
        quaternion = joint.GetAttribute(f"physics:localRot{index}").Get()
        position = joint.GetAttribute(f"physics:localPos{index}").Get()
        rotation = R.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()])
        return _transform(position, rotation.as_matrix())
