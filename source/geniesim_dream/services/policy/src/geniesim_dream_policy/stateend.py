"""Convert Dream policy waypoints to the world model's 29D action format."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _framework_math():
    root = os.environ.get("GENIESIM_DREAM_COSMOS_FRAMEWORK")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    prefix = "cosmos_framework.data.generator.action"
    return (
        importlib.import_module(f"{prefix}.agibot_fk"),
        importlib.import_module(f"{prefix}.pose_utils"),
    )


def _g2_urdf_path() -> Path:
    raw = os.environ.get("GENIESIM_DREAM_G2_URDF")
    if not raw:
        raise ValueError("Set GENIESIM_DREAM_G2_URDF to the G2 kinematics URDF")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"G2 URDF is missing: {path}")
    return path


def _g2_kinematics_xml(urdf_path: Path | None = None) -> str:
    """Build MuJoCo-loadable kinematics-only XML from the committed G2 URDF."""
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path or _g2_urdf_path()).getroot()
    mujoco_element = root.find("mujoco")
    if mujoco_element is None:
        mujoco_element = ET.Element("mujoco")
        root.insert(0, mujoco_element)
    compiler_element = mujoco_element.find("compiler")
    if compiler_element is None:
        compiler_element = ET.SubElement(mujoco_element, "compiler")
    compiler_element.attrib["fusestatic"] = "false"

    for link_element in root.findall("link"):
        for child_element in list(link_element):
            if child_element.tag in {"visual", "collision"}:
                link_element.remove(child_element)

    return ET.tostring(root, encoding="unicode")


GRIPPER_TRAVEL_DEG = 120.0


HEAD_IDENTITY_REL = np.asarray([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


def split_canvas(canvas: Image.Image, head_rows: int) -> dict[str, np.ndarray]:
    width, height = canvas.size
    cut = min(head_rows, height)
    pixels = np.asarray(canvas.convert("RGB"))
    return {
        "head": pixels[:cut],
        "left_wrist": pixels[cut:, : width // 2],
        "right_wrist": pixels[cut:, width // 2 :],
    }


def retime_actions(
    waypoints: np.ndarray, speed: float, out_len: int, now: np.ndarray
) -> tuple[np.ndarray, int]:
    anchored = np.concatenate([now[None], waypoints]).astype(np.float32)
    dst_t = (np.arange(out_len, dtype=np.float64) + 1.0) * speed
    out = np.empty((out_len, anchored.shape[1]), dtype=np.float32)
    grid = np.arange(len(anchored), dtype=np.float64)
    for d in range(anchored.shape[1]):
        out[:, d] = np.interp(dst_t, grid, anchored[:, d])
    return (out, int(np.count_nonzero(dst_t > len(waypoints))))


class StateEndEncoder:
    def __init__(self, head_waist: np.ndarray, head_source: str = "identity"):
        self._head_waist = np.asarray(head_waist, dtype=np.float32).reshape(4)
        self._head_source = head_source

    def state20(self, wm: np.ndarray) -> np.ndarray:
        wm = np.asarray(wm, dtype=np.float32).reshape(-1, 16)
        out = np.zeros((len(wm), 20), dtype=np.float32)
        out[:, 0:7] = wm[:, 0:7]
        out[:, 7:14] = wm[:, 8:15]
        out[:, 14] = np.clip(wm[:, 7], 0.0, 1.0) * GRIPPER_TRAVEL_DEG
        out[:, 15] = np.clip(wm[:, 15], 0.0, 1.0) * GRIPPER_TRAVEL_DEG
        out[:, 16:20] = self._head_waist
        return out

    def __call__(self, waypoints: np.ndarray) -> np.ndarray:
        fk_module, pose_module = _framework_math()
        state = self.state20(waypoints)
        steps = len(state) - 1
        if steps < 1:
            raise ValueError("need at least two waypoints to form one relative action step")
        zeros_pos = np.zeros((len(state), 3), dtype=np.float32)
        zeros_quat = np.zeros((len(state), 4), dtype=np.float32)
        fk = fk_module.compute_fk_transforms_batch(state, "agibot_world_gripper")
        fk = fk_module.apply_robot_base_motion_to_poses(fk, zeros_pos, zeros_quat)
        fk = fk_module.apply_agibot_gripper_to_opencv(
            fk, fk_module.AGIBOT_WORLD_GRIPPER_TO_OPENCV_BY_WRIST
        )
        rel = {
            name: pose_module.pose_abs_to_rel(
                fk[name], rotation_format="rot6d", pose_convention="backward_framewise"
            )
            for name in ("head_camera", "right_wrist", "left_wrist")
        }
        head = (
            np.repeat(HEAD_IDENTITY_REL[None], steps, axis=0)
            if self._head_source == "identity"
            else rel["head_camera"]
        )
        grip = fk_module.convert_gripper_state_to_open_fraction(state[1:, 14:16])
        return np.concatenate(
            [head, rel["right_wrist"], grip[:, 1:2], rel["left_wrist"], grip[:, 0:1]], axis=-1
        ).astype(np.float32)


G2_GRIPPER_OPEN_RAD = 0.91

_G2_CACHE: dict = {}


def _load_g2_model():
    if "model" not in _G2_CACHE:
        import mujoco

        model = mujoco.MjModel.from_xml_string(_g2_kinematics_xml())
        _G2_CACHE["model"] = model
        _G2_CACHE["data"] = mujoco.MjData(model)
    return (_G2_CACHE["model"], _G2_CACHE["data"])


_G2_T_END = np.array(
    [[0.0, 0.0, 1.0, 0.09], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


_G2_HEAD_IDENTITY = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0], dtype=np.float32)


class G2StateEndEncoder:
    L_JOINTS = tuple((f"idx{21 + i}_arm_l_joint{i + 1}" for i in range(7)))
    R_JOINTS = tuple((f"idx{61 + i}_arm_r_joint{i + 1}" for i in range(7)))
    H_JOINTS = tuple((f"idx{11 + i}_head_joint{i + 1}" for i in range(3)))
    B_JOINTS = tuple((f"idx{1 + i:02d}_body_joint{i + 1}" for i in range(5)))

    def __init__(self, head_body: np.ndarray):
        self._head_body = np.asarray(head_body, dtype=np.float64).reshape(8)

    def __call__(self, waypoints: np.ndarray) -> np.ndarray:
        import mujoco

        fk_module, pose_module = _framework_math()
        model, data = _load_g2_model()

        def adr(name: str) -> int:
            return model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]

        a_l = [adr(n) for n in self.L_JOINTS]
        a_r = [adr(n) for n in self.R_JOINTS]
        a_h = [adr(n) for n in self.H_JOINTS]
        a_b = [adr(n) for n in self.B_JOINTS]
        body_l = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_l_link7")
        body_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "arm_r_link7")
        wm = np.asarray(waypoints, dtype=np.float64).reshape(-1, 16)
        steps = len(wm) - 1
        if steps < 1:
            raise ValueError("need at least two waypoints to form one relative action step")
        poses = np.zeros((len(wm), 2, 4, 4), dtype=np.float64)
        for i, row in enumerate(wm):
            data.qpos[:] = 0
            for k in range(7):
                data.qpos[a_l[k]] = row[k]
                data.qpos[a_r[k]] = row[8 + k]
            for k in range(3):
                data.qpos[a_h[k]] = self._head_body[k]
            for k in range(5):
                data.qpos[a_b[k]] = self._head_body[3 + k]
            mujoco.mj_forward(model, data)
            for side, body in ((0, body_l), (1, body_r)):
                t = poses[i, side]
                t[3, 3] = 1.0
                t[:3, :3] = data.xmat[body].reshape(3, 3)
                t[:3, 3] = data.xpos[body]
                poses[i, side] = t @ _G2_T_END
                wrist = "left_wrist" if side == 0 else "right_wrist"
                poses[i, side, :3, :3] = (
                    poses[i, side, :3, :3]
                    @ fk_module.AGIBOT_WORLD_GRIPPER_TO_OPENCV_BY_WRIST[wrist]
                )
        rel = {
            side: pose_module.pose_abs_to_rel(
                poses[:, side], rotation_format="rot6d", pose_convention="backward_framewise"
            )
            for side in (0, 1)
        }
        head = np.repeat(_G2_HEAD_IDENTITY[None], steps, axis=0)
        grip = np.clip(-wm[1:, [7, 15]] / G2_GRIPPER_OPEN_RAD, 0.0, 1.0).astype(np.float32)
        return np.concatenate([head, rel[1], grip[:, 1:2], rel[0], grip[:, 0:1]], axis=-1).astype(
            np.float32
        )
