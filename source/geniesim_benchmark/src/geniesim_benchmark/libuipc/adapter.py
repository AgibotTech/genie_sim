# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Wuji G2 g2a_sim CoRobot wire adaptation, independent of ROS and physics SDKs.

The twenty native-radian values per hand follow the source policy order. Welded
head/waist outputs are validated and retained, but cannot change the scene welds.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np
import yaml

ARM_JOINTS = tuple(
    f"idx{start+i:02d}_arm_{side}_joint{i+1}" for side, start in (("l", 21), ("r", 61)) for i in range(7)
)
WAIST_JOINTS = tuple(f"idx{i:02d}_body_joint{i}" for i in range(1, 6))
HEAD_JOINTS = tuple(f"idx{10+i:02d}_head_joint{i}" for i in range(1, 4))
GRIPPER_JOINTS = tuple(
    f"hand_{side}_{finger}_joint{joint}"
    for side in ("l", "r")
    for finger in ("thumb", "index", "middle", "ring", "pinky")
    for joint in range(1, 5)
)
DRIVE_JOINTS = ARM_JOINTS + GRIPPER_JOINTS


def finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _rows(channel, name, width, required=False):
    if isinstance(channel, dict):
        if channel.get("kind") != "JOINT_ABS":
            raise ValueError(f"{name}: only JOINT_ABS is supported")
        channel = channel.get("values")
    elif required:
        raise ValueError(f"{name}: expected a JOINT_ABS channel")
    if not isinstance(channel, (list, tuple)) or not channel:
        raise ValueError(f"{name}: values must be a nonempty sequence")
    if any(
        not isinstance(row, (list, tuple)) or len(row) != width or not all(finite(v) for v in row) for row in channel
    ):
        raise ValueError(f"{name}: every row requires {width} finite numeric values")
    return tuple(tuple(float(v) for v in row) for row in channel)


def _empty(channel):
    if isinstance(channel, dict):
        channel = channel.get("values")
    return channel is None or isinstance(channel, (list, tuple)) and not channel


@dataclass(frozen=True)
class ActionChunk:
    targets: tuple
    history_interval: int
    optional_channels: Mapping


def parse_result(result):
    """Validate the entire received chunk before returning a single target."""
    if not isinstance(result, dict):
        raise ValueError("inference result must be a mapping")
    channels = {
        name: _rows(result.get(name), name, width, required=name.endswith("arm"))
        for name, width in (("left_arm", 7), ("right_arm", 7), ("left_effector", 20), ("right_effector", 20))
    }
    optional = {
        name: _rows(result[name], name, width)
        for name, width in (("waist", 5), ("head", 3))
        if not _empty(result.get(name))
    }
    if not _empty(result.get("wheels")):
        raise ValueError("wheels actions are unsupported in the fixed-base Wuji scene")
    count = len(channels["left_arm"])
    if any(len(rows) != count for rows in (*channels.values(), *optional.values())):
        raise ValueError("all action channels must have the same nonzero row count")
    interval = result.get("hist_frame_interval", 0)
    if type(interval) is not int or interval < 0:
        raise ValueError("hist_frame_interval must be a nonnegative integer")
    targets = tuple(
        MappingProxyType(
            dict(
                zip(
                    DRIVE_JOINTS,
                    channels["left_arm"][i]
                    + channels["right_arm"][i]
                    + channels["left_effector"][i]
                    + channels["right_effector"][i],
                )
            )
        )
        for i in range(count)
    )
    return ActionChunk(targets, interval, MappingProxyType(optional))


def validate_identity(identity):
    if set(identity["joint_names"]) != set(DRIVE_JOINTS) or len(identity["joint_names"]) != 54:
        raise ValueError("Wuji policy requires the actual 54 controlled arm/hand joints")


def validate_task_snapshot(observation, asset_identity):
    """Require every verified task mesh and dynamic authored body in each frame.

    The generic client also supports rigid-only IPC scenes. This application
    supplies the selected scene inventory instead of inventing task rules there.
    Mesh IDs are the verified YAML object indices used by the backend snapshot.
    """
    meshes = asset_identity["meshes"]
    expected = {
        (index, mesh["name"], mesh["prim_path"], mesh["vertices"])
        for index, mesh in enumerate(meshes)
        if mesh["kind"] in ("cloth_mesh", "tet_mesh")
    }
    if not expected:
        raise ValueError("Wuji task has no required deformable inventory")
    snapshot = observation["snapshot"]
    deformables = snapshot.get("deformables", ())
    actual = {
        (mesh.get("mesh_id"), mesh.get("name"), mesh.get("prim_path"), len(mesh.get("points", ())))
        for mesh in deformables
    }
    if actual != expected or len(deformables) != len(expected):
        raise ValueError("snapshot deformables differ from the full verified task inventory")
    required = {mesh["body_prim_path"] for mesh in meshes if mesh["kind"] == "rigid_mesh"}
    if not required.issubset(snapshot["bodies"]):
        raise ValueError("snapshot is missing a verified task rigid-body binding")


def load_fixed_joints(scene_file):
    scene = yaml.safe_load(Path(scene_file).read_text())
    initial = scene["robot"]["init_joint_pos"]
    if any(not finite(initial.get(name)) for name in WAIST_JOINTS + HEAD_JOINTS):
        raise ValueError("scene must contain all eight finite welded joint initial angles in degrees")
    return {name: math.radians(initial[name]) for name in WAIST_JOINTS + HEAD_JOINTS}


def _relative_pose(base, end):
    if len(base) != 7 or len(end) != 7 or not all(finite(v) for v in (*base, *end)):
        raise ValueError("invalid synchronized WXYZ body pose")
    # The public engine already composes mesh->authored-root offsets. Only
    # reference-frame conversion remains; output policy orientation is XYZW.
    bq = np.asarray(base[3:], dtype=float)[[1, 2, 3, 0]]
    eq = np.asarray(end[3:], dtype=float)[[1, 2, 3, 0]]
    if min(np.linalg.norm(bq), np.linalg.norm(eq)) < 1e-12:
        raise ValueError("invalid synchronized body quaternion")
    bq /= np.linalg.norm(bq)
    eq /= np.linalg.norm(eq)
    inv = bq * [-1, -1, -1, 1]

    def multiply(a, b):
        return np.r_[a[3] * b[:3] + b[3] * a[:3] + np.cross(a[:3], b[:3]), a[3] * b[3] - np.dot(a[:3], b[:3])]

    delta = np.r_[np.asarray(end[:3]) - np.asarray(base[:3]), 0]
    return {"position": multiply(multiply(inv, delta), bq)[:3].tolist(), "orientation": multiply(inv, eq).tolist()}


def _compose_fixed_pose(body, local):
    if len(body) != 7 or len(local) != 7 or not all(finite(v) for v in (*body, *local)):
        raise ValueError("Invalid frozen body/fixed-frame WXYZ pose")
    bq = np.asarray(body[3:], dtype=float)
    lq = np.asarray(local[3:], dtype=float)
    if min(np.linalg.norm(bq), np.linalg.norm(lq)) < 1e-12:
        raise ValueError("Invalid fixed-frame quaternion")
    bq /= np.linalg.norm(bq)
    lq /= np.linalg.norm(lq)
    v = np.asarray(local[:3], dtype=float)
    xyz = (
        np.asarray(body[:3], dtype=float)
        + v
        + 2 * bq[0] * np.cross(bq[1:], v)
        + 2 * np.cross(bq[1:], np.cross(bq[1:], v))
    )
    quaternion = np.r_[
        bq[0] * lq[0] - np.dot(bq[1:], lq[1:]), bq[0] * lq[1:] + lq[0] * bq[1:] + np.cross(bq[1:], lq[1:])
    ]
    return [*xyz, *quaternion]


def end_pose(snapshot, *, frame_bindings=None):
    required = ("base_link", "arm_base_link", "arm_l_end_link", "arm_r_end_link")
    poses = {}
    if frame_bindings is not None:
        if (
            type(frame_bindings.get("version")) is not int
            or frame_bindings.get("version") != 1
            or frame_bindings.get("pose_schema") != "xyz+wxyz"
            or frame_bindings.get("matrix_convention") != "column-vector"
            or set(frame_bindings.get("frames", {})) != set(required)
        ):
            raise ValueError("Invalid verified policy frame bindings")
        for name, binding in frame_bindings["frames"].items():
            if binding["body_path"] not in snapshot["bodies"]:
                raise ValueError("Frozen snapshot is missing the policy frame's declared rigid ancestor")
            poses[name] = _compose_fixed_pose(snapshot["bodies"][binding["body_path"]], binding["body_from_frame_pose"])
    for path, pose in (() if frame_bindings is not None else snapshot["bodies"].items()):
        name = path.rsplit("/", 1)[-1]
        if name not in required:
            continue
        if name in poses:
            raise ValueError(f"ambiguous synchronized robot body: {name}")
        poses[name] = pose
    if set(poses) != set(required):
        raise ValueError("synchronized snapshot is missing robot end-pose reference bodies")
    return {
        frame: {
            side: _relative_pose(poses[frame], poses[link])
            for side, link in (("left_arm", "arm_l_end_link"), ("right_arm", "arm_r_end_link"))
        }
        for frame in ("base_link", "arm_base_link")
    }


def wire_images(images):
    output = {}
    if set(images) != {"head", "hand_left", "hand_right"}:
        raise ValueError("Wuji inference requires all three synchronized cameras")
    for name, image in images.items():
        data = image["image_data"]
        if isinstance(data, str):
            data = base64.b64decode(data, validate=True)
        if image.get("encoding") != "JPEG" or not isinstance(data, (bytes, bytearray)) or not data:
            raise ValueError("policy image must contain JPEG bytes")
        output[name] = {
            "encoding": "JPEG",
            "image_data": bytes(data),
            "width": image["width"],
            "height": image["height"],
        }
    return output


def build_payload(
    observation,
    fixed_joints,
    *,
    instruction,
    task_name,
    episode_idx=0,
    history=(),
    history_interval=0,
    frame_bindings=None,
):
    state = observation["snapshot"]

    def read(names):
        values = [fixed_joints[name] if name in fixed_joints else state["joints"].get(name) for name in names]
        if not all(finite(v) for v in values):
            raise ValueError("observation is missing a finite required robot joint")
        return [float(v) for v in values]

    timestamp = state["captured_wall_time_ns"]
    params = dict(
        timestamps={"head": timestamp, "states": timestamp},
        images=wire_images(observation["images"]),
        states={
            "head_joint_states": read(HEAD_JOINTS),
            "waist_joint_states": read(WAIST_JOINTS),
            "arm_joint_states": read(ARM_JOINTS),
            "gripper_states": read(GRIPPER_JOINTS),
            "end_pose": end_pose(state, frame_bindings=frame_bindings),
        },
        prompt=instruction,
        robot_type="g2a_sim",
        task_name=task_name,
        episode_idx=episode_idx,
        episode_done=False,
        task_progress=[],
    )
    if history_interval:
        params["history"] = {"interval": history_interval, "images": list(history)}
    return {"method": "infer", "params": params}


class InferenceClient:
    """One direct msgpack/WebSocket infer exchange; no metadata handshake."""

    def __init__(self, host, timeout_s=120):
        if not isinstance(host, str) or not host.strip():
            raise ValueError("an explicit inference endpoint is required")
        self.uri = host if host.startswith(("ws://", "wss://")) else "ws://" + host
        self.timeout_s = timeout_s
        self._ws = None

    def connect(self):
        if self._ws is None:
            from websockets.sync.client import connect

            # 14.x forwards unknown **kwargs to socket.create_connection; async
            # keepalive options therefore cannot be selected by **kwargs presence.
            self._ws = connect(
                self.uri,
                compression=None,
                max_size=None,
                open_timeout=self.timeout_s,
                close_timeout=min(self.timeout_s, 10.0),
            )

    def infer(self, payload, *, archive=None):
        import msgpack

        wire = msgpack.packb(payload, use_bin_type=True)
        if archive is not None:
            archive = Path(archive)
            archive.mkdir(parents=True, exist_ok=False)
            (archive / "request.msgpack").write_bytes(wire)
            for name, image in payload["params"]["images"].items():
                (archive / (name + ".jpg")).write_bytes(image["image_data"])
            for index, frame in enumerate(payload["params"].get("history", {}).get("images", [])):
                for name, image in frame.items():
                    (archive / f"history_{index:04d}_{name}.jpg").write_bytes(image["image_data"])
        self.connect()
        self._ws.send(wire)
        raw = self._ws.recv(timeout=self.timeout_s)
        if archive is not None:
            (archive / ("response.txt" if isinstance(raw, str) else "response.msgpack")).write_bytes(
                raw.encode() if isinstance(raw, str) else raw
            )
        if isinstance(raw, str):
            raise RuntimeError(f"inference service error: {raw}")
        response = msgpack.unpackb(raw, raw=False)
        if not isinstance(response, dict) or response.get("error") or "result" not in response:
            raise ValueError("inference reply requires a successful result envelope")
        return response["result"]

    def close(self):
        if self._ws is not None:
            self._ws.close()
            self._ws = None


def policy_frame_bindings(manifest_path):
    """Verify fixed URDF/USD chains from policy frames to declared rigid bodies.

    The manifest explicitly maps snapshot body paths under robot_prefix to the
    source USD's default prim. Empty fixed links are coordinate frames, never
    synthetic bodies. Matrices use column vectors: world_frame=world_body@local.
    """
    import math
    import xml.etree.ElementTree as ET
    from pxr import Usd, UsdGeom, UsdPhysics, Gf

    path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(path.read_text())
    root = Path(manifest["base_path"]).resolve(strict=True)

    def asset(value, relative):
        result = Path(value)
        result = (result if result.is_absolute() else relative / result).resolve(strict=True)
        if not result.is_relative_to(root):
            raise ValueError("Policy frame robot resource is outside selected assets")
        return result

    robot = manifest["robot"]
    usd = asset(manifest["robot_usda"], root)
    urdf = asset(robot["urdf"], path.parent)
    stage = Usd.Stage.Open(str(usd))
    if stage is None or stage.GetCompositionErrors() or not stage.GetDefaultPrim():
        raise ValueError("Invalid policy frame robot USD composition")
    if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0, atol=1e-12):
        raise ValueError("Policy frame robot USD must use metres")
    inventory = robot["body_paths"]
    if not isinstance(inventory, dict) or not inventory or len(set(inventory.values())) != len(inventory):
        raise ValueError("Ambiguous policy frame rigid body inventory")
    snapshot_root = "/" + manifest["robot_prefix"].strip("/")
    usd_root = str(stage.GetDefaultPrim().GetPath())

    def usd_path(body_path):
        if not body_path.startswith(snapshot_root + "/"):
            raise ValueError("Policy body path is outside the manifest robot root")
        return usd_root + body_path[len(snapshot_root) :]

    for name, body_path in inventory.items():
        prim = stage.GetPrimAtPath(usd_path(body_path))
        if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetName() != name:
            raise ValueError("Policy body inventory does not match declared rigid USD paths")
    if {str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)} != {
        usd_path(body_path) for body_path in inventory.values()
    }:
        raise ValueError("Policy manifest rigid inventory differs from the robot USD")
    tree = ET.parse(urdf).getroot()
    links = [link.get("name") for link in tree.findall("link")]
    if len(set(links)) != len(links):
        raise ValueError("Duplicate policy frame URDF links")
    if not set(inventory).issubset(links):
        raise ValueError("Policy rigid inventory contains unknown URDF links")
    joints = {}
    for joint in tree.findall("joint"):
        child = joint.find("child").get("link")
        if child in joints:
            raise ValueError("Ambiguous policy frame URDF parent")
        joints[child] = joint

    def origin(joint):
        tag = joint.find("origin")
        xyz = np.asarray([float(v) for v in (tag.get("xyz", "0 0 0") if tag is not None else "0 0 0").split()])
        rpy = np.asarray([float(v) for v in (tag.get("rpy", "0 0 0") if tag is not None else "0 0 0").split()])
        if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite(xyz).all() or not np.isfinite(rpy).all():
            raise ValueError("Invalid fixed policy frame URDF origin")
        roll, pitch, yaw = rpy
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        result = np.eye(4)
        result[:3, :3] = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        result[:3, 3] = xyz
        return result

    cache = UsdGeom.XformCache()
    frames = {}
    for frame in ("base_link", "arm_base_link", "arm_l_end_link", "arm_r_end_link"):
        if frame not in links:
            raise ValueError("Required policy frame is missing from robot URDF")
        current = frame
        chain = []
        seen = set()
        while current not in inventory:
            if current in seen or current not in joints:
                raise ValueError("Policy frame has no declared rigid ancestor")
            seen.add(current)
            joint = joints[current]
            if joint.get("type") != "fixed":
                raise ValueError("Policy frame binding crosses a moving URDF joint")
            chain.append((current, joint))
            current = joint.find("parent").get("link")
        body_path = inventory[current]
        frame_path = body_path + "".join("/" + link for link, _ in reversed(chain))
        intermediate = body_path
        for link, _ in reversed(chain):
            intermediate += "/" + link
            fixed_prim = stage.GetPrimAtPath(usd_path(intermediate))
            fixed_xform = UsdGeom.Xformable(fixed_prim)
            if (
                not fixed_xform
                or fixed_prim.HasAPI(UsdPhysics.RigidBodyAPI)
                or fixed_xform.GetResetXformStack()
                or any(op.GetNumTimeSamples() for op in fixed_xform.GetOrderedXformOps())
            ):
                raise ValueError("Policy fixed USD chain is missing, rigid, animated or resets its parent transform")
        body = stage.GetPrimAtPath(usd_path(body_path))
        prim = stage.GetPrimAtPath(usd_path(frame_path))
        if not prim or not prim.IsA(UsdGeom.Xformable):
            raise ValueError("Required fixed policy frame is missing from robot USD")
        relative = cache.GetLocalToWorldTransform(prim) * cache.GetLocalToWorldTransform(body).GetInverse()
        matrix = np.asarray(relative, dtype=float).T
        expected = np.eye(4)
        for _, joint in reversed(chain):
            expected = expected @ origin(joint)
        if not np.isfinite(matrix).all() or not np.allclose(matrix, expected, atol=1e-6, rtol=0):
            raise ValueError("Policy fixed frame USD and URDF transforms differ")
        transform = Gf.Transform(relative)
        quat = transform.GetRotation().GetQuat()
        frames[frame] = {
            "frame_path": frame_path,
            "body_path": body_path,
            "body_name": current,
            "fixed_joint_chain": [joint.get("name") for _, joint in reversed(chain)],
            "body_from_frame": matrix.tolist(),
            "body_from_frame_pose": [*transform.GetTranslation(), quat.GetReal(), *quat.GetImaginary()],
        }
    return dict(
        version=1,
        pose_schema="xyz+wxyz",
        matrix_convention="column-vector",
        frames=frames,
        robot_usda=str(usd),
        robot_urdf=str(urdf),
    )
