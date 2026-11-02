# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Isaac Sim render bridge for measured native MuJoCo geometry poses."""

import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time

import numpy as np

from .render_protocol import send_message, receive_message
from .viewer import ViewerClosed
from .robot_render import native_to_usd_joint
from .cameras import camera_state, inspection_cameras


def assets_directory():
    configured = (
        os.environ.get("RUBIK_ASSETS_ROOT")
        or os.environ.get("GENIESIM_ASSETS_PATH")
        or os.environ.get("SIM_ASSETS")
    )
    if not configured:
        try:
            from geniesim_assets import ASSETS_PATH
        except ImportError as error:
            raise RuntimeError("Set GENIESIM_ASSETS_PATH or SIM_ASSETS to the public asset directory") from error
        configured = ASSETS_PATH
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError("Public asset directory is missing: " + str(root))
    return root


def export_render_model(env, directory):
    model, cameras = env.model, env.config.cameras
    directory.mkdir(parents=True, exist_ok=False)
    ids = np.flatnonzero((model.geom_group != 2) & (model.geom_rgba[:, 3] > 0) & (model.geom_bodyid < env.cube.root))
    # The collection background provides the visible floor; the native plane
    # remains physical but must not cover the HDR background in policy images.
    ids = np.array([], dtype=int)
    geoms = []
    for index in ids:
        kind = int(model.geom_type[index])
        if kind not in (0, 2, 6, 7):
            raise ValueError(f"Unsupported visual geometry {model.geom(index).name}: {kind}")
        geoms.append(
            {
                "id": int(index),
                "name": model.geom(index).name,
                "type": kind,
                "size": model.geom_size[index].tolist(),
                "rgba": model.geom_rgba[index].tolist(),
                "mesh": int(model.geom_dataid[index]) if kind == 7 else -1,
            }
        )
    assets_root = assets_directory()
    input_root = env.scene_catalog.extract(directory / "input", assets_root=assets_root)
    cube_bindings = [
        {"id": int(body), "name": model.body(body).name.replace(":", "_")}
        for body in [env.cube.root] + [b for b, _ in env.cube.items]
    ]
    robot_config = json.loads(
        files("geniesim_benchmark").joinpath("app/robot_cfg/G2_crsB_wuji.json").read_text()
    )
    robot_usd = assets_root / robot_config["robot"]["robot_usd"]
    if not robot_usd.is_file():
        raise RuntimeError("Standard G2_crsB_wuji USD asset is missing: " + str(robot_usd))
    environment_usd = (
        assets_root / "background/sphere_hdr_backgrounds/sphere_hdr_shucaichang/background.usda"
    )
    hdr_texture = assets_root / "background/common/HDR/shucaichang.hdr"
    if not environment_usd.is_file() or not hdr_texture.is_file():
        raise RuntimeError("Collection-matched shucaichang HDR background assets are missing")
    scene = {
        "environment_usd": str(environment_usd),
        "environment_usd_sha256": hashlib.sha256(environment_usd.read_bytes()).hexdigest(),
        "hdr_texture": str(hdr_texture),
        "hdr_texture_sha256": hashlib.sha256(hdr_texture.read_bytes()).hexdigest(),
        "robot_usd": str(robot_usd),
        "robot_usd_sha256": hashlib.sha256(robot_usd.read_bytes()).hexdigest(),
        "required_robot_joints": [native_to_usd_joint(name) for name in env.action_contract()["joint_names"]],
        "clip_range": [float(model.vis.map.znear * model.stat.extent), float(model.vis.map.zfar * model.stat.extent)],
        "geoms": geoms,
        "cameras": cameras,
        "inspection_cameras": inspection_cameras(),
        "cube_bindings": cube_bindings,
        "scene_usda": str(input_root / env.current_scene.relative_path),
        "source_scene_id": env.current_scene.identifier,
        "source_scene_sha256": env.current_scene.sha256,
        "render_scene_sha256": hashlib.sha256((input_root / env.current_scene.relative_path).read_bytes()).hexdigest(),
        "assets_root": str(assets_root),
    }
    (directory / "scene.json").write_text(json.dumps(scene, indent=2))
    return ids


class IsaacRenderer:
    def __init__(self, env, headless, output):
        self.env = env
        self.headless = headless
        self.directory = Path(output) / "isaac-render"
        self.ids = export_render_model(env, self.directory)
        self.cube_bodies = [env.cube.root] + [b for b, _ in env.cube.items]
        configured_launcher = (
            env.config.isaac_python or os.environ.get("RUBIK_ISAAC_PYTHON", "") or "omni_python"
        )
        launcher = shutil.which(configured_launcher)
        if not launcher:
            raise RuntimeError(
                f"Isaac Sim Python launcher not found: {configured_launcher}. "
                "Use a combo image with omni_python or set RUBIK_ISAAC_PYTHON or rubik.isaac_python."
            )
        self.temporary = tempfile.TemporaryDirectory(prefix="rubik-isaac-")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket_path = str(Path(self.temporary.name) / "render.sock")
        self.listener.bind(self.socket_path)
        self.listener.listen(1)
        self.listener.settimeout(0.5)
        self.connection = None
        self.process = None
        self.log = (self.directory / "isaac.log").open("w")
        worker = Path(__file__).with_name("isaac_worker.py")
        command = [launcher, str(worker), "--socket", self.socket_path, "--directory", str(self.directory)]
        if headless:
            command.append("--headless")
        self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240
            print(f"Starting Isaac Sim renderer; startup log: {self.directory / 'isaac.log'}", flush=True)
            while self.connection is None:
                try:
                    self.connection, _ = self.listener.accept()
                except socket.timeout:
                    if self.process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError(f"Isaac Sim renderer startup failed; inspect {self.directory / 'isaac.log'}")
            self.connection.settimeout(max(1, deadline - time.monotonic()))
            send_message(self.connection, self.snapshot(capture=False))
            response = receive_message(self.connection)
            if response.get("error"):
                raise RuntimeError(response["error"])
            self.metadata = response["metadata"]
            self.connection.settimeout(60)
            print("Isaac Sim renderer ready: RTX images and live USD state synchronization", flush=True)
        except BaseException:
            self.close()
            raise

    def snapshot(self, capture=False):
        data = self.env.data
        joint_positions = {
            key: float(np.deg2rad(value)) for key, value in self.env.initial_joints["init_joint_pos"].items()
        }
        joint_positions.update(
            {
                native_to_usd_joint(name): float(value)
                for name, value in zip(self.env.action_contract()["joint_names"], data.qpos[self.env.action_q])
            }
        )
        anchors = {}
        for body in range(self.env.model.nbody):
            name = self.env.model.body(body).name
            target = None
            if name in ("base_link", "head_link3"):
                target = name
            elif name in ("l_wrist", "r_wrist"):
                target = "hand_" + name[0] + "_base_link"
            elif name and name.startswith(("l_", "r_")) and "link" in name:
                target = "hand_" + name.replace("_finger", "")
            if target:
                matrix = np.eye(4)
                matrix[:3, :3] = data.xmat[body].reshape(3, 3)
                matrix[:3, 3] = data.xpos[body]
                anchors["/genie/" + target] = matrix.tolist()
        return {
            "robot_joint_positions": joint_positions,
            "robot_anchors": anchors,
            "method": "state",
            "capture": capture,
            "simulation_time": float(data.time),
            "geom_positions": np.asarray(data.geom_xpos[self.ids], dtype=np.float64).tobytes(),
            "geom_rotations": np.asarray(data.geom_xmat[self.ids], dtype=np.float64).tobytes(),
            "cube_positions": np.asarray(data.xpos[self.cube_bodies], dtype=np.float64).tobytes(),
            "cube_rotations": np.asarray(data.xmat[self.cube_bodies], dtype=np.float64).tobytes(),
            "inspection_cameras": {
                name: camera_state(self.env.model, data, camera) for name, camera in inspection_cameras().items()
            },
            "cameras": {
                name: camera_state(self.env.model, data, camera) for name, camera in self.env.config.cameras.items()
            },
        }

    def is_running(self):
        return self.process is not None and self.process.poll() is None

    def request(self, capture):
        if not self.is_running():
            raise ViewerClosed("Isaac Sim visualization closed")
        try:
            state = self.snapshot(capture)
            send_message(self.connection, state)
            response = receive_message(self.connection)
        except (EOFError, BrokenPipeError, ConnectionResetError) as error:
            raise ViewerClosed("Isaac Sim visualization closed") from error
        if response.get("error"):
            raise RuntimeError(response["error"])
        if (
            response["simulation_time"] != state["simulation_time"]
            or response["maximum_pose_error"] > 1e-8
            or response["maximum_robot_pose_error"] > 2e-5
            or response["maximum_camera_pose_error"] > 2e-5
        ):
            raise RuntimeError("Isaac Sim renderer did not synchronize the requested physical state")
        return response

    def images(self):
        response = self.request(True)
        return {
            name: np.frombuffer(value["rgb"], dtype=np.uint8).reshape(value["shape"]).copy()
            for name, value in response["images"].items()
        }

    def sync(self):
        self.request(False)

    def pace(self, started, simulation_interval):
        self.sync()
        remaining = simulation_interval - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)

    def wait_until_closed(self):
        while self.is_running():
            time.sleep(0.05)

    def close(self):
        if self.connection is not None:
            try:
                send_message(self.connection, {"method": "close"})
            except OSError:
                pass
            self.connection.close()
            self.connection = None
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.listener.close()
        self.log.close()
        self.temporary.cleanup()
