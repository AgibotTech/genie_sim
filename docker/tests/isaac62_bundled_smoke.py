#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit GPU/ROS acceptance for system Python using bundled Isaac Sim 6.2.

Run directly with system python3 after sourcing ROS Jazzy. Only the lightweight
bundled-SDK adapter may provide Isaac metadata; the duplicate pip SDK is absent.
This script is not collected by pytest and does not modify images or sources.
The probe uses the engine's actual version, layout, launch and URDF conversion
helpers, then checks a local physics/render scene and independent ROS peer.
"""

import argparse
import faulthandler
from importlib import metadata, util
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time
import traceback

TOPIC = "/geniesim_bundled_acceptance"


def engine_sources() -> None:
    """Use the checked-out engine helpers even before a colcon build."""
    engine = Path(__file__).resolve().parents[2] / "source/geniesim_ros/src/ros_ws/src"
    for package in ("genie_sim_engine_isaacsim", "genie_sim_engine_newton", "genie_sim_engine_core"):
        sys.path.insert(0, str(engine / package))


def assemble(output: Path) -> int:
    """Exercise the production AS3 converter in its required isolated process."""
    engine_sources()
    from genie_sim_engine_isaacsim.assemble._robot_impl import _convert_urdf_to_usd, _isaacsim_version

    assert _isaacsim_version() == (6, 2)
    directory = output / "assembly"
    directory.mkdir(exist_ok=True)
    urdf = directory / "robot.urdf"
    urdf.write_text("""<robot name="robot">
  <link name="base_link">
    <inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.2 0.2 0.2"/></geometry></visual>
    <collision><geometry><box size="0.2 0.2 0.2"/></geometry></collision>
  </link>
  <link name="arm_link">
    <inertial><mass value="0.2"/><inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><geometry><box size="0.1 0.1 0.2"/></geometry></visual>
    <collision><geometry><box size="0.1 0.1 0.2"/></geometry></collision>
  </link>
  <joint name="arm_joint" type="revolute">
    <parent link="base_link"/><child link="arm_link"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>\n""")
    (directory / "robot_raw.urdf").write_text(urdf.read_text())
    _convert_urdf_to_usd(str(urdf), str(directory / "robot.usda"))
    return 0


def ros_peer(output: Path) -> int:
    """Receive live simulator pose/image messages using system ROS alone."""
    import rclpy
    from sensor_msgs.msg import Image
    from std_msgs.msg import String

    rclpy.init()
    node = rclpy.create_node("bundled_isaac_external_peer")
    ack = node.create_publisher(String, f"{TOPIC}/ack", 10)
    seen = {"python": sys.executable, "rclpy": rclpy.__file__}

    def receive_pose(message):
        seen["pose"] = json.loads(message.data)
        ack.publish(String(data="external-ack"))

    def receive_image(message):
        assert message.encoding == "rgba8"
        assert len(message.data) == message.height * message.step
        seen["image"] = {
            "width": message.width,
            "height": message.height,
            "encoding": message.encoding,
            "bytes": len(message.data),
        }

    node.create_subscription(String, f"{TOPIC}/pose", receive_pose, 10)
    node.create_subscription(Image, f"{TOPIC}/rgb", receive_image, 10)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if "pose" in seen and "image" in seen:
                (output / "ros-peer.json").write_text(json.dumps(seen, indent=2) + "\n")
                return 0
        raise RuntimeError("Timed out waiting for the simulator's pose and image")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def peer_environment(root: Path) -> dict[str, str]:
    """Keep ROS paths but remove the Isaac environment from the external peer."""
    env = os.environ.copy()
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "LD_PRELOAD", "PATH"):
        if name in env:
            env[name] = os.pathsep.join(
                entry for entry in env[name].split(os.pathsep) if entry and not Path(entry).is_relative_to(root)
            )
    for name in ("ISAAC_PATH", "CARB_APP_PATH", "EXP_PATH", "PYTHONEXE"):
        env.pop(name, None)
    return env


def standalone(output: Path) -> int:
    """Qualify SDK Torch outside Kit, including CUDA, Triton and coexistence."""
    import torch
    import torch.nn.functional as functional
    import torchvision
    import torchaudio
    import rclpy
    import warp as wp
    import newton
    from cuda.bindings import runtime as cudart

    root = Path("/isaac-sim")
    site = Path(sysconfig.get_path("purelib"))
    assert Path(sys.executable).samefile("/usr/bin/python3")
    assert "carb" not in sys.modules and "isaacsim" not in sys.modules
    for module in (torch, torchvision, torchaudio):
        assert Path(module.__file__).is_relative_to(root), module.__file__
        assert not (site / module.__name__).exists(), f"Duplicate system payload: {module.__name__}"
        provider = metadata.distribution(module.__name__)
        assert provider.read_text("geniesim-bundled.json"), provider
        assert all(
            not Path(provider.locate_file(path)).resolve().is_relative_to(root) for path in provider.files
        ), "Provider owns SDK payload"
    assert torch.version.cuda == "12.8", torch.__version__
    runtime_status, runtime_version = cudart.cudaRuntimeGetVersion()
    assert int(runtime_status) == 0 and 12000 <= runtime_version < 13000, (runtime_status, runtime_version)
    x = torch.randn(128, 128, device="cuda", requires_grad=True)
    matrix = x @ x.T
    matrix.sum().backward()
    assert bool(torch.isfinite(x.grad).all())
    conv = functional.conv2d(torch.ones(1, 1, 16, 16, device="cuda"), torch.ones(1, 1, 3, 3, device="cuda"))
    assert bool(torch.allclose(conv, torch.full_like(conv, 9)))
    compiled = torch.compile(lambda value: value.sin() + value.square(), fullgraph=True)
    assert bool(torch.allclose(compiled(x), x.sin() + x.square(), atol=1e-5))
    torch.cuda.synchronize()
    wp.init()
    assert wp.get_device("cuda:0").is_cuda
    retired = {"nvidia-cublas", "nvidia-cuda-runtime", "nvidia-cusolver", "nvidia-cusparse", "nvidia-nvjitlink"}
    installed = {dist.metadata["Name"].lower().replace("_", "-") for dist in metadata.distributions()}
    assert not retired & installed, f"CUDA 13 packages remain: {retired & installed}"
    assert not (site / "nvidia/cu13").exists(), "CUDA 13 runtime directory remains"
    assert util.find_spec("uipc") is None, "Incompatible native IPC payload is installed"
    assert Path(rclpy.__file__).is_relative_to("/opt/ros/jazzy")
    result = {
        "pass": True,
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_path": torch.__file__,
        "torchvision": torchvision.__file__,
        "torchaudio": torchaudio.__file__,
        "cuda": torch.version.cuda,
        "cuda_python_runtime": runtime_version,
        "device": torch.cuda.get_device_name(0),
        "matmul_autograd_cudnn_triton": "passed",
        "warp": wp.__version__,
        "newton": newton.__version__,
        "rclpy": rclpy.__file__,
        "cuda13_packages": sorted(retired & installed),
        "native_ipc": "unavailable: supplied wheels require CUDA 13",
        "kit_loaded": "carb" in sys.modules,
    }
    assert not result["kit_loaded"]
    (output / "standalone.json").write_text(json.dumps(result, indent=2) + "\n")
    print("BUNDLED_TORCH_STANDALONE_PASS", json.dumps(result), flush=True)
    return 0


def smoke(output: Path) -> int:
    """Check PhysX, spatially varying RGB, metric depth, and external ROS delivery."""
    root = Path(os.environ.get("ISAAC_PATH", "/isaac-sim")).resolve()
    assert Path(sys.executable).samefile("/usr/bin/python3"), sys.executable
    assert sys.version_info[:2] == (3, 12), sys.version
    spec = util.find_spec("isaacsim")
    assert spec and spec.origin and Path(spec.origin).resolve().is_relative_to(root), spec
    manifest = metadata.distribution("isaacsim").read_text("geniesim-bundled.json")
    assert manifest and Path(json.loads(manifest)["root"]).resolve() == root, "Missing bundled SDK adapter"
    system_site = Path(sysconfig.get_path("purelib"))
    assert not (system_site / "isaacsim").exists(), "Duplicate pip SDK is present"
    versions = {}
    for dist in metadata.distributions(path=[str(system_site)]):
        name = dist.metadata["Name"].lower().replace("_", "-").replace(".", "-")
        versions.setdefault(name, set()).add(dist.version)
    duplicates = {name: sorted(values) for name, values in versions.items() if len(values) > 1}
    assert not duplicates, f"Conflicting system pip metadata: {duplicates}"
    import torch as standalone_torch

    assert Path(standalone_torch.__file__).is_relative_to(root), standalone_torch.__file__
    assert "carb" not in sys.modules, "Torch import unexpectedly activated Kit"

    engine_sources()
    from genie_sim_engine_isaacsim.kit.launch_config import prepare_launch_config

    from isaacsim import SimulationApp

    launch = prepare_launch_config(
        {
            "headless": True,
            "width": 64,
            "height": 48,
            "anti_aliasing": 0,
            "disable_viewport_updates": True,
        }
    )
    print("LAUNCH", launch, flush=True)
    app = SimulationApp(launch)
    peer = None
    peer_log = None
    node = None
    code = 1
    try:
        import carb

        print("CAPTURE_COMPAT", carb.settings.get_settings().get("/rtx/hydra/supportMultiTickRate"), flush=True)
        assert carb.settings.get_settings().get("/rtx/hydra/supportMultiTickRate") is False
        import numpy as np
        import torch
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics
        import rclpy
        from sensor_msgs.msg import Image
        from std_msgs.msg import String

        assert Path(carb.__file__).is_relative_to(root), carb.__file__
        assert torch is standalone_torch, "Kit replaced the standalone Torch provider"
        assert Path(UsdGeom.__file__).is_relative_to(root), UsdGeom.__file__
        assert Path(rclpy.__file__).is_relative_to("/opt/ros/jazzy"), rclpy.__file__
        cuda_sum = torch.tensor([1.0, 2.0, 3.0], device="cuda:0").sum().item()
        assert cuda_sum == 6.0, cuda_sum
        converted = Usd.Stage.Open(str(output / "assembly/robot.usda"))
        assert converted and converted.GetDefaultPrim(), "AS3 conversion produced no robot"
        assert any(prim.IsA(UsdPhysics.RevoluteJoint) for prim in converted.Traverse()), "Missing converted joint"
        from genie_sim_engine_isaacsim.kit.bootstrap import resolve_isaac_paths

        sdk_paths = resolve_isaac_paths()
        assert all(item["exists"] for item in sdk_paths.values()), sdk_paths
        world = World(physics_dt=1 / 120, rendering_dt=1 / 60, backend="numpy", device="cpu")
        stage = omni.usd.get_context().get_stage()
        floor = UsdGeom.Cube.Define(stage, "/World/Floor")
        floor.CreateSizeAttr(1)
        transform = UsdGeom.Xformable(floor.GetPrim())
        transform.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
        transform.AddScaleOp().Set(Gf.Vec3d(10, 10, 0.1))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Cube",
                name="cube",
                position=np.array([0, 0, 1.0]),
                scale=np.array([0.5, 0.5, 0.5]),
                color=np.array([1.0, 0.1, 0.1]),
                mass=1,
            )
        )
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000)
        camera = UsdGeom.Camera.Define(stage, "/World/Camera")
        camera.CreateFocalLengthAttr(18)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
        UsdGeom.Xformable(camera.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, 3))
        world.reset()
        for _ in range(240):
            world.step(render=False)
        position, _ = cube.get_world_pose()
        assert 0.20 < float(position[2]) < 0.30, position
        product = rep.create.render_product("/World/Camera", (64, 48))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([product])
        depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth.attach([product])
        world.step(render=True)
        app.update()
        capture_deadline = time.monotonic() + 30
        while time.monotonic() < capture_deadline:
            world.step(render=True)
            app.update()
            frame = np.asarray(rgb.get_data())
            distances = np.asarray(depth.get_data())
            if frame.shape == (48, 64, 4) and distances.shape == (48, 64):
                if frame[:, :, :3].mean(axis=2).std() > 1:
                    break
        assert frame.shape == (48, 64, 4), frame.shape
        spatial_std = float(frame[:, :, :3].mean(axis=2).std())
        assert spatial_std > 1, spatial_std
        assert distances.shape == (48, 64), distances.shape
        assert np.isclose(float(distances[24, 32]), 3 - float(position[2]) - 0.25, atol=0.05), distances[24, 32]
        rclpy.init()
        node = rclpy.create_node("bundled_isaac_system_python_probe")
        publisher = node.create_publisher(String, f"{TOPIC}/pose", 10)
        image_publisher = node.create_publisher(Image, f"{TOPIC}/rgb", 10)
        acknowledgements = []
        node.create_subscription(String, f"{TOPIC}/ack", lambda msg: acknowledgements.append(msg.data), 10)
        (output / "ros-peer.json").unlink(missing_ok=True)
        peer_log = (output / "ros-peer.log").open("w")
        peer = subprocess.Popen(
            ["/usr/bin/python3", str(Path(__file__).resolve()), "--ros-peer", "--output-dir", str(output)],
            env=peer_environment(root),
            stdout=peer_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            publisher.publish(String(data=json.dumps({"position": position.tolist(), "physics_steps": 240})))
            image = Image()
            image.width, image.height, image.encoding, image.step = 64, 48, "rgba8", 64 * 4
            image.data = frame.astype(np.uint8).tobytes()
            image_publisher.publish(image)
            rclpy.spin_once(node, timeout_sec=0.05)
            if acknowledgements and peer.poll() == 0:
                break
            if peer.poll() not in (None, 0):
                raise RuntimeError(f"External ROS peer exited with {peer.returncode}; see ros-peer.log")
        assert acknowledgements and peer.wait(timeout=2) == 0, "Missing external ROS acknowledgement/delivery"
        result = {
            "pass": True,
            "assembly": "AS3 revolute joint validated",
            "sdk_paths": sdk_paths,
            "python": sys.executable,
            "python_version": sys.version.split()[0],
            "adapter_version": metadata.version("isaacsim"),
            "sdk_build": json.loads(manifest)["sdk_version"],
            "isaacsim": spec.origin,
            "usd": UsdGeom.__file__,
            "carb": carb.__file__,
            "rclpy": rclpy.__file__,
            "numpy": np.__file__,
            "torch": torch.__version__,
            "torch_path": torch.__file__,
            "cuda_tensor_sum": cuda_sum,
            "cube_position": position.tolist(),
            "physics_steps": 240,
            "rgb_shape": list(frame.shape),
            "rgb_spatial_std": spatial_std,
            "depth_centre_m": float(distances[24, 32]),
            "external_ros_ack": acknowledgements[0],
        }
        np.save(output / "rgb.npy", frame)
        np.save(output / "depth.npy", distances)
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("BUNDLED_SYSTEM_ROS_PASS", json.dumps(result), flush=True)
        rgb.detach()
        depth.detach()
        world.stop()
        code = 0
    except Exception:
        traceback.print_exc()
    finally:
        if peer is not None and peer.poll() is None:
            peer.terminate()
            try:
                peer.wait(timeout=3)
            except subprocess.TimeoutExpired:
                peer.kill()
                peer.wait()
        if peer_log is not None:
            peer_log.close()
        if node is not None:
            node.destroy_node()
            rclpy.shutdown()
        app.close(exit_code=code)
    return code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ros-peer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--assemble-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--standalone", action="store_true", help="Check bundled Torch/CUDA without Kit")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.ros_peer:
        (args.output_dir / "result.json").unlink(missing_ok=True)
    faulthandler.dump_traceback_later(60, repeat=True)
    if args.ros_peer:
        return ros_peer(args.output_dir)
    if args.assemble_child:
        return assemble(args.output_dir)
    if args.standalone:
        return standalone(args.output_dir)
    with (args.output_dir / "assembly.log").open("w") as log:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--assemble-child", "--output-dir", str(args.output_dir)],
            check=True,
            timeout=120,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return smoke(args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
