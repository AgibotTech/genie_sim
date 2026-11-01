# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import os
import sys
import time
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_directory)

from common.base_utils.logger import logger

if os.path.exists(os.path.join(root_directory, "git_commit_info.txt")):
    with open(os.path.join(root_directory, "git_commit_info.txt"), "r") as f:
        logger.info("############GIT COMMIT INFO##########")
        logger.info(f.read())
        logger.info("############GIT COMMIT INFO##########")
else:
    logger.warning(f"git_commit_info.txt not found in {root_directory}")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
)
parser.add_argument("--enable_physics", action="store_true", default=True)
parser.add_argument("--enable_render", action="store_true", default=False)
parser.add_argument(
    "--enable_curobo",
    action="store_true",
    default=True,
)
parser.add_argument("--publish_ros", action="store_true", default=False)
parser.add_argument(
    "--physics_step",
    type=int,
    default=60,
    help="PhysX stepping frequency in Hz. Higher values (such as 120/200) can improve contact solver stability. "
         "Recording/render frequency is independent and controlled by --rendering_step and task_info['fps'].",
)
parser.add_argument(
    "--rendering_step",
    type=int,
    default=30,
    help="Rendering frequency in Hz (main-loop render decisions and World rendering_dt). "
         "Independent of --physics_step; the default remains 30.",
)
parser.add_argument(
    "--enable_depth",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--livestream",
    type=int,
    default=0,
    choices=[0, 1, 2],
    help="0: no livestream, 1: Public Network, 2: Private Network",
)
parser.add_argument(
    "--backend",
    type=str,
    default="cpu",
    choices=["cpu", "gpu"],
    help="Physics backend. 'gpu' = PhysX on CUDA + torch tensors (default); "
         "'cpu' = PhysX on CPU + numpy tensors. RTX render always runs on CUDA.",
)
parser.add_argument(
    "--curobo_batch_size",
    type=int,
    default=10,
    help="cuRobo plan_batch / IK batch parallelism. Default: 20. Larger values may improve planning success but use more GPU memory; "
         "smaller values reduce memory pressure. Set once at server startup and fixed during execution.",
)
parser.add_argument(
    "--enable_deformable",
    action="store_true",
    default=False,
    help="Enable FEM surface deformable (cloth / soft body) support. When enabled: "
         "(1) force backend=gpu (FEM tensor views require CUDA); "
         "(2) set carb /persistent/physics/enableDeformableBeta=True before creating World; "
         "(3) enable GPU dynamics, broadphase=GPU and GpuMaxDeformableSurfaceContacts=4M after creating World; "
         "(4) scan the scene and create SurfaceDeformableBodyView after reset. "
         "Rigid-body tasks are unaffected; prims without deformable schemas are skipped.",
)

args = parser.parse_args()

# Deformable tensor views require the CUDA physics pipeline.
if args.enable_deformable and args.backend != "gpu":
    logger.warning(
        f"--enable_deformable forces backend=gpu (requested backend={args.backend}); "
        f"FEM SurfaceDeformableBodyView does not support the CPU pipeline."
    )
    args.backend = "gpu"

import sim_backend
from sim_backend import BackendConfig, Device

# --- v2 curobo warp ordering fix -------------------------------------------
# Isaac Sim startup replaces Warp with its bundled version, while cuRobo v2
# collision kernels use wp.func(module=...), which requires Warp 1.10 or later.
# Import cuRobo before SimulationApp so kernels register with the environment
# Warp version; later imports can reuse the registered kernels.
# This avoids registration with the older bundled Warp.
import warp as _wp  # noqa: F401 -- load the environment Warp first.
import curobo as _curobo  # noqa: F401 -- trigger cuRobo version and top-level imports.
# --- v2 curobo runtime flag: enable cuda_graph_reset ---
# CUDA graph reset is disabled unless the cuRobo runtime flag is set.
# Enable it explicitly before constructing CUDA graphs.
# This requires CUDA 12.0 or later; configure it before any cuRobo object
# can enter a graph-building path.
import curobo._src.runtime as _curobo_runtime  # noqa: F401
_curobo_runtime.cuda_graph_reset = True
# Trigger the collision-kernel registration imports.
from curobo._src.motion.motion_planner_batch import (  # noqa: F401
    BatchMotionPlanner as _BatchMotionPlanner,
)
# --- end v2 curobo warp ordering fix ---------------------------------------

# `--backend gpu` → (CUDA, "torch"),`--backend cpu` → (CPU, "numpy").
# CUDA physics requires torch tensors; CPU physics uses numpy.
# Select the matching World backend explicitly.
_physics_device, _tensors_as = (
    (Device.CUDA, "torch") if args.backend == "gpu" else (Device.CPU, "numpy")
)

public_ip_env = os.environ.get("PUBLIC_IP", "127.0.0.1")
extra_args = [
    "--/persistent/rtx/modes/rt2/enabled=true",
]
if args.livestream >= 1:
    # Note: Only one livestream extension can be enabled at a time
    if args.livestream == 1:
        # WebRTC public network
        extra_args += [
            f"--/app/livestream/publicEndpointAddress={public_ip_env}",
            "--/app/livestream/port=49100",
            "--enable",
            "omni.services.livestream.nvcf",
        ]
    elif args.livestream == 2:
        # WebRTC private network
        extra_args += [
            "--enable",
            "omni.services.livestream.nvcf",
        ]
    else:
        logger.warning("Invalid livestream mode")

simulation_app = sim_backend.init_app(
    backend_config=BackendConfig(
        physics_device=_physics_device,
        render_device=Device.CUDA,   # RTX rendering requires a GPU.
        tensors_as=_tensors_as,
    ),
    headless=args.headless,
    rtx_mode="pathtrace",            # "RealTimePathTracing"
    extensions=() if not args.publish_ros else ("isaacsim.ros2.bridge",),
    async_rendering=False,           # Keep async rendering disabled for synchronized capture.
    extra_args=extra_args,
    hide_ui=args.headless and args.livestream == 0,
    # Enable deformable beta before creating World via the extra carb settings
    # passed to init_app, after its fixed settings.
    rtx_settings=(
        {"/persistent/physics/enableDeformableBeta": True}
        if args.enable_deformable else None
    ),
)
# SimulationApp is now running and Isaac Sim imports are available.
# cuRobo kernels were registered before startup; subsequent imports reuse them.
from isaacsim.core.api import World
import omni

from server.command_controller import CommandController
from server.grpc_server import GrpcServer
from server.ui_builder import UIBuilder

# Override curobo batch size before any CuroboMotion is constructed (UIBuilder may
# CUROBO_BATCH_SIZE is a module-level constant read by name at its call sites.
# Updating it before constructing planners applies the requested batch size.
from server.motion_generator import motion_gen_reacher
motion_gen_reacher.CUROBO_BATCH_SIZE = args.curobo_batch_size
logger.info(f"curobo batch size set to {motion_gen_reacher.CUROBO_BATCH_SIZE}")


physics_dt = (float)(1 / args.physics_step)
rendering_dt = (float)(1 / args.rendering_step)
world = World(
    stage_units_in_meters=1,
    physics_dt=physics_dt,
    rendering_dt=rendering_dt,
    device=sim_backend.physics_device_str(),
    backend=sim_backend.tensors_as(),
)
physx_interface = omni.physx.get_physx_interface()
# Override CPU setting to use GPU
# physx_interface.overwrite_gpu_setting(1)

# world._physics_context.enable_gpu_dynamics(flag=True)

# FEM deformables: enable GPU dynamics and broadphase after World construction.
# Use larger contact buffers for deformable scenes.
if args.enable_deformable:
    physics_ctx = world.get_physics_context()
    physics_ctx.enable_gpu_dynamics(True)
    try:
        physics_ctx.set_broadphase_type("GPU")
    except Exception as exc:
        logger.warning(f"set_broadphase_type('GPU') failed (nonfatal): {exc}")
    try:
        from pxr import PhysxSchema
        stage = omni.usd.get_context().get_stage()
        for _p in stage.Traverse():
            if _p.GetTypeName() == "PhysicsScene":
                PhysxSchema.PhysxSceneAPI.Apply(_p) \
                    .CreateGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1048576)
                break
    except Exception as exc:
        logger.warning(f"Failed to set GpuMaxDeformableSurfaceContacts (nonfatal): {exc}")
    logger.info(
        "FEM surface deformables enabled (GPU dynamics + broadphase=GPU + GpuMaxDeformableSurfaceContacts=4M); "
        "cloth_view will be created after world.reset completes in init_robot."
    )

ui_builder = UIBuilder(world=world, debug=args.debug, enable_deformable=args.enable_deformable)
server_function = CommandController(
    ui_builder=ui_builder,
    enable_physics=args.enable_physics,
    enable_curobo=args.enable_curobo,
    publish_ros=args.publish_ros,
    rendering_step=args.rendering_step,
    debug=args.debug,
    output_depth=args.enable_depth,
    enable_render=args.enable_render,
)
def callback_physics(step_size):
    if server_function:
        server_function.callback_physics()

ui_builder.my_world.add_physics_callback("on_physics", callback_fn=callback_physics)
rpc_server = GrpcServer(server_function=server_function)
rpc_server.start()

step = 0
last_physics_time = 0
last_render_time = 0
logger.info("**** server ready ****")
start_time = time.time()
current_time = 0
while simulation_app.is_running():
    try:
        with rpc_server.server_function._timing_context("ui_builder.my_world.step"):
            ui_builder.my_world.step(render=False)
            current_time = ui_builder.my_world.current_time
            need_render = False
            if (
                last_render_time == 0
                or last_render_time > current_time
                or current_time - last_render_time >= rendering_dt
            ) and args.enable_render:
                need_render = True
                last_render_time = current_time
            if need_render:
                ui_builder.my_world.render()
        if rpc_server:
            rpc_server.server_function.on_physics_step()
            if rpc_server.server_function.exit:
                break
        if not ui_builder.my_world.is_playing():
            # World not playing yet (waiting on the gRPC client). Yield the GIL so the
            # gRPC server thread stays responsive instead of busy-spinning this loop.
            time.sleep(0.001)
            continue
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)
rpc_server.server_function.print_timing_stats()
end_time = time.time()
total_sim_time = sum([t[1] - t[0] for t in rpc_server.server_function.recording_times]) if len(rpc_server.server_function.recording_times) > 0 else 0
logger.info(f"Total time: {end_time - start_time} seconds, recording_times: {rpc_server.server_function.recording_times}, total_sim_time: {total_sim_time}")
simulation_app.close()
