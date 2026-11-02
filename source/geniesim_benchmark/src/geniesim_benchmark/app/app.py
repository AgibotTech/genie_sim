# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import signal
import sys
import traceback
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import geniesim_benchmark.utils.system_utils as system_utils
from geniesim_benchmark.config.params import *

system_utils.check_and_fix_env()

ps = ParameterServer()
declare_dataclass_params(Config, ps)
ps.set_parameters_from_yaml(system_utils.config_path() + "/config.yaml")
ps.override_from_cli()
cfg = load_dataclass(Config, ps)


from geniesim_benchmark.app.workflow import AppLauncher

app_launcher = AppLauncher(cfg.app)
simulation_app = app_launcher.app

import carb
import omni
import omni.timeline

from geniesim_benchmark.app.world_step import step_world

# Workaround for omni.replicator.core 1.12.27 (TODO marked "real fix in kit 109"):
# annotator_utils._resize_data_for_overscan() reads /rtx/dataWindowNDC/{0..3}
# without a None fallback, so when those settings are unset (e.g. headless +
# no replicator graph) it does None - None and crashes annotator data fetch.
# Force fitOutputToDataWindow=true to short-circuit the branch, and pin the
# NDC values to "no overscan" as defensive defaults.
_carb_settings = carb.settings.get_settings()
_carb_settings.set("/rtx/dataWindow/fitOutputToDataWindow", True)
_carb_settings.set("/rtx/dataWindowNDC/0", 0.0)
_carb_settings.set("/rtx/dataWindowNDC/1", 0.0)
_carb_settings.set("/rtx/dataWindowNDC/2", 1.0)
_carb_settings.set("/rtx/dataWindowNDC/3", 1.0)

# Apply the collision-cooking and rendering settings before loading task scenes.
_carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)
_carb_settings.set("/omni/replicator/asyncRendering", False)
_carb_settings.set("/app/asyncRendering", False)
_carb_settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)


# Global variables
import time

_frame_count = 0
_last_time = time.time()
_SHUTDOWN_TIMEOUT_SECONDS = 10.0
# Main-loop period while physics is paused for inference (20 Hz).
_PAUSED_MAIN_LOOP_PERIOD_S = 0.05
_SIMULATION_SPEED = int(os.environ.get("GENIESIM_SIMULATION_SPEED", "1"))
if _SIMULATION_SPEED not in (1, 2):
    raise ValueError("GENIESIM_SIMULATION_SPEED must be 1 or 2")

from isaacsim.core.utils import extensions

extensions.enable_extension("isaacsim.ros2.bridge")


def _extension_available(name):
    import omni.kit.app

    manager = omni.kit.app.get_app().get_extension_manager()
    return any(ext.get("name") == name for ext in manager.get_extensions())


# BehaviorScripts mounted via OmniScriptingAPI (e.g. the plug/phone attractor on
# benchmark_plug_*) only run when this extension is loaded. Isaac Sim 6.0 moved
# the runtime from omni.kit.scripting to this core extension. Isaac Sim 5.1 does
# not ship it, so 5.1 runs keep their existing extension set.
if _extension_available("omni.behavior.scripting.core"):
    extensions.enable_extension("omni.behavior.scripting.core")


def wait_rclpy(timeout=10, tick=0.1):
    """Block until rclpy can be imported, or raise after <timeout> seconds."""
    start = time.time()
    while True:
        try:
            import rclpy

            return rclpy
        except ModuleNotFoundError:
            if time.time() - start > timeout:
                raise RuntimeError("rclpy still not available")
            time.sleep(tick)


rclpy = wait_rclpy()
# ROS is opt-in: only bring up a context for the teleop / enable_ros flow.
# Eval / benchmark runs stay ROS-free — no context, no nodes, no pub/sub.
if cfg.app.enable_ros:
    rclpy.init()

from isaacsim.core.api import World
from geniesim_benchmark.app.controllers import APICore
from geniesim_benchmark.app.task_manager import TaskManager
from geniesim_benchmark.app.workflow.ui_builder import UIBuilder


def _shutdown_queues_ready(api_core):
    render = api_core.render_receipt()
    return api_core.physics_queue_empty() and render["drained"], render


def _require_render_drain(api_core, exit_code, shutdown_errors):
    render = api_core.render_receipt()
    if not render["drained"]:
        shutdown_errors.append("render scheduler did not drain before runtime close")
        exit_code = exit_code or 1
    return exit_code, render


def main():
    """Main function."""

    if cfg.app.enable_deformable:
        # Surface/volume deformables in IsaacSim 5.x are a beta feature gated by
        # this persistent setting. Without it the PhysxSurfaceDeformableBodyAPI on
        # the cloth body is silently ignored and the mesh behaves like a static /
        # rigid prim. Must be set before the World (and its physics scene) is built.
        carb.settings.get_settings().set_bool("/persistent/physics/enableDeformableBeta", True)

    world = World(
        stage_units_in_meters=1,
        physics_dt=1.0 / cfg.app.physics_step,
        rendering_dt=_SIMULATION_SPEED / cfg.app.rendering_step,
        # FEM surface-deformable simulation data is only available through the GPU
        # tensor backend; the World must run on cuda or create_surface_deformable_
        # body_view reports 0 sim nodes. Only set it for deformable tasks so every
        # rigid-only task keeps the backend it already runs on.
        device="cuda:0" if cfg.app.enable_deformable else None,
    )
    if cfg.app.enable_gpu_dynamics:
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
        world._physics_context.enable_gpu_dynamics(flag=True)
        world._physics_context.enable_ccd(flag=True)
    if cfg.app.enable_deformable:
        # GPU broadphase is required for convex<->cloth collision. Set it on the
        # physics context up front (before any play()) so the very first
        # Workspace load doesn't hit "Non-GPU-compatible convex mesh is not able
        # to collide with cloth".
        world._physics_context.set_broadphase_type("GPU")
    ui_builder = UIBuilder(world=world)
    api_core = APICore(ui_builder=ui_builder, config=cfg)
    task_manager = TaskManager(
        api_core=api_core,
        benchmark_config=cfg.benchmark,
    )

    def request_signal_shutdown(signum, _frame):
        signal_name = signal.Signals(signum).name
        api_core.request_shutdown(f"signal:{signal_name}", signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_signal_shutdown)

    def callback_physics(step_size):
        global _frame_count, _last_time
        _frame_count += 1
        now = time.time()
        elapsed = now - _last_time
        if elapsed >= 1.0:
            hz = _frame_count / elapsed
            print(f"[Physics Callback] {hz:.2f} Hz")
            _frame_count = 0
            _last_time = now

        if task_manager:
            task_manager.api_core.physics_step(step_size)
            if cfg.app.enable_deformable:
                task_manager.api_core.cloth_pin_step(step_size)
            task_manager.api_core.on_ros_tick(step_size)

    ui_builder.my_world.add_physics_callback("on_physics", callback_fn=callback_physics)
    from geniesim_benchmark.app.main_loop_watchdog import MainLoopWatchdog
    from geniesim_benchmark.app.inference_wait import InferenceWaitGate

    watchdog = MainLoopWatchdog(cfg.app.main_loop_stall_timeout_s)
    watchdog.start()
    inference_wait = InferenceWaitGate(cfg.app.headless and os.environ.get("GENIESIM_IDLE_KIT_DURING_INFERENCE") == "1")
    task_manager.start()

    step = 0
    exit_code = 0
    shutdown_started_at = None
    shutdown_deadline = None
    exit_drain_deadline = None
    worker_joined = False
    shutdown_errors = []
    try:
        while simulation_app.is_running():
            if not api_core.exit and not api_core.shutdown_requested() and inference_wait.wait(api_core):
                continue
            loop_started_at = time.monotonic()
            watchdog.step_started()
            step_world(
                ui_builder.my_world,
                task_manager.api_core.frame_render_enabled,
                omni.timeline.get_timeline_interface(),
            )
            watchdog.step_finished()
            task_manager.api_core.render_step()
            if api_core.physics_paused:
                api_core.wait_while_physics_paused(loop_started_at + _PAUSED_MAIN_LOOP_PERIOD_S - time.monotonic())

            if api_core.shutdown_requested() and shutdown_started_at is None:
                shutdown_request = api_core.shutdown_request()
                print(
                    f"Shutdown requested: {shutdown_request['reason']}; stopping the benchmark",
                    flush=True,
                )
                shutdown_started_at = time.monotonic()
                shutdown_deadline = shutdown_started_at + _SHUTDOWN_TIMEOUT_SECONDS
                task_manager.request_shutdown(shutdown_request["reason"])

            if shutdown_started_at is not None:
                if time.monotonic() >= shutdown_deadline:
                    error = RuntimeError(f"benchmark shutdown exceeded {_SHUTDOWN_TIMEOUT_SECONDS:.0f}s")
                    if api_core.fatal_error is None:
                        api_core.fatal_error = error
                        api_core.fatal_exit_code = 1
                    shutdown_errors.append(str(error))
                    break
                if task_manager.is_alive():
                    continue
                worker_joined = True
                queues_ready, _render_receipt = _shutdown_queues_ready(api_core)
                if not queues_ready:
                    continue
                task_manager.api_core.post_process()
                break

            if task_manager.api_core.exit:
                if exit_drain_deadline is None:
                    exit_drain_deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
                queues_ready, _render_receipt = _shutdown_queues_ready(api_core)
                if not queues_ready and time.monotonic() < exit_drain_deadline:
                    continue
                if not queues_ready:
                    shutdown_errors.append(
                        f"physics/render queues did not drain within {_SHUTDOWN_TIMEOUT_SECONDS:.0f}s"
                    )
                task_manager.api_core.post_process()
                break

            if not ui_builder.my_world.is_playing() and not task_manager.api_core.physics_paused:
                if step % 100 == 0:
                    print("**** simulation paused ****")
                step += 1

                continue
    except KeyboardInterrupt:
        api_core.request_shutdown("keyboard_interrupt", signal.SIGINT)
        shutdown_errors.append("KeyboardInterrupt escaped the signal shutdown handler")
        exit_code = 1
    except BaseException:
        exit_code = 1
        traceback.print_exc()
    finally:
        # Shutdown runs its own deadlines; the watchdog only guards loop steps.
        watchdog.stop()
        print("Shutting down...")
        if task_manager.is_alive():
            if not api_core.shutdown_requested():
                api_core.request_shutdown("app_finalization")
            task_manager.request_shutdown(api_core.shutdown_request()["reason"])
        if shutdown_deadline is None:
            worker_join_timeout = _SHUTDOWN_TIMEOUT_SECONDS
        else:
            worker_join_timeout = max(0.0, shutdown_deadline - time.monotonic())
        worker_joined = task_manager.join(timeout=worker_join_timeout)
        if not worker_joined:
            shutdown_errors.append("task worker did not exit before shutdown deadline")
            exit_code = 1
        try:
            ui_builder.my_world.stop()
        except Exception as error:
            world_stop_status = f"failed: {type(error).__name__}: {error}"
            shutdown_errors.append(world_stop_status)
            exit_code = 1
        try:
            task_manager.api_core.stop_all_recording()
        except Exception as error:
            recording_stop_status = f"failed: {type(error).__name__}: {error}"
            shutdown_errors.append(recording_stop_status)
            exit_code = 1
        try:
            task_manager.api_core.shutdown_ros()
        except Exception as error:
            ros_shutdown_status = f"failed: {type(error).__name__}: {error}"
            shutdown_errors.append(ros_shutdown_status)
            exit_code = 1
        if task_manager.api_core.fatal_error is not None:
            exit_code = task_manager.api_core.fatal_exit_code or 1
        exit_code, _render_receipt = _require_render_drain(api_core, exit_code, shutdown_errors)
        try:
            simulation_app.close()
        except SystemExit as error:
            code = error.code
            exit_code = exit_code or (code if isinstance(code, int) else (0 if code is None else 1))
        except BaseException as error:
            print(f"Simulation app close failed: {type(error).__name__}: {error}", flush=True)
            exit_code = exit_code or 1
        print(f"Shutdown complete, exit_code={exit_code}", flush=True)
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
