# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
import signal
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


# Global variables
import time

_frame_count = 0
_last_time = time.time()

from isaacsim.core.utils import extensions

extensions.enable_extension("isaacsim.ros2.bridge")


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


def main():
    """Main function."""

    world = World(
        stage_units_in_meters=1,
        physics_dt=1.0 / cfg.app.physics_step,
        rendering_dt=1.0 / cfg.app.rendering_step,
    )
    if cfg.app.enable_gpu_dynamics:
        physx_interface = omni.physx.get_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
        world._physics_context.enable_gpu_dynamics(flag=True)
        world._physics_context.enable_ccd(flag=True)
    ui_builder = UIBuilder(world=world)
    task_manager = TaskManager(
        api_core=APICore(ui_builder=ui_builder, config=cfg),
        benchmark_config=cfg.benchmark,
    )

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
            task_manager.api_core.physics_step()
            task_manager.api_core.on_ros_tick(step_size)

    ui_builder.my_world.add_physics_callback("on_physics", callback_fn=callback_physics)
    task_manager.start()

    step = 0
    shutdown_requested = False
    signal_handlers = {}

    def request_shutdown(signum, _frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    if cfg.app.enable_ros:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal_handlers[signum] = signal.signal(signum, request_shutdown)
    try:
        while not shutdown_requested and simulation_app.is_running():
            ui_builder.my_world.step(render=task_manager.api_core.frame_render_enabled)
            task_manager.api_core.render_step()

            if task_manager.api_core.exit:
                task_manager.api_core.post_process()
                break

            if not ui_builder.my_world.is_playing():
                if step % 100 == 0:
                    print("**** simulation paused ****")
                step += 1

                continue
    except KeyboardInterrupt:
        print("main loop: KeyboardInterrupt received")
    finally:
        print("Shutting down...")
        try:
            task_manager.join(timeout=10)
            try:
                if cfg.app.enable_ros:
                    ui_builder.my_world.stop()
            finally:
                task_manager.api_core.stop_all_recording()
        finally:
            try:
                task_manager.api_core.shutdown_ros()
            finally:
                try:
                    simulation_app.close()
                finally:
                    for signum, handler in signal_handlers.items():
                        signal.signal(signum, handler)
        print("Shutdown complete")


if __name__ == "__main__":
    main()
