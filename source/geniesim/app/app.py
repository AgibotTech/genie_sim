# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os, sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

import geniesim.utils.system_utils as system_utils
from geniesim.config.params import *

system_utils.check_and_fix_env()

ps = ParameterServer()
for f in fields(Config):
    ps.declare_parameter(f.name, None)
ps.set_parameters_from_yaml(system_utils.config_path() + "/config.yaml")
ps.override_from_cli()
cfg = load_dataclass(Config, ps)


from geniesim.app.workflow import AppLauncher

app_launcher = AppLauncher(cfg.app)
simulation_app = app_launcher.app

import carb
import omni


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
rclpy.init()

from isaacsim.core.api import World
from geniesim.app.controllers import APICore
from geniesim.app.task_manager import TaskManager
from geniesim.app.workflow.ui_builder import UIBuilder


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
    while simulation_app.is_running():
        ui_builder.my_world.step(render=True)
        task_manager.api_core.render_step()

        if task_manager.api_core.exit:
            task_manager.api_core.post_process()
            break

        if not ui_builder.my_world.is_playing():
            if step % 100 == 0:
                print("**** simulation paused ****")
            step += 1

            continue

    simulation_app.close()


if __name__ == "__main__":
    main()
