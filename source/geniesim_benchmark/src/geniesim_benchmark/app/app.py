# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
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

# Local calibration maintenance only; never enabled for ordinary execution.
if os.environ.get("GENIESIM_DOMAIN_EVIDENCE_USE") == "calibration" and not cfg.app.headless:
    extensions.enable_extension("omni.kit.window.script_editor")


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


def _write_shutdown_receipt(receipt):
    evidence_dir = os.environ.get("GENIESIM_CONTROL_EVIDENCE_DIR", "").strip()
    if not evidence_dir:
        return None
    path = Path(evidence_dir) / "control-runtime-shutdown.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
    return str(path)


def _write_runtime_application_identity():
    evidence_dir = os.environ.get("GENIESIM_CONTROL_EVIDENCE_DIR", "").strip()
    if not evidence_dir:
        return None
    path = Path(evidence_dir) / "control-runtime-application.json"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value = {
        "kind": "geniesim_control_runtime_application",
        "requested_ui": os.environ.get("GENIESIM_CONTROL_UI", ""),
        "headless": bool(cfg.app.headless),
        "render_mode": cfg.app.render_mode,
        "simulation_speed": _SIMULATION_SPEED,
        "physics_dt_s": 1.0 / cfg.app.physics_step,
        "rendering_dt_s": _SIMULATION_SPEED / cfg.app.rendering_step,
    }
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return value


def _shutdown_queues_ready(api_core):
    render = api_core.render_receipt()
    return api_core.physics_queue_empty() and render["drained"], render


def _require_render_drain(api_core, exit_code, shutdown_errors):
    render = api_core.render_receipt()
    if not render["drained"]:
        shutdown_errors.append("render scheduler did not drain before runtime close")
        exit_code = exit_code or 1
    return exit_code, render


def _require_shutdown_standstill(api_core, exit_code, shutdown_errors):
    final_physics = api_core.shutdown_receipt()
    if (
        final_physics.get("raw_motion", {}).get("complete") is not True
        or final_physics.get("standstill", {}).get("status") != "confirmed"
    ):
        shutdown_errors.append("runtime stopped without a complete confirmed standstill window")
        return exit_code or 1
    return exit_code


def _build_shutdown_receipt(
    *,
    exit_code,
    api_core,
    task_manager,
    shutdown_queue_drained,
    shutdown_settle_ticks,
    render_receipt,
    worker_joined,
    world_stop_status,
    recording_stop_status,
    ros_shutdown_status,
    app_close_status,
    shutdown_errors,
):
    return {
        "kind": "geniesim_control_runtime_shutdown",
        "receipt_id": os.environ.get("GENIESIM_CONTROL_RECEIPT_ID"),
        "status": "failed" if exit_code else "completed",
        "exit_code": exit_code,
        "request": api_core.shutdown_request(),
        "task_manager": task_manager.receipt(),
        "physics": {
            "queue_drained": shutdown_queue_drained,
            "settle_ticks": shutdown_settle_ticks,
            **api_core.shutdown_receipt(),
        },
        "render": render_receipt,
        "cleanup": {
            "worker_joined": worker_joined,
            "world_stop": world_stop_status,
            "recording_stop": recording_stop_status,
            "ros_shutdown": ros_shutdown_status,
            "simulation_app_close": app_close_status,
        },
        "errors": shutdown_errors,
        "cleanup_finished_at_unix": time.time(),
    }


def main():
    """Main function."""

    _write_runtime_application_identity()
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
    telemetry = None
    from geniesim_benchmark.app.main_loop_watchdog import MainLoopWatchdog
    from geniesim_benchmark.app.inference_wait import InferenceWaitGate

    watchdog = MainLoopWatchdog(cfg.app.main_loop_stall_timeout_s)
    watchdog.start()
    inference_wait = InferenceWaitGate(
        cfg.app.headless and os.environ.get("GENIESIM_IDLE_KIT_DURING_INFERENCE") == "1")
    task_manager.start()

    step = 0
    exit_code = 0
    shutdown_started_at = None
    shutdown_deadline = None
    shutdown_settle_ticks = 0
    shutdown_standstill = None
    shutdown_queue_drained = False
    exit_drain_deadline = None
    worker_joined = False
    world_stop_status = "not_started"
    ros_shutdown_status = "not_started"
    recording_stop_status = "not_started"
    app_close_status = "not_started"
    shutdown_errors = []
    try:
        if cfg.app.teleop_telemetry_endpoint and not os.environ.get("GENIESIM_TELEMETRY_TOKEN"):
            # The map stream is operator-only; a scored run without the supervisor token must still evaluate, just without the stream.
            print("[TeleopMap] GENIESIM_TELEMETRY_TOKEN unset; pose stream disabled")
        elif cfg.app.teleop_telemetry_endpoint:
            from geniesim_benchmark.app.controllers.teleop_telemetry import TeleopTelemetry

            if cfg.benchmark.enable_vec:
                raise ValueError("Teleop map telemetry currently requires a single environment")
            telemetry = TeleopTelemetry(
                api_core,
                cfg.app.teleop_telemetry_endpoint,
                cfg.app.teleop_telemetry_rate_hz,
                os.environ["SIM_REPO_ROOT"],
                cfg.benchmark.task_name,
            )
            print(f"[TeleopMap] Pose stream enabled: {telemetry.endpoint} ({cfg.app.teleop_telemetry_rate_hz:g} Hz)")
        while simulation_app.is_running():
            if not api_core.exit and not api_core.shutdown_requested() and inference_wait.wait(api_core):
                continue
            loop_started_at = time.monotonic()
            watchdog.step_started()
            ui_builder.my_world.step(render=task_manager.api_core.frame_render_enabled)
            watchdog.step_finished()
            task_manager.api_core.render_step()
            if telemetry is not None:
                telemetry.sample(paused=not ui_builder.my_world.is_playing())
            if api_core.physics_paused:
                api_core.wait_while_physics_paused(loop_started_at + _PAUSED_MAIN_LOOP_PERIOD_S - time.monotonic())

            if api_core.shutdown_requested() and shutdown_started_at is None:
                shutdown_request = api_core.shutdown_request()
                print(
                    f"Shutdown requested: {shutdown_request['reason']}; stopping control safely",
                    flush=True,
                )
                shutdown_started_at = time.monotonic()
                shutdown_deadline = shutdown_started_at + _SHUTDOWN_TIMEOUT_SECONDS
                shutdown_standstill = api_core.create_shutdown_motion_journal()
                task_manager.request_shutdown(shutdown_request["reason"])

            if shutdown_started_at is not None:
                if time.monotonic() >= shutdown_deadline:
                    error = RuntimeError(f"control shutdown exceeded {_SHUTDOWN_TIMEOUT_SECONDS:.0f}s")
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
                    shutdown_settle_ticks = 0
                    continue
                shutdown_queue_drained = True
                if shutdown_standstill is None:
                    # No calibrated motion profile (every robot except G2_90d):
                    # there is no standstill contract to confirm.
                    task_manager.api_core.post_process()
                    break
                sample = api_core.measure_shutdown_motion()
                stopped, motion, standstill = shutdown_standstill.observe(sample)
                shutdown_settle_ticks = standstill["stable_samples"]
                api_core.update_shutdown_receipt(
                    motion=motion,
                    standstill=standstill,
                    physics_queue_drained=True,
                    settle_ticks=shutdown_settle_ticks,
                )
                if not stopped:
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
                    shutdown_errors.append(f"physics/render queues did not drain within {_SHUTDOWN_TIMEOUT_SECONDS:.0f}s")
                shutdown_queue_drained = queues_ready
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
        if telemetry is not None:
            try:
                telemetry.close()
            except Exception as error:
                shutdown_errors.append(f"telemetry close failed: {error}")
                exit_code = 1
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
            world_stop_status = "completed"
        except Exception as error:
            world_stop_status = f"failed: {type(error).__name__}: {error}"
            shutdown_errors.append(world_stop_status)
            exit_code = 1
        try:
            task_manager.api_core.stop_all_recording()
            recording_stop_status = "completed"
        except Exception as error:
            recording_stop_status = f"failed: {type(error).__name__}: {error}"
            shutdown_errors.append(recording_stop_status)
            exit_code = 1
        try:
            task_manager.api_core.shutdown_ros()
            ros_shutdown_status = "completed"
        except Exception as error:
            ros_shutdown_status = f"failed: {type(error).__name__}: {error}"
            shutdown_errors.append(ros_shutdown_status)
            exit_code = 1
        if task_manager.api_core.fatal_error is not None:
            exit_code = task_manager.api_core.fatal_exit_code or 1
        exit_code, render_receipt = _require_render_drain(api_core, exit_code, shutdown_errors)
        if shutdown_standstill is not None:
            exit_code = _require_shutdown_standstill(api_core, exit_code, shutdown_errors)
        receipt = _build_shutdown_receipt(
            exit_code=exit_code,
            api_core=api_core,
            task_manager=task_manager,
            shutdown_queue_drained=shutdown_queue_drained,
            shutdown_settle_ticks=shutdown_settle_ticks,
            render_receipt=render_receipt,
            worker_joined=worker_joined,
            world_stop_status=world_stop_status,
            recording_stop_status=recording_stop_status,
            ros_shutdown_status=ros_shutdown_status,
            app_close_status=app_close_status,
            shutdown_errors=shutdown_errors,
        )
        from geniesim_benchmark.control_evidence import close_runtime_with_receipt

        exit_code = close_runtime_with_receipt(simulation_app, receipt, _write_shutdown_receipt)
        print(f"Shutdown complete, exit_code={exit_code}", flush=True)
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
