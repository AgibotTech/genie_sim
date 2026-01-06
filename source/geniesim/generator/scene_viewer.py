#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import subprocess
import re
import time
import threading
import traceback
import queue
import argparse
from pathlib import Path

# Add project path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

import geniesim.utils.system_utils as system_utils
from isaacsim import SimulationApp

system_utils.check_and_fix_env()

simulation_app = SimulationApp(
    {
        "headless": False,
    }
)
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.prims import delete_prim, get_prim_at_path, get_prim_children


import omni
import carb
from pxr import UsdLux, Sdf

# File monitoring related
LLM_RESULT_PATH = os.path.join(os.path.dirname(__file__), "LLM_RESULT.py")
GENERATOR_SCRIPT = os.path.join(project_root, "scripts", "run_generator.sh")
last_mtime = 0
world = None
lock = threading.Lock()


def parse_scene_path_from_output(output: str) -> str:
    """Parse scene_path from subprocess output"""
    # Match "step3: save scene to {scene_path}..." format
    # Support multiple possible formats
    patterns = [
        r"step3:\s*save\s+scene\s+to\s+(.+?)\s*\.\.\.",
        r"step3:\s*save\s+scene\s+to\s+(.+?)$",
        r"save\s+scene\s+to\s+(.+?)\s*\.\.\.",
    ]

    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
        if match:
            scene_path = match.group(1).strip()
            # Remove possible quotes
            scene_path = scene_path.strip("\"'")
            if scene_path:
                return scene_path
    return None


def load_scene_to_isaac_sim(scene_path: str, world: World, auto_play: bool = False):
    """Import generated USDA file to Isaac Sim, add dome light, and start play"""

    if not os.path.exists(scene_path):
        print(f"Error: Scene file does not exist: {scene_path}")
        return False

    try:
        with lock:
            # Get current stage
            stage = get_current_stage()
            if stage is None:
                print("Error: Unable to get current stage")
                return False

            # Delete old World content (if exists)
            world_prim = get_prim_at_path("/World")
            has_world_prim = world_prim.IsValid()
            if world_prim.IsValid():
                # Delete all child prims under World (keep World itself)
                children = get_prim_children(world_prim)
                for child in children:
                    delete_prim(child.GetPath())

            print(f"Importing scene: {scene_path}")

            # Import scene to /World
            if not has_world_prim:
                add_reference_to_stage(usd_path=scene_path, prim_path="/World")
            else:
                delete_prim(world_prim.GetPath())
                add_reference_to_stage(usd_path=scene_path, prim_path="/World")

            print("✅ Scene successfully imported to Isaac Sim")

            # Add Dome Light
            dome_light_path = "/World/DomeLight"
            dome_light_prim = get_prim_at_path(dome_light_path)

            if not dome_light_prim.IsValid():
                # Create Dome Light
                dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path(dome_light_path))
                if dome_light:
                    # Set dome light attributes
                    dome_light.CreateIntensityAttr().Set(1000)
                    dome_light.CreateColorTemperatureAttr().Set(6500.0)
                    dome_light.CreateEnableColorTemperatureAttr().Set(True)
                    print("✅ Dome Light added")
                else:
                    print("⚠️ Failed to create Dome Light")
            else:
                print("ℹ️ Dome Light already exists")

            # Automatically start play
            try:
                if auto_play and not world.is_playing():
                    world.play()
                    print("✅ Play started")
                elif not auto_play:
                    world.pause()
                    print("✅ Play paused")
            except Exception as e:
                print(f"⚠️ Error starting play: {e}")

            return True

    except Exception as e:
        print(f"❌ Error importing scene: {e}")
        traceback.print_exc()
        return False


def run_generator_subprocess():
    """Execute generator script in background thread and capture output"""

    print("\nDetected LLM_RESULT.py file change, starting generator...")

    try:
        # Execute subprocess and capture output
        process = subprocess.Popen(
            [GENERATOR_SCRIPT],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        output_lines = []
        scene_path = None

        # Print output in real-time
        for line in process.stdout:
            line = line.rstrip()
            print(f"[Generator] {line}")
            output_lines.append(line)

            # Try to parse scene_path
            if scene_path is None:
                parsed_path = parse_scene_path_from_output(line)
                if parsed_path:
                    scene_path = parsed_path
                    print(f"✅ Parsed scene path: {scene_path}")

        # Wait for process to complete
        return_code = process.wait()

        if return_code != 0:
            print(f"❌ Generator execution failed, return code: {return_code}")
            print("Full output:")
            print("\n".join(output_lines))
            return None

        # If path not parsed from output, try parsing from full output
        if scene_path is None:
            full_output = "\n".join(output_lines)
            scene_path = parse_scene_path_from_output(full_output)

        if scene_path is None:
            print("❌ Unable to parse scene path from output")
            print("Full output:")
            print("\n".join(output_lines))
            return None

        # Ensure path is absolute
        if not os.path.isabs(scene_path):
            # Try relative to assets path
            assets_path = system_utils.assets_path()
            abs_scene_path = os.path.join(assets_path, scene_path)
            if os.path.exists(abs_scene_path):
                scene_path = abs_scene_path
            else:
                # Try relative to project root
                abs_scene_path = os.path.join(project_root, scene_path)
                if os.path.exists(abs_scene_path):
                    scene_path = abs_scene_path
                else:
                    # Try relative to generator directory
                    generator_dir = os.path.dirname(__file__)
                    abs_scene_path = os.path.join(generator_dir, scene_path)
                    if os.path.exists(abs_scene_path):
                        scene_path = abs_scene_path
                    else:
                        print(f"❌ Unable to find scene file: {scene_path}")
                        print(f"Attempted paths:")
                        print(f"  - {os.path.join(assets_path, scene_path)}")
                        print(f"  - {os.path.join(project_root, scene_path)}")
                        print(f"  - {os.path.join(generator_dir, scene_path)}")
                        return None

        return scene_path

    except Exception as e:
        print(f"❌ Error executing generator: {e}")
        traceback.print_exc()
        return None


def run_generator(auto_play: bool = False):
    """Execute generator in main thread (subprocess in background thread, USD import in main thread)"""

    # Execute subprocess in background thread
    def run_in_thread():
        scene_path = run_generator_subprocess()
        if scene_path:
            # Put scene path and auto_play flag in queue, import in main thread
            scene_path_queue.put((scene_path, auto_play))

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()


def check_file_changed():
    """Check if file has changed"""
    global last_mtime

    try:
        if not os.path.exists(LLM_RESULT_PATH):
            return False

        current_mtime = os.path.getmtime(LLM_RESULT_PATH)
        if current_mtime != last_mtime:
            return True
        return False
    except Exception as e:
        print(f"Error checking file: {e}")
        return False


# Queue for executing generator in main thread
generator_queue = queue.Queue()
# Queue for storing parsed scene paths and auto_play flag
scene_path_queue = queue.Queue()


def file_monitor_thread():
    """File monitoring thread"""
    global last_mtime

    # Initialize last_mtime
    if os.path.exists(LLM_RESULT_PATH):
        last_mtime = os.path.getmtime(LLM_RESULT_PATH)

    print(f"Starting to monitor file: {LLM_RESULT_PATH}")

    while simulation_app.is_running():
        try:
            if check_file_changed():
                last_mtime = os.path.getmtime(LLM_RESULT_PATH)
                # Put task in queue, execute in main thread
                carb.log_info("File change detected, executing generator...")
                generator_queue.put(True)

            time.sleep(1)  # Check once per second

        except Exception as e:
            print(f"File monitoring thread error: {e}")
            traceback.print_exc()
            time.sleep(1)


def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scene viewer for Isaac Sim")
    parser.add_argument(
        "--auto-play",
        action="store_true",
        default=False,
        help="Automatically start play when scene is loaded (default: False)",
    )
    args = parser.parse_args()

    auto_play = args.auto_play

    print("Starting Isaac Sim...")

    # Create World
    world = World(
        stage_units_in_meters=1,
        physics_dt=1.0 / 60,
        rendering_dt=1.0 / 60,
    )

    print("✅ Isaac Sim started")
    print(f"Monitoring file: {LLM_RESULT_PATH}")
    print(f"Auto-play: {auto_play}")
    print("Waiting for file changes...")

    # Start file monitoring thread
    monitor_thread = threading.Thread(target=file_monitor_thread, daemon=True)
    monitor_thread.start()

    # Main loop
    step = 0
    printed_pause = False
    while simulation_app.is_running():
        # Check if there are pending generator tasks
        try:
            if not generator_queue.empty():
                generator_queue.get_nowait()
                run_generator(auto_play)
        except queue.Empty:
            pass

        # Check if there are pending scene paths to import
        try:
            if not scene_path_queue.empty():
                scene_path, scene_auto_play = scene_path_queue.get_nowait()
                load_scene_to_isaac_sim(scene_path, world, scene_auto_play)
        except queue.Empty:
            pass

        world.step(render=True)
        if not world.is_playing():
            if not printed_pause:
                print("**** Simulation paused ****")
                printed_pause = True
            step += 1
            continue
        else:
            printed_pause = False

    print("Closing Isaac Sim...")
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, exiting...")
        simulation_app.close()
    except Exception as e:
        print(f"Program error: {e}")
        traceback.print_exc()
        simulation_app.close()
