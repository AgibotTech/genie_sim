# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Supervise an isolated Wuji collection session."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import select
import socket
import subprocess
import sys
import tempfile
import time

import yaml

from geniesim_cli._style import CYAN, GREEN, RED, RST
from geniesim_teleop.wuji.contracts import HandCommands, load_task
from geniesim_teleop.wuji.model import generate_urdf

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config/wuji_teleop.yaml"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Wuji task YAML, container-side path")
    result.add_argument("--source", choices=("sdk", "retargeting"), default="sdk")
    result.add_argument("--left-sn", default="")
    result.add_argument("--right-sn", default="")
    result.add_argument("--sdk-user", default="")
    network = result.add_mutually_exclusive_group()
    network.add_argument("--interface", default="")
    network.add_argument(
        "--host-ip", "--host_ip", default="", help="Local Pico-LAN IPv4; required for a multi-IP interface"
    )
    result.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Pico UDP port (default: 8080, matching the fixed AIDEA App port)",
    )
    result.add_argument("--domain-id", type=int, default=91, help="Dedicated ROS domain; domain 0 is refused")
    result.add_argument(
        "--external-simulator", action="store_true", help="Use an already running Wuji simulator in this domain"
    )
    result.add_argument("--headless", action="store_true")
    result.add_argument(
        "--dry-run", action="store_true", help="Validate task/model and print commands without ROS or devices"
    )
    result.add_argument(
        "--check", action="store_true", help="Also check runtime imports, domain and UDP port; do not start"
    )
    return result


def validate(args):
    if not 1 <= args.domain_id <= 232:
        raise ValueError("Use a dedicated ROS domain between 1 and 232")
    if not 1024 <= args.port <= 65535:
        raise ValueError("Use a Pico port between 1024 and 65535")
    if args.source == "sdk" and not (args.left_sn or args.right_sn):
        raise ValueError("SDK mode requires --left-sn and/or --right-sn")
    if args.source == "retargeting" and (args.left_sn or args.right_sn or args.sdk_user):
        raise ValueError("Glove serial numbers and --sdk-user require --source sdk")
    if args.left_sn and args.left_sn == args.right_sn:
        raise ValueError("Left and right gloves must have different serial numbers")
    task = load_task(args.config)
    with tempfile.TemporaryDirectory(prefix="wuji-validate-") as directory:
        model = generate_urdf(task, Path(directory) / "robot.urdf")
        HandCommands(model)
        task.mc_initial_states(model)
    return task


def commands(args, config, run_dir, host_ip):
    def module(name, *extra):
        return [sys.executable, "-m", name, *map(str, extra)]

    hand_topic = "/teleop/wuji/hand_command" if args.source == "sdk" else "/wuji_hand/joint_command"
    result = []
    if not args.external_simulator:
        result.append(
            (
                "simulator",
                module(
                    "geniesim_cli",
                    "benchmark",
                    "run",
                    config,
                    f"--app.headless={str(args.headless).lower()}",
                    "--benchmark.model_arc=pi",
                    "--benchmark.policy_class=DemoPolicy",
                    "--benchmark.env_class=DummyEnv",
                    "--benchmark.interactive=true",
                    "--benchmark.record=false",
                ),
            )
        )
    result.extend(
        [
            (
                "bridge",
                module(
                    "geniesim_teleop.wuji.bridge",
                    "--config",
                    config,
                    "--hand-topic",
                    hand_topic,
                    "--model-urdf",
                    run_dir / "robot.urdf",
                ),
            ),
            (
                "mc",
                module(
                    "geniesim_teleop.wuji.mc",
                    "--config",
                    config,
                    "--cache-dir",
                    run_dir / "mc",
                    "--model-urdf",
                    run_dir / "robot.urdf",
                ),
            ),
            (
                "pico",
                module("geniesim_teleop.wuji.pico", "--config", config, "--host_ip", host_ip, "--port", args.port),
            ),
        ]
    )
    if args.source == "sdk":
        result.append(
            (
                "hand",
                module(
                    "geniesim_teleop.wuji.sdk",
                    "--config",
                    config,
                    "--left-sn",
                    args.left_sn,
                    "--right-sn",
                    args.right_sn,
                    "--sdk-user",
                    args.sdk_user,
                    "--output-topic",
                    hand_topic,
                ),
            )
        )
    return result


def runtime_preflight(args):
    from geniesim_teleop.utils.network import select_host_ip

    host_ip = select_host_ip(args.interface, args.host_ip)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((host_ip, args.port))
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import geniesim_teleop.wuji.bridge; import geniesim_teleop.wuji.pico",
        ],
        check=True,
    )
    if args.source == "sdk":
        probe = subprocess.run(
            [sys.executable, "-c", "from wuji_sdk import HandModel, Handedness, RetargetSession, SdkManager"],
            capture_output=True,
            text=True,
        )
        if probe.returncode:
            raise RuntimeError(
                f"Wuji SDK unavailable in control Python {sys.executable}:\n{probe.stderr}\n"
                "Rebuild the image with geniesim docker6.2 build, "
                "then recreate the container. Updating the checkout does not update installed packages."
            )
    if not args.external_simulator and shutil.which(os.environ.get("GENIESIM_WUJI_SIM_PY", "python3.12")) is None:
        raise ValueError("Simulator interpreter unavailable; set GENIESIM_WUJI_SIM_PY inside the container")
    import rclpy

    rclpy.init()
    node = rclpy.create_node("wuji_preflight")
    try:
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        topics = ["/joint_command", "/hal/joint_cmd", "/wbc/retarget"]
        if not args.external_simulator:
            topics.append("/joint_states")
        occupied = [topic for topic in topics if node.count_publishers(topic)]
        if occupied:
            raise ValueError(f"ROS domain {args.domain_id} is occupied: {occupied}; choose another --domain-id")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return host_ip


def stop_children(children, simulator=None):
    """Signal only process groups created by this supervisor, then reap them."""
    if simulator is not None:
        try:
            os.killpg(simulator.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    controls = [child for child in children if child is not simulator]
    graceful = False
    for sig, timeout in ((signal.SIGINT, 30.0), (signal.SIGTERM, 5.0), (signal.SIGKILL, 1.0)):
        for child in reversed(controls):
            try:
                os.killpg(child.pid, sig)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + timeout
        for child in controls:
            try:
                child.wait(timeout=max(0.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        if all(child.poll() is not None for child in controls):
            graceful = sig != signal.SIGKILL
            break
    if simulator is not None:
        simulator.wait()
    return graceful


def mark_recordings(directory, existing, keep):
    """Mark only new finalized bags from this session; never rewrite earlier episodes."""
    for path in sorted(directory.glob("*/recording_info.json")):
        if path in existing or not path.parent.name.isdigit():
            continue
        if not (path.parent / "metadata.yaml").is_file():
            raise RuntimeError(f"Recording did not finalize; leaving it unmarked: {path.parent}")
        info = json.loads(path.read_text())
        if "teleop_result" not in info:
            info["teleop_result"] = keep
            path.write_text(json.dumps(info, indent=4, ensure_ascii=False))
            print(f"Recording {'kept' if keep else 'discarded'}: {path.parent}", flush=True)


def run(args, task, host_ip):
    repo = task.benchmark_root.parents[3]
    output = repo / "output/wuji_teleop"
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=time.strftime("%Y%m%d_%H%M%S_"), dir=output))
    config = run_dir / "task.yaml"
    generate_urdf(task, run_dir / "robot.urdf")
    settings = yaml.safe_load(task.config_path.read_text())
    settings.setdefault("teleop", {})["mc_model_urdf"] = "robot.urdf"
    settings["benchmark"].update(
        model_arc="pi", policy_class="DemoPolicy", env_class="DummyEnv", interactive=True, record=False
    )
    settings["benchmark"]["output_dir"] = str(run_dir / "data")
    config.write_text(yaml.safe_dump(settings, sort_keys=False))
    env = os.environ.copy()
    env.update(
        GENIESIM_SKIP_AUTOBOOT="1",
        GENIESIM_TELEOP_CONFIG=str(config),
        GENIESIM_TELEOP_INTERFACE="",
        GENIESIM_TELEOP_HOST_IP="",
        PYTHONUNBUFFERED="1",
    )
    children, logs = [], []
    stopping = False
    keep = None
    recording_dir = repo / "output/recording_data" / task.sub_task_name
    existing_recordings = {
        directory / "recording_info.json" for directory in recording_dir.glob("*") if directory.is_dir()
    }

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        launch_commands = commands(args, config, run_dir, host_ip)
        for name, argv in launch_commands:
            if stopping:
                break
            log = (run_dir / f"{name}.log").open("w")
            logs.append(log)
            child_env = env.copy()
            if name == "simulator":
                child_env["GENIESIM_PY_CMD"] = env.get("GENIESIM_WUJI_SIM_PY", "python3.12")
                # Keep the control-only ROS/NumPy overlay out of the Isaac process.
                child_env["PYTHONPATH"] = ":".join(
                    part
                    for part in env.get("PYTHONPATH", "").split(":")
                    if part
                    and part != "/usr/lib/python3/dist-packages"
                    and not part.startswith("/opt/ros/")
                    and "/app/share/" not in part
                )
            child = subprocess.Popen(
                argv,
                env=child_env,
                cwd=repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            children.append(child)
            print(f"{CYAN}{name}: PID {child.pid}, {run_dir / (name + '.log')}{RST}", flush=True)
        (run_dir / "session.json").write_text(
            json.dumps(
                {
                    "domain_id": args.domain_id,
                    "pico_address": f"{host_ip}:{args.port}",
                    "pids": [child.pid for child in children],
                    "config": str(config),
                },
                indent=2,
            )
        )
        print(
            f"{GREEN}Wuji processes started; Pico {host_ip}:{args.port}, ROS domain {args.domain_id}. Ctrl-C exits.{RST}",
            flush=True,
        )
        print("Press Pico B to start recording.", flush=True)
        if not args.external_simulator:
            print("Enter Y + Enter to keep, N + Enter to discard and stop. Ctrl-C leaves data unmarked.", flush=True)
        if args.source == "retargeting":
            print(f"External retargeter: ROS_DOMAIN_ID={args.domain_id}, output /wuji_hand/joint_command", flush=True)
        while not stopping:
            if not args.external_simulator and sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                decision = sys.stdin.readline().strip().lower()
                if decision in ("y", "n"):
                    keep = decision == "y"
                    break
            for (name, _), child in zip(launch_commands, children):
                if child.poll() is not None:
                    log_path = run_dir / f"{name}.log"
                    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-20:])
                    raise RuntimeError(f"Wuji {name} exited ({child.returncode}); inspect {log_path}\n{tail}")
            time.sleep(0.2)
    finally:
        simulator = children[0] if children and not args.external_simulator else None
        if simulator is not None:
            print(f"{CYAN}Stopping session; waiting for recordings to finish writing.{RST}", flush=True)
        graceful = stop_children(children, simulator)
        for log in logs:
            log.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if keep is not None:
            if graceful and simulator is not None and simulator.returncode == 0:
                mark_recordings(recording_dir, existing_recordings, keep)
            else:
                print(f"{RED}Simulator shutdown did not complete cleanly; recording remains unmarked.{RST}", flush=True)
        print(f"{CYAN}Wuji session stopped. Logs: {run_dir}{RST}", flush=True)


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        task = validate(args)
        os.environ["ROS_DOMAIN_ID"] = str(args.domain_id)
        if args.dry_run:
            print(f"ROS_DOMAIN_ID={args.domain_id}; robot={task.robot_cfg}; source={args.source}")
            for name, argv in commands(args, task.config_path, Path("<session>"), args.host_ip or "<Pico-LAN-IP>"):
                print(f"{name}: {shlex.join(argv)}")
            return
        with open(Path(tempfile.gettempdir()) / f"geniesim-wuji-domain-{args.domain_id}.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            host_ip = runtime_preflight(args)
            if args.check:
                print(
                    f"{GREEN}Wuji preflight passed: {host_ip}:{args.port}, domain {args.domain_id}; no devices started.{RST}"
                )
                return
            run(args, task, host_ip)
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"{RED}Wuji startup failed: {exc}{RST}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
