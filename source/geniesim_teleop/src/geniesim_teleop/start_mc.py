#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0


"""Prepare a task-specific, no-tool MC model from the simulator's URDF."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import yaml

from geniesim_teleop.config.task_config import load_teleop_task, robot_family

APP_ROOT = Path(__file__).resolve().parent / "app"


class MCDumper(yaml.SafeDumper):
    # MC's YAML reader requires sequence items indented beneath mapping keys.
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def write_yaml(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(value, Dumper=MCDumper, sort_keys=False))


def prepare_mc(task, app_root=APP_ROOT, cache_root=None):
    family = robot_family(task.robot_cfg)
    if not family.startswith("G2"):
        raise ValueError(f"The bundled MC supports G2; got {task.robot_cfg}")
    # The shipped binary only supports the G2 platform selector. This is a
    # runtime alias; geometry still comes from the selected task, including crsB.
    mc_robot = "G2"
    app_root = Path(app_root).resolve()
    cache_root = Path(cache_root).resolve() if cache_root else app_root / "bin/.cache/mc"
    profile = cache_root / Path(task.robot_cfg).stem / task.task_name
    # MC resolves models relative to its executable. Each task therefore needs
    # a runtime root containing its selected model.
    (profile / "bin").mkdir(parents=True, exist_ok=True)
    for filename in ("genie_motion_control", "start_mc.sh"):
        shutil.copy2(app_root / "bin" / filename, profile / "bin" / filename)
    for directory in ("lib", "vendors", "conan"):
        target = profile / directory
        if not target.is_symlink():
            target.symlink_to(app_root / directory, target_is_directory=True)
    share = profile / "share"
    share.mkdir(exist_ok=True)
    for child in (app_root / "share").iterdir():
        if child.name in ("robot_model", "genie_robot_description"):
            continue
        target = share / child.name
        if not target.is_symlink():
            target.symlink_to(child, target_is_directory=child.is_dir())
    config_dir = profile / "configuration"
    robot_dir = config_dir / "robot" / mc_robot
    # This directory is generated only; rebuild it so removed overrides do not
    # survive a task/config switch through an old cache.
    if robot_dir.exists():
        shutil.rmtree(robot_dir)
    shutil.copytree(app_root / "configuration/robot/G2", robot_dir)

    # Keep origins, axes, limits, inertias and frames exactly as in the sim URDF.
    # The no-tool teleop controller does not use collision checking; omitting
    # geometry also avoids resolving meshes from a different robot package.
    tree = ET.parse(task.urdf_path)
    for link in tree.getroot().findall("link"):
        for child in list(link):
            if child.tag in ("visual", "collision", "collision_checking"):
                link.remove(child)
    model_dir = share / "robot_model/urdf/G2"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "G2.urdf"
    ET.indent(tree, space="  ")
    tree.write(model_path, encoding="utf-8", xml_declaration=True)

    joints = {joint.get("name"): joint for joint in tree.getroot().findall("joint")}
    for group_name in ("left_arm", "right_arm", "waist"):
        path = robot_dir / f"model/{group_name}.yaml"
        actuator = yaml.safe_load(path.read_text())
        if group_name == "waist":
            actuator["end_frame_names"] = ["arm_base_link"]
        for group in actuator["joints"]:
            for joint in group:
                limits = joints[joint["name"]].find("limit")
                joint["position"] = [float(limits.get("lower")), float(limits.get("upper"))]
                joint["velocity"] = float(limits.get("velocity"))
                # Some simulation URDFs use effort=0 as an unspecified value.
                if float(limits.get("effort")) > 0:
                    joint["torque"] = min(joint["torque"], float(limits.get("effort")))
                if "position_ik" in joint:
                    joint["position_ik"] = [
                        max(joint["position_ik"][0], joint["position"][0]),
                        min(joint["position_ik"][1], joint["position"][1]),
                    ]
        write_yaml(path, actuator)

    srdf_path = model_dir / "G2.srdf"
    ET.ElementTree(ET.Element("robot", name=tree.getroot().get("name", "genie"))).write(
        srdf_path,
        encoding="utf-8",
        xml_declaration=True,
    )
    write_yaml(
        robot_dir / "system_parameter/default.yaml",
        {
            "serial_urdf_file": model_path.name,
            "serial_srdf_file": srdf_path.name,
        },
    )
    quark_path = robot_dir / "quark/quark.yaml"
    quark = yaml.safe_load(quark_path.read_text())
    quark["config"]["dir"] = str(robot_dir)
    for publisher in quark["publishers"]:
        if publisher["name"] == "general_joint_command_ros":
            publisher["enable"] = True
    write_yaml(quark_path, quark)

    settings_path = robot_dir / "se3_track_setting/default.yaml"
    settings = yaml.safe_load(settings_path.read_text())
    settings["vr_task_setting"]["enable_collision_check"] = False
    settings["relexed_ik_setting"]["collision_penalty"] = 0
    # With the bundled solver, the default IK damping (2/1) stalls a small
    # reachable waist pose at the CRS open_door posture (~11 mm residual).
    # Set both IK damping terms to zero; the PVT/LPF layer retains actuator
    # velocity, acceleration and jerk limits.
    settings["relexed_ik_setting"]["damp_v_cost"] = 0.0
    settings["relexed_ik_setting"]["damp_a_cost"] = 0.0
    write_yaml(settings_path, settings)

    manifest = {
        "task": task.task_name,
        "robot_cfg": task.robot_cfg,
        "source_urdf": str(task.urdf_path),
        "source_sha256": hashlib.sha256(task.urdf_path.read_bytes()).hexdigest(),
        "mc_urdf": str(model_path),
        "mc_robot": mc_robot,
        "mc_configuration": str(config_dir),
        "mc_launcher": str(profile / "bin/start_mc.sh"),
    }
    (profile / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=os.environ.get("GENIESIM_TELEOP_CONFIG"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, help="Writable directory for task-specific MC runtimes")
    args = parser.parse_args()
    task = load_teleop_task(args.config)
    manifest = prepare_mc(task, cache_root=args.cache_dir)
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.prepare_only:
        script = Path(manifest["mc_launcher"])
        os.execv(
            "/bin/bash",
            ["bash", str(script), f"--robot={manifest['mc_robot']}", "-C", manifest["mc_configuration"], "--no-tool"],
        )


if __name__ == "__main__":
    main()
