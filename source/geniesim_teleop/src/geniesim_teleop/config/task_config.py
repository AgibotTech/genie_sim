# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve the task's robot without importing Isaac Sim or ROS."""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import runpy
import tempfile

import yaml


def benchmark_root():
    repo = os.environ.get("SIM_REPO_ROOT")
    source = Path(repo) / "source" if repo else Path(__file__).resolve().parents[4]
    root = source / "geniesim_benchmark/src/geniesim_benchmark"
    if not root.is_dir():
        import importlib.util

        spec = importlib.util.find_spec("geniesim_benchmark")
        if spec and spec.submodule_search_locations:
            root = Path(spec.submodule_search_locations[0])
        else:
            raise ValueError("Cannot locate geniesim_benchmark; set SIM_REPO_ROOT to the repository root")
    return root


def robot_family(robot_cfg):
    name = Path(robot_cfg).stem
    for family in ("G2_crsB_90d", "G2_crsB_omnipicker", "G2_omnipicker", "G2_crsB_wuji"):
        if name == family or name.startswith(family + "_"):
            return family
    raise ValueError(f"Unsupported teleop robot: {robot_cfg}")


@dataclass
class TeleopTask:
    config_path: Path
    benchmark_root: Path
    task_name: str
    sub_task_name: str
    robot_cfg: str
    robot_config: dict
    urdf_path: Path

    def initial_states(self):
        states = runpy.run_path(str(self.benchmark_root / "benchmark/config/robot_init_states.py"))["TASK_INFO_DICT"]
        family = robot_family(self.robot_cfg)
        for name in (self.sub_task_name, self.task_name):
            state = states.get(name, {}).get(family)
            if state is not None:
                for key, size in (("init_arm", 14), ("body_state", 5), ("head_state", 3)):
                    values = state.get(key, [])
                    if len(values) != size or not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Task {name}/{family} requires {size} finite {key} joint positions")
                return deepcopy(state)
        raise ValueError(f"No initial states for {self.task_name}/{self.sub_task_name} with {family}")

    def mc_initial_states(self, model_urdf=None):
        """Preserve task wrist poses while preparing one IK-compatible reset."""
        from geniesim_teleop.config.arm_reset import prepare_arm_reset

        state = self.initial_states()
        if robot_family(self.robot_cfg) == "G2_crsB_wuji":
            if model_urdf is None:
                settings = yaml.safe_load(self.config_path.read_text())
                model_urdf = settings.get("teleop", {}).get("mc_model_urdf")
            if model_urdf is None:
                # Use the canonical Wuji MC model for direct launches too.
                from geniesim_teleop.wuji.model import generate_urdf

                with tempfile.TemporaryDirectory(prefix="wuji-reset-") as directory:
                    model = generate_urdf(self, Path(directory) / "robot.urdf")
                    state["init_arm"] = prepare_arm_reset(model, state["init_arm"])
                return state
        model = Path(model_urdf).expanduser() if model_urdf is not None else self.urdf_path
        if not model.is_absolute():
            model = self.config_path.parent / model
        state["init_arm"] = prepare_arm_reset(model, state["init_arm"])
        return state


def load_teleop_task(config_path=None, robot_cfg=None):
    root = benchmark_root()
    config_path = (
        Path(config_path or os.environ.get("GENIESIM_TELEOP_CONFIG") or root / "config/teleop.yaml")
        .expanduser()
        .resolve()
    )
    benchmark = yaml.safe_load(config_path.read_text())["benchmark"]
    task_name = benchmark["task_name"]
    task = json.loads((root / "benchmark/config/eval_tasks" / f"{task_name}.json").read_text())
    selected = task["robot"]["robot_cfg"]
    if robot_cfg and robot_cfg != selected:
        raise ValueError(f"--robot_cfg {robot_cfg} conflicts with task {task_name}: {selected}")
    robot_family(selected)
    robot_dir = root / "app/robot_cfg"
    robot_config = json.loads((robot_dir / selected).read_text())
    urdf = robot_dir / robot_config["robot"]["urdf_name"].lstrip("/")
    if not urdf.is_file():
        raise ValueError(f"Task robot URDF does not exist: {urdf}")
    return TeleopTask(
        config_path, root, task_name, benchmark.get("sub_task_name", ""), selected, robot_config, urdf.resolve()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("GENIESIM_TELEOP_CONFIG"))
    args = parser.parse_args()
    try:
        task = load_teleop_task(args.config)
        task.mc_initial_states()
        print(f"Task: {task.task_name}/{task.sub_task_name}; robot: {task.robot_cfg}")
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
