# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Generate reproducible Wuji chef evaluation scenes without mutating teleop inputs."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil

import yaml

JOINT_GROUPS = {
    "left": tuple(f"idx{21 + i:02d}_arm_l_joint{i + 1}" for i in range(7)),
    "right": tuple(f"idx{61 + i:02d}_arm_r_joint{i + 1}" for i in range(7)),
    "body": tuple(f"idx{i + 1:02d}_body_joint{i + 1}" for i in range(5)),
    "head": tuple(f"idx{11 + i:02d}_head_joint{i + 1}" for i in range(3)),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pose(path: Path) -> tuple[dict, dict[str, dict[str, float]]]:
    pose = yaml.safe_load(path.read_text())
    if not isinstance(pose, dict) or pose.get("units") != "radians":
        raise ValueError("pose must declare units: radians")
    joints = {}
    for group, names in JOINT_GROUPS.items():
        values = pose.get(group)
        if not isinstance(values, list) or len(values) != len(names):
            raise ValueError(f"pose {group} must contain exactly {len(names)} joint values")
        for name, value in zip(names, values):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"pose {name} must be a finite number in radians")
            if not math.isfinite(math.degrees(value)):
                raise ValueError(f"pose {name} overflows scene degrees")
            joints[name] = {"radians": float(value), "degrees": math.degrees(value)}
    return pose, joints


def _scene_pose(scene: dict) -> tuple[dict, dict[str, dict[str, float]]]:
    """Describe effective arm/body/head values without changing explicit scene keys."""
    initial = scene["robot"].get("init_joint_pos")
    if not isinstance(initial, dict):
        raise ValueError("scene must contain a robot.init_joint_pos mapping")
    pose = {"pose_id": "source_scene", "units": "radians"}
    joints = {}
    for group, names in JOINT_GROUPS.items():
        pose[group] = []
        for name in names:
            value = initial.get(name, 0.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"scene {name} must be a finite number in degrees")
            radians = math.radians(value)
            joints[name] = {"radians": radians, "degrees": float(value)}
            pose[group].append(radians)
    return pose, joints


def _scene_text(source: str, *, x: float, y: float, joints: dict[str, dict[str, float]] | None = None) -> str:
    """Replace base XY; preserve the complete joint block unless a pose was supplied."""
    start = source.index("  init_base_pose:\n")
    end = source.index("  init_joint_pos:\n", start)
    base = yaml.safe_load(source)["robot"]["init_base_pose"]
    base.update(x=x, y=y)
    new_base = "  # Seeded base XY in world metres; z and yaw retain the source values.\n"
    new_base += "  init_base_pose:\n"
    new_base += f"    x: {x:.4f}\n    y: {y:.4f}\n"
    new_base += f"    z: {base['z']!r}\n    theta: {base['theta']!r}\n"
    if joints is None:
        new_base += "  # Original source joint initialization and unspecified zero defaults retained.\n"
    else:
        new_base += "  # User pose in degrees (input radians); original hand initialization retained.\n"
    source = source[:start] + new_base + source[end:]
    if joints is None:
        return source
    # The source's init_joint_pos is the last robot field. Stop before the
    # first root-level line, including comments, to retain the camera/docs.
    match = re.search(r"(?ms)^  init_joint_pos:\n.*?(?=^[^ \n]|\Z)", source)
    if match is None:
        raise ValueError("scene must contain a robot.init_joint_pos block")
    initial = yaml.safe_load(source)["robot"]["init_joint_pos"]
    initial.update({name: value["degrees"] for name, value in joints.items()})
    replacement = "  init_joint_pos:\n" + "".join(f"    {name}: {value!r}\n" for name, value in initial.items()) + "\n"
    return source[: match.start()] + replacement + source[match.end() :]


def generate_layouts(
    *,
    output_dir: Path,
    source_scene: Path,
    source_launcher: Path,
    source_evaluation: Path,
    pose_path: Path | None = None,
    rules_path: Path,
    count: int = 20,
    seed: int = 20261021,
) -> dict:
    """Write a portable bundle following the supplied script's XY-only rule.

    The shell script is reference data, never executed. All paths in the
    bundle are relative; each unique scene stem owns a separate staged USD
    and manifest. With no pose input, preserve the source's explicit joint
    block and implicit zero defaults. Refuse replacement of an existing output directory.
    """
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("count must be a positive integer no greater than 1000")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    inputs = {
        "scene": Path(source_scene).expanduser().resolve(strict=True),
        "launcher": Path(source_launcher).expanduser().resolve(strict=True),
        "evaluation": Path(source_evaluation).expanduser().resolve(strict=True),
        "rules": Path(rules_path).expanduser().resolve(strict=True),
    }
    if pose_path is not None:
        inputs["pose"] = Path(pose_path).expanduser().resolve(strict=True)
    rules = inputs["rules"].read_text()
    ranges = re.findall(r"^BASE_XY_RANGE\s*=\s*([0-9.eE+-]+)\b", rules, re.MULTILINE)
    if len(ranges) != 1 or float(ranges[0]) != 0.05 or "random.uniform(-BASE_XY_RANGE, BASE_XY_RANGE)" not in rules:
        raise ValueError("rules must contain the supported BASE_XY_RANGE = 0.05 uniform XY rule")
    if 'f"{d:.4f}"' not in rules:
        raise ValueError("rules must quantize XY to four decimal places")
    scene_text = inputs["scene"].read_text()
    scene = yaml.safe_load(scene_text)
    launcher = yaml.safe_load(inputs["launcher"].read_text())
    evaluation = yaml.safe_load(inputs["evaluation"].read_text())
    if scene["robot"]["robot_name"] != "G2_crsB_wuji" or evaluation.get("runtime") != "newton":
        raise ValueError("layouts require the G2_crsB_wuji scene and runtime: newton evaluation")
    preserve_source_pose = pose_path is None
    if preserve_source_pose:
        pose, joints = _scene_pose(scene)
        pose_snapshot = yaml.safe_dump(pose, allow_unicode=True, sort_keys=False).encode()
    else:
        pose, joints = _pose(inputs["pose"])
        pose_snapshot = inputs["pose"].read_bytes()
    if evaluation.get("episodes") != 1:
        raise ValueError("each layout requires one fresh-engine episode")
    params = launcher["genie_sim_engine"]["ros__parameters"]
    if params.get("physics_engine_visualizer") != "isaacsim":
        raise ValueError("Wuji layouts require the Isaac Sim renderer")
    if params.get("step_mode") != "lockstep" or params.get("step_lockstep_ticks") != 1:
        raise ValueError("Wuji layouts require lockstep with one tick per action")
    if not params.get("wok_evaluation_json"):
        raise ValueError("Wuji layouts require the original task scoring configuration")
    plugin_name = scene["newton"]["scene_plugin"]
    inputs["plugin"] = (inputs["scene"].parent / plugin_name).resolve(strict=True)
    if Path(plugin_name).name != plugin_name:
        raise ValueError("source scene_plugin must be a sibling filename")
    rng = random.Random(seed)
    samples = []
    for _ in range(count):
        raw = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
        xy = tuple(float(f"{value:.4f}") for value in raw)
        if any(xy == previous[1] for previous in samples):
            raise ValueError("seed produced duplicate four-decimal XY; choose another seed")
        samples.append((raw, xy))
    manifest = {
        "version": 1,
        "task_name": evaluation["task_name"],
        "count": count,
        "seed": seed,
        "pose_id": pose.get("pose_id"),
        "initialization": {
            "mode": "source_scene" if preserve_source_pose else "explicit_pose",
            "source": "scene" if preserve_source_pose else "pose",
            "explicit_joint_values_preserved": preserve_source_pose,
            "implicit_zero_joints": (
                [name for name in joints if name not in scene["robot"]["init_joint_pos"]]
                if preserve_source_pose
                else []
            ),
            "pose_snapshot": {
                "path": "initial_pose.yaml",
                "sha256": hashlib.sha256(pose_snapshot).hexdigest(),
            },
        },
        "generalization": {
            "x_range_m": [-0.05, 0.05],
            "y_range_m": [-0.05, 0.05],
            "decimal_places": 4,
            "assignment": "replace",
            "theta_randomized": False,
        },
        "sources": {key: {"name": path.name, "sha256": _sha256(path)} for key, path in inputs.items()},
        "joints": joints,
        "layouts": [],
    }
    # Assemble next to the destination and publish only a complete bundle.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        for folder in ("scenes", "launchers", "evaluations"):
            (staging / folder).mkdir()
        shutil.copyfile(inputs["plugin"], staging / "scenes" / plugin_name)
        (staging / "initial_pose.yaml").write_bytes(pose_snapshot)
        shutil.copyfile(inputs["rules"], staging / "generalization_rules.sh")
        for layout_id, (raw, (x, y)) in enumerate(samples):
            stem_prefix = "scene_wuji_chef_oldpose_layout" if preserve_source_pose else "scene_wuji_chef_layout"
            stem = f"{stem_prefix}_{layout_id:02d}"
            paths = {
                "scene": f"scenes/{stem}.yaml",
                "launcher": f"launchers/{layout_id:02d}.yaml",
                "evaluation": f"evaluations/{layout_id:02d}.yaml",
            }
            text = _scene_text(scene_text, x=x, y=y, joints=None if preserve_source_pose else joints)
            expected = copy.deepcopy(scene)
            expected["robot"]["init_base_pose"].update(x=x, y=y)
            if not preserve_source_pose:
                expected["robot"]["init_joint_pos"].update({name: value["degrees"] for name, value in joints.items()})
            if yaml.safe_load(text) != expected:
                raise ValueError("generated scene changed fields outside the requested initial pose and base XY")
            (staging / paths["scene"]).write_text(text)
            run_launcher = copy.deepcopy(launcher)
            run_launcher["genie_sim_engine"]["ros__parameters"][
                "isaac_render_log_path"
            ] = f"layout_{layout_id:02d}_isaac.log"
            (staging / paths["launcher"]).write_text(yaml.safe_dump(run_launcher, allow_unicode=True, sort_keys=False))
            run_evaluation = copy.deepcopy(evaluation)
            run_evaluation["launcher"] = f"../{paths['launcher']}"
            (staging / paths["evaluation"]).write_text(
                yaml.safe_dump(run_evaluation, allow_unicode=True, sort_keys=False)
            )
            manifest["layouts"].append(
                {
                    "layout_id": layout_id,
                    **paths,
                    "manifest_subpath": f"background/common/runtime/{stem}/manifest.json",
                    "base_pose": expected["robot"]["init_base_pose"],
                    "raw_base_xy": list(raw),
                    "sha256": {key: _sha256(staging / path) for key, path in paths.items()},
                }
            )
        (staging / "layouts.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        with (staging / "layouts.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["layout_id", "x_m", "y_m", "z_m", "theta_rad", "scene", "evaluation"])
            for layout in manifest["layouts"]:
                base = layout["base_pose"]
                writer.writerow(
                    [
                        layout["layout_id"],
                        base["x"],
                        base["y"],
                        base["z"],
                        base["theta"],
                        layout["scene"],
                        layout["evaluation"],
                    ]
                )
        pose_description = (
            "The complete source joint block is preserved; `initial_pose.yaml` describes its effective 22 joints "
            "in radians, including unspecified zero defaults."
            if preserve_source_pose
            else "The explicit pose supplies 22 joint angles in radians in `initial_pose.yaml` and degrees in scenes."
        )
        (staging / "README.md").write_text(
            f"# Wuji chef evaluation layouts\n\n{count} layouts; seed `{seed}`. "
            "Robot base X/Y independently follow U(-0.05, 0.05) metres on a square, rounded to four decimals. "
            "Base Z/yaw, hands, physics, scoring and action budget retain the source settings. "
            f"{pose_description}\n\n"
            "Run in the prepared ROS/Newton/Isaac Sim environment:\n\n"
            "```bash\ngeniesim benchmark run-layouts --bundle /absolute/path/to/this/bundle "
            "--infer-host HOST:PORT --output-dir /absolute/path/to/new/results\n```\n\n"
            "Use `--dry-run` to inspect the launch plan. Each layout starts a fresh engine. "
            "Results, logs and Isaac Sim observer video are saved separately per layout. "
            "`layouts.json` records source hashes, full joint values and layout config hashes; "
            "`layouts.csv` lists the sampled poses. Generation alone does not validate physical task success.\n"
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _bringup_config() -> Path:
    for ancestor in Path(__file__).resolve().parents:
        directory = ancestor / "source/geniesim_ros/src/ros_ws/src/genie_sim_bringup/config"
        if directory.is_dir():
            return directory
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("genie_sim_bringup")) / "config"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rules", type=Path, required=True, help="reference run_chef_scene.sh; never executed")
    parser.add_argument(
        "--pose", type=Path, help="explicit 22-joint radians override; default preserves the source scene pose"
    )
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument(
        "--evaluation", type=Path, default=Path(__file__).parents[1] / "config/g2crsbwuji_dexmanip_wok_flip.yaml"
    )
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    options = parser.parse_args(argv)
    config_dir = _bringup_config() if options.scene is None or options.launcher is None else None
    try:
        manifest = generate_layouts(
            output_dir=options.output_dir,
            source_scene=options.scene or config_dir / "scene_flat_g2_sp_chef_wuji_eval.yaml",
            source_launcher=options.launcher or config_dir / "launcher_newton_wuji_eval.yaml",
            source_evaluation=options.evaluation,
            pose_path=options.pose,
            rules_path=options.rules,
            count=options.count,
            seed=options.seed,
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(
        f"Generated {manifest['count']} Wuji chef layouts in {options.output_dir.resolve()} (seed {manifest['seed']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
