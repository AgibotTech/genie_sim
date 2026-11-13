# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
from re import L
import yaml
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import Any, Dict, Type, TypeVar
import geniesim_benchmark.utils.system_utils as system_utils

T = TypeVar("T")


class ParameterServer:
    def __init__(self):
        self._params: Dict[str, Any] = {}

    def declare_parameter(self, name: str, default_value: Any = None):
        if name not in self._params:
            self._params[name] = default_value
        return self._params[name]

    def set_parameters_from_yaml(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        self._update_from_dict("", yaml_data)

    def _update_from_dict(self, prefix: str, data: Dict[str, Any]):
        for k, v in data.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                self._update_from_dict(full_key, v)
            else:
                self._params[full_key] = v

    def override_from_cli(self, cli_args=None):
        def _str2bool(v):
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("true", "1", "yes", "on")
            return bool(v)

        parser = argparse.ArgumentParser()

        default_config_path = system_utils.config_path() + "/config.yaml"
        parser.add_argument(
            "--config",
            type=str,
            default=default_config_path,
            metavar="PATH",
            help="path to 'config.yaml' with setup config, " "default: ./config/config.yaml",
        )

        for key, value in self._params.items():
            if isinstance(value, bool):
                parser.add_argument(f"--{key}", type=_str2bool, nargs="?", const=True, default=None)
            else:
                arg_type = type(value) if value is not None else str
                parser.add_argument(f"--{key}", type=arg_type, default=None)
        args, unknown = parser.parse_known_args(cli_args)
        if args.config and default_config_path != args.config:
            self.set_parameters_from_yaml(args.config)

        for key, value in vars(args).items():
            if key == "config" or value is None:
                continue

            self._params[key] = value

    def get(self, name: str, default: Any = None):
        return self._params.get(name, default)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._params)


# -------------------- Automatic dataclass mapping --------------------
def load_dataclass(cls: Type[T], ps: ParameterServer, prefix: str = "") -> T:
    """Recursively build dataclass from ParameterServer"""
    kwargs = {}
    for f in fields(cls):
        key = f"{prefix}.{f.name}" if prefix else f.name
        if is_dataclass(f.type):
            # Nested dataclass
            kwargs[f.name] = load_dataclass(f.type, ps, key)
        else:
            kwargs[f.name] = ps.get(key, f.default)
    return cls(**kwargs)


# -------------------- Dataclass definitions --------------------
@dataclass
class AppConfig:
    headless: bool = False
    livestream: int = -1

    physics_step: int = 120
    rendering_step: int = 60
    enable_curobo: bool = False
    reset_fallen: bool = False
    enable_ros: bool = False
    record_img: bool = False
    record_video: bool = False
    render_mode: str = "RaytracedLighting"
    disable_physics: bool = False
    data_convert: bool = False
    enable_gpu_dynamics: bool = False
    enable_rate_limit: bool = False
    enable_playback: bool = False
    enable_pub_depth_camera: bool = False
    on_demand_render: bool = False
    # FEM cloth/deformable support: the beta PhysX flag, a cuda World, GPU
    # broadphase and the per-tick cloth-pin update. Separate from
    # enable_gpu_dynamics so tasks that only want PhysX GPU rigid-body dynamics
    # keep the backend they already run on.
    enable_deformable: bool = False
    show_pin_sphere: bool = False  # draw the cloth-pin debug sphere (cloth-fold)
    log_jaw_pocket_counts: bool = False  # periodically log live left/right jaw_pocket inside_count (cloth-fold)
    # Keep the robot's gripper collision meshes active during cloth-fold. Default
    # False: cloth_pin disables them so the cloth is driven purely by the pin (not
    # by gripper-vs-cloth contact/friction). Set True to let the grippers physically
    # collide with the cloth.
    enable_gripper_collision: bool = False
    # Optional independent pose stream for the keyboard mock server's map.
    # Exit the process (code 1) when one main-loop world.step() runs longer
    # than this; it can hang forever inside Kit's native app.update(). 0 disables.
    main_loop_stall_timeout_s: float = 300.0


@dataclass
class BenchmarkConfig:
    num_episode: int = 1
    policy_class: str = "DemoPolicy"
    env_class: str = "DummyEnv"
    task_name: str = ""
    sub_task_name: str = ""
    output_dir: str = "output"
    # Resume committed serial episodes in the existing task output directory.
    resume: bool = False
    # Optional scheduler job/model identity; must remain unchanged across retries.
    resume_id: str = ""
    fps: int = 30
    record: bool = False
    # Save replay state with recordings; the third-person camera remains opt-in.
    record_state: bool = True
    record_world_camera: bool = False
    record_video_dir: str = ""
    infer_host: str = "localhost:8999"
    model_arc: str = "pi"
    interactive: bool = False
    seed: int = 1
    preview: bool = False
    enable_vec: int = 0
    shared_cam_render_frames: int = 8
    shared_cam_first_render_frames: int = 8
    # Number of scene instances to sample (deterministically by `seed`).
    # 0 (default) means "use all instances under llm_task/<sub_task>/".
    num_instances: int = 0
    # Exact comma-separated instance ids, in caller order. An explicit value
    # takes precedence over num_instances.
    instance_ids: str = ""
    # realprobe evals target the real robot: CoRobotPolicy drops the "_sim"
    # suffix from the reported robot_type (e.g. g2a_sim -> g2a) so the
    # inference server sees the real-robot embodiment tag. Also settable from
    # the eval json's robot.realprobe flag.
    realprobe: bool = False
    # Number of linearly-interpolated substeps per raw action-chunk waypoint
    # executed by CoRobotPolicy. <=1 disables interpolation (raw passthrough).
    policy_action_interp_substeps: int = 1
    # Wait this many seconds for the arm to settle before collecting the
    # next observation for inference.
    infer_settle_timeout: float = 1.0
    # Overlay semantically-aligned paraphrased instructions from a sidecar
    # config (keyed by sub_task_name + instance_id). Off by default.
    language_perturbation: bool = False
    language_perturbation_config: str = ""
    # Pause the simulation timeline (via omni.timeline) while the policy
    # blocks on an inference round-trip, so objects don't drift under
    # gravity during the wait. Disabled by default.
    pause_physics_during_inference: bool = False
    # Override only the configured gripper control-joint drives on the runtime
    # USD stage. -1 preserves the shared asset values; angular drives use N·m.
    gripper_drive_max_force: float = -1.0


@dataclass
class LayoutConfig:
    seed: int = 0
    autogen_ratio: float = 0.5
    num_obj: int = 1


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)


# -------------------- Recursive declaration helper --------------------
def declare_dataclass_params(cls: Type, ps: ParameterServer, prefix: str = ""):
    for f in fields(cls):
        key = f"{prefix}.{f.name}" if prefix else f.name
        if is_dataclass(f.type):
            declare_dataclass_params(f.type, ps, key)
        else:
            default = f.default if f.default is not MISSING else None
            ps.declare_parameter(key, default)


# -------------------- main --------------------
if __name__ == "__main__":
    ps = ParameterServer()

    # 1. Declare default parameters (recursively register all nested dataclass fields)
    declare_dataclass_params(Config, ps)

    # 2. Load from YAML file
    ps.set_parameters_from_yaml("config.yaml")

    # 3. CLI override
    #    Example: python params.py --app.physics_step 240 --benchmark.task_name "new_task"
    ps.override_from_cli()

    # 4. Automatically build dataclass
    cfg = load_dataclass(Config, ps)

    print("Final parameter dictionary:", ps.as_dict())
    print("Dataclass object:", cfg)
