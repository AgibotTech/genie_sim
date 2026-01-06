# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
from re import L
import yaml
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Type, TypeVar
import geniesim.utils.system_utils as system_utils

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


@dataclass
class BenchmarkConfig:
    num_episode: int = 1
    policy_class: str = "DemoPolicy"
    env_class: str = "DummyEnv"
    task_name: str = ""
    sub_task_name: str = ""
    output_dir: str = "output"
    fps: int = 30
    record: bool = False
    infer_host: str = "127.0.0.1"
    infer_port: str = 8999
    model_arc: str = "pi"
    enable_ros: bool = False
    interactive: bool = False


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


# -------------------- main --------------------
if __name__ == "__main__":
    ps = ParameterServer()

    # 1. Declare default parameters (only need dataclass default values, no need to declare all)
    for f in fields(Config):
        ps.declare_parameter(f.name, None)

    # 2. Load from YAML file
    ps.set_parameters_from_yaml("config.yaml")

    # 3. CLI override
    #    Example: python params.py --app.physics_step 240 --benchmark.task_name "new_task"
    ps.override_from_cli()

    # 4. Automatically build dataclass
    cfg = load_dataclass(Config, ps)

    print("Final parameter dictionary:", ps.as_dict())
    print("Dataclass object:", cfg)
