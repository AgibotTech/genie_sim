# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Connect the normal chef task entrypoint to independent layout engines."""

import json
from pathlib import Path
import tempfile
import sys

import yaml

from .layout_batch import load_plans, run_batch
from .layouts import generate_layouts
from ..rt_engine.session import resolve_assets, resolve_bringup_file


def prepare_instances(config_path, config, output, assets, infer_host):
    """Freeze the effective config and generate distinct, reproducible layouts."""
    benchmark = config["benchmark"]
    count, seed = benchmark["num_instances"], benchmark.get("seed", 20261021)
    if type(count) is not int or not 1 <= count <= 1000:
        raise ValueError("benchmark.num_instances must be in [1, 1000]")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    effective = output / "config.yaml"
    effective.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    bundle = output / "bundle"
    generate_layouts(
        output_dir=bundle,
        source_scene=resolve_bringup_file(config["scene"], config_path),
        source_launcher=resolve_bringup_file(config["launcher"], config_path),
        source_evaluation=effective,
        count=count,
        seed=seed,
    )
    return load_plans(bundle, assets, output / "instances", infer_host, python_cmd=sys.executable)


def run_instances(options, config_path, config):
    if options.manifest is not None:
        raise ValueError("layout instances require fresh engines; do not attach a shared manifest")
    assets = resolve_assets(config_path, options.assets_root)
    if options.dry_run:
        with tempfile.TemporaryDirectory(prefix="wuji-layout-plan-") as temporary:
            plans = prepare_instances(config_path, config, Path(temporary) / "plan", assets, options.infer_host)
            print(json.dumps({"num_instances": len(plans), "layouts": [
                {"instance_id": p.layout_id, "scene": yaml.safe_load(p.scene.read_text())["robot"]["init_base_pose"]}
                for p in plans
            ]}, indent=2))
        return 0
    plans = prepare_instances(config_path, config, options.output_dir, assets, options.infer_host)
    summary = run_batch(plans, startup_timeout_s=config["ready_timeout_s"])
    return int(any(row["status"] != "evaluated" for row in summary["layouts"]))
