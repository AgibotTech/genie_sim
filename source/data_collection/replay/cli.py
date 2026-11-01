"""Parse replay options before Isaac Sim is initialized."""

import argparse
import json
from pathlib import Path


def parse_cli_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run replay rounds and convert the recorded output."
    )
    parser.add_argument(
        "--config_json",
        default="replay/config_default.json",
        help="Replay configuration JSON path.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run without a GUI by default; use --no-headless to show it.",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Recording directory containing state.json and recording_info.json.",
    )
    parser.add_argument(
        "--recording_root",
        default="recording_data",
        help="Root directory for replay recordings.",
    )
    parser.add_argument(
        "--deformable-smoothing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override replay_config.no_deformable_smooth.",
    )
    parser.add_argument(
        "--cloth-subdivision-levels",
        type=int,
        choices=range(4),
        default=None,
        help="Cloth rendering subdivision level, from 0 to 3.",
    )
    args = parser.parse_args(argv)
    if not Path(args.input_dir).is_dir():
        parser.error(f"Input directory not found: {args.input_dir}")
    return args


def load_top_config(path):
    config_path = Path(path)
    if not config_path.is_file():
        raise SystemExit(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise SystemExit(f"Config must be a JSON object: {config_path}")
    return config
