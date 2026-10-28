# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""SDK-version-specific Kit launch requirements, shared by rendering and physics."""

from importlib import metadata
import os
from pathlib import Path
import sys


def prepare_launch_config(config: dict) -> dict:
    """Keep explicit Replicator captures working with Isaac Sim 6.2.

    The 6.2 multi-tick renderer stalls after stage replacement and fails to
    advance to Replicator's scheduled frame. The same scene with legacy tick
    scheduling returns RGB and metre-valued depth, including mixed-height tiled
    captures. Set this before Kit starts so its initial viewport uses the same
    scheduling contract as later render products. Legacy 5.1/4.5 stays unchanged.
    This is an SDK compatibility requirement, not a physics or pacing knob.
    """
    result = dict(config)
    try:
        root = os.environ.get("ISAAC_PATH")
        if root and Path(sys.executable).resolve().is_relative_to(Path(root).resolve() / "kit/python"):
            # Bundled Python has no wheel metadata and may inherit another SDK's paths.
            installed_version = (Path(root) / "VERSION").read_text().strip()
        else:
            installed_version = metadata.version("isaacsim")
        version = tuple(int(part) for part in installed_version.split(".")[:2])
    except (metadata.PackageNotFoundError, OSError, ValueError):
        return result
    if version == (6, 2):
        args = list(result.get("extra_args", []))
        args.append("--/rtx/hydra/supportMultiTickRate=false")
        result["extra_args"] = args
    return result
