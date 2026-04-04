# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Install metadata and dependencies.

Only local ``sharp`` is declared as a pip dependency. DA360 is loaded at runtime
from ``external/DA360`` (or ``--da360-root`` / ``GENIESIM_DA360_ROOT``).
"""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

_ROOT = Path(__file__).resolve().parent
_EXT = _ROOT.parent / "external"


def _path_dep(dist_name: str, folder: str) -> str:
    loc = (_EXT / folder).resolve()
    if not loc.is_dir():
        raise FileNotFoundError(
            f"Missing {dist_name} checkout: {loc}\n"
            "Expected layout: <parent>/external/{ml-sharp,DA360} next to this package."
        )
    return f"{dist_name} @ {loc.as_uri()}"


setup(
    name="geniesim-world",
    version="0.1.0",
    description="Add-on for Apple ml-sharp: depth-guided predictor + panorama pipeline CLI (no vendored sharp).",
    long_description=(_ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        _path_dep("sharp", "ml-sharp"),
        "click>=8.0",
        "einops>=0.7",
        "numpy>=1.23",
        "opencv-python-headless>=4.8",
        "pillow>=9.0",
        "scipy>=1.9",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        "xformers==0.0.32.post2",
        "tqdm>=4.60",
    ],
    extras_require={
        "dev": ["ruff>=0.4", "pytest>=7.0"],
    },
    entry_points={
        "console_scripts": [
            "geniesim_world=geniesim_world.cli_pano:cli",
        ],
    },
)
