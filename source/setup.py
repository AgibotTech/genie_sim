# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0


"""Installation script for the 'geniesim' python package."""

# 1. install isaacsim
# reference: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html#installing-isaac-lab

# 2. pip install -e ./source

# OTHERS:
# conda install -c conda-forge libstdcxx-ng=13

import os
from setuptools import setup

SETUP_PATH = os.path.dirname(os.path.realpath(__file__))
REQUIREMENTS_PATH = os.path.join(SETUP_PATH, "../requirements.txt")


def read_requirements(file_path):
    requirements = []
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirements.append(line)
    return requirements


# Minimum dependencies required prior to installation
REQUIREMENTS = read_requirements(REQUIREMENTS_PATH)
# REQUIREMENTS.append(f"ik_solver @ file://{WHEEL_PATH}")

skip_deps = os.environ.get("SKIP_DEPS", "0") == "1"
if skip_deps:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = REQUIREMENTS

# Installation operation
setup(
    name="GenieSim",
    author="Genie Sim Team",
    maintainer="Genie Sim Team",
    url="https://github.com/AgibotTech/genie_sim",
    version="3.0.0",
    description="geniesim",
    keywords=["agibot", "genie", "sim", "benchmark"],
    license="Mozilla Public License Version 2.0",
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=INSTALL_REQUIRES,
    packages=["geniesim"],
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
