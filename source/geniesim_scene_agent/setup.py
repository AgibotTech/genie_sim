# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Package the Python tools with their skills, references, and task examples."""

from pathlib import Path
import shutil

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py


class BuildWithResources(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        destination = Path(self.build_lib) / "geniesim_scene_agent" / "resources"

        def ignore(directory, names):
            skipped = {name for name in names if name == "__pycache__" or name.endswith((".pyc", ".pyo"))}
            if Path(directory) == root / "examples":
                skipped.add("assets")
            return skipped

        for name in ("skills", "examples", "docs", "docker", "THIRD_PARTY_LICENSES"):
            shutil.copytree(root / name, destination / name, dirs_exist_ok=True, ignore=ignore)
        for name in ("README.md", "AGENTS.md", "LICENSE", "VERSION", ".dockerignore", "THIRD_PARTY_NOTICES.md"):
            shutil.copy2(root / name, destination / name)


setup(
    packages=find_packages(include=["scene_agent_assets*", "scene_agent_scene*"]) + find_packages(where="src"),
    package_dir={"geniesim_scene_agent": "src/geniesim_scene_agent"},
    cmdclass={"build_py": BuildWithResources},
)
