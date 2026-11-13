# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Check the shipped optional source packager's universal release contract."""

from pathlib import Path
import runpy
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recipe(tmp_path, monkeypatch):
    pytest.importorskip("tomllib")  # The source recipe requires CPython 3.12.
    subprocess.run(
        ["git", "apply", "--include=tools/wheel_cuda128/*.py", str(
            ROOT / "3rdparty/libuipc/patches/0002-agibot-full-abd-cuda128-wheel.patch"
        )],
        cwd=tmp_path,
        check=True,
    )
    tools = tmp_path / "tools/wheel_cuda128"
    monkeypatch.syspath_prepend(str(tools))
    return runpy.run_path(str(tools / "package_wheel.py"))


def test_universal_source_wheel_has_no_architecture_in_filename(recipe, tmp_path):
    config = recipe["Config"](
        arch="universal", **{name: tmp_path for name in (
            "source", "build", "python", "cuda", "clean_lib", "output", "work"
        )}
    )
    assert config.wheel_name == (
        "pyuipc-0.9.0+pdfopt.ef5bfa2.cu128.licensefix2.pathfix1-cp312-cp312-linux_x86_64.whl"
    )
    assert config.architectures == ["89", "120"]
    recipe["validate_cubin_architectures"]("sm_89.cubin sm_120.cubin", config.architectures)
    for incomplete in ("sm_89.cubin", "sm_120.cubin", "sm_89.cubin sm_120.cubin sm_90.cubin"):
        with pytest.raises(RuntimeError, match="cubin architectures differ"):
            recipe["validate_cubin_architectures"](incomplete, config.architectures)
