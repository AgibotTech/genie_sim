# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Check persistent repository settings without touching host configuration or the network."""

import configparser
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("configure_mirrors", ROOT / "docker/configure_mirrors.py")
mirrors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mirrors)

CHINA_APT = "https://mirrors.tuna.tsinghua.edu.cn/ubuntu"
CHINA_PIP = "https://pypi.tuna.tsinghua.edu.cn/simple"
UPSTREAM_PIP = "https://pypi.org/simple"


@pytest.fixture(params=["list", "sources"])
def ubuntu(tmp_path, request):
    directory = tmp_path / "etc/apt/sources.list.d"
    directory.mkdir(parents=True)
    if request.param == "list":
        path = tmp_path / "etc/apt/sources.list"
        path.write_text(
            "# Ubuntu repositories\n"
            "deb [arch=amd64] http://us.archive.ubuntu.com/ubuntu noble main universe\n"
            "deb-src http://security.ubuntu.com/ubuntu noble-security main # security\n"
        )
    else:
        path = directory / "ubuntu.sources"
        path.write_text(
            "Types: deb deb-src\n"
            "URIs: http://us.archive.ubuntu.com/ubuntu\n"
            "Suites: noble noble-updates noble-backports\n"
            "Components: main universe restricted multiverse\n"
            "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n\n"
            "Types: deb\n"
            "URIs: http://security.ubuntu.com/ubuntu\n"
            "Suites: noble-security\n"
            "Components: main universe\n"
            "Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
        )
    third_party = directory / "ros2.list"
    third_party.write_text("deb http://packages.ros.org/ros2/ubuntu noble main\n")
    return tmp_path, path, third_party


def test_base_selection_persists_then_derived_inherits_and_can_restore(ubuntu):
    root, path, third_party = ubuntu
    untouched = third_party.read_bytes()
    mirrors.configure(root, "upstream", UPSTREAM_PIP)
    upstream = path.read_bytes()
    mirrors.configure(root, CHINA_APT, CHINA_PIP)
    assert CHINA_APT in path.read_text()
    assert "archive.ubuntu.com" not in path.read_text()
    selected = path.read_bytes()
    mirrors.configure(root)
    mirrors.configure(root, "", "")
    assert path.read_bytes() == selected
    assert _pip_index(root) == CHINA_PIP
    mirrors.configure(root, "upstream", UPSTREAM_PIP)
    assert path.read_bytes() == upstream
    assert _pip_index(root) == UPSTREAM_PIP
    assert third_party.read_bytes() == untouched


def _pip_index(root):
    config = configparser.ConfigParser()
    config.read(root / "etc/pip.conf")
    return config.get("global", "index-url")


def test_reapplying_selection_preserves_added_components_and_other_pip_options(ubuntu):
    root, path, _ = ubuntu
    pip = root / "etc/pip.conf"
    pip.write_text("[global]\ntimeout = 60\nextra-index-url = https://pypi.nvidia.com\n")
    mirrors.configure(root, CHINA_APT, CHINA_PIP)
    path.write_text(path.read_text().replace("main universe", "main universe multiverse"))
    mirrors.configure(root, CHINA_APT, CHINA_PIP)
    once = path.read_bytes()
    mirrors.configure(root, CHINA_APT, CHINA_PIP)
    assert path.read_bytes() == once
    assert "main universe multiverse" in path.read_text()
    assert "timeout = 60" in pip.read_text()
    assert "extra-index-url = https://pypi.nvidia.com" in pip.read_text()
    assert pip.stat().st_mode & 0o777 == 0o644


def test_inline_third_party_stanzas_remain_unchanged(ubuntu):
    root, path, _ = ubuntu
    if path.suffix == ".sources":
        unrelated = (
            "\nTypes: deb\nURIs: http://packages.ros.org/ros2/ubuntu\n"
            "Suites: noble\nComponents: main\nSigned-By: /usr/share/keyrings/ros-archive-keyring.gpg\n"
        )
    else:
        unrelated = "deb http://packages.ros.org/ros2/ubuntu noble main\n"
    path.write_text(path.read_text() + unrelated)
    mirrors.configure(root, CHINA_APT)
    mirrors.configure(root, "upstream")
    assert unrelated in path.read_text()


def test_mixed_deb822_suites_restore_archive_and_security_separately(tmp_path):
    path = tmp_path / "etc/apt/sources.list.d/ubuntu.sources"
    path.parent.mkdir(parents=True)
    path.write_text(
        f"Types: deb\nURIs: {CHINA_APT}\nSuites: noble noble-security\n"
        "Components: main\nSigned-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg\n"
    )
    mirrors.configure(tmp_path, "upstream")
    assert "http://archive.ubuntu.com/ubuntu\nSuites: noble\n" in path.read_text()
    assert "http://security.ubuntu.com/ubuntu\nSuites: noble-security\n" in path.read_text()


def test_ports_restore_uses_the_ports_archive(ubuntu, monkeypatch):
    root, path, _ = ubuntu
    monkeypatch.setattr(mirrors.platform, "machine", lambda: "aarch64")
    mirrors.configure(root, CHINA_APT)
    mirrors.configure(root, "upstream")
    assert "http://ports.ubuntu.com/ubuntu-ports" in path.read_text()


@pytest.mark.parametrize("bad", ["file:///tmp/index", "not-a-url", "https://pypi.org/simple\nextra"])
def test_invalid_values_fail_before_changing_sources(ubuntu, bad):
    root, path, _ = ubuntu
    original = path.read_bytes()
    with pytest.raises(ValueError, match="HTTP"):
        mirrors.configure(root, CHINA_APT, bad)
    assert path.read_bytes() == original


def test_missing_ubuntu_sources_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="No managed Ubuntu"):
        mirrors.configure(tmp_path, CHINA_APT)


def test_pip_file_overrides_user_configuration_but_allows_environment_override(tmp_path):
    mirrors.configure(tmp_path, pip_index=CHINA_PIP)
    home = tmp_path / "runtime-user"
    user_config = home / ".config/pip/pip.conf"
    user_config.parent.mkdir(parents=True)
    user_config.write_text(f"[global]\nindex-url = {UPSTREAM_PIP}\n")
    env = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")}
    env.update(HOME=str(home), XDG_CONFIG_HOME=str(home / ".config"), PIP_CONFIG_FILE=str(tmp_path / "etc/pip.conf"))
    command = [
        sys.executable,
        "-c",
        "from pip._internal.commands import create_command; "
        "print(create_command('install').parse_args([])[0].index_url)",
    ]
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=True)
    assert result.stdout.strip() == CHINA_PIP
    result = subprocess.run(
        command, env=dict(env, PIP_INDEX_URL=UPSTREAM_PIP), text=True, capture_output=True, check=True
    )
    assert result.stdout.strip() == UPSTREAM_PIP


@pytest.mark.parametrize("name", ["Dockerfile", "Dockerfile.5.1", "Dockerfile.4.5"])
def test_base_recipes_install_the_helper_before_using_package_managers(name):
    text = (ROOT / "docker" / name).read_text()
    assert "ARG APT_MIRROR=upstream" in text
    assert "ARG PIP_INDEX_URL=https://pypi.org/simple" in text
    assert "ENV PIP_CONFIG_FILE=/etc/pip.conf" in text
    assert text.index("COPY docker/configure_mirrors.py") < text.index("RUN apt-get update")
    assert 'geniesim-configure-mirrors --apt "${APT_MIRROR}"' in text
    assert "pip config set" not in text


@pytest.mark.parametrize("name", ["Dockerfile.post-train", "Dockerfile.post-train-teleop"])
def test_derived_recipes_inherit_before_installing_dependencies(name):
    text = (ROOT / "docker" / name).read_text()
    assert "ARG PIP_INDEX_URL\n" in text
    assert "ARG APT_MIRROR\n" in text
    assert "ARG PIP_INDEX_URL=" not in text
    assert text.index("RUN geniesim-configure-mirrors") < text.index("RUN pip3 install")
