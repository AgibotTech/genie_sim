# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise Docker dispatch and the shared launcher at the process boundary."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def cli(tmp_path):
    calls = tmp_path / "calls.jsonl"
    assets = tmp_path / "assets with spaces"
    assets.mkdir()
    (assets / "pyproject.toml").write_text('[project]\nname = "geniesim_assets"\n')
    docker = tmp_path / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['DOCKER_CALLS'])\n"
        "previous = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []\n"
        "args = sys.argv[1:]\n"
        "with path.open('a') as stream: stream.write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'inspect':\n"
        " if '.Config.Labels' in args[2]:\n"
        "  print(os.environ.get('FAKE_OWNER', os.environ['GENIESIM_REPO_ROOT']) + '|' + "
        "os.environ.get('FAKE_PROFILE', 'post-train'))\n"
        " elif '.State.StartedAt' in args[2]: print('2026-01-01T00:00:00.000000000Z')\n"
        " else:\n"
        "  state = os.environ.get('FAKE_STATE', '')\n"
        "  if any(call[0] == 'run' for call in previous): state = os.environ.get('FAKE_AFTER_RUN', 'running')\n"
        "  if not state: sys.exit(1)\n"
        "  print(state)\n"
        "elif args[0] == 'logs':\n"
        " if os.environ.get('FAKE_READY', '1') == '1': print('GENIESIM_READY')\n"
        " elif os.environ.get('FAKE_OLD_READY') == '1':\n"
        "  if '--since' not in args or args[args.index('--since') + 1] != '2026-01-01T00:00:00.000000000Z':\n"
        "   print('GENIESIM_READY')\n"
        "elif args[0] == 'build': sys.exit(int(os.environ.get('FAKE_BUILD_CODE', '0')))\n"
    )
    docker.chmod(0o755)
    gpu = tmp_path / "nvidia-smi"
    gpu.write_text("#!/bin/sh\nprintf '8.9\\n'\n")
    gpu.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GENIESIM_", "RLINF_", "TELEOP_"))}
    env.update(
        PATH=f"{tmp_path}:{env['PATH']}",
        HOME=str(tmp_path),
        PYTHONPATH=str(ROOT / "source/geniesim_cli/src"),
        GENIESIM_REPO_ROOT=str(ROOT),
        GENIESIM_ASSETS_SRC=str(assets),
        GENIESIM_CACHE_ROOT=str(tmp_path / "cache"),
        DOCKER_CALLS=str(calls),
        NO_COLOR="1",
    )

    def run(*args, stdin="", **overrides):
        calls.unlink(missing_ok=True)
        result = subprocess.run(
            [sys.executable, "-m", "geniesim_cli", *args],
            cwd=tmp_path,
            env=dict(env, **overrides),
            capture_output=True,
            text=True,
            input=stdin,
            timeout=10,
        )
        commands = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, commands

    return run, assets


@pytest.mark.parametrize(
    ("profile", "dockerfile", "image", "override"),
    [
        ("post-train", "Dockerfile.post-train", "geniesim-post-train:server", "BASE_IMAGE=runtime:local"),
        (
            "post-train-teleop",
            "Dockerfile.post-train-teleop",
            "geniesim-post-train:teleop",
            "SERVER_IMAGE=server:local",
        ),
    ],
)
def test_profile_build_uses_the_selected_parent_layer(cli, profile, dockerfile, image, override):
    run, _ = cli
    result, commands = run("docker", "--profile", profile, "build", "--build-arg", override)
    assert result.returncode == 0, result.stdout + result.stderr
    assert commands == [
        ["build", "-f", str(ROOT / "docker" / dockerfile), "-t", image, "--build-arg", override, str(ROOT)]
    ]


@pytest.mark.parametrize("verb", ["docker", "docker6.2"])
def test_default_and_explicit_docker_build_62(cli, verb):
    run, _ = cli
    result, commands = run(verb, "build", "--no-resolve", "--build-arg", "TEST_BUILD_ARG=1")
    assert result.returncode == 0, result.stdout + result.stderr
    assert commands[0][2] == str(ROOT / "docker/Dockerfile")
    assert commands[0][4] == "registry.agibot.com/genie-sim/geniesim4:latest"


def test_profile_propagates_build_failure(cli):
    run, _ = cli
    result, _ = run("docker", "--profile", "post-train", "build", FAKE_BUILD_CODE="17")
    assert result.returncode == 17
    assert "Built " not in result.stdout


@pytest.mark.parametrize(
    "profile, name", [("post-train", "post_train_server"), ("post-train-teleop", "post_train_teleop")]
)
def test_profiles_share_uid_mounts_and_seeded_shader_caches(cli, profile, name):
    run, assets = cli
    result, commands = run("docker", "--profile", profile, "up", "--headless", GENIESIM_IMAGE="review:local")
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(call for call in commands if call[0] == "run")
    assert command[command.index("--name") + 1] == name
    assert command[command.index("--shm-size") + 1] == "8g"
    assert command[-4:] == ["review:local", "tail", "-f", "/dev/null"]
    assert f"{assets}:/geniesim_assets:rw" in command
    assert f"{ROOT}:/workspace:rw" in command
    assert f"HOST_UID={os.getuid()}" in command
    assert f"HOST_GID={os.getgid()}" in command
    assert "GENIESIM_PY_CMD=python3" in command
    assert "ROS_DISTRO=jazzy" in command
    assert f"org.geniesim.repo={ROOT}" in command
    assert f"org.geniesim.profile={profile}" in command
    assert f"{ROOT}/docker/entrypoint.sh:/usr/local/bin/geniesim-entrypoint:ro" in command
    assert "--entrypoint" not in command
    assert not any("/usr/local/bin/geniesim-profile-entrypoint" in arg for arg in command)
    assert f"source={name}-isaacsim-kit-cache,target=/isaac-sim/kit/cache" in command
    assert f"source={name}-ovrtx-cache,target=/usr/local/lib/python3.12/dist-packages/ovrtx/bin/cache" in command
    assert "-e" in command and "DISPLAY" not in command


@pytest.mark.parametrize("verb", ["docker", "docker4.5", "docker5.1", "docker6.2"])
def test_legacy_running_container_does_not_wait_for_profile_readiness(cli, verb):
    run, _ = cli
    result, commands = run(
        verb,
        "up",
        stdin="\n",
        FAKE_STATE="running",
        FAKE_READY="0",
        GENIESIM_START_TIMEOUT="invalid",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "is already running" in result.stdout
    assert len(commands) == 1 and commands[0][:3] == ["inspect", "-f", "{{.State.Status}}"]


@pytest.mark.parametrize(
    "verb, name, image, python, ros",
    [
        ("docker", "geniesim4", "geniesim4:latest", "python3", "jazzy"),
        ("docker5.1", "geniesim3", "geniesim3:latest", "omni_python", "jazzy"),
        ("docker4.5", "geniesim2", "geniesim2:latest", "python3", "humble"),
        ("docker6.2", "geniesim4", "geniesim4:latest", "python3", "jazzy"),
    ],
)
def test_legacy_up_preserves_runtime_defaults_and_workspace_override(cli, verb, name, image, python, ros):
    run, assets = cli
    workspace = f"{ROOT}/docker/.."
    result, commands = run(
        verb,
        "up",
        "--headless",
        stdin="\n",
        GENIESIM_WORKSPACE=workspace,
        GENIESIM_START_TIMEOUT="invalid",
        GENIESIM_SHM_SIZE="8g",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    command = next(call for call in commands if call[0] == "run")
    assert command[command.index("--name") + 1] == name
    assert command[-4:] == [f"registry.agibot.com/genie-sim/{image}", "tail", "-f", "/dev/null"]
    assert f"{workspace}:/workspace:rw" in command
    assert f"{workspace}/docker/entrypoint.sh:/usr/local/bin/geniesim-entrypoint:ro" in command
    assert f"{assets}:/geniesim_assets:rw" in command
    assert f"GENIESIM_PY_CMD={python}" in command
    assert f"ROS_DISTRO={ros}" in command
    assert "--shm-size" not in command and "--entrypoint" not in command and "--label" not in command


@pytest.mark.parametrize("action", ["up", "down", "into", "logs"])
def test_legacy_extra_arguments_keep_the_original_dispatch(cli, action):
    run, _ = cli
    result, _ = run("docker", action, "--unused", FAKE_STATE="running")
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_help_still_checks_the_host_environment(cli):
    run, _ = cli
    result, commands = run("docker", "--help", GENIESIM_IN_CONTAINER="1")
    assert result.returncode == 1
    assert "cannot be used inside the container" in result.stdout
    assert not commands


@pytest.mark.parametrize("verb", ["docker", "docker6.2"])
def test_docker62_help_does_not_require_acknowledgment(cli, verb):
    run, _ = cli
    result, commands = run(verb, "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"geniesim {verb}" in result.stdout
    assert "registry.agibot.com/genie-sim/geniesim4:latest" in result.stdout
    assert "issues/201" not in result.stdout
    assert "Press [ENTER]" not in result.stdout
    assert not commands


def test_docker51_help_keeps_its_combo_prebuilt_notice(cli):
    run, _ = cli
    result, commands = run("docker5.1", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "registry.agibot.com/genie-sim/geniesim3:latest" in result.stdout
    assert "issues/201" in result.stdout
    assert not commands


@pytest.mark.parametrize("action", ["up", "down", "into", "logs"])
def test_default_lifecycle_matches_explicit_docker62(cli, action):
    run, _ = cli
    default, default_commands = run("docker", action, FAKE_STATE="running")
    explicit, explicit_commands = run("docker6.2", action, FAKE_STATE="running")
    assert default.returncode == explicit.returncode == 0
    assert default_commands == explicit_commands
    assert all("geniesim3" not in argument for command in default_commands for argument in command)
    assert any("geniesim4" in command for command in default_commands)


@pytest.mark.parametrize("verb", ["up", "down", "into", "logs"])
@pytest.mark.parametrize("mismatch", [{"FAKE_OWNER": "/another/checkout"}, {"FAKE_PROFILE": "post-train-teleop"}])
def test_profile_refuses_to_manage_another_checkout_or_profile(cli, verb, mismatch):
    run, _ = cli
    result, commands = run("docker", "--profile", "post-train", verb, FAKE_STATE="running", **mismatch)
    assert result.returncode != 0
    assert "ownership" in result.stdout
    assert commands and all(call[0] == "inspect" for call in commands)


def test_profile_into_uses_the_host_uid(cli):
    run, _ = cli
    result, commands = run("docker", "--profile", "post-train", "into", FAKE_STATE="running")
    assert result.returncode == 0, result.stdout + result.stderr
    assert commands[-1] == [
        "exec",
        "-it",
        "-u",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/home/isaac-sim",
        "-w",
        "/workspace",
        "post_train_server",
        "bash",
        "-l",
    ]


def test_profile_down_removes_its_own_container(cli):
    run, _ = cli
    result, commands = run("docker", "--profile", "post-train", "down", FAKE_STATE="running")
    assert result.returncode == 0, result.stdout + result.stderr
    assert commands[-2:] == [["stop", "post_train_server"], ["rm", "-f", "post_train_server"]]


@pytest.mark.parametrize(
    "args",
    [
        ["docker", "--profile", "unknown", "up"],
        ["docker", "--profile"],
        ["docker5.1", "--profile", "post-train", "up"],
        ["docker", "--profile", "post-train", "up", "--typo"],
    ],
)
def test_invalid_selection_or_flags_cannot_start_a_container(cli, args):
    run, _ = cli
    result, commands = run(*args)
    assert result.returncode != 0
    assert not commands


@pytest.mark.parametrize("after_run, timeout", [("exited", "300"), ("running", "1")])
def test_up_does_not_report_success_before_entrypoint_readiness(cli, after_run, timeout):
    run, _ = cli
    result, _ = run(
        "docker",
        "--profile",
        "post-train",
        "up",
        "--headless",
        FAKE_READY="0",
        FAKE_AFTER_RUN=after_run,
        GENIESIM_START_TIMEOUT=timeout,
    )
    assert result.returncode != 0
    assert "did not become ready" in result.stdout


def test_profile_help_does_not_prompt_or_call_docker(cli):
    run, _ = cli
    result, commands = run("docker", "--profile", "post-train", "--help")
    assert result.returncode == 0
    assert "Dockerfile.post-train" in result.stdout
    assert not commands


def test_profile_invalid_timeout_cannot_create_a_container(cli):
    run, _ = cli
    result, commands = run("docker", "--profile", "post-train", "up", "--headless", GENIESIM_START_TIMEOUT="invalid")
    assert result.returncode == 2
    assert all(call[0] != "run" for call in commands)


def test_ready_marker_does_not_hide_an_exited_container(cli):
    run, _ = cli
    result, _ = run("docker", "--profile", "post-train", "up", "--headless", FAKE_AFTER_RUN="exited")
    assert result.returncode != 0
    assert "did not become ready" in result.stdout


def test_repeated_up_waits_for_an_existing_container(cli):
    run, _ = cli
    result, commands = run(
        "docker",
        "--profile",
        "post-train",
        "up",
        "--headless",
        FAKE_STATE="running",
        FAKE_READY="0",
        GENIESIM_START_TIMEOUT="1",
    )
    assert result.returncode != 0
    assert "did not become ready" in result.stdout
    assert all(call[0] != "run" for call in commands)


@pytest.mark.parametrize("profile", ["post-train", "post-train-teleop"])
def test_repeated_up_ignores_ready_markers_from_before_restart(cli, profile):
    run, _ = cli
    result, commands = run(
        "docker",
        "--profile",
        profile,
        "up",
        "--headless",
        FAKE_PROFILE=profile,
        FAKE_STATE="running",
        FAKE_READY="0",
        FAKE_OLD_READY="1",
        GENIESIM_START_TIMEOUT="1",
    )
    assert result.returncode != 0
    assert "did not become ready" in result.stdout
    assert all(call[0] != "run" for call in commands)
