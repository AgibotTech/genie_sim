# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Manage GenieSim Docker containers across Isaac Sim variants and post-training profiles.

Dispatched by four CLI commands:

* ``geniesim docker``    — default alias for ``docker5.1``.
* ``geniesim docker6.2`` — Isaac Sim 6.2 variant (geniesim4).
* ``geniesim docker5.1`` — Isaac Sim 5.1 variant (geniesim3).
* ``geniesim docker4.5`` — Isaac Sim 4.5 variant (geniesim2 E.O.L.).
* ``geniesim docker --profile post-train`` — post-training simulation server layer.
* ``geniesim docker --profile post-train-teleop`` — optional teleoperation layer.

The verbs map onto the shell scripts that live next to the Dockerfile in
``<repo_root>/docker/`` (resolved at runtime via :func:`_docker_dir`).
We intentionally keep the heavy lifting in shell — the Python layer is
just a typed dispatcher with consistent styling and env-var injection
(notably ``HOST_UID`` / ``HOST_GID``, used by ``entrypoint.sh`` to remap
the canned ``isaac-sim`` user inside the NVIDIA Isaac Sim base image).

Subcommands:

* ``build``   — ``docker build`` with the variant's Dockerfile from the repo root.
* ``up``      — start the container (GUI by default; ``--headless`` to skip X11).
* ``down``    — stop and remove the container.
* ``into``    — ``docker exec -it <container> bash``.
* ``logs``    — ``docker logs [-f] <container>``.

Container name and image tag are variant-specific (see :data:`VARIANTS`). Both can
be overridden via env vars ``GENIESIM_CONTAINER`` / ``GENIESIM_IMAGE`` for
parity with the shell scripts the verbs delegate to.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from geniesim_cli import _env, _mirrors
from geniesim_cli._style import BOLD, CYAN, DIM, GREEN, MAGENTA, RED, RST, WHITE, YELLOW

# ---------------------------------------------------------------------------
# Public variant registry
#
# ``Variant`` + ``VARIANTS`` + ``get_variant`` + ``variant_env`` are the
# stable API for downstream tooling that layers on top of geniesim_cli —
# in particular ``geniesim_cli_industrial``, which reuses these env vars
# so both CLIs launch containers with the same runtime baseline. Do not
# rename or repurpose fields without a deprecation path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """Build + runtime config for one Isaac Sim variant.

    Fields
    ------
    dockerfile : filename under ``dir`` (relative to the repo root).
    image      : docker image tag used for pull/build/run.
    container  : default container name for ``up``/``into``.
    env        : runtime env vars passed to ``start.sh`` and forwarded
                 into the container by ``entrypoint.sh``. Read-only view
                 (``MappingProxyType``); mutations raise. Excluded from
                 ``__hash__`` because mappings aren't hashable — kept in
                 ``__eq__`` so two Variants with different env still
                 compare unequal (hash collisions are permitted).
    implemented: ``False`` for variants that are not ready for general use
                 (Dockerfile incomplete, image not published). ``run()``
                 prints a banner and waits on ENTER before dispatching --
                 it is an acknowledgement gate, **not** a refusal.
    dir        : docker subdir (currently always ``"docker"``; kept as a
                 field for future flexibility, e.g. per-variant subdir).
    isaacsim_series : Isaac Sim **series spec** (e.g. ``"6.X"``) resolved to
                 concrete ``ISAACSIM_TAG`` / ``ISAACSIM_PIP_VERSION`` build
                 args by ``geniesim_cli._isaacsim`` at build time. ``None``
                 means "don't resolve" — the Dockerfile's literal ``ARG``
                 defaults stand. Appended last and optional, so downstream
                 positional construction keeps working.
    """

    dockerfile: str
    image: str
    container: str
    env: Mapping[str, str] = field(default_factory=dict, hash=False)
    implemented: bool = True
    dir: str = "docker"
    isaacsim_series: str | None = None

    def __post_init__(self) -> None:
        # Enforce read-only env at construction time so ``dataclasses.replace(v,
        # env={...})`` — which bypasses the registry-level ``MappingProxyType``
        # wrapping — still produces an immutable Variant.
        if not isinstance(self.env, MappingProxyType):
            object.__setattr__(self, "env", MappingProxyType(dict(self.env)))


VARIANTS: Mapping[str, Variant] = MappingProxyType(
    {
        "": Variant(
            dockerfile="Dockerfile",
            image="registry.agibot.com/genie-sim/geniesim4:latest",
            container="geniesim4",
            implemented=True,
            # Pin the requested base respin. The resolver explicitly pairs
            # tag ``6.2.0.1`` with the published cp312 wheel ``6.2.0.0``;
            # the Dockerfile carries the same pair for offline builds.
            isaacsim_series="6.2.0.1",
            env=MappingProxyType(
                {
                    "ROS_DISTRO": "jazzy",
                    "GENIESIM_PY_CMD": "python3",
                    "GENIESIM_BREAK_SYSTEM_PKGS": "1",
                    "GENIESIM_CHOWN_KIT_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit",
                    "GENIESIM_EXTRA_CACHE_DIRS": "kit pip numba",
                    "GENIESIM_ISAACSIM_KIT_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache",
                    "GENIESIM_OVRTX_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/ovrtx/bin/cache",
                }
            ),
        ),
        "4.5": Variant(
            dockerfile="Dockerfile.4.5",
            image="registry.agibot.com/genie-sim/geniesim2:latest",
            container="geniesim2",
            env=MappingProxyType(
                {
                    "ROS_DISTRO": "humble",
                    "GENIESIM_PY_CMD": "python3",
                    "GENIESIM_BREAK_SYSTEM_PKGS": "1",
                    "GENIESIM_CHOWN_KIT_PATH": "/usr/local/lib/python3.10/dist-packages/isaacsim/kit",
                    "GENIESIM_EXTRA_CACHE_DIRS": "",
                    "GENIESIM_ISAACSIM_KIT_CACHE_PATH": "/usr/local/lib/python3.10/dist-packages/isaacsim/kit/cache",
                    "GENIESIM_OVRTX_CACHE_PATH": "/usr/local/lib/python3.10/dist-packages/ovrtx/bin/cache",
                }
            ),
        ),
        "5.1": Variant(
            dockerfile="Dockerfile.5.1",
            image="registry.agibot.com/genie-sim/geniesim3:latest",
            container="geniesim3",
            env=MappingProxyType(
                {
                    "ROS_DISTRO": "jazzy",
                    "GENIESIM_PY_CMD": "omni_python",
                    "GENIESIM_BREAK_SYSTEM_PKGS": "0",
                    "GENIESIM_CHOWN_KIT_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit",
                    "GENIESIM_EXTRA_CACHE_DIRS": "kit pip numba",
                    "GENIESIM_ISAACSIM_KIT_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/isaacsim/kit/cache",
                    "GENIESIM_OVRTX_CACHE_PATH": "/usr/local/lib/python3.12/dist-packages/ovrtx/bin/cache",
                }
            ),
        ),
    }
)


def get_variant(name: str) -> Variant:
    """Return the :class:`Variant` for ``name``.

    ``name`` is the variant key (``""``, ``"4.5"``, ``"5.1"``). Raises
    ``KeyError`` with a helpful message on unknown names. Prefer this
    over ``VARIANTS[name]`` at the boundary of another package so the
    error path stays consistent.
    """
    try:
        return VARIANTS[name]
    except KeyError:
        known = ", ".join(sorted(f"'{k}'" for k in VARIANTS)) or "(none)"
        raise KeyError(f"unknown docker variant '{name}' (known: {known})") from None


def variant_env(name: str) -> Mapping[str, str]:
    """Return the runtime env dict for a variant.

    Narrow accessor for downstream CLIs (``geniesim_cli_industrial``)
    that only need the env vars — they don't have to know about the
    rest of the :class:`Variant` shape.
    """
    return get_variant(name).env


PROFILES: Mapping[str, Variant] = MappingProxyType(
    {
        "post-train": Variant(
            dockerfile="Dockerfile.post-train",
            image="geniesim-post-train:server",
            container="post_train_server",
            env={
                **variant_env(""),
                "GENIESIM_SHM_SIZE": "8g",
            },
        ),
        "post-train-teleop": Variant(
            dockerfile="Dockerfile.post-train-teleop",
            image="geniesim-post-train:teleop",
            container="post_train_teleop",
            env={
                **variant_env(""),
                "GENIESIM_SHM_SIZE": "8g",
            },
        ),
    }
)


# Legacy dict-of-dicts alias — kept for one deprecation cycle so
# out-of-tree callers that still poke at ``_VARIANTS`` don't break
# silently. New code should use :data:`VARIANTS` / :func:`get_variant`
# / :func:`variant_env`.
_VARIANTS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
        name: MappingProxyType(
            {
                "dir": v.dir,
                "dockerfile": v.dockerfile,
                "image": v.image,
                "container": v.container,
                "env": v.env,
                "implemented": v.implemented,
            }
        )
        for name, v in VARIANTS.items()
    }
)

_active_variant: str = ""
_active_profile: str = ""
# Command word the user typed at the CLI ("docker", "docker5.1", ...).
# Status/help banners echo this so the user sees the form they invoked,
# not the resolved variant alias.
_invoked_as: str = "docker"


def _cmd_label() -> str:
    return _invoked_as


def _selected_variant() -> Variant:
    return PROFILES[_active_profile] if _active_profile else get_variant(_active_variant)


def _workspace() -> Path:
    return Path(os.environ.get("GENIESIM_WORKSPACE", str(_repo_root()))).expanduser().resolve()


def _image() -> str:
    v = _selected_variant()
    return os.environ.get("GENIESIM_IMAGE", v.image)


def _container() -> str:
    v = _selected_variant()
    return os.environ.get("GENIESIM_CONTAINER", v.container)


def _repo_root() -> Path:
    """Locate the repo root that contains the docker/ dir."""
    override = _env.repo_root()
    if override:
        return Path(override).resolve()

    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "docker" / "Dockerfile").is_file():
            return candidate

    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if (ancestor / "docker" / "Dockerfile").is_file():
            return ancestor

    print(f"{RED}❌ Could not locate the repo root (no docker/Dockerfile found).{RST}")
    print(f"   {DIM}Set {BOLD}$GENIESIM_REPO_ROOT{RST}{DIM} or run from inside the repo.{RST}")
    sys.exit(1)


def _docker_dir() -> Path:
    v = _selected_variant()
    return _repo_root() / v.dir


def _find_assets_src() -> str:
    import importlib.util

    spec = importlib.util.find_spec("geniesim_assets")
    if spec and spec.submodule_search_locations:
        loc = Path(list(spec.submodule_search_locations)[0])
    elif spec and spec.origin:
        loc = Path(spec.origin).parent
    else:
        return ""
    for p in [loc, *loc.parents]:
        if (p / "pyproject.toml").is_file():
            return str(p)
    return ""


def _ensure_docker_cli() -> None:
    if os.environ.get("GENIESIM_IN_CONTAINER") == "1":
        print(f"{RED}❌ 'geniesim {_cmd_label()}' cannot be used inside the container.{RST}")
        print(f"   {DIM}Run this command on the host machine.{RST}")
        sys.exit(1)
    if shutil.which("docker") is None:
        print(f"{RED}❌ docker is not on PATH.{RST}")
        print(f"   {DIM}Install Docker Engine first: https://docs.docker.com/engine/install/{RST}")
        sys.exit(1)


def _container_state() -> str | None:
    """Return ``'running'`` / ``'exited'`` / ``...`` or ``None`` if absent."""
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", _container()],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    if _active_profile:
        owner = subprocess.run(
            [
                "docker",
                "inspect",
                "-f",
                '{{index .Config.Labels "org.geniesim.repo"}}|{{index .Config.Labels "org.geniesim.profile"}}',
                _container(),
            ],
            capture_output=True,
            text=True,
        )
        if owner.returncode != 0 or owner.stdout.strip() != f"{_workspace()}|{_active_profile}":
            print(f"{RED}❌ Container {_container()} has different checkout/profile ownership.{RST}")
            print(f"   {DIM}Choose another GENIESIM_CONTAINER or use the original checkout/profile.{RST}")
            sys.exit(1)
    return result.stdout.strip() or None


def _print_usage() -> None:
    label = _cmd_label()
    v = _selected_variant()
    print(f"{BOLD}{MAGENTA}🐳 geniesim {label}{RST}")
    print()
    print(f"{BOLD}Usage:{RST} geniesim {label} {CYAN}<subcommand>{RST} [args...]")
    print()
    print(f"{BOLD}Subcommands:{RST}")
    dockerfile = f"{v.dir}/{v.dockerfile}" if _active_profile else "docker/Dockerfile"
    print(f"  {CYAN}build{RST} [--mirrors NAME | --china] [docker-build args...]  🔨 Build {BOLD}{dockerfile}{RST}")
    print(f"  {CYAN}up{RST} [--headless]                  🚀 Start the container (GUI by default)")
    print(f"  {CYAN}down{RST}                              🛑 Stop and remove the container")
    print(f"  {CYAN}into{RST}                              📖 Drop into a shell in the running container")
    print(f"  {CYAN}logs{RST} [-f]                         📋 Show container logs ({DIM}-f to follow{RST})")
    print()
    print(f"{BOLD}Flags:{RST}")
    if _active_profile or label == "docker":
        print(
            f"  {WHITE}--profile post-train|post-train-teleop{RST}  {DIM}place before the subcommand of 'geniesim docker'{RST}"
        )
    print(f"  {WHITE}--mirrors NAME{RST}  {DIM}select a preset from docker/mirrors.toml{RST}")
    print(f"  {WHITE}--china{RST}         {DIM}alias for --mirrors china{RST}")
    mirror_default = "inherit parent image" if _active_profile else "upstream"
    print(f"  {DIM}Mirror default: {mirror_default}; explicit --build-arg values take precedence.{RST}")
    v_series = _selected_variant().isaacsim_series
    if v_series:
        print(
            f"  {WHITE}--isaacsim{RST} {DIM}<spec>{RST}  "
            f"{DIM}Isaac Sim series to resolve the pins from (default: {v_series}){RST}"
        )
        print(f"  {WHITE}--no-resolve{RST}       {DIM}skip version resolution; use the Dockerfile ARG defaults{RST}")
    if not _active_profile and _active_variant != "4.5":
        print(
            f"  {WHITE}--build-arg LIBUIPC_SM=89|120{RST}  "
            f"{DIM}override host GPU 0 detection for native IPC preinstallation{RST}"
        )
    print()
    print(f"{BOLD}Environment overrides:{RST}")
    print(f"  {WHITE}GENIESIM_IMAGE{RST}      {DIM}image tag (default: {v.image}){RST}")
    print(f"  {WHITE}GENIESIM_CONTAINER{RST}  {DIM}container name (default: {v.container}){RST}")
    workspace_default = "repo root" if _active_profile else "cwd"
    print(
        f"  {WHITE}GENIESIM_WORKSPACE{RST}  {DIM}host path bind-mounted at /workspace (default: {workspace_default}){RST}"
    )
    if _active_profile:
        print(f"  {WHITE}GENIESIM_ASSETS_SRC{RST} {DIM}host asset package path (default: editable installation){RST}")
    print(f"  {WHITE}GENIESIM_CACHE_ROOT{RST} {DIM}host path for Isaac Sim cache (default: ~/docker/isaac-sim){RST}")
    if _active_profile:
        print(f"  {WHITE}GENIESIM_START_TIMEOUT{RST} {DIM}entrypoint readiness timeout in seconds (default: 300){RST}")
    if _selected_variant().isaacsim_series:
        print(f"  {WHITE}GENIESIM_ISAACSIM_TAG{RST}         {DIM}pin the NGC base tag, skipping resolution{RST}")
        print(f"  {WHITE}GENIESIM_ISAACSIM_PIP_VERSION{RST} {DIM}pin the PyPI wheel (set both or neither){RST}")


# The Dockerfiles install Isaac Sim into the distro's system python3, so the
# wheel's Python tag is a property of the base image, not of the host.
_VARIANT_PYTHON_TAG = {"": "cp312", "5.1": "cp312", "4.5": "cp310"}


def _isaacsim_build_args(spec: str | None) -> list[str]:
    """``--build-arg`` pair for the Isaac Sim pins, or ``[]`` to leave the
    Dockerfile's literal ``ARG`` defaults alone.

    Three ways to get a concrete pair, in precedence order:

    1. ``GENIESIM_ISAACSIM_TAG`` / ``GENIESIM_ISAACSIM_PIP_VERSION`` — both
       must be set; skips the network entirely.
    2. ``spec`` resolved against nvcr.io + pypi.nvidia.com.
    3. Nothing — the ``ARG`` defaults in the Dockerfile apply.

    A resolution failure is a warning, never fatal: offline builds must keep
    behaving exactly as they did before the resolver existed.
    """
    tag = os.environ.get("GENIESIM_ISAACSIM_TAG")
    wheel = os.environ.get("GENIESIM_ISAACSIM_PIP_VERSION")
    if tag and wheel:
        print(f"   {DIM}Isaac Sim:{RST}  {CYAN}tag {tag}, wheel {wheel}{RST} {DIM}(from env){RST}")
    elif tag or wheel:
        print(f"   {YELLOW}⚠️  set both GENIESIM_ISAACSIM_TAG and GENIESIM_ISAACSIM_PIP_VERSION, or neither{RST}")
        return []
    elif spec is None:
        return []
    else:
        from geniesim_cli import _isaacsim

        python_tag = _VARIANT_PYTHON_TAG.get(_active_variant)
        resolved = _isaacsim.resolve(spec, python_tag=python_tag)
        if resolved is None:
            print(f"   {YELLOW}⚠️  could not resolve Isaac Sim {spec}; using the Dockerfile ARG defaults{RST}")
            return []
        tag, wheel = resolved
        print(f"   {DIM}Isaac Sim:{RST}  {CYAN}{spec} → tag {tag}, wheel {wheel}{RST}")

    # Echoed above on purpose: with a floating spec, the build log is the
    # only record of what the pins actually resolved to.
    return ["--build-arg", f"ISAACSIM_TAG={tag}", "--build-arg", f"ISAACSIM_PIP_VERSION={wheel}"]


def _libuipc_build_args(forwarded: list[str]) -> list[str]:
    """Select the host's GPU 0 unless Docker receives an explicit architecture."""
    override = None
    for index, token in enumerate(forwarded):
        value = forwarded[index + 1] if token == "--build-arg" and index + 1 < len(forwarded) else ""
        if token.startswith("--build-arg="):
            value = token.split("=", 1)[1]
        if value.split("=", 1)[0] == "LIBUIPC_SM":
            override = value.split("=", 1)[1] if "=" in value else os.environ.get("LIBUIPC_SM", "")
    if override is not None:
        if override not in {"89", "120"}:
            raise RuntimeError("LIBUIPC_SM must be 89 or 120")
        print(f"   {DIM}libuipc:{RST}    {CYAN}sm_{override} (explicit build argument){RST}")
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--id=0", "--query-gpu=compute_cap", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            "Cannot detect host GPU 0; pass --build-arg LIBUIPC_SM=89 or --build-arg LIBUIPC_SM=120"
        ) from error
    sm = {"8.9": "89", "12.0": "120"}.get(result.stdout.strip())
    if sm is None:
        raise RuntimeError(f"Unsupported host GPU capability {result.stdout.strip()!r}; libuipc supports SM 89 and 120")
    print(f"   {DIM}libuipc:{RST}    {CYAN}sm_{sm} (host GPU 0){RST}")
    return ["--build-arg", f"LIBUIPC_SM={sm}"]


def _do_build(extra: list[str]) -> None:
    repo_root = _repo_root()
    v = _selected_variant()
    dockerfile = _docker_dir() / v.dockerfile
    image = _image()

    # Pull our own flags out of the user-supplied extras. Anything else is
    # forwarded verbatim to `docker build` (e.g. --no-cache, --progress).
    preset = None
    resolve_pins = True
    series = v.isaacsim_series
    forwarded: list[str] = []
    expect_series = False
    tokens = iter(extra)
    for tok in tokens:
        if expect_series:
            series, expect_series = tok, False
        elif tok == "--china" or tok == "--mirrors" or tok.startswith("--mirrors="):
            if tok == "--china":
                selected = "china"
            elif tok == "--mirrors":
                selected = next(tokens, "")
            else:
                selected = tok.partition("=")[2]
            if not selected or selected.startswith("--") or (preset is not None and selected != preset):
                print(f"{RED}❌ Select one mirror preset with --mirrors NAME or --china.{RST}")
                sys.exit(2)
            preset = selected
        elif tok == "--build-arg":
            forwarded.extend([tok, next(tokens, "")])
        elif tok == "--no-resolve":
            resolve_pins = False
        elif tok == "--isaacsim":
            expect_series = True
        elif tok.startswith("--isaacsim="):
            series = tok.split("=", 1)[1]
        else:
            forwarded.append(tok)
    if expect_series:
        print(f"{RED}❌ Error: --isaacsim needs a series spec (e.g. --isaacsim 6.X){RST}")
        sys.exit(2)

    preset = preset or (None if _active_profile else "upstream")
    try:
        supported = _mirrors.declared_args(dockerfile)
        settings = _mirrors.load_preset(repo_root / "docker/mirrors.toml", preset) if preset else {}
        defaults = {name: value for name, value in settings.items() if name in supported}
        values, forwarded, explicit = _mirrors.merge_build_args(defaults, forwarded)
    except (OSError, ValueError) as exc:
        print(f"{RED}❌ {exc}{RST}")
        sys.exit(2)

    print(f"{BOLD}{MAGENTA}🔨 geniesim {_cmd_label()} build{RST}")
    print(f"   {DIM}Repo root:{RST}  {CYAN}{repo_root}{RST}")
    print(f"   {DIM}Dockerfile:{RST} {CYAN}{dockerfile}{RST}")
    print(f"   {DIM}Tag:{RST}        {CYAN}{image}{RST}")
    print(f"   {DIM}Mirrors:{RST}    {CYAN}{preset or 'inherit parent image'}{RST}")
    for name in sorted(_mirrors.MIRROR_ARGS & values.keys()):
        source = "--build-arg" if name in explicit else f"preset {preset}"
        print(
            f"   {DIM}{name}:{RST} {CYAN}{values[name] if values[name] is not None else '(from environment)'}{RST} {DIM}({source}){RST}"
        )

    # Resolved inside the banner block: the "Isaac Sim:" line it prints
    # belongs with Repo root / Dockerfile / Tag, above the blank separator.
    resolved = _isaacsim_build_args(series if resolve_pins else None)
    if not _active_profile and _VARIANT_PYTHON_TAG.get(_active_variant) == "cp312":
        try:
            resolved += _libuipc_build_args(_mirrors.as_flags(values))
        except RuntimeError as error:
            print(f"{RED}❌ {error}{RST}")
            sys.exit(2)
    resolved_values, _, _ = _mirrors.merge_build_args({}, resolved)
    values = {**resolved_values, **values}
    build_args = _mirrors.as_flags(values)
    print()

    cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile),
        "-t",
        image,
        *build_args,
        *forwarded,
        str(repo_root),
    ]
    print(f"   {YELLOW}$ {' '.join(cmd)}{RST}")
    print()
    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "1"
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"   {RED}❌ docker build failed (exit {result.returncode}){RST}")
        sys.exit(result.returncode)
    print(f"   {GREEN}✅ Built {image}{RST}")


def _do_up(headless: bool) -> None:
    state = _container_state()
    if state == "running":
        if not _active_profile:
            print(f"{YELLOW}⚠️  Container {BOLD}{_container()}{RST}{YELLOW} is already running.{RST}")
            print(
                f"   {DIM}Use{RST} {CYAN}geniesim {_cmd_label()} into{RST} {DIM}to enter it, or{RST} {CYAN}geniesim {_cmd_label()} down{RST} {DIM}to recreate.{RST}"
            )
            return
        print(f"{YELLOW}⚠️  Container {_container()} is already running; waiting for entrypoint readiness.{RST}")
    elif state is not None:
        print(f"{YELLOW}⚠️  Container {BOLD}{_container()}{RST}{YELLOW} exists (state: {state}).{RST}")
        print(f"   {DIM}Run{RST} {CYAN}geniesim {_cmd_label()} down{RST} {DIM}first.{RST}")
        sys.exit(1)

    script = _docker_dir() / "start.sh"
    if not script.is_file():
        print(f"{RED}❌ Missing {script}{RST}")
        sys.exit(1)

    print(f"{BOLD}{MAGENTA}🚀 geniesim {_cmd_label()} up{RST} {DIM}({'headless' if headless else 'gui'}){RST}")
    print(f"   {DIM}Image:{RST}     {CYAN}{_image()}{RST}")
    print(f"   {DIM}Container:{RST} {CYAN}{_container()}{RST}")
    print()

    env = os.environ.copy()
    env.setdefault(_env.GENIESIM_IMAGE.name, _image())
    env.setdefault(_env.GENIESIM_CONTAINER.name, _container())
    if _active_profile:
        env[_env.GENIESIM_WORKSPACE.name] = str(_workspace())
        if not env.get(_env.GENIESIM_ASSETS_SRC.name):
            env[_env.GENIESIM_ASSETS_SRC.name] = _find_assets_src()
    else:
        env.setdefault(_env.GENIESIM_WORKSPACE.name, str(_repo_root()))
        env.setdefault("GENIESIM_ASSETS_SRC", _find_assets_src())
    env["GENIESIM_DOCKER_PROFILE"] = _active_profile
    # Inject variant-specific values for start.sh and entrypoint.sh.
    # Use direct assignment (not setdefault) so the variant always wins over
    # any same-named vars the user may have exported on the host (e.g. ROS_DISTRO).
    env.setdefault("GENIESIM_VARIANT_LABEL", _cmd_label())
    for key, val in _selected_variant().env.items():
        env[key] = val
    cmd = ["bash", str(script)]
    if state == "running":
        cmd.append("--wait-only")
    if headless:
        cmd.append("--headless")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"   {RED}❌ up failed (exit {result.returncode}){RST}")
        sys.exit(result.returncode)


def _do_down() -> None:
    name = _container()
    state = _container_state()
    if state is None:
        print(f"{DIM}No container named {BOLD}{name}{RST}{DIM}; nothing to do.{RST}")
        return
    print(f"{BOLD}{MAGENTA}🛑 geniesim {_cmd_label()} down{RST}")
    print(f"   {DIM}Container:{RST} {CYAN}{name}{RST} {DIM}(state: {state}){RST}")
    if state == "running":
        subprocess.run(["docker", "stop", name], check=False)
    subprocess.run(["docker", "rm", "-f", name], check=False)
    print(f"   {GREEN}✅ Removed{RST}")


def _do_into() -> None:
    name = _container()
    state = _container_state()
    if state != "running":
        print(f"{RED}❌ Container {BOLD}{name}{RST}{RED} is not running (state: {state}).{RST}")
        print(f"   {DIM}Start it with{RST} {CYAN}geniesim {_cmd_label()} up{RST}{DIM}.{RST}")
        sys.exit(1)
    uid = os.getuid()
    gid = os.getgid()
    os.execvp(
        "docker",
        [
            "docker",
            "exec",
            "-it",
            "-u",
            f"{uid}:{gid}",
            "-e",
            "HOME=/home/isaac-sim",
            "-w",
            "/workspace",
            name,
            "bash",
            "-l",
        ],
    )


def _do_logs(follow: bool) -> None:
    name = _container()
    state = _container_state()
    if state is None:
        print(f"{RED}❌ No container named {BOLD}{name}{RST}{RED}.{RST}")
        sys.exit(1)
    cmd = ["docker", "logs"]
    if follow:
        cmd.append("-f")
    cmd.append(name)
    os.execvp(cmd[0], cmd)


def run(args: list[str], variant: str = "", invoked_as: str = "docker") -> None:
    global _active_variant, _active_profile, _invoked_as
    _active_variant = variant
    _active_profile = ""
    _invoked_as = invoked_as
    if invoked_as == "docker" and args[:1] == ["--profile"]:
        if len(args) < 2 or args[1] not in PROFILES:
            print(f"{RED}❌ Use 'geniesim docker --profile post-train|post-train-teleop <subcommand>'.{RST}")
            sys.exit(2)
        _active_profile = args[1]
        _invoked_as = f"docker --profile {_active_profile}"
        args = args[2:]

    if _active_profile:
        if not args or args[0] in ("-h", "--help", "help"):
            _print_usage()
            return
        sub, *rest = args
        allowed = {"up": {"--headless"}, "down": set(), "into": set(), "logs": {"-f", "--follow"}}
        if sub in allowed and any(arg not in allowed[sub] for arg in rest):
            print(f"{RED}❌ Unsupported arguments for {sub}: {' '.join(rest)}{RST}")
            sys.exit(2)
    _ensure_docker_cli()

    # ---- experimental variant warning -----------------------------------
    # Generic gate for any variant marked ``implemented=False``: let
    # developers who explicitly want to try it through, but make them
    # acknowledge first so failures aren't mistaken for bugs in setup.
    # Profile-aware via ``_selected_variant()`` so an unimplemented
    # post-train profile gates like an Isaac variant does.
    if not _selected_variant().implemented:
        v = _selected_variant()
        print(
            f"\n{YELLOW}"
            f"╔════════════════════════════════════════════════════════════════╗\n"
            f"║  🚧  UNVERIFIED variant: {invoked_as + ' (' + v.container + ')':<38}║\n"
            f"╠════════════════════════════════════════════════════════════════╣\n"
            f"║  This variant is not fully implemented yet. Its image is not   ║\n"
            f"║  published and its runtime is unverified — build it yourself   ║\n"
            f"║  and expect rough edges. See docker/README.md.                 ║\n"
            f"║                                                                ║\n"
            f"║  For a stable container use geniesim docker (→ docker5.1).     ║\n"
            f"╚════════════════════════════════════════════════════════════════╝"
            f"{RST}\n"
        )
        try:
            input(f"{YELLOW}Press [ENTER] to continue, or Ctrl+C to abort...{RST}")
        except (KeyboardInterrupt, EOFError):
            print(f"\n{RED}Aborted by user.{RST}\n")
            sys.exit(130)

    if not args or args[0] in ("-h", "--help", "help"):
        _print_usage()
        return

    sub = args[0]
    rest = args[1:]

    if sub == "build":
        _do_build(rest)
        return
    if sub == "up":
        headless = "--headless" in rest
        _do_up(headless)
        return
    if sub == "down":
        _do_down()
        return
    if sub == "into":
        _do_into()
        return
    if sub == "logs":
        follow = "-f" in rest or "--follow" in rest
        _do_logs(follow)
        return

    print(f"{RED}❌ Error: unknown docker subcommand '{sub}'{RST}")
    print()
    _print_usage()
    sys.exit(1)
