# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Central registry for all ``GENIESIM_*`` environment variables.

This module is the **single source of truth** for every project-owned
environment variable used anywhere in the ``main`` repo. Other modules
must import typed accessors from here instead of reading ``os.environ``
directly with bare string literals — that keeps defaults, semantics,
and documentation consolidated in one place.

Design
------
* Each variable is described by an :class:`EnvVar` record:
  ``name``, ``description``, ``default``, ``category``, ``consumers``.
* :data:`REGISTRY` holds every record, keyed by its canonical name.
* Typed accessor functions (``workspace()``, ``image()``,
  ``container_name()``, ...) wrap ``os.environ.get(...)`` with the
  documented default and type coercion (paths → :class:`pathlib.Path`,
  ``"1"`` flags → :class:`bool`, ...).
* :func:`dump` returns a ``[(EnvVar, current_value)]`` snapshot used by
  ``geniesim env`` and ``geniesim status`` to print the current view.

Migration aliases
-----------------
``SIM_REPO_ROOT`` → :func:`repo_path` (canonical name
``GENIESIM_REPO_PATH``) and ``SIM_ASSETS`` → :func:`assets_path`
(canonical name ``GENIESIM_ASSETS_PATH``). The accessors transparently
fall back to the legacy ``SIM_*`` names when the new ones are unset, so
existing shell entrypoints keep working during the rename.

Name collision
--------------
``GENIESIM_CONTAINER`` historically meant *both* the docker container's
name (a string, e.g. ``"geniesim"``) *and* a boolean flag asserting
"running inside the container" (e.g. ``GENIESIM_CONTAINER=1``). This
module disambiguates by introducing :func:`in_container` backed by the
new ``GENIESIM_IN_CONTAINER`` env var; :func:`container_name` keeps the
string semantics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Registry record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvVar:
    """Schema for a single registered environment variable.

    The ``default`` is documentation-only; accessor functions decide
    how (and when) to materialise it. ``consumers`` is the list of
    files that read the variable, kept here so ``grep`` against this
    module suffices for an audit.
    """

    name: str
    description: str
    default: str = "(unset)"
    category: str = "misc"
    consumers: tuple[str, ...] = field(default_factory=tuple)

    def get(self, fallback: Optional[str] = None) -> Optional[str]:
        """Return the raw ``os.environ`` value, or ``fallback`` if unset."""
        return os.environ.get(self.name, fallback)


# ---------------------------------------------------------------------------
# Registry — every project-owned env var lives here
# ---------------------------------------------------------------------------


REGISTRY: dict[str, EnvVar] = {}


def _register(var: EnvVar) -> EnvVar:
    REGISTRY[var.name] = var
    return var


# --- workspace / repo layout ------------------------------------------------

GENIESIM_WORKSPACE = _register(
    EnvVar(
        name="GENIESIM_WORKSPACE",
        description="Colcon workspace + host path bind-mounted at /workspace inside the container.",
        default="$(pwd)",
        category="workspace",
        consumers=(
            "geniesim_cli/cli.py",
            "geniesim_cli/_workspace.py",
            "geniesim_cli/commands/status.py",
            "geniesim_cli/commands/docker.py",
            "docker/start.sh",
        ),
    )
)

GENIESIM_REPO_ROOT = _register(
    EnvVar(
        name="GENIESIM_REPO_ROOT",
        description="Override for the repo root that contains docker/Dockerfile.",
        default="(auto-detected from cwd or geniesim_cli source location)",
        category="workspace",
        consumers=("geniesim_cli/commands/docker.py", "geniesim_cli/commands/post_train.py"),
    )
)

GENIESIM_REPO_PATH = _register(
    EnvVar(
        name="GENIESIM_REPO_PATH",
        description=(
            "Repo root used by geniesim_benchmark / rlinf_geniesim / data_collection "
            "for path resolution. Renamed from SIM_REPO_ROOT; legacy name still read as fallback."
        ),
        default="(see ``repo_path()``)",
        category="workspace",
        consumers=(
            "geniesim_benchmark/utils/system_utils.py",
            "geniesim_cli/commands/post_train.py",
            "geniesim_benchmark/evaluator/generators/auto_score.py",
            "geniesim_benchmark/benchmark/policy/pipolicy.py",
            "rlinf_geniesim/renderer/rl_renderer.py",
            "rlinf_geniesim/scripts/sim_server.py",
            "data_collection/common/base_utils/ros_nodes/*.py",
            "source/rlinf_geniesim/scripts/run_rlinf.sh",
            "source/rlinf_geniesim/scripts/entrypoint_geniesim_rlinf.sh",
        ),
    )
)

GENIESIM_ASSETS_PATH = _register(
    EnvVar(
        name="GENIESIM_ASSETS_PATH",
        description=(
            "Asset root used by geniesim_benchmark and data_collection. "
            "Renamed from SIM_ASSETS; legacy name still read as fallback."
        ),
        default="(unset — required at runtime)",
        category="workspace",
        consumers=(
            "geniesim_benchmark/teleop/replay_state.py",
            "geniesim_benchmark/plugins/ader/action/common_actions.py",
            "geniesim_benchmark/plugins/tgs/taskgen_utils.py",
            "data_collection/server/command_controller.py",
            "data_collection/client/agent/omniagent.py",
            "data_collection/client/layout/*.py",
        ),
    )
)

# --- docker -----------------------------------------------------------------

GENIESIM_IMAGE = _register(
    EnvVar(
        name="GENIESIM_IMAGE",
        description="Docker image tag used by ``geniesim docker build|up``.",
        default="registry.agibot.com/genie-sim/geniesim3:latest",
        category="docker",
        consumers=(
            "geniesim_cli/commands/docker.py",
            "docker/start.sh",
        ),
    )
)

GENIESIM_CONTAINER = _register(
    EnvVar(
        name="GENIESIM_CONTAINER",
        description="Container name used by ``geniesim docker {up,down,into,logs}``.",
        default="geniesim",
        category="docker",
        consumers=(
            "geniesim_cli/commands/docker.py",
            "docker/start.sh",
            "docker/into.sh",
        ),
    )
)

GENIESIM_IN_CONTAINER = _register(
    EnvVar(
        name="GENIESIM_IN_CONTAINER",
        description=(
            "Boolean flag (``1`` / ``0``) set by docker entrypoints to indicate "
            "code is running inside a GenieSim container. Replaces the previous "
            "boolean overload of GENIESIM_CONTAINER."
        ),
        default="0",
        category="docker",
        consumers=(
            "source/rlinf_geniesim/scripts/run_rlinf.sh",
            "source/rlinf_geniesim/scripts/dockerfile_geniesim_rlinf",
        ),
    )
)

GENIESIM_CACHE_ROOT = _register(
    EnvVar(
        name="GENIESIM_CACHE_ROOT",
        description="Host path mounted as the Isaac Sim cache root inside the container.",
        default="$HOME/docker/isaac-sim",
        category="docker",
        consumers=(
            "geniesim_cli/commands/docker.py",
            "docker/start.sh",
        ),
    )
)

GENIESIM_ASSETS_SRC = _register(
    EnvVar(
        name="GENIESIM_ASSETS_SRC",
        description="Host asset package used by the post-training Docker profiles.",
        default="(auto-detect editable installation)",
        category="docker",
        consumers=("geniesim_cli/commands/docker.py", "docker/start.sh"),
    )
)

GENIESIM_START_TIMEOUT = _register(
    EnvVar(
        name="GENIESIM_START_TIMEOUT",
        description="Maximum seconds to wait for post-training container readiness, including repeated up calls.",
        default="300",
        category="docker",
        consumers=("docker/start.sh",),
    )
)

GENIESIM_BASHRC_SEED = _register(
    EnvVar(
        name="GENIESIM_BASHRC_SEED",
        description="Marker tag wrapping the entrypoint-injected ~/.bashrc block (idempotency).",
        default="(literal sentinel; shell-only)",
        category="docker",
        consumers=("docker/entrypoint.sh",),
    )
)

# --- build / packaging ------------------------------------------------------

GENIESIM_ROS_SKIP_BUILD = _register(
    EnvVar(
        name="GENIESIM_ROS_SKIP_BUILD",
        description="Set to ``1`` to skip the colcon stage during ``pip install geniesim_ros``.",
        default="(unset)",
        category="build",
        consumers=("geniesim_ros/setup.py",),
    )
)

GENIESIM_COMPILE = _register(
    EnvVar(
        name="GENIESIM_COMPILE",
        description="Set to ``1`` to trigger Cython compilation during the geniesim wheel build.",
        default="(unset)",
        category="build",
        consumers=("geniesim_cli/commands/deploy.py", "geniesim/setup.py"),
    )
)

# --- geniesim_world / DA360 -------------------------------------------------

GENIESIM_DA360_ROOT = _register(
    EnvVar(
        name="GENIESIM_DA360_ROOT",
        description="DA360 repo root (contains networks/).",
        default="<repo>/external/DA360",
        category="world",
        consumers=("geniesim_world/cli_pano.py",),
    )
)

GENIESIM_DA360_CHECKPOINT = _register(
    EnvVar(
        name="GENIESIM_DA360_CHECKPOINT",
        description="DA360 weights file (.pth).",
        default="<da360-root>/DA360_large.pth",
        category="world",
        consumers=("geniesim_world/cli_pano.py",),
    )
)

GENIESIM_REALESRGAN_BIN = _register(
    EnvVar(
        name="GENIESIM_REALESRGAN_BIN",
        description="Absolute path to realesrgan-ncnn-vulkan binary.",
        default="(unset — autodetected on PATH)",
        category="world",
        consumers=("geniesim_world/cli_pano.py",),
    )
)


# ---------------------------------------------------------------------------
# Typed accessors
# ---------------------------------------------------------------------------


_DEFAULT_IMAGE = "registry.agibot.com/genie-sim/geniesim3:latest"
_DEFAULT_CONTAINER = "geniesim"


def _truthy(value: Optional[str]) -> bool:
    """Standard ``1/true/yes/on`` truthy semantics for env-var flags."""
    if value is None:
        return False
    return value.strip().lower() in ("1", "true", "yes", "on", "y")


# --- workspace / repo -------------------------------------------------------


def workspace() -> Optional[str]:
    """Return ``$GENIESIM_WORKSPACE`` if set, else ``None``.

    Callers decide their own fallback (``cwd``, the bundled ``ros_ws``,
    etc.) — keeping that policy out of this module avoids a circular
    import with :mod:`geniesim_cli._workspace`.
    """
    return GENIESIM_WORKSPACE.get()


def workspace_or_cwd() -> str:
    """Convenience: ``$GENIESIM_WORKSPACE`` if set, otherwise ``cwd``."""
    return workspace() or str(Path.cwd())


def workspace_origin() -> str:
    """Return ``"$GENIESIM_WORKSPACE"`` or ``"cwd"`` — for human-readable status output."""
    return "$GENIESIM_WORKSPACE" if workspace() else "cwd"


def repo_root() -> Optional[str]:
    """Return ``$GENIESIM_REPO_ROOT`` if set, else ``None``."""
    return GENIESIM_REPO_ROOT.get()


def repo_path() -> Optional[str]:
    """Return ``$GENIESIM_REPO_PATH``, falling back to legacy ``$SIM_REPO_ROOT``."""
    return GENIESIM_REPO_PATH.get() or os.environ.get("SIM_REPO_ROOT")


def assets_path() -> Optional[str]:
    """Return ``$GENIESIM_ASSETS_PATH``, falling back to legacy ``$SIM_ASSETS``."""
    return GENIESIM_ASSETS_PATH.get() or os.environ.get("SIM_ASSETS")


# --- docker -----------------------------------------------------------------


def image() -> str:
    """Return ``$GENIESIM_IMAGE`` or the default tag."""
    return GENIESIM_IMAGE.get(fallback=_DEFAULT_IMAGE) or _DEFAULT_IMAGE


def container_name() -> str:
    """Return ``$GENIESIM_CONTAINER`` (string container name) or the default."""
    return GENIESIM_CONTAINER.get(fallback=_DEFAULT_CONTAINER) or _DEFAULT_CONTAINER


def in_container() -> bool:
    """``True`` iff ``$GENIESIM_IN_CONTAINER`` is truthy.

    This is the **disambiguated** replacement for the legacy boolean
    use of ``GENIESIM_CONTAINER=1``. Updated entrypoints
    (``source/rlinf_geniesim/scripts/run_rlinf.sh``, ``source/rlinf_geniesim/scripts/dockerfile_geniesim_rlinf``)
    set the new name; the legacy one is no longer consulted.
    """
    return _truthy(GENIESIM_IN_CONTAINER.get())


def cache_root() -> str:
    """Return ``$GENIESIM_CACHE_ROOT`` or the default ``$HOME/docker/isaac-sim``."""
    explicit = GENIESIM_CACHE_ROOT.get()
    if explicit:
        return explicit
    return str(Path.home() / "docker" / "isaac-sim")


# --- build / packaging ------------------------------------------------------


def ros_skip_build() -> bool:
    """``True`` iff ``$GENIESIM_ROS_SKIP_BUILD=1`` (skip colcon stage)."""
    return _truthy(GENIESIM_ROS_SKIP_BUILD.get())


def compile_enabled() -> bool:
    """``True`` iff ``$GENIESIM_COMPILE=1`` (Cython build)."""
    return _truthy(GENIESIM_COMPILE.get())


# --- geniesim_world ---------------------------------------------------------


def da360_root() -> Optional[str]:
    """Return ``$GENIESIM_DA360_ROOT`` if set, else ``None``."""
    return GENIESIM_DA360_ROOT.get()


def da360_checkpoint() -> Optional[str]:
    """Return ``$GENIESIM_DA360_CHECKPOINT`` if set, else ``None``."""
    return GENIESIM_DA360_CHECKPOINT.get()


def realesrgan_bin() -> Optional[str]:
    """Return ``$GENIESIM_REALESRGAN_BIN`` if set, else ``None``."""
    return GENIESIM_REALESRGAN_BIN.get()


# ---------------------------------------------------------------------------
# Inspection helpers (used by ``geniesim env`` / ``geniesim status``)
# ---------------------------------------------------------------------------


def dump() -> list[tuple[EnvVar, Optional[str]]]:
    """Return ``[(EnvVar, raw current value)]`` for every registered var."""
    return [(var, var.get()) for var in REGISTRY.values()]


def by_category() -> dict[str, list[EnvVar]]:
    """Group registered vars by ``EnvVar.category`` for tabular display."""
    groups: dict[str, list[EnvVar]] = {}
    for var in REGISTRY.values():
        groups.setdefault(var.category, []).append(var)
    return groups
