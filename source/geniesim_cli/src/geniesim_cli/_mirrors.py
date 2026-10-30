# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Load build mirror presets and merge Docker arguments without duplicates."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

try:
    import tomllib
except ImportError:
    import tomli as tomllib


MIRROR_ARGS = frozenset({"ISAACSIM_REGISTRY", "APT_MIRROR", "PIP_INDEX_URL"})


def load_preset(path: Path, name: str) -> dict[str, str]:
    """Read a named preset using the Dockerfiles' existing build-argument names."""
    try:
        with path.open("rb") as stream:
            presets = tomllib.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read mirror presets from {path}: {exc}") from exc
    if name not in presets:
        raise ValueError(f"Unknown mirror preset {name!r}; available: {', '.join(sorted(presets))}")
    preset = presets[name]
    if not isinstance(preset, dict) or not preset:
        raise ValueError(f"Mirror preset {name!r} must be a nonempty table")
    for key, value in preset.items():
        if key not in MIRROR_ARGS or not isinstance(value, str):
            raise ValueError(f"Invalid mirror setting {key!r} in preset {name!r}")
        if key == "ISAACSIM_REGISTRY" and (not value or "://" in value or any(c.isspace() for c in value)):
            raise ValueError(f"Expected a registry/path prefix for {key} in preset {name!r}")
        if key in {"APT_MIRROR", "PIP_INDEX_URL"} and value and not (key == "APT_MIRROR" and value == "upstream"):
            parsed = urlsplit(value)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc or any(c.isspace() for c in value):
                raise ValueError(f"Invalid repository URL for {key} in preset {name!r}")
    return dict(preset)


def declared_args(dockerfile: Path) -> set[str]:
    """Return the argument names accepted by the selected recipe."""
    return set(re.findall(r"^\s*ARG\s+([A-Za-z_][A-Za-z0-9_]*)", dockerfile.read_text(), re.MULTILINE))


def merge_build_args(
    defaults: dict[str, str], forwarded: list[str]
) -> tuple[dict[str, str | None], list[str], set[str]]:
    """Apply explicit Docker arguments last, retaining bare environment-backed args."""
    values: dict[str, str | None] = dict(defaults)
    remaining = []
    explicit = set()
    tokens = iter(forwarded)
    for token in tokens:
        if token == "--build-arg":
            try:
                argument = next(tokens)
            except StopIteration:
                raise ValueError("--build-arg needs NAME or NAME=VALUE") from None
        elif token.startswith("--build-arg="):
            argument = token.partition("=")[2]
        else:
            remaining.append(token)
            continue
        name, separator, value = argument.partition("=")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid build argument {argument!r}")
        values[name] = value if separator else None
        explicit.add(name)
    return values, remaining, explicit


def as_flags(values: dict[str, str | None]) -> list[str]:
    """Encode each effective build argument exactly once."""
    return [
        token
        for name, value in values.items()
        for token in ("--build-arg", name if value is None else f"{name}={value}")
    ]
