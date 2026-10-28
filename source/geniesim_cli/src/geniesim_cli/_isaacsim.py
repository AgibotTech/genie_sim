# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Resolve an Isaac Sim **series spec** (e.g. ``6.X``) into the two concrete
version strings the Dockerfiles take as build args.

Why this module exists
----------------------
``docker/Dockerfile`` needs two strings that must name the same Isaac Sim
release:

- ``ISAACSIM_TAG`` — the NGC base image tag, e.g. ``6.2.0.1``
- ``ISAACSIM_PIP_VERSION`` — the PyPI wheel version, e.g. ``6.2.0.0`` (4)

Independent pins previously let ``Dockerfile`` ship wheels and a base
from different Isaac releases. A series spec collapses them to one declaration (``6.X``) that is
resolved here, on the host, before ``docker build`` runs.

Two upstream facts drive the design, both measured rather than assumed:

1. **There is no floating tag to point ``FROM`` at.** All 127 tags on
   ``nvcr.io/nvidia/isaac-sim`` are fully-qualified versions plus
   ``.sig`` / ``.sbom`` / ``.vex`` attestation tags. ``latest``, ``6``,
   ``6.2``, ``5.1`` are all absent -- and ``FROM`` cannot wildcard anyway.
   So the wildcard has to be resolved on our side.

2. **Image and wheel versions need an explicit correspondence.** The
   ``6.2.0.1`` image respin uses the published ``6.2.0.0`` wheel. The tag
   space and the wheel space do not correspond by any string rule::

       NGC tags   6.2.0  6.2.0.1
       PyPI       6.2.0.0

   There is no ``6.2.0.1`` wheel; historically tag ``4.2.0`` paired with
   wheel ``4.2.0.2``. We pick the highest tag with
   an available matching wheel, including the approved ``6.2.0.1`` to
   ``6.2.0.0`` pairing, per :func:`_wheel_matches_tag`.

Contract
--------
:func:`resolve` never raises and never blocks for long: every failure mode
(offline, HTTP error, unparseable index, no match) returns ``None``, and the
caller is expected to fall back to the Dockerfile's literal ``ARG`` defaults.
That keeps an unreachable network degrading to today's exact behaviour
instead of breaking the build. This is the only network access anywhere in
``geniesim_cli``, which is why it is opt-outable (``--no-resolve``) and
short-timeout.

Standalone use, for debugging a spec without starting a build::

    python3 -m geniesim_cli._isaacsim 6.X
    python3 -m geniesim_cli._isaacsim 6.X --python-tag cp312
    python3 -m geniesim_cli._isaacsim 6.2.0.1 --python-tag cp312
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request

# NGC registry. The tags list needs a bearer token; ``proxy_auth`` issues an
# anonymous one for a public repository, which is the same thing ``docker
# pull`` does under the hood.
_NGC_REPO = "nvidia/isaac-sim"
_NGC_AUTH = f"https://nvcr.io/proxy_auth?scope=repository:{_NGC_REPO}:pull"
_NGC_TAGS = f"https://nvcr.io/v2/{_NGC_REPO}/tags/list"

# NVIDIA's PEP 503 simple index. It exposes no PEP 658 metadata (anchors
# carry only ``#sha256=``), so wheel *filenames* are the whole data source --
# which is enough, since version and Python tag are both in the name.
_PYPI_INDEX = "https://pypi.nvidia.com/isaacsim/"

# A release tag: 3 or 4 numeric components. Rejects the ``.sig`` / ``.sbom`` /
# ``.vex`` attestation tags and prerelease tags in one go.
_TAG_RE = re.compile(r"\d+(?:\.\d+){2,3}\Z")

# ``isaacsim-6.2.0.0-cp312-none-manylinux_2_35_x86_64.whl``
_WHEEL_RE = re.compile(r"isaacsim-(?P<version>\d+(?:\.\d+)+)-(?P<python_tag>cp\d+)-")

# A spec is either a series (``6.X``, ``6.x``, ``6``, ``6.2.X``) or a fully
# qualified version (``6.2.0``). ``X`` is accepted in either case.
_SPEC_RE = re.compile(r"\A(?P<prefix>\d+(?:\.\d+)*)(?:\.[xX*])?\Z")

_TAG_WHEEL_PAIRS = {"6.2.0.1": "6.2.0.0"}


def _fetch(url: str, timeout: float, headers: dict[str, str] | None = None) -> bytes | None:
    """GET ``url``, or ``None`` on any failure. Never raises."""
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


def _version_key(version: str) -> tuple[int, ...]:
    """``"6.2.0"`` -> ``(6, 2, 0)``. For sorting -- string sort would put
    ``6.10.0`` before ``6.2.0``."""
    return tuple(int(part) for part in version.split("."))


def parse_spec(spec: str) -> str | None:
    """Normalize a spec to its numeric prefix, or ``None`` if unparseable.

    ``6.X`` / ``6.x`` / ``6`` -> ``"6"``; ``6.2.X`` -> ``"6.2"``;
    ``6.2.0`` -> ``"6.2.0"`` (exact, see :func:`_tag_in_series`).
    """
    match = _SPEC_RE.match(spec.strip())
    return match.group("prefix") if match else None


def _tag_in_series(tag: str, prefix: str) -> bool:
    """Is ``tag`` in the series named by ``prefix``?

    Component-wise, not ``startswith`` -- otherwise prefix ``6.2`` would
    also claim a hypothetical ``6.20.0``. A prefix that is itself a full
    3-or-4 component version matches only that exact tag.
    """
    tag_parts = tag.split(".")
    prefix_parts = prefix.split(".")
    if len(prefix_parts) >= 3:
        return tag_parts == prefix_parts
    return len(tag_parts) >= len(prefix_parts) and tag_parts[: len(prefix_parts)] == prefix_parts


def _wheel_matches_tag(wheel: str, tag: str) -> bool:
    """The tag-to-wheel correspondence rule.

    A 3-component tag ``T`` matches wheel ``W`` iff ``W`` is ``T`` plus one
    more component (``6.2.0`` -> ``6.2.0.0``); a 4-component tag matches iff
    the two are equal, except for explicitly approved image-to-wheel pairs.
    ``6.2.0.1`` is paired with the published ``6.2.0.0`` wheel; other respins
    do not inherit that exception.
    """
    if tag in _TAG_WHEEL_PAIRS:
        return wheel == _TAG_WHEEL_PAIRS[tag]
    tag_parts = tag.split(".")
    wheel_parts = wheel.split(".")
    if len(tag_parts) == 3:
        return len(wheel_parts) == 4 and wheel_parts[:3] == tag_parts
    return wheel_parts == tag_parts


def ngc_tags(timeout: float = 5.0) -> list[str]:
    """Release tags on ``nvcr.io/nvidia/isaac-sim``, or ``[]`` on failure."""
    auth = _fetch(_NGC_AUTH, timeout)
    if auth is None:
        return []
    try:
        token = json.loads(auth)["token"]
    except (ValueError, KeyError, TypeError):
        return []

    body = _fetch(_NGC_TAGS, timeout, headers={"Authorization": f"Bearer {token}"})
    if body is None:
        return []
    try:
        tags = json.loads(body).get("tags") or []
    except (ValueError, AttributeError):
        return []
    return [tag for tag in tags if isinstance(tag, str) and _TAG_RE.match(tag)]


def pypi_wheels(timeout: float = 5.0) -> dict[str, set[str]]:
    """``{wheel version: {python tag, ...}}`` from pypi.nvidia.com, or ``{}``.

    The Python tag is carried along because it is a real guard: 6.x wheels
    are cp312, 5.x are cp311 and 4.x are cp310, so a series spec pointed at
    the wrong Dockerfile is catchable here in a second rather than inside a
    long build.
    """
    body = _fetch(_PYPI_INDEX, timeout)
    if body is None:
        return {}
    wheels: dict[str, set[str]] = {}
    for match in _WHEEL_RE.finditer(body.decode("utf-8", "replace")):
        wheels.setdefault(match.group("version"), set()).add(match.group("python_tag"))
    return wheels


def resolve(spec: str, *, python_tag: str | None = None, timeout: float = 5.0) -> tuple[str, str] | None:
    """Resolve ``spec`` to ``(ISAACSIM_TAG, ISAACSIM_PIP_VERSION)``.

    Returns ``None`` -- never raises -- if the spec is unparseable, either
    index is unreachable, or no tag in the series has a matching wheel (for
    ``python_tag``, when given). Callers fall back to the Dockerfile's
    ``ARG`` defaults.
    """
    prefix = parse_spec(spec)
    if prefix is None:
        return None

    tags = ngc_tags(timeout)
    wheels = pypi_wheels(timeout)
    if not tags or not wheels:
        return None

    candidates = sorted(
        (tag for tag in tags if _tag_in_series(tag, prefix)),
        key=_version_key,
        reverse=True,
    )
    for tag in candidates:
        matching = [
            version
            for version, python_tags in wheels.items()
            if _wheel_matches_tag(version, tag) and (python_tag is None or python_tag in python_tags)
        ]
        if matching:
            return tag, max(matching, key=_version_key)
    return None


def _main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(f"usage: python3 -m {__spec__.name} <series-spec> [--python-tag cpXYZ]", file=sys.stderr)
        print("example: 6.X, 6.2.X, 6.2.0.1", file=sys.stderr)
        return 0 if argv else 2

    spec = argv[0]
    python_tag = None
    if "--python-tag" in argv:
        index = argv.index("--python-tag")
        if index + 1 >= len(argv):
            print("error: --python-tag needs a value", file=sys.stderr)
            return 2
        python_tag = argv[index + 1]

    resolved = resolve(spec, python_tag=python_tag)
    if resolved is None:
        tags, wheels = ngc_tags(), pypi_wheels()
        if not tags or not wheels:
            reason = "could not reach nvcr.io and/or pypi.nvidia.com"
        elif parse_spec(spec) is None:
            reason = f"unparseable spec {spec!r}"
        else:
            reason = f"no tag in series {spec} has a matching wheel"
            if python_tag:
                reason += f" for {python_tag}"
        print(f"unresolved: {reason}", file=sys.stderr)
        return 1

    tag, wheel = resolved
    print(f"ISAACSIM_TAG={tag}")
    print(f"ISAACSIM_PIP_VERSION={wheel}")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
