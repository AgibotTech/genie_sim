# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Backwards-compatible re-export shim.

The canonical location for :class:`Manifest` and ``MANIFEST_VERSION``
is :mod:`genie_sim_engine_core.schema.manifest`. This module preserves
the pre-schema import path
(:mod:`genie_sim_engine_core.assemble.manifest`) so older stages that
imported ``Manifest`` here still work.

``RAW_MANIFEST_SENTINEL`` — retired after every producer moved to
typed :class:`Manifest.merge_slice` writes. The name is kept as a
tombstone (raises :class:`RuntimeError` on use) so any straggler
consumer surfaces at PR time instead of silently no-op'ing.
"""

from __future__ import annotations

from ..schema.errors import SchemaError, SchemaVersionError  # noqa: F401 — re-exported
from ..schema.manifest import (
    MANIFEST_VERSION,
    CameraManifest,  # noqa: F401 — re-exported
    LidarManifest,  # noqa: F401 — re-exported
    Manifest,
)


class _RetiredSentinel:
    """Placeholder for the retired ``RAW_MANIFEST_SENTINEL``.

    Kept as a class-level tombstone rather than a module-level string
    so any lingering ``slice_[RAW_MANIFEST_SENTINEL] = ...`` construction
    raises at construction time with a message pointing to the schema.
    """

    def __repr__(self) -> str:  # pragma: no cover — humans only
        return "<RAW_MANIFEST_SENTINEL: retired; use typed Manifest.merge_slice>"

    def __hash__(self) -> int:
        raise RuntimeError(
            "RAW_MANIFEST_SENTINEL was retired when every assemble stage moved "
            "to typed Manifest.merge_slice writes. Return the typed slice "
            "directly (e.g. `manifest_slice={'physics': {engine: {...}}, "
            "'cameras': [...]}`)"
        )


RAW_MANIFEST_SENTINEL = _RetiredSentinel()


__all__ = [
    "MANIFEST_VERSION",
    "Manifest",
    "RAW_MANIFEST_SENTINEL",
]
