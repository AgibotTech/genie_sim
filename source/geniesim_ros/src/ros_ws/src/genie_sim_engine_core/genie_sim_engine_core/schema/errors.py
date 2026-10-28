# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Schema-parsing errors raised by :mod:`genie_sim_engine_core.schema`.

Kept in a leaf module so any caller that only wants to catch schema
failures can ``from genie_sim_engine_core.schema.errors import
SchemaError`` without pulling in ``pxr`` / ``yaml`` transitively via the
scene/manifest modules.
"""

from __future__ import annotations


class SchemaError(ValueError):
    """Raised when scene YAML or manifest JSON does not parse cleanly.

    Carries the source path (``path``) so re-raised errors from deep in
    the parse tree still point back at the file the operator can edit.
    Callers should format ``str(exc)`` for humans; the ``path`` attribute
    is the machine-readable pointer.
    """

    def __init__(self, message: str, *, path: object = None) -> None:
        super().__init__(message)
        self.path = path


class SchemaVersionError(SchemaError):
    """Raised when a versioned schema file has a version this build doesn't accept.

    Distinct from ``SchemaError`` so launchers can offer a "re-run
    assemble" hint on version drift but a "fix your YAML" hint on other
    schema failures.
    """


__all__ = ["SchemaError", "SchemaVersionError"]
