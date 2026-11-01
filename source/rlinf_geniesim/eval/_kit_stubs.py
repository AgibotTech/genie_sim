# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Import-only substitutes for benchmark dependencies on Isaac Kit.

Checkers use ApiCoreShim for runtime state, but their import chain also
references APICore and Isaac stage helpers. Provide placeholders only when
Kit is unavailable; calling a Kit-only placeholder raises immediately.
Installation is idempotent and never replaces an already imported module."""

from __future__ import annotations

import importlib.util
import sys
import types


class _KitOnly:
    """Placeholder that satisfies from-imports but raises when used."""

    def __init__(self, what: str) -> None:
        self._what = what

    def __call__(self, *a, **k):
        raise RuntimeError(
            f"{self._what} called at runtime — this is a kit-only stub "
            f"(the rlinf checker path must not depend on it)"
        )

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _KitOnly(f"{self._what}.{name}")


def _loud_module(fullname: str) -> types.ModuleType:
    m = types.ModuleType(fullname)
    m.__path__ = []  # let the import machinery treat it as a package

    def _ga(name: str, _f: str = fullname):
        if name.startswith("__"):
            raise AttributeError(name)
        return _KitOnly(f"{_f}.{name}")

    m.__getattr__ = _ga  # type: ignore[method-assign]
    sys.modules[fullname] = m
    return m


def install_kit_stubs(logger=None) -> list:
    """Install placeholders for unavailable checker dependencies.

    Return names installed by this call. Preserve modules already loaded or
    successfully imported; repeated calls install nothing further.
    """
    installed: list = []

    # Try real imports: lazy package specs may exist without a usable Kit runtime.

    # 1. api_core wholesale (its module body needs pxr.PhysxSchema/rclpy/omni).
    ctl = "geniesim_benchmark.app.controllers"
    if ctl not in sys.modules and f"{ctl}.api_core" not in sys.modules:
        try:
            importlib.import_module(f"{ctl}.api_core")
        except Exception:  # noqa: BLE001 — kit-less: any failure → stub
            api_mod = types.ModuleType(f"{ctl}.api_core")

            class APICore:  # placeholder NAME only — see module docstring
                pass

            api_mod.APICore = APICore
            pkg = types.ModuleType(ctl)
            pkg.APICore = APICore
            pkg.api_core = api_mod
            sys.modules[ctl] = pkg
            sys.modules[f"{ctl}.api_core"] = api_mod
            installed.append(f"{ctl}.api_core")

    # Import stage utilities before substituting placeholders for unavailable modules.
    for name in ("isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.stage"):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue  # genuinely importable — leave the real module alone
        except Exception:  # noqa: BLE001 — any failure means "not usable here"
            pass
        _loud_module(name)
        installed.append(name)

    if installed and logger is not None:
        try:
            logger(
                f"[kit-stubs] no Isaac-kit runtime in this process — "
                f"stubbed for checker import: {installed} "
                f"(any RUNTIME use of these will raise loudly)"
            )
        except Exception:  # noqa: BLE001 — logging must never break eval
            pass
    return installed


__all__ = ["install_kit_stubs"]
