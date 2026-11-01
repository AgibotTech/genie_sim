# Copyright (c) 2023-2026, AgiBot Inc.
# SPDX-License-Identifier: MPL-2.0
"""Compatibility loader for trusted local NumPy asset pickles."""

import pickle
from pathlib import Path


class _NumpyAssetUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2 serializes arrays through numpy._core; NumPy 1.26 exposes
        # the same reconstruction helpers through numpy.core.
        if module == "numpy._core" or module.startswith("numpy._core."):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_asset_pickle(path: str | Path):
    """Load a trusted local asset pickle across NumPy 1.x/2.x module paths."""
    with open(path, "rb") as stream:
        return _NumpyAssetUnpickler(stream).load()
