"""Load OpenPI's supplied Transformers replacements before pi0.5 imports."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Import order matters: modeling_gemma does ``from .configuration_gemma import
# GemmaConfig``, so the config module has to be seeded first.
OVERLAY_MODULES: tuple[tuple[str, str], ...] = (
    ("transformers.models.gemma.configuration_gemma", "models/gemma/configuration_gemma.py"),
    ("transformers.models.gemma.modeling_gemma", "models/gemma/modeling_gemma.py"),
    ("transformers.models.paligemma.modeling_paligemma", "models/paligemma/modeling_paligemma.py"),
    ("transformers.models.siglip.modeling_siglip", "models/siglip/modeling_siglip.py"),
    # openpi's own installation check. It only asserts the transformers version,
    # but PI0Pytorch.__init__ refuses to build if it cannot be imported.
    ("transformers.models.siglip.check", "models/siglip/check.py"),
)

_INSTALLED_MARKER = "_openpi_transformers_overlay"


def overlay_root() -> Path:
    """Directory holding openpi's replacement transformers modules."""
    import openpi.models_pytorch as models_pytorch

    # A namespace package, so ``__file__`` is None.
    return Path(next(iter(models_pytorch.__path__))) / "transformers_replace"


def venv_is_patched() -> bool:
    """Whether site-packages already holds openpi's replacements.

    ``check.py`` exists only in openpi's copies, so finding it on the import path
    is a sufficient marker, and ``find_spec`` answers without importing anything.
    """
    try:
        return importlib.util.find_spec("transformers.models.siglip.check") is not None
    except (ImportError, AttributeError, ValueError):
        return False


def is_installed() -> bool:
    """Whether the patched modules are what this process will get."""
    module = sys.modules.get(OVERLAY_MODULES[0][0])
    return getattr(module, _INSTALLED_MARKER, False) or venv_is_patched()


def install() -> Path | None:
    """Make sure the patched transformers modules are in effect.

    Returns the directory the modules were seeded from, or ``None`` if seeding was
    unnecessary because the venv is already patched. Idempotent.
    """
    if is_installed():
        return None if venv_is_patched() else overlay_root()

    # transformers itself must be importable before its submodules are faked.
    import transformers  # noqa: F401

    root = overlay_root()
    if not root.is_dir():
        raise FileNotFoundError(f"openpi's transformers_replace directory is missing at {root}")

    already = [name for name, _ in OVERLAY_MODULES if name in sys.modules]
    if already:
        raise RuntimeError(
            "transformers submodules were imported before the openpi overlay was "
            f"installed, so the stock versions are already in use: {already}. Either "
            "call install() before importing openpi.models_pytorch, or patch the venv "
            "once with scripts/install_venv_transformers.sh, after which the call "
            "order stops mattering."
        )

    for name, relative in OVERLAY_MODULES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"overlay module {name} is missing at {path}")
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not load overlay module {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        # Register before executing so intra-overlay relative imports find it.
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
        setattr(module, _INSTALLED_MARKER, True)

    return root
