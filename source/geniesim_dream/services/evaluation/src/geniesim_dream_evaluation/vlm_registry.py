"""Per-task-family VLM weight registry with lazy switching.

Default: preload only ``qwen3_5``. Other keys (e.g. ``qwen3_vl`` for doll)
are loaded on demand; the previous resident model is released first.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from geniesim_dream_evaluation.task_router import try_classify_task_family

logger = logging.getLogger(__name__)

LoadModelFn = Callable[[str], tuple[Any, Any]]
DEFAULT_MODEL_KEY = "qwen3_5"
DEFAULT_JUDGE_FLAGS: dict[str, Any] = {
    "enable_task_route": True,
    "generic_prompt_mode": "one_shot",
}
_GENERIC_PROMPT_MODES = frozenset({"one_shot", "three_stage"})


def default_models_config_path() -> Path:
    configured = os.environ.get("DREAM_VLM_MODELS_CONFIG")
    if not configured:
        raise ValueError("Set --models-config or DREAM_VLM_MODELS_CONFIG")
    return Path(configured).expanduser()


@dataclass(frozen=True)
class ModelBundle:
    key: str
    path: Path
    model: Any
    processor: Any

    @property
    def vlm_model_id(self) -> str:
        return self.path.name


def _release_resident() -> None:
    """Collect released model objects before returning cached CUDA memory."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        logger.debug("cuda empty_cache skipped", exc_info=True)


@dataclass
class VlmModelRegistry:
    config: dict[str, Any]
    paths_by_key: dict[str, Path]
    load_model_fn: LoadModelFn
    default_family: str = "generic"
    default_model_key: str = DEFAULT_MODEL_KEY
    active_key: str | None = None
    _bundle: ModelBundle | None = field(default=None, repr=False)

    def family_for_instruction(self, instruction: str) -> str:
        family = try_classify_task_family(instruction)
        if family is None:
            return self.default_family
        return family.value

    def model_key_for_family(self, family: str) -> str:
        mapping = self.config.get("by_task_family") or {}
        key = mapping.get(family) or mapping.get(self.default_family)
        if not key:
            raise KeyError(f"No model key configured for task_family={family!r}")
        if key not in self.paths_by_key:
            raise KeyError(f"Model key {key!r} has no configured path")
        return str(key)

    def ensure_bundle(self, key: str) -> ModelBundle:
        if key not in self.paths_by_key:
            raise KeyError(f"Unknown model key: {key!r}")
        path = self.paths_by_key[key]
        if self._bundle is not None and self._bundle.path == path:
            if self._bundle.key != key:
                self._bundle = ModelBundle(
                    key=key,
                    path=path,
                    model=self._bundle.model,
                    processor=self._bundle.processor,
                )
            self.active_key = key
            return self._bundle

        if self._bundle is not None:
            logger.info(
                "Switching VLM weights %s (%s) -> %s (%s)",
                self.active_key,
                self._bundle.path,
                key,
                path,
            )
            # Drop the owning bundle before GC; a local copy would keep its
            # GPU tensors alive during cleanup and the next model's placement.
            self._bundle = None
            self.active_key = None
            _release_resident()

        if not path.exists():
            raise FileNotFoundError(
                f"VLM checkpoint missing for {key}: {path} "
                f"(create the soft link under checkpoints/ or edit models config)"
            )
        model, processor = self.load_model_fn(str(path))
        self._bundle = ModelBundle(key=key, path=path, model=model, processor=processor)
        self.active_key = key
        return self._bundle

    def pick_for_instruction(self, instruction: str) -> tuple[str, ModelBundle]:
        family = self.family_for_instruction(instruction)
        key = self.model_key_for_family(family)
        return family, self.ensure_bundle(key)

    def loaded_paths(self) -> dict[str, str]:
        if self._bundle is None or self.active_key is None:
            return {}
        return {self.active_key: str(self._bundle.path)}

    def configured_paths(self) -> dict[str, str]:
        return {key: str(path) for key, path in self.paths_by_key.items()}

    @property
    def judge_flags(self) -> dict[str, Any]:
        return parse_judge_flags(self.config)


def parse_judge_flags(config: dict[str, Any] | None) -> dict[str, Any]:
    """Read judge behavior flags; missing section/fields use run.py defaults."""
    raw = (config or {}).get("judge")
    if raw is None:
        return dict(DEFAULT_JUDGE_FLAGS)
    if not isinstance(raw, dict):
        raise ValueError("judge must be an object")
    enable = raw.get("enable_task_route", DEFAULT_JUDGE_FLAGS["enable_task_route"])
    if not isinstance(enable, bool):
        raise ValueError("judge.enable_task_route must be a bool")
    mode = raw.get("generic_prompt_mode", DEFAULT_JUDGE_FLAGS["generic_prompt_mode"])
    if mode not in _GENERIC_PROMPT_MODES:
        raise ValueError(
            f"judge.generic_prompt_mode must be one of {sorted(_GENERIC_PROMPT_MODES)}, got {mode!r}"
        )
    return {"enable_task_route": enable, "generic_prompt_mode": str(mode)}


def load_models_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("models"), dict) or not payload["models"]:
        raise ValueError(f"{path} must contain a non-empty models object")
    if not isinstance(payload.get("by_task_family"), dict) or not payload["by_task_family"]:
        raise ValueError(f"{path} must contain a non-empty by_task_family object")
    parse_judge_flags(payload)
    return payload


def resolve_model_path(raw: str, *, root: Path | None = None) -> Path:
    expanded = os.path.expandvars(raw)
    if "${" in expanded:
        raise ValueError(f"Unresolved VLM model path: {raw}")
    path = Path(expanded).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = root or Path.cwd()
    return (base / path).resolve()


def referenced_model_keys(config: dict[str, Any]) -> list[str]:
    keys = sorted({str(v) for v in (config.get("by_task_family") or {}).values()})
    missing = [key for key in keys if key not in (config.get("models") or {})]
    if missing:
        raise KeyError(f"by_task_family references unknown model keys: {missing}")
    return keys


def load_model_registry(
    config_path: Path,
    *,
    load_model_fn: LoadModelFn | None = None,
    root: Path | None = None,
    force_model_path: Path | None = None,
    preload_keys: list[str] | None = None,
    default_model_key: str = DEFAULT_MODEL_KEY,
) -> VlmModelRegistry:
    """Build registry and optionally preload keys (default: only ``qwen3_5``).

    If ``force_model_path`` is set, every model key resolves to that checkpoint.
    """
    from geniesim_dream_evaluation.vlm_backend import load_model as default_load

    loader = load_model_fn or default_load
    config = load_models_config(config_path)
    default_family = str(config.get("default_family") or "generic")
    keys = referenced_model_keys(config)
    base = root or config_path.parent

    paths_by_key: dict[str, Path] = {}
    for key in keys:
        if force_model_path is not None:
            paths_by_key[key] = force_model_path.resolve()
        else:
            paths_by_key[key] = resolve_model_path(str(config["models"][key]), root=base)

    if default_model_key not in paths_by_key:
        raise KeyError(
            f"default_model_key {default_model_key!r} is not in configured models {sorted(paths_by_key)}"
        )

    to_preload = list(preload_keys) if preload_keys is not None else [default_model_key]
    for key in to_preload:
        if key not in paths_by_key:
            raise KeyError(f"preload key {key!r} is not configured")

    registry = VlmModelRegistry(
        config=config,
        paths_by_key=paths_by_key,
        load_model_fn=loader,
        default_family=default_family,
        default_model_key=default_model_key,
    )
    for key in to_preload:
        registry.ensure_bundle(key)
    return registry
