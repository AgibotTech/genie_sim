"""
Fuzzer helpers for reading material variants and applying them to scenes.
- parse: read material_config.json and enumerate combinations
- set: apply materials, lighting, and backgrounds to scene clones
- Fuzzer: cycle through combinations across replay rounds
"""
from .core import Fuzzer
from .parse import collect_material_config, material_choices_from_config, shuffle_material_config
from .set import (
    collect_scene_material_info,
    discover_lights_under_env,
    randomize_scene_materials,
    set_background,
    set_lights_random,
    set_materials_by_choices,
)

__all__ = [
    "Fuzzer",
    "collect_material_config",
    "material_choices_from_config",
    "shuffle_material_config",
    "collect_scene_material_info",
    "randomize_scene_materials",
    "set_materials_by_choices",
    "set_lights_random",
    "set_background",
    "discover_lights_under_env",
]
