"""
Read material variants from configuration and enumerate possible combinations.
"""
import itertools
import json
import os
import random

from common.base_utils.logger import logger


def shuffle_material_config(
    material_config_by_entity: dict,
    rng: random.Random | None = None,
) -> dict:
    """
    Shuffle each object's material choices in a new dict without modifying input.
    Environments then consume the shuffled choices in order.
    """
    rng = rng or random.Random()
    out = {}
    for obj_id, part_materials in material_config_by_entity.items():
        out[obj_id] = {}
        for part_rel_path, material_look_paths in part_materials.items():
            if isinstance(material_look_paths, list) and material_look_paths:
                lst = list(material_look_paths)
                rng.shuffle(lst)
                out[obj_id][part_rel_path] = lst
            else:
                out[obj_id][part_rel_path] = material_look_paths
    return out


def _asset_id_from_usd_path(obj_usd: str) -> str:
    """Get the asset ID from a USD path, such as table_000/Aligned.usda."""
    return os.path.basename(os.path.dirname(obj_usd))


def collect_material_config(
    sim_assets: str,
    object_asset_dict: dict,
    no_fuzzing_set: set[str] | None = None,
    objects_dir_name: str = "Objects",
) -> dict:
    """
    Load material_config.json for each asset in object_asset_dict.
    Skip assets listed in no_fuzzing_set. Return choices by object and part;
    object IDs include the /entity suffix.
    """
    no_fuzzing_set = no_fuzzing_set or set()
    objects_segment = f"/{objects_dir_name}/"
    material_config_by_entity = {}
    for prim_path, obj_usd in object_asset_dict.items():
        if objects_segment not in prim_path:
            continue
        asset_id = _asset_id_from_usd_path(obj_usd)
        if asset_id in no_fuzzing_set:
            continue
        obj_id = prim_path.split(objects_segment)[-1].strip("/").split("/")[0]
        obj_id += "/entity"
        rel_path = obj_usd.split("/assets/")[-1] if "/assets/" in obj_usd else obj_usd
        full_usd = (
            obj_usd
            if os.path.isabs(obj_usd) and os.path.isfile(obj_usd)
            else os.path.join(sim_assets, rel_path)
        )
        if not os.path.isfile(full_usd):
            continue
        asset_dir = os.path.dirname(full_usd)
        config_path = os.path.join(asset_dir, "material_config.json")
        if not os.path.isfile(config_path):
            continue
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.warning("Failed to load material_config.json for %s: %s", obj_id, e)
            continue
        if not isinstance(config, dict):
            continue
        material_config_by_entity[obj_id] = {}
        for part_rel_path, material_look_paths in config.items():
            if isinstance(material_look_paths, list) and material_look_paths:
                material_config_by_entity[obj_id][part_rel_path] = list(material_look_paths)
    return material_config_by_entity


def material_choices_from_config(material_config_by_entity: dict) -> list[dict]:
    """
    Enumerate material combinations from material_config_by_entity.
    Each choice maps an object and part to a material index for one environment.
    """
    if not material_config_by_entity:
        return [{}]
    keys = []
    choices_per_key = []
    for obj_id, part_materials in material_config_by_entity.items():
        for part_rel_path, material_look_paths in part_materials.items():
            if material_look_paths:
                keys.append((obj_id, part_rel_path))
                choices_per_key.append(list(range(len(material_look_paths))))
    if not keys:
        return [{}]
    combos = list(itertools.product(*choices_per_key))
    out = []
    for combo in combos:
        cfg = {}
        for (obj_id, part_rel_path), idx in zip(keys, combo):
            cfg.setdefault(obj_id, {})[part_rel_path] = idx
        out.append(cfg)
    return out

