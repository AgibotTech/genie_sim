"""
Apply selected materials, lights, and backgrounds to replay environments.
Distant lights vary in angle and intensity; other lights vary in intensity.
"""
import os
import random
from typing import Any

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, PhysxSchema

from common.base_utils.logger import logger

def _get_isolated_part_prim_path(env_path: str) -> str:
    return env_path.rstrip("/") + "_isolated"

def _disable_physics_under_prim(stage, prim_path: str) -> None:
    """Disable collision and rigid-body behaviour below prim_path."""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim or not root_prim.IsValid():
        return
    for prim in Usd.PrimRange(root_prim):
        path = str(prim.GetPath())
        # Remove articulation and rigid-body APIs before disabling collision.
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
        if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, path)
        if rigid_api:
            rigid_api.GetRigidBodyEnabledAttr().Set(False)
        collision_api = UsdPhysics.CollisionAPI.Get(stage, path)
        if collision_api:
            collision_api.GetCollisionEnabledAttr().Set(False)

def set_background(
    stage,
    env_paths: list[str],
    background_lists: list[dict[str, Any]],
    sim_assets: str = "",
    log=None,
    shuffle_list: bool = True,
) -> None:
    """
    Set a background on each environment and remove physics under its reference.
    Reuse backgrounds cyclically when there are fewer than environments.
    Each background provides a USD path, position, and rotation. When
    shuffle_list is false, keep the supplied order across rounds.
    """
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.core.prims import SingleXFormPrim as XFormPrim
    from isaacsim.core.utils.prims import delete_prim

    sim_assets = sim_assets or os.environ.get("SIM_ASSETS", "")
    log = log or logger
    if not background_lists:
        log.info("set_background: no backgrounds, skip")
        return
    n_bg = len(background_lists)
    if shuffle_list:
        random.shuffle(background_lists)
    log.info(f"set_background: background_lists: {background_lists}")
    for env_index, env_path in enumerate(env_paths):
        bg_cfg = background_lists[env_index % n_bg]
        usd_rel = bg_cfg.get("usd") if isinstance(bg_cfg, dict) else str(bg_cfg)
        if not usd_rel:
            raise ValueError(f"set_background: env {env_path} background has no usd")
        full_usd = usd_rel if os.path.isabs(usd_rel) and os.path.isfile(usd_rel) else os.path.join(sim_assets, usd_rel)
        if not os.path.isfile(full_usd):
            raise ValueError(f"set_background: USD not found {full_usd}")
        env_isolated_part_prim_path = _get_isolated_part_prim_path(env_path)
        if not stage.GetPrimAtPath(env_isolated_part_prim_path).IsValid():
            env_xform = XFormPrim(prim_path=env_path)
            env_origin_position, env_origin_rotation = env_xform.get_world_pose()
            xform = XFormPrim(prim_path=env_isolated_part_prim_path, position=env_origin_position, orientation=env_origin_rotation)
        background_path = env_isolated_part_prim_path + "/background_fuzz"
        existing = stage.GetPrimAtPath(background_path)
        if existing and existing.IsValid():
            delete_prim(existing.GetPath())
        add_reference_to_stage(full_usd, background_path)
        position = bg_cfg.get("position", [0, 0, 0]) if isinstance(bg_cfg, dict) else [0, 0, 0]
        rotation = bg_cfg.get("rotation", [1, 0, 0, 0]) if isinstance(bg_cfg, dict) else [1, 0, 0, 0]
        xform = XFormPrim(prim_path=background_path, translation=position, orientation=rotation)
        log.info(f"set_background for env {env_path}: {background_path} {position} {rotation}")


def set_materials_by_choices(
    stage,
    env_paths: list,
    material_config_by_entity: dict,
    env_id_to_choices: dict,
    log=None,
    objects_dir_name: str = "Objects",
) -> None:
    """
    Bind materials for each environment using env_id_to_choices.
    env_id_to_choices[env_id] = { obj_id: { part_rel_path: material_index } }
    """
    log = log or logger
    for env_index, env_path in enumerate(env_paths):
        choices = env_id_to_choices.get(env_index, {})
        for obj_id, part_materials in material_config_by_entity.items():
            object_root = f"{env_path}/{objects_dir_name}/{obj_id}"
            obj_choices = choices.get(obj_id, {})
            for part_rel_path, material_look_paths in part_materials.items():
                if not material_look_paths:
                    continue
                idx = obj_choices.get(part_rel_path, env_index % len(material_look_paths))
                material_look = material_look_paths[idx % len(material_look_paths)]
                target_prim_path = f"{object_root}/{part_rel_path}"
                material_prim_path = f"{object_root}/{material_look}"
                target_prim = stage.GetPrimAtPath(target_prim_path)
                material_prim = stage.GetPrimAtPath(material_prim_path)
                if not target_prim.IsValid():
                    log.debug("Skip material bind: target prim not found %s", target_prim_path)
                    continue
                if not material_prim.IsValid():
                    log.debug("Skip material bind: material prim not found %s", material_prim_path)
                    continue
                material = UsdShade.Material(material_prim)
                if not material:
                    continue
                try:
                    binding_api = UsdShade.MaterialBindingAPI.Apply(target_prim)
                    binding_api.Bind(material)
                except Exception as e:
                    log.warning(
                        "Failed to bind material %s -> %s: %s",
                        material_prim_path,
                        target_prim_path,
                        e,
                    )


def collect_scene_material_info(stage, env_paths: list[str]) -> dict[str, list[str]]:
    """
    Scan meshes in each environment and its isolated section for materials.
    Candidates include peers of the bound material and materials beginning
    with ml_{mesh_name}. Return only meshes with multiple choices.
    """
    all_results: dict[str, list[str]] = {}
    processed_meshes: set[str] = set()

    def _scan_under(root_prim):
        if not root_prim or not root_prim.IsValid():
            return
        for prim in Usd.PrimRange(root_prim):
            if "/G2/" in str(prim.GetPath()) or "/G1/" in str(prim.GetPath()):
                continue
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh_path = str(prim.GetPath())
            if mesh_path in processed_meshes:
                continue
            processed_meshes.add(mesh_path)

            parent_prim = prim.GetParent()
            mesh_name = parent_prim.GetName() if parent_prim and parent_prim.IsValid() else ""
            current_mesh_name = prim.GetName() if prim and prim.IsValid() else ""
            ml_prefix = f"ml_{mesh_name}" if mesh_name else ""
            ml_current_prefix = f"ml_{current_mesh_name}" if current_mesh_name else ""

            material_paths: list[str] = []
            material_path_set: set[str] = set()

            def _add_material(path: str):
                if path not in material_path_set:
                    material_path_set.add(path)
                    material_paths.append(path)

            binding_api = UsdShade.MaterialBindingAPI(prim)
            material, _binding_rel = binding_api.ComputeBoundMaterial()

            if material:
                mat_path = str(material.GetPath())
                mat_prim = stage.GetPrimAtPath(mat_path)
                if mat_prim.IsValid():
                    mat_parent = mat_prim.GetParent()
                    if mat_parent and mat_parent.IsValid():
                        for sub_prim in Usd.PrimRange(mat_parent):
                            if sub_prim.IsA(UsdShade.Material):
                                sub_name = sub_prim.GetName()
                                sub_path = str(sub_prim.GetPath())
                                if ml_prefix and sub_name.startswith(ml_prefix):
                                    _add_material(sub_path)
                                elif ml_current_prefix and sub_name.startswith(ml_current_prefix):
                                    _add_material(sub_path)
                                elif not ml_prefix:
                                    _add_material(sub_path)

            if not material_paths:
                direct_binding_rel = binding_api.GetDirectBindingRel()
                if direct_binding_rel:
                    for target_path in direct_binding_rel.GetTargets():
                        target_prim = stage.GetPrimAtPath(target_path)
                        if target_prim.IsValid():
                            for sub_prim in Usd.PrimRange(target_prim):
                                if sub_prim.IsA(UsdShade.Material):
                                    sub_name = sub_prim.GetName()
                                    sub_path = str(sub_prim.GetPath())
                                    if ml_prefix and sub_name.startswith(ml_prefix):
                                        _add_material(sub_path)
                                    elif not ml_prefix:
                                        _add_material(sub_path)

            if len(material_paths) > 1:
                all_results[mesh_path] = material_paths

    for env_path in env_paths:
        env_prim = stage.GetPrimAtPath(env_path)
        _scan_under(env_prim)
        isolated_path = _get_isolated_part_prim_path(env_path)
        isolated_prim = stage.GetPrimAtPath(isolated_path)
        if isolated_prim and isolated_prim.IsValid():
            _scan_under(isolated_prim)
    logger.info(f"collect_scene_material_info: found {len(all_results)} meshes with >1 materials: {list(all_results.keys())}")
    return all_results


def randomize_scene_materials(
    stage,
    material_info: dict[str, list[str]],
    rng: random.Random,
    log=None,
) -> None:
    """
    Randomly choose and bind one discovered material per mesh.
    """
    log = log or logger
    for mesh_path, material_paths in material_info.items():
        mesh_prim = stage.GetPrimAtPath(mesh_path)
        if not mesh_prim or not mesh_prim.IsValid():
            continue
        chosen = rng.choice(material_paths)
        material = UsdShade.Material.Get(stage, Sdf.Path(chosen))
        if not material:
            log.warning("randomize_scene_materials: material not found %s", chosen)
            continue
        try:
            UsdShade.MaterialBindingAPI(mesh_prim).UnbindAllBindings()
            UsdShade.MaterialBindingAPI(mesh_prim).Bind(material)
        except Exception as e:
            log.warning("randomize_scene_materials: bind failed %s -> %s: %s", chosen, mesh_path, e)


def _is_light_prim(prim) -> bool:
    """Return whether the prim is a supported UsdLux light."""
    if not prim or not prim.IsValid():
        return False
    return prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.SphereLight) or prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DiskLight)


def discover_lights_under_env(stage, env_path: str) -> list[str]:
    """Return paths of supported lights below an environment."""
    lights = []
    env_prim = stage.GetPrimAtPath(env_path)
    if not env_prim or not env_prim.IsValid():
        return lights
    for prim in Usd.PrimRange(env_prim):
        if _is_light_prim(prim):
            lights.append(prim.GetPath().pathString)
    return lights


def discover_root_lights(stage) -> list[str]:
    """Return supported lights directly below the stage root."""
    lights = []
    try:
        root = stage.GetPseudoRoot()
        for prim in root.GetChildren():
            if _is_light_prim(prim):
                lights.append(prim.GetPath().pathString)
    except Exception as e:
        logger.error("discover_root_lights: %s", e)
        
    return lights


def set_lights_random(
    stage,
    env_paths: list,
    rng: random.Random,
    log=None,
) -> None:
    """
    Randomize lights in each environment relative to their original values.
    Scale intensity by 0.5 to 1.5 and rotate distant lights by up to 45 degrees.
    """
    log = log or logger
    ROTATE_DELTA = 45.0
    INTENSITY_SCALE_LO, INTENSITY_SCALE_HI = 0.5, 1.5

    def _get_original_intensity(prim) -> float:
        try:
            lux = None
            if prim.IsA(UsdLux.DomeLight):
                lux = UsdLux.DomeLight(prim)
            if prim.IsA(UsdLux.DistantLight):
                lux = UsdLux.DistantLight(prim)
            if prim.IsA(UsdLux.SphereLight):
                lux = UsdLux.SphereLight(prim)
            if prim.IsA(UsdLux.RectLight):
                lux = UsdLux.RectLight(prim)
            if prim.IsA(UsdLux.DiskLight):
                lux = UsdLux.DiskLight(prim)
            if lux:
                attr = lux.GetIntensityAttr()
                if attr:
                    return float(attr.Get())
        except Exception as e:
            logger.error("get_original_intensity: %s", e)
        return 1.0

    def _get_original_rotation(prim) -> tuple[float, float, float]:
        try:
            xform = UsdGeom.Xformable(prim)
            if xform:
                for op in xform.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        v = op.Get()
                        return (float(v[0]), float(v[1]), float(v[2]))
        except Exception as e:
            logger.error("get_original_rotation: %s", e)
        return (0.0, 0.0, 0.0)

    def _apply_to_light(light_path: str, prim, rotation, intensity) -> None:
        if not prim or not prim.IsValid():
            return
        is_distant = prim.IsA(UsdLux.DistantLight)
        try:
            if is_distant and rotation is not None:
                xform = UsdGeom.Xformable(prim)
                if xform:
                    ops = xform.GetOrderedXformOps()
                    rotate_op = None
                    for op in ops:
                        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                            rotate_op = op
                            break
                    if rotate_op:
                        rotate_op.Set(Gf.Vec3f(*rotation))
                    else:
                        xform.AddRotateXYZOp().Set(Gf.Vec3f(*rotation))
        except Exception as e:
            log.error("set_lights_random: rotation failed %s: %s", light_path, e)
        try:
            lux = None
            if prim.IsA(UsdLux.DomeLight):
                lux = UsdLux.DomeLight(prim)
            if prim.IsA(UsdLux.DistantLight):
                lux = UsdLux.DistantLight(prim)
            if prim.IsA(UsdLux.SphereLight):
                lux = UsdLux.SphereLight(prim)
            if prim.IsA(UsdLux.RectLight):
                lux = UsdLux.RectLight(prim)
            if prim.IsA(UsdLux.DiskLight):
                lux = UsdLux.DiskLight(prim)
            if lux and intensity is not None:
                attr = lux.GetIntensityAttr()
                if attr:
                    attr.Set(float(intensity))
        except Exception as e:
            log.error("set_lights_random: intensity failed %s: %s", light_path, e)

    for env_index, env_path in enumerate(env_paths):
        light_paths = discover_lights_under_env(stage, env_path)
        env_isolated_part_prim_path = _get_isolated_part_prim_path(env_path)
        light_paths.extend(discover_lights_under_env(stage, env_isolated_part_prim_path))
        for light_path in light_paths:
            prim = stage.GetPrimAtPath(light_path)
            if not prim or not prim.IsValid():
                continue
            orig_intensity = _get_original_intensity(prim)
            intensity = orig_intensity * rng.uniform(INTENSITY_SCALE_LO, INTENSITY_SCALE_HI)
            rotation = None
            if prim.IsA(UsdLux.DistantLight):
                ox, oy, oz = _get_original_rotation(prim)
                rotation = (
                    ox + rng.uniform(-ROTATE_DELTA, ROTATE_DELTA),
                    oy + rng.uniform(-ROTATE_DELTA, ROTATE_DELTA),
                    oz + rng.uniform(-ROTATE_DELTA, ROTATE_DELTA),
                )
            _apply_to_light(light_path, prim, rotation, intensity)

    root_lights = discover_root_lights(stage)
    for light_path in root_lights:
        prim = stage.GetPrimAtPath(light_path)
        if not prim or not prim.IsValid():
            continue
        orig_intensity = _get_original_intensity(prim)
        intensity = orig_intensity * rng.uniform(INTENSITY_SCALE_LO, INTENSITY_SCALE_HI)
        rotation = None
        if prim.IsA(UsdLux.DistantLight):
            ox, oy, oz = _get_original_rotation(prim)
            rotation = (
                ox + rng.uniform(-ROTATE_DELTA, ROTATE_DELTA),
                oy + rng.uniform(-ROTATE_DELTA, ROTATE_DELTA),
                oz + rng.uniform(-ROTATE_DELTA, ROTATE_DELTA),
            )
        _apply_to_light(light_path, prim, rotation, intensity)
