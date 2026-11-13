# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Load only the selected checker and its runtime dependencies."""

from importlib import import_module

_MODULES = {
    "OnShelf": "onshelf",
    "Inside": "inside",
    "InBBox": "inbbox",
    "PickUpOnGripper": "pickup_on_gripper",
    "Ontop": "ontop",
    "OnTopCount": "ontop_count",
    "FluidInside": "fluid_inside",
    "Onfloor": "onfloor",
    "Cover": "cover",
    "CheckParticleInBBox": "check_particle_in_bbox",
    "PushPull": "push_pull",
    "Follow": "follow",
    "TriggerAction": "trigger_action",
    "CheckStainClean": "check_stain_clean",
    "GripperPassing": "gripper_passing",
    "Approach": "approach",
    "VLM": "vlm",
    "ScoopPopcorn": "popcorn",
    "LiftUp": "liftup",
    "Upright": "upright",
    "Rotated": "rotated",
    "StableGrasp": "stable_grasp",
    "ChassisAtTarget": "chassis_at_target",
    "RobotNearObject": "robot_near_object",
    "Stack": "stack",
    "RelativePositionChecker": "relative_position_checker",
    "MixedRules": "mixed_rules",
    "PlaceOnRivet": "place_on_rivet",
    "PlugInserted": "plug_inserted",
    "Hang": "hang",
    "InkOverlap": "ink_overlap",
    "PourVolume": "pour_volume",
    "FoldCycle1": "folded",
    "FoldCycle2": "folded",
    "FoldCycle3": "folded",
    "FoldCycle1Tshirt006": "folded_tshirt006",
    "FoldCycle2Tshirt006": "folded_tshirt006",
    "FoldCycle3Tshirt006": "folded_tshirt006",
    "ScissorsHandleInserted": "scissors_handle",
    "ScissorsHandleActuated": "scissors_handle",
    "Erased": "erased",
    "WujiFoldLeftSide": "wuji_fold",
    "WujiFoldRightSide": "wuji_fold",
    "WujiFoldHemToCollar": "wuji_fold",
    "WujiLiftCloth": "wuji_hanger",
    "WujiLiftHanger": "wuji_hanger",
    "WujiHangerNearCollar": "wuji_hanger",
    "WujiHangerInserted": "wuji_hanger",
    "WujiGarmentHung": "wuji_hanger",
    "WujiBagPlaced": "wuji_bag",
    "WujiBottle1InBin": "wuji_bag",
    "WujiBottle2InBin": "wuji_bag",
    "WujiBothBottlesInBag": "wuji_bag",
    "WujiBagCinched": "wuji_bag",
}

__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
    globals()[name] = value
    return value
