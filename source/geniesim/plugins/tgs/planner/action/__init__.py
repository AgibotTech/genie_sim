# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from .grasp import GraspStage, PickStage
from .place import PlaceStage

ACTION_STAGE = {"grasp": GraspStage, "pick": PickStage, "place": PlaceStage}


def build_stage(action):
    if action not in ACTION_STAGE:
        raise NotImplementedError
    return ACTION_STAGE[action]
