# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import os
from .common_actions import *
from .custom import (
    PickUpOnGripper,
    Inside,
    FluidInside,
    Onfloor,
    Ontop,
    Cover,
    CheckParticleInBBox,
    PushPull,
    Follow,
    OnShelf,
    TriggerAction,
    CheckStainClean,
    GripperPassing,
    Approach,
    VLM,
)
import geniesim.utils.system_utils as system_utils


def get_definition_filename(behavior_activity, instance, task_definitions_path):
    ori_path = os.path.join(
        task_definitions_path,
        behavior_activity,
        f"problem{instance}.json",
    )

    if os.path.exists(ori_path):
        return ori_path

    return os.path.join(
        task_definitions_path,
        "default_problem.json",
    )


def scan_config(filename=None):
    if filename is not None:
        with open(filename) as f:
            eval_config = json.load(f)
    else:
        raise ValueError("No input JSON provided.")

    return eval_config


def get_object_scope(object_terms):
    scope = {}
    for object_cat in object_terms:
        for object_inst in object_terms[object_cat]:
            scope[object_inst] = None
    return scope


def do_parsing(env, task_definitions_path):
    sub_task_name = env.init_task_config.get("sub_task_name", "")
    if sub_task_name == "":
        acts_filename = get_definition_filename(env.params.task_name, env.params.instance, task_definitions_path)
        cfg = scan_config(acts_filename)
        problem_name = cfg["Problem"]
        objects = cfg["Objects"]
        task_progress = []
        acts = parse_action(cfg["Acts"][0], task_progress, env)
        return problem_name, objects, acts, task_progress
    else:
        instance_id = env.init_task_config["scene"]["scene_instance_id"]
        path = os.path.join(
            system_utils.benchmark_conf_path(),
            "llm_task",
            sub_task_name,
            str(instance_id),
        )
        try:
            with open(
                os.path.join(path, "problems.json"),
                "r",
            ) as f:
                problems = json.load(f)
                idx = env.task.current_episode_id % len(env.task.instructions)
                problem = problems["problem" + str(idx + 1)]
                problem_name = problem["Problem"]
                task_progress = []
                objects = []
                acts = parse_action(problem["Acts"][0], task_progress, env)
                return problem_name, objects, acts, task_progress
        except:
            print(f"No problems found for {path}, use default")
            default_problem = os.path.join(task_definitions_path, "default_problem.json")
            cfg = scan_config(default_problem)
            problem_name = cfg["Problem"]
            objects = cfg["Objects"]
            task_progress = []
            acts = parse_action(cfg["Acts"][0], task_progress, env)
            return problem_name, objects, acts, task_progress


def record_act_obj(act, task_progress):
    task_progress.append(
        {
            "class_name": act.__class__.__name__,
            "id": hex(id(act)),
            "progress": "",
            "acion_obj": act,
        }
    )


def parse_action(obj: dict, task_progress, env) -> ActionBase:
    for key, value in obj.items():
        if key == "ActionList":
            act = ActionList(env)
            record_act_obj(act, task_progress)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env))
            return act
        elif key == "ActionSetWaitAny":
            act = ActionSetWaitAny(env)
            record_act_obj(act, task_progress)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env))
            return act
        elif key == "ActionSetWaitAll":
            act = ActionSetWaitAll(env)
            record_act_obj(act, task_progress)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env))
            return act
        elif key.startswith("ActionSetWaitSome"):
            tail = key[len("ActionSetWaitSome") :].lstrip("_")
            try:
                num = int(tail)
            except ValueError:
                num = 1
            act = ActionSetWaitSome(env, num)
            record_act_obj(act, task_progress)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env))
            return act
        elif key == "ActionWaitForTime":
            act = ActionWaitForTime(env, wait_time=value)
            record_act_obj(act, task_progress)
            return act
        elif key == "PickUpOnGripper":
            params = value.split("|")
            act = PickUpOnGripper(env, params[0], params[1])
            record_act_obj(act, task_progress)
            return act
        elif key == "Timeout":
            act = TimeOut(env, time_out=float(value))
            record_act_obj(act, task_progress)
            return act
        elif key == "Inside":
            params = value.split("|")
            act = Inside(env, params[0], params[1], params[2])
            record_act_obj(act, task_progress)
            return act
        elif key == "FluidInside":
            params = value.split("|")
            act = FluidInside(env, params[0], params[1])
            record_act_obj(act, task_progress)
            return act
        elif key == "StepOut":
            act = StepOut(env, max_step=value)
            record_act_obj(act, task_progress)
            return act
        elif key == "Onfloor":
            params = value.split("|")
            act = Onfloor(env, obj_name=params[0], height=params[1])
            record_act_obj(act, task_progress)
            return act
        elif key == "Ontop":
            params = value.split("|")
            act = Ontop(env, params[0], params[1])
            record_act_obj(act, task_progress)
            return act
        elif key == "Cover":
            params = value.split("|")
            act = Cover(env, params[0], params[1])
            record_act_obj(act, task_progress)
            return act
        elif key == "PushPull":
            params = value.split("|")
            act = PushPull(
                env,
                params[0],
                params[1],
                params[2],
                int(params[3]) if len(params) > 3 else 0,
            )
            record_act_obj(act, task_progress)
            return act
        elif key == "Follow":
            params = value.split("|")
            act = Follow(env, params[0], params[1], params[2])
            record_act_obj(act, task_progress)
            return act
        elif key == "Approach":
            params = value.split("|")
            act = Approach(env, float(params[0]), float(params[1]), float(params[2]))
            record_act_obj(act, task_progress)
            return act
        elif key == "CheckParticleInBBox":
            params = value.split("|")
            values = params[1].split(",")
            act = CheckParticleInBBox(
                env,
                int(params[0]),
                [
                    float(values[0]),
                    float(values[1]),
                    float(values[2]),
                    float(values[3]),
                    float(values[4]),
                    float(values[5]),
                ],
            )
            record_act_obj(act, task_progress)
            return act
        elif key == "OnShelf":
            params = value.split("|")
            act = OnShelf(
                env,
                params[0],
                params[1],
                params[2],
                params[3],
            )
            record_act_obj(act, task_progress)
            return act
        elif key == "TriggerAction":
            params = value.split("|")
            act = TriggerAction(env, params[0], params[1])
            record_act_obj(act, task_progress)
            return act
        elif key == "CheckStainClean":
            params = value.split("|")
            act = CheckStainClean(env, params[0], int(params[1]))
            record_act_obj(act, task_progress)
            return act
        elif key == "GripperPassing":
            params = value.split("|")
            act = GripperPassing(env, params[0], params[1] == "true")
            record_act_obj(act, task_progress)
            return act
        elif key == "VLM":
            params = value.split("|")
            if len(params) == 1:
                act = VLM(env, params[0], "")
            else:
                act = VLM(env, params[0], params[1])
            record_act_obj(act, task_progress)
            return act
        else:
            raise ValueError(f"Unknown action type: {key}")
    return None
