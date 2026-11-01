# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import os
from .common_actions import *
from .custom import (
    PickUpOnGripper,
    Inside,
    InBBox,
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
    LiftUp,
    Upright,
    Rotated,
    StableGrasp,
    ChassisAtTarget,
    Stack,
    RelativePositionChecker,
    MixedRules,
    PlaceOnRivet,
    RobotNearObject,
    Erased,
    FoldCycle1,
    FoldCycle2,
    FoldCycle3,
    FoldCycle1Tshirt006,
    FoldCycle2Tshirt006,
    FoldCycle3Tshirt006,
    PlugInserted,
    Hang,
    InkOverlap,
    PourVolume,
    ScissorsHandleInserted,
    ScissorsHandleActuated,
)
import geniesim_benchmark.utils.system_utils as system_utils
from geniesim_benchmark.plugins.output_system.policy_budget import PolicyBudgetError


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
    from geniesim_benchmark.plugins.output_system.policy_budget import (
        configured_scoring_action, select_scoring_problem,
    )

    # ScoreLayout is an explicit problem-level selection. Reset it on every
    # episode so legacy problems never inherit another episode's layout.
    env.task.score_layout_name = None
    sub_task_name = env.init_task_config.get("sub_task_name", "")
    robot_cfg = getattr(env, "robot_cfg", None) or env.init_task_config.get("robot_cfg")
    if sub_task_name == "":
        acts_filename = get_definition_filename(env.params.task_name, env.params.instance, task_definitions_path)
        cfg = select_scoring_problem(scan_config(acts_filename), robot_cfg)
        env.task.score_layout_name = cfg.get("ScoreLayout")
        problem_name = cfg["Problem"]
        objects = cfg["Objects"]
        task_progress = []
        acts = parse_action(configured_scoring_action(cfg), task_progress, env)
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
                problem = select_scoring_problem(problems["problem" + str(idx + 1)], robot_cfg)
                env.task.score_layout_name = problem.get("ScoreLayout")
                problem_name = problem["Problem"]
                task_progress = []
                objects = []
                acts = parse_action(configured_scoring_action(problem), task_progress, env)
                return problem_name, objects, acts, task_progress
        except Exception as e:
            if isinstance(e, PolicyBudgetError):
                raise
            print(f"No problems found for {path} ({e!r}), use default")
            default_problem = os.path.join(task_definitions_path, "default_problem.json")
            cfg = select_scoring_problem(scan_config(default_problem), robot_cfg)
            env.task.score_layout_name = cfg.get("ScoreLayout")
            problem_name = cfg["Problem"]
            objects = cfg["Objects"]
            task_progress = []
            acts = parse_action(configured_scoring_action(cfg), task_progress, env)
            return problem_name, objects, acts, task_progress


def record_act_obj(act, task_progress, spec=None):
    task_progress.append(
        {
            "class_name": act.__class__.__name__,
            "id": hex(id(act)),
            "progress": "",
            "acion_obj": act,
            # The declaration string of a leaf checker (object ids, bbox,
            # gripper, budget); containers carry their children instead.
            "spec": spec if isinstance(spec, (str, int, float)) else None,
        }
    )


def parse_action(obj: dict, task_progress, env, stage_step_out=None) -> ActionBase:
    for key, value in obj.items():
        if key == "PolicyDispatchBudget":
            from .policy_dispatch_budget import PolicyDispatchBudget

            try:
                required = {"version", "step", "action"}
                if (not isinstance(value, dict) or not required <= set(value)
                        or set(value) - required - {"points", "step_out"}):
                    raise ValueError("PolicyDispatchBudget requires version, step, action; optional points, step_out")
                if "step_out" in value and (stage_step_out is None or value["step_out"] != stage_step_out):
                    raise ValueError("PolicyDispatchBudget.step_out must match the enclosing native StepOut")
                stage = parse_action(value["action"], task_progress, env, stage_step_out)
                act = PolicyDispatchBudget(env, stage, value["step"], value["version"],
                                           points=value.get("points"), step_out=value.get("step_out"))
                record_act_obj(act, task_progress, value)
                # Declare every stage before execution, retaining unknown counts
                # for stages that never start. No native SCORE is added here.
                task_progress[-1]["progress"] = act.progress_info
                return act
            except Exception as exc:
                raise PolicyBudgetError(f"Invalid PolicyDispatchBudget: {exc}") from exc
        if key == "PointInVolume":
            from .custom.point_in_volume import PointInVolume

            act = PointInVolume(env, **value)
            record_act_obj(act, task_progress, value)
            return act
        if key == "OnSurface":
            from .custom.on_surface import OnSurface

            act = OnSurface(env, **value)
            record_act_obj(act, task_progress, value)
            return act
        if key == "ActionSetSuccessFirst":
            from .success_first import ActionSetSuccessFirst

            if not isinstance(value, list) or not value:
                raise ValueError("ActionSetSuccessFirst requires a nonempty action list")
            act = ActionSetSuccessFirst(env)
            record_act_obj(act, task_progress, value)
            local_budgets = [item["StepOut"] for item in value if "StepOut" in item]
            inherited_budget = local_budgets[0] if len(local_budgets) == 1 else stage_step_out
            for item in value:
                if not isinstance(item, dict) or len(item) != 1:
                    raise ValueError("ActionSetSuccessFirst requires single-key actions")
                act.add_action(parse_action(item, task_progress, env, inherited_budget))
            return act
        if key == "ThroughOpening":
            from .custom.through_opening import ThroughOpening

            act = ThroughOpening(env, **value)
            record_act_obj(act, task_progress, value)
            return act
        if key == "ShaftInserted":
            from .custom.shaft_inserted import ShaftInserted

            act = ShaftInserted(env, **value)
            record_act_obj(act, task_progress, value)
            return act
        if key in ("InsidePrecise", "CoverPrecise"):
            from .custom.precise import InsidePrecise, CoverPrecise

            act = {"InsidePrecise": InsidePrecise, "CoverPrecise": CoverPrecise}[key](env, **value)
            record_act_obj(act, task_progress, value)
            return act
        if key == "ActionList":
            act = ActionList(env)
            record_act_obj(act, task_progress, value)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env, stage_step_out))
            return act
        elif key == "ActionSetWaitAny":
            act = ActionSetWaitAny(env)
            record_act_obj(act, task_progress, value)
            local_budgets = [item["StepOut"] for item in value if "StepOut" in item]
            inherited_budget = local_budgets[0] if len(local_budgets) == 1 else stage_step_out
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env, inherited_budget))
            return act
        elif key == "ActionSetWaitAll":
            act = ActionSetWaitAll(env)
            record_act_obj(act, task_progress, value)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env, stage_step_out))
            return act
        elif key.startswith("ActionSetWaitSome"):
            tail = key[len("ActionSetWaitSome") :].lstrip("_")
            try:
                num = int(tail)
            except ValueError:
                num = 1
            act = ActionSetWaitSome(env, num)
            record_act_obj(act, task_progress, value)
            for item in value:
                for k, v in item.items():
                    act.add_action(parse_action({k: v}, task_progress, env, stage_step_out))
            return act
        elif key == "ActionWaitForTime":
            act = ActionWaitForTime(env, wait_time=value)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "PickUpOnGripper":
            params = value.split("|")
            act = PickUpOnGripper(env, params[0], params[1])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Timeout":
            act = TimeOut(env, time_out=float(value))
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Inside":
            params = value.split("|")
            act = Inside(env, params[0], params[1], params[2])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Erased":
            # Format: "<prim>|<threshold>"; prim is the eraser's entity prim
            # (eraser:erasedFraction) or the board's Chalk mesh (chalk:active).
            params = value.split("|")
            act = Erased(env, params[0], params[1] if len(params) >= 2 else 0.95)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "InBBox":
            # Format: "object_id|center_x,center_y,center_z|len_x,len_y,len_z"
            params = value.split("|")
            center_values = params[1].split(",")
            size_values = params[2].split(",")
            bbox_center = [float(center_values[0]), float(center_values[1]), float(center_values[2])]
            bbox_size = [float(size_values[0]), float(size_values[1]), float(size_values[2])]
            act = InBBox(env, params[0], bbox_center, bbox_size)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "FluidInside":
            params = value.split("|")
            act = FluidInside(env, params[0], params[1])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "StepOut":
            act = StepOut(env, max_step=value)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "FoldCycle1":
            # value is an optional cloth name ("auto" or ""); the checker auto-finds
            # the surface deformable cloth, so we just pass it through.
            act = FoldCycle1(env, cloth_name=value if value else "auto")
            record_act_obj(act, task_progress, value)
            return act
        elif key == "FoldCycle2":
            act = FoldCycle2(env, cloth_name=value if value else "auto")
            record_act_obj(act, task_progress, value)
            return act
        elif key == "FoldCycle3":
            act = FoldCycle3(env, cloth_name=value if value else "auto")
            record_act_obj(act, task_progress, value)
            return act
        elif key == "FoldCycle1Tshirt006":
            act = FoldCycle1Tshirt006(env, cloth_name=value if value else "auto")
            record_act_obj(act, task_progress, value)
            return act
        elif key == "FoldCycle2Tshirt006":
            act = FoldCycle2Tshirt006(env, cloth_name=value if value else "auto")
            record_act_obj(act, task_progress, value)
            return act
        elif key == "FoldCycle3Tshirt006":
            act = FoldCycle3Tshirt006(env, cloth_name=value if value else "auto")
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Onfloor":
            params = value.split("|")
            act = Onfloor(env, obj_name=params[0], height=params[1])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Ontop":
            params = value.split("|")
            kwargs = {}
            if len(params) >= 3:
                kwargs["area_denom"] = params[2]
            if len(params) >= 4:
                kwargs["threshold"] = float(params[3])
            act = Ontop(env, params[0], params[1], **kwargs)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Cover":
            params = value.split("|")
            act = Cover(env, params[0], params[1])
            record_act_obj(act, task_progress, value)
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
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Follow":
            params = value.split("|")
            act = Follow(env, params[0], params[1], params[2])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Approach":
            params = value.split("|")
            act = Approach(env, float(params[0]), float(params[1]), float(params[2]))
            record_act_obj(act, task_progress, value)
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
            record_act_obj(act, task_progress, value)
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
            record_act_obj(act, task_progress, value)
            return act
        elif key == "TriggerAction":
            params = value.split("|")
            act = TriggerAction(env, params[0], params[1])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "CheckStainClean":
            params = value.split("|")
            act = CheckStainClean(env, params[0], int(params[1]))
            record_act_obj(act, task_progress, value)
            return act
        elif key == "GripperPassing":
            params = value.split("|")
            act = GripperPassing(env, params[0], params[1] == "true")
            record_act_obj(act, task_progress, value)
            return act
        elif key == "VLM":
            params = value.split("|")
            task_id = params[0]
            interval = params[1]
            act = VLM(env, task_id, int(interval))
            record_act_obj(act, task_progress, value)
            return act
        elif key == "LiftUp":
            params = value.split("|")
            act = LiftUp(env, params[0], float(params[1]))
            record_act_obj(act, task_progress, value)
            return act
        elif key == "InkOverlap":
            # Format: "ink_prim|guide_prim|pass_score|sigma_precision|cov_full_dist
            #          |cov_zero_dist|guide_half_width|ink_half_width|quality_exponent
            #          |resample_step"
            params = value.split("|")
            kwargs = {}
            for idx, name in enumerate(
                (
                    "pass_score",
                    "sigma_precision",
                    "cov_full_dist",
                    "cov_zero_dist",
                    "guide_half_width",
                    "ink_half_width",
                    "quality_exponent",
                    "resample_step",
                ),
                start=2,
            ):
                if len(params) > idx:
                    kwargs[name] = float(params[idx])
            act = InkOverlap(env, params[0], params[1], **kwargs)
            record_act_obj(act, task_progress)
            return act
        elif key == "PourVolume":
            # Format: "<particle_obj>|<receiver_obj>|<source_obj>|<min>|<max>"
            #          "|<expected_total>|<settle_checks>|<settle_tol>|<max_tilt_deg>"
            params = value.split("|")
            kwargs = {}
            for idx, name in enumerate(
                (
                    "min_particles",
                    "max_particles",
                    "expected_total",
                    "settle_checks",
                    "settle_tol",
                    "max_tilt_deg",
                ),
                start=3,
            ):
                if len(params) > idx and params[idx] != "":
                    kwargs[name] = params[idx]
            if len(params) < 5:
                raise ValueError("PourVolume needs at least particle|receiver|source|min|max")
            act = PourVolume(env, params[0], params[1], params[2], **kwargs)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Upright":
            params = value.split("|")
            allow_flipped = params[2].lower() == "true" if len(params) > 2 else False
            act = Upright(env, params[0], float(params[1]), allow_flipped)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Rotated":
            # Format: "obj|angle_deg" or "obj|angle_deg|ax,ay,az" (optional
            # signed world axis; rotation about other axes is projected out).
            params = value.split("|")
            axis = None
            if len(params) > 2 and params[2].strip():
                axis = [float(v) for v in params[2].split(",")]
            act = Rotated(env, params[0], float(params[1]), axis=axis)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "StableGrasp":
            params = value.split("|")
            kwargs = {}
            if len(params) >= 3:
                kwargs["distance_threshold"] = float(params[2])
            if len(params) >= 4:
                kwargs["pose_diff_pos_threshold"] = float(params[3])
            if len(params) >= 5:
                kwargs["pose_diff_rot_threshold_rad"] = float(params[4])
            if len(params) >= 6:
                kwargs["min_movement_threshold"] = float(params[5])
            act = StableGrasp(env, params[0], params[1], **kwargs)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "ChassisAtTarget":
            # Format: "[x,y,yaw]|[x_thresh,y_thresh,yaw_thresh]"
            params = value.split("|")
            act = ChassisAtTarget(env, params[0], params[1])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Stack":
            # Format: "[a,b,c]|[x,y]" - object IDs and XY threshold
            params = value.split("|")
            act = Stack(env, params[0], params[1])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "RelativePosition":
            # Format: "obj_A|obj_B|relation" or "obj_A|obj_B|relation|threshold"
            params = value.split("|")
            threshold = params[3] if len(params) > 3 else None
            act = RelativePositionChecker(env, params[0], params[1], params[2], threshold)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "MixedRules":
            rules = value["rules"]
            check_interval = value.get("check_interval", 1)
            mode = value.get("mode", "or")
            kwargs = {"strict": value["strict"]} if "strict" in value else {}
            if "carry_distance" in value:
                kwargs["carry_distance"] = value["carry_distance"]
            act = MixedRules(env, rules, check_interval, mode, **kwargs)
            record_act_obj(act, task_progress, value)
            return act
        elif key == "PlaceOnRivet":
            if isinstance(value, dict):
                act = PlaceOnRivet(env, **value)
                record_act_obj(act, task_progress, value)
                return act
            # Format: "active_obj|passive_obj|rel_x,rel_y,rel_z|qw,qx,qy,qz|xy_tol|z_tol|orient_tol|still_thresh|still_steps"
            params = value.split("|")
            act = PlaceOnRivet(
                env,
                active_obj=params[0],
                passive_obj=params[1],
                target_rel_pos=params[2],
                target_quat=params[3],
                xy_tol=float(params[4]) if len(params) > 4 else 0.02,
                z_tol=float(params[5]) if len(params) > 5 else 0.01,
                orient_tol=float(params[6]) if len(params) > 6 else 0.15,
                still_thresh=float(params[7]) if len(params) > 7 else 0.02,
                still_steps=int(params[8]) if len(params) > 8 else 15,
            )
            record_act_obj(act, task_progress, value)
            return act
        elif key == "RobotNearObject":
            params = value.split("|")
            act = RobotNearObject(env, params[0], params[1])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "PlugInserted":
            # Format: "plug_obj" | "plug_obj|joint_name" | "plug_obj|joint_name|hold_ticks"
            params = value.split("|")
            act = PlugInserted(
                env,
                plug_name=params[0],
                joint_name=params[1] if len(params) > 1 and params[1] else "PhoneAttachJoint",
                hold_ticks=int(params[2]) if len(params) > 2 else 1,
            )
            record_act_obj(act, task_progress, value)
            return act
        elif key == "Hang":
            # Format: "obj_name|min_height[|stable_checks[|tolerance]]"
            params = value.split("|")
            act = Hang(env, params[0], float(params[1]), *[float(p) for p in params[2:4]])
            record_act_obj(act, task_progress, value)
            return act
        elif key == "ScissorsHandleInserted":
            # Structured format:
            #   {"object": "benchmark_scissors_006", "hold_ticks": 3}
            act = ScissorsHandleInserted(
                env,
                obj_name=value["object"],
                hold_ticks=value.get("hold_ticks", 3),
            )
            record_act_obj(act, task_progress, value)
            return act
        elif key == "ScissorsHandleActuated":
            # Structured format:
            #   {"object": "benchmark_scissors_006",
            #    "min_travel_ratio": 0.15, "hold_ticks": 2}
            act = ScissorsHandleActuated(
                env,
                obj_name=value["object"],
                hold_ticks=value.get("hold_ticks", 2),
                min_travel_ratio=value.get("min_travel_ratio", 0.15),
            )
            record_act_obj(act, task_progress, value)
            return act
        else:
            raise ValueError(f"Unknown action type: {key}")
    return None
