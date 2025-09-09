# Copyright (c) 2023-2025, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
import os, sys
import numpy as np
import glob
import json, uuid
from pathlib import Path
from collections import defaultdict

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from geniesim.utils.logger import Logger

logger = Logger()  # Create singleton instance

from envs.demo_env import DemoEnv
from envs.dummy_env import DummyEnv
from geniesim.robot.genie_robot import IsaacSimRpcRobot
from policy.base import BasePolicy
from policy.demopolicy import DemoPolicy
from policy.baselinepolicy import BaselinePolicy
from geniesim.layout.task_generate import TaskGenerator
from hooks.task import TaskMetric, TaskHook
from geniesim.utils.error_code import ErrorCode
from geniesim.utils.eval_utils import *

import geniesim.utils.system_utils as system_utils
import rclpy
import time

system_utils.check_and_fix_env()


def get_hook_callbacks(policy):
    hooks = [
        TaskHook(policy),
    ]

    return (
        [hk.start_callback for hk in hooks],
        [hk.step_callback for hk in hooks],
        [hk.end_callback for hk in hooks],
        [hk.gather_results for hk in hooks],
    )


class TaskBenchmark(object):
    def __init__(self, policy: BasePolicy, args):
        self.policy = policy
        self.single_evaluate_ret = None
        self.output_dir = args.output_dir
        self.tasks = self.check_task(args)
        self.episodes_per_instance = args.num_episode
        self.args = args
        self.task_config = None
        self.record = args.record
        self.fps = args.fps

    def check_task(self, args):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if args.task_name != "":
            self.task_name = args.task_name
        else:
            raise ValueError("Invalid task_name")
        tasks = sorted(
            [
                os.path.splitext(item)[0]
                for item in os.listdir(
                    os.path.join(system_utils.benchmark_ader_path(), "eval_tasks")
                )
            ]
        )
        if self.task_name == "all":
            logger.info("Evaluating agent on all tasks")
        elif self.task_name in tasks:
            tasks = [self.task_name]
            logger.info("Evaluating agent on the given task: {}".format(tasks[0]))
        else:
            raise ValueError("Invalid task_name: {}".format(self.task_name))
        return tasks

    def evaluate_policy(self):
        for task in self.tasks:
            # load task config
            task_config_file = os.path.join(
                system_utils.benchmark_ader_path(), "eval_tasks", task + ".json"
            )
            self.task_config = system_utils.load_json(task_config_file)
            self.task_config["specific_task_name"] = task

            # benchmark result
            out_dir = os.path.join(self.output_dir, task)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            evaluate_results = []
            episode = 0
            per_episode_metrics = {}
            scene_instance_ids = [0]
            for instance_id in scene_instance_ids:
                # one instance
                self.task_config["scene"]["scene_instance_id"] = instance_id

                task_generator = TaskGenerator(self.task_config)
                task_folder = os.path.join(
                    system_utils.benchmark_root_path(),
                    "saved_task/%s" % (self.task_config["task"]),
                )
                task_generator.generate_tasks(
                    save_path=task_folder,
                    task_num=self.episodes_per_instance,
                    task_name=self.task_config["task"],
                )
                robot_position = task_generator.robot_init_pose["position"]
                robot_rotation = task_generator.robot_init_pose["quaternion"]
                self.task_config["robot"]["robot_init_pose"][
                    "position"
                ] = robot_position
                self.task_config["robot"]["robot_init_pose"][
                    "quaternion"
                ] = robot_rotation
                specific_task_files = glob.glob(task_folder + "/*.json")
                for episode_id in range(self.episodes_per_instance):
                    self.single_evaluate_ret = EVAL_TEMPLATE
                    # one episode
                    episode_file = specific_task_files[episode_id]
                    per_episode_metrics[episode] = self.evaluate_episode(episode_file)
                    episode += 1
                    summarize_scores(self.single_evaluate_ret, self.task_name)
                    evaluate_results.append(self.single_evaluate_ret)

            # output evaluate_results
            dump_eval_result(out_dir, evaluate_results)

    def evaluate_episode(self, episode_file):
        # Create agent to be evaluated
        if "robot" not in self.task_config:
            robot_cfg = "G1_120s.json"
        else:
            robot_cfg = self.task_config["robot"]["robot_cfg"]
        # init robot and scene

        robot = IsaacSimRpcRobot(
            robot_cfg=robot_cfg,
            scene_usd=self.task_config["scene"]["scene_usd"],
            client_host=self.args.client_host,
            position=self.task_config["robot"]["robot_init_pose"]["position"],
            rotation=self.task_config["robot"]["robot_init_pose"]["quaternion"],
            gripper_control_type=self.args.gripper_control_type,
        )

        if self.args.env_class == "DemoEnv":
            env = DemoEnv(robot, episode_file, self.task_config, self.policy)
        else:
            env = DummyEnv(robot, episode_file, self.task_config)
        init_pose = self.task_config["robot"].get("init_arm_pose")
        if init_pose:
            robot.set_init_pose(init_pose)

        (
            start_callbacks,
            step_callbacks,
            end_callbacks,
            data_callbacks,
        ) = get_hook_callbacks(self.policy)

        for callback in start_callbacks:  # before task
            callback(env, None)

        observaion = env.reset()  # 1st frame
        self.single_evaluate_ret["task_name"] = self.task_config["task"]
        self.single_evaluate_ret["model_path"] = ""
        self.single_evaluate_ret["start_time"] = system_utils.TIMENOW()

        if self.record:
            env.start_recording(
                task_name=self.task_name,
                camera_prim_list=[],
                fps=self.fps,
            )

        try:
            env.do_eval_action()
            while rclpy.ok():
                action = self.policy.act(observaion, step_num=env.current_step)
                for callback in step_callbacks:  # during task
                    callback(env, action)

                time.sleep(10)

                observaion, done, need_update, task_progress = env.step(action)

                logger.info(f"STEP {env.current_step}")
                if need_update:
                    self.update_eval_ret(task_progress)

                self.policy.sim_ros_node.loop_rate.sleep()

                if done:
                    break
        except KeyboardInterrupt:
            self.single_evaluate_ret["result"]["code"] = int(
                ErrorCode.ABNORMAL_INTERRUPTION.value
            )
            self.single_evaluate_ret["result"]["step"] = env.current_step
            self.single_evaluate_ret["result"]["msg"] = "expect: KeyboardInterrupt"

        for callback in end_callbacks:  # during task
            callback(env, action)
        self.single_evaluate_ret["end_time"] = system_utils.TIMENOW()

        if self.record:
            env.robot.client.stop_recording()

        robot.client.Exit()

        metrics_summary = {}
        return metrics_summary

    def convert_code(self, info):
        if info["done_cond_name"] == "Timeout":
            return int(ErrorCode.OUT_OF_MAX_STEP.value)
        elif info["done_cond_name"] == "PredicateGoal":
            return int(ErrorCode.UNKNOWN_ERROR.value)
        return int(ErrorCode.INIT_VALUE.value)

    def assemble_ret(self, info):
        self.single_evaluate_ret["result"]["step"] = info["final_step"]
        if info["success"] == True:
            self.single_evaluate_ret["result"]["code"] = int(ErrorCode.SUCCESS.value)
        else:
            self.single_evaluate_ret["result"]["code"] = self.convert_code(info)
            self.single_evaluate_ret["result"]["msg"] = ""

    def update_eval_ret(self, task_progress):
        self.single_evaluate_ret["result"]["progress"] = task_progress

    def other_output(self, log_file, summary_log_file, per_episode_metrics):
        with open(log_file, "w+") as f:
            json.dump(per_episode_metrics, f)
            logger.info("Per episode eval results saved to %s" % log_file)

            aggregated_metrics = {}
            success_score = []
            simulator_time = []
            kinematic_disarrangement = []
            logical_disarrangement = []
            distance_navigated = []
            displacement_of_hands = []

            task_to_mean_success_score = defaultdict(list)
            task_scores = []

            for episode, metric in per_episode_metrics.items():
                task_to_mean_success_score[metric["task"]].append(
                    metric["q_score"]["final"]
                )

            for task, scores in task_to_mean_success_score.items():
                task_scores.append(np.mean(scores))

            task_scores = sorted(task_scores, reverse=True)

            for episode, metric in per_episode_metrics.items():
                success_score.append(metric["q_score"]["final"])
                simulator_time.append(metric["time"]["simulator_time"])
                kinematic_disarrangement.append(
                    metric["kinematic_disarrangement"]["relative"]
                )
                logical_disarrangement.append(
                    metric["logical_disarrangement"]["relative"]
                )
                distance_navigated.append(
                    np.sum(metric["agent_distance"]["timestep"]["body"])
                )
                displacement_of_hands.append(
                    np.sum(metric["grasp_distance"]["timestep"]["left_hand"])
                    + np.sum(metric["grasp_distance"]["timestep"]["right_hand"])
                )

            aggregated_metrics["Success Score"] = np.mean(success_score)
            aggregated_metrics["Success Score Top 5"] = np.mean(
                np.array(task_scores)[:5]
            )
            aggregated_metrics["Simulated Time"] = np.mean(simulator_time)
            aggregated_metrics["Kinematic Disarrangement"] = np.mean(
                kinematic_disarrangement
            )
            aggregated_metrics["Logical Disarrangement"] = np.mean(
                logical_disarrangement
            )
            aggregated_metrics["Distance Navigated"] = np.mean(distance_navigated)
            aggregated_metrics["Displacement of Hands"] = np.mean(displacement_of_hands)
            with open(summary_log_file, "w+") as f:
                json.dump(aggregated_metrics, f)
            logger.info("Aggregated eval results saved to %s" % summary_log_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--client_host",
        type=str,
        default="localhost:50051",
        help="The client host",
    )
    parser.add_argument(
        "--num_episode",
        type=int,
        default=1,
        help="Set number of episodes to run",
    )
    parser.add_argument(
        "--policy_class",
        type=str,
        default="DemoPolicy",
        choices=["DemoPolicy", "BaselinePolicy"],
        help="Choose the policy class",
    )
    parser.add_argument(
        "--env_class",
        type=str,
        default="DummyEnv",
        choices=["DemoEnv", "DummyEnv"],
        help="Choose the task env",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="iros_stamp_the_seal",
        help="Specify the task to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(system_utils.benchmark_root_path(), "output"),
        help="Set output directory",
    )
    parser.add_argument(
        "--gripper_control_type",
        type=int,
        default=0,
        help="Set gripper control type, 0-position control, 1-velocity control",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Set the fps of the recording"
    )
    parser.add_argument("--record", action="store_true", help="Enable data recording")
    args = parser.parse_args()

    logger.info(
        "Evaluating agent of type {} on {}".format(args.policy_class, args.task_name)
    )

    if args.policy_class == "DemoPolicy":
        policy = DemoPolicy(task_name=args.task_name)
    elif args.policy_class == "BaselinePolicy":
        policy = BaselinePolicy(task_name=args.task_name)
    else:
        raise ValueError("Invalid policy class: {}".format(args.policy_class))

    benchmark = TaskBenchmark(policy, args)
    benchmark.evaluate_policy()  # Evaluate agent on the benchmark
    policy.shutdown()


if __name__ == "__main__":
    main()
