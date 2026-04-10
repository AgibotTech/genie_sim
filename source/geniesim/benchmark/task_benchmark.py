# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
from re import sub
import os
import sys
import glob
import shutil
from pathlib import Path

import numpy as np
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from geniesim.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from geniesim.plugins.output_system import TaskEvaluation, EvaluationSummary
from geniesim.plugins.tgs import ObjectSampler, TaskGenerator
from geniesim.utils.data_courier import DataCourier
import geniesim.utils.system_utils as system_utils
from geniesim.utils.name_utils import robot_type_mapping
from geniesim.benchmark.envs.demo_env import DemoEnv
from geniesim.benchmark.policy.demopolicy import DemoPolicy
from geniesim.benchmark.hooks.task import TaskHook
from geniesim.app.controllers.api_core import APICore
from geniesim.utils.generalization_utils import update_init_env

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
    def __init__(self, args, api_core: APICore):
        self.args = args
        self.api_core: APICore = api_core
        self.policy = None
        self.env = None
        self.tasks = self.check_task()
        self.task_config = None

        seed = getattr(args, "seed", 0)
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

        self.data_courier = DataCourier(self.api_core, self.api_core.enable_ros, self.args.model_arc)

    def check_task(self):
        if self.args.task_name != "":
            self.task_name = self.args.task_name
        else:
            raise ValueError("args.task_name is empty.")
        tasks = sorted(
            [
                os.path.splitext(item)[0]
                for item in os.listdir(os.path.join(system_utils.benchmark_conf_path(), "eval_tasks"))
            ]
        )
        if self.task_name == "all":
            logger.info("Evaluating agent on all tasks")
        elif self.task_name in tasks:
            tasks = [self.task_name]
            logger.info("Evaluating agent on the given task: {}".format(tasks[0]))
        else:
            logger.warning("Use self-defined task_name: {}".format(self.task_name))
        return tasks

    def config_task(self):
        if self.args.model_arc in ["pi", "abs_pose"]:
            self.task_mode = "infer"
        else:
            self.task_mode = "empty"

    def evaluate_policy(self):
        self.config_task()
        for task in self.tasks:
            # load task config
            task_config_file = os.path.join(system_utils.benchmark_conf_path(), "eval_tasks", task + ".json")
            robot_init_pose_file = os.path.join(system_utils.benchmark_conf_path(), "robot_init_pose.json")
            self.task_config = system_utils.load_json(task_config_file)
            self.robot_init_pose = system_utils.load_json(robot_init_pose_file)
            self.task_config["specific_task_name"] = task
            self.task_config["sub_task_name"] = self.args.sub_task_name
            self.instruction = self.task_config.get("instruction", "")
            logger.info(f"sub_task_name: {self.args.sub_task_name}")

            if hasattr(self.api_core, "sub_task_name"):
                self.api_core.sub_task_name = self.args.sub_task_name
                if hasattr(self.api_core, "benchmark_ros_node") and self.api_core.benchmark_ros_node is not None:
                    if hasattr(self.api_core.benchmark_ros_node, "set_sub_task_name"):
                        self.api_core.benchmark_ros_node.set_sub_task_name(self.args.sub_task_name)
                        logger.info(f"Republished sub_task_name to ROS: {self.args.sub_task_name}")

            task_generator = TaskGenerator(self.task_config)
            task_folder = os.path.join(
                system_utils.benchmark_root_path(),
                "saved_task/%s" % (self.task_config["task"]),
            )

            gen_config = self.task_config.get("generalization", {})

            task_generator.generate_tasks(
                save_path=task_folder,
                task_name=self.task_config["task"],
                gen_config=gen_config,
            )
            robot_position = task_generator.robot_init_pose["position"]
            robot_rotation = task_generator.robot_init_pose["quaternion"]
            self.task_config["robot"]["robot_init_pose"]["position"] = robot_position
            self.task_config["robot"]["robot_init_pose"]["quaternion"] = robot_rotation
            if "robot" not in self.task_config:
                robot_cfg = "G1_120s.json"
            else:
                robot_cfg = self.task_config["robot"]["robot_cfg"]

            self.task_config["robot_cfg"] = robot_type_mapping(robot_cfg.split(".")[0])
            self.data_courier.set_robot_cfg(self.task_config["robot_cfg"])

            episode_idx = 0
            scene_instance_ids = [0]
            sub_task_name = self.args.sub_task_name
            if sub_task_name != "":
                sub_task_path = os.path.join(system_utils.benchmark_conf_path(), "llm_task", sub_task_name)
                scene_instance_ids = sorted([int(name) for name in os.listdir(sub_task_path) if name.isdigit()])

            self.evaluate_summary = EvaluationSummary(
                os.path.join(system_utils.benchmark_output_path()), task, sub_task_name
            )
            for instance_id in scene_instance_ids:
                # one instance
                self.task_config["scene"]["scene_instance_id"] = instance_id
                specific_task_files = sorted(glob.glob(task_folder + "/*.json"))

                self.create_policy()
                self.create_env(specific_task_files[0], instance_id)
                time.sleep(0.5)
                self.api_core.collect_init_physics()

                seed = getattr(self.args, "seed", 0)
                for file_idx, episode_file in enumerate(specific_task_files):
                    logger.info(f"EPISODE FILE: {episode_file}")
                    self.episode_content = system_utils.load_json(episode_file)
                    gen_seed = seed + instance_id * 1000 + file_idx
                    np.random.seed(gen_seed)
                    update_init_env(self.env, self.task_config, self.episode_content)
                    self.env.apply_generalization(self.api_core, self.task_config)

                    if self.args.record:
                        self.set_record_topics()
                        self.env.start_recording(
                            camera_prim_list=[],
                            fps=self.args.fps,
                            extra_prim_paths=[],
                            record_topic_list=self.record_topic_list,
                        )

                    episode_count = 1 if self.args.preview else len(self.env.task.instructions)
                    episode_seed = seed + instance_id
                    for episode_id in range(self.args.num_episode * episode_count):
                        np.random.seed(episode_seed)
                        self.env.set_current_task(episode_id)
                        if self.instruction != "":
                            self.env.task.set_instruction(self.instruction)
                        current_instruction = self.env.task.get_instruction()
                        single_te = TaskEvaluation(task_name=self.task_name, sub_task_name=sub_task_name)
                        single_te.update_from_dict(
                            {
                                "task_name": self.task_config["task"],
                                "task_type": "benchmark",
                                "robot_type": robot_cfg.split(".")[0],
                                "start_time": system_utils.get_local_time(),
                                "model_type": self.args.model_arc,
                                "task_instruction": current_instruction[0],
                            }
                        )
                        self.evaluate_summary.update_current(single_te)
                        self.data_courier.pub_static_info_msg(
                            self.evaluate_summary.to_static_msg_pub(
                                episode_idx, self.args.num_episode, self.data_courier.sim_time()
                            )
                        )
                        self.data_courier.pub_dynamic_info_msg(self.evaluate_summary.to_dynamic_msg_pub())
                        # one episode
                        self.evaluate_episode(robot_cfg, single_te)
                        self.policy.set_episode_idx(episode_idx)
                        if self.args.record:
                            self.api_core.benchmark_ros_node.pub_episode_done(episode_idx)
                            self.api_core.benchmark_ros_node.wait_episode_ack(timeout=30.0)

                        episode_idx += 1
                        self.evaluate_summary.make_cache()

                    if self.args.record:
                        self.env.stop_recording()

                self.env.stop()
            self.api_core.stop()
            try:
                task_folder_abs = os.path.abspath(task_folder)
                if os.path.isdir(task_folder_abs):
                    shutil.rmtree(task_folder_abs)
                    logger.info("Removed task folder: %s" % task_folder_abs)
            except Exception as e:
                logger.warning("Failed to remove task folder %s: %s" % (task_folder, e))

    @staticmethod
    def _parse_infer_host(infer_host: str):
        """Parse infer_host in 'host:port' form to (host, port). Default port 8999 if omitted."""
        if ":" in infer_host:
            host, port_str = infer_host.rsplit(":", 1)
            return host.strip(), int(port_str.strip())
        return infer_host.strip(), 8999

    def create_policy(self):
        host, port = self._parse_infer_host(self.args.infer_host)
        logger.info(f"Infer service address: {host}:{port}")
        if self.args.model_arc == "pi" or self.args.model_arc == "abs_pose":
            from geniesim.benchmark.policy.pipolicy import PiPolicy

            self.policy = PiPolicy(
                task_name=self.args.task_name,
                host_ip=host,
                port=port,
                sub_task_name=self.args.sub_task_name,
                preview=self.args.preview,
            )
        elif self.args.model_arc == "":
            from geniesim.benchmark.policy.base import BasePolicy

            self.policy = BasePolicy(
                task_name=self.args.task_name,
                sub_task_name=self.args.sub_task_name,
            )
        elif self.args.policy_class == "DemoPolicy":
            self.policy = DemoPolicy(
                task_name=self.args.task_name,
                sub_task_name=self.args.sub_task_name,
            )
        else:

            if self.args.policy_class == "DemoPolicy":
                self.policy = DemoPolicy(
                    task_name=self.args.task_name,
                    sub_task_name=self.args.sub_task_name,
                )
            elif False:
                # placeholder, customize your own policy here
                pass
            else:
                raise ValueError("Invalid policy class: {}".format(self.args.policy_class))
        self.policy.set_data_courier(self.data_courier)

    def gen_layouts(self, mode):
        assets_folder = os.path.join(system_utils.assets_path(), "objects", "gm")
        if os.path.exists(assets_folder):
            self.obj_sampler = ObjectSampler(
                self.api_core,
                self.task_config["task"],
                self.task_config.get("problem_instance", 0),
                assets_folder,
            )
            if mode == "instance":
                scene_info = self.obj_sampler.generate_scenes_from_instance()
            elif mode == "input":
                scene_info = self.obj_sampler.generate_scenes_from_input()
        else:
            logger.warning(f"Assets folder {assets_folder} does not exist")
            scene_info = {}
        return scene_info

    def create_env(self, episode_file, instance_id):
        if "robot" not in self.task_config:
            robot_cfg = "G1_120s.json"
        else:
            robot_cfg = self.task_config["robot"]["robot_cfg"]

        # init robot and scene
        sub_usd_path = ""
        if self.args.sub_task_name != "":
            sub_usd_path = os.path.join(
                system_utils.benchmark_conf_path(),
                "llm_task",
                self.args.sub_task_name,
                str(instance_id),
                "scene.usda",
            )

        self.api_core.init_robot_cfg(
            robot_cfg,
            self.task_config["scene"]["scene_usd"],
            self.task_config["robot"]["robot_init_pose"]["position"],
            self.task_config["robot"]["robot_init_pose"]["quaternion"],
            sub_usd_path,
        )

        scene_info = None
        if self.args.sub_task_name == "":
            scene_info = self.gen_layouts(mode="instance")

        if self.args.model_arc == "pi":
            from geniesim.benchmark.envs.pi_env import PiEnv

            self.env = PiEnv(self.api_core, episode_file, self.task_config)
        elif self.args.model_arc == "abs_pose":
            from geniesim.benchmark.envs.abs_pose_env import AbsPoseEnv

            self.env = AbsPoseEnv(self.api_core, episode_file, self.task_config)
        elif self.args.model_arc == "":
            from geniesim.benchmark.envs.dummy_env import DummyEnv

            self.env = DummyEnv(self.api_core, episode_file, self.task_config)
        elif self.args.env_class == "DemoEnv":
            self.env = DemoEnv(self.api_core, episode_file, self.task_config, self.policy)
        elif self.args.env_class == "DummyEnv":
            self.env = DummyEnv(self.api_core, episode_file, self.task_config)
        else:
            raise ValueError("Invalid env_class {self.args.env_class}")

        self.env.set_data_courier(self.data_courier)
        self.env.set_scene_info(scene_info)

    def set_record_topics(self):
        if "G1" in self.task_config["robot"]["robot_cfg"]:
            self.record_topic_list = [
                "/tf",
                "/joint_states",
                "/record/camera_rgb",
                "/record/head_camera_rgb",
                "/record/left_camera_rgb",
                "/record/right_camera_rgb",
                "/record/static_info",
            ]
        elif "G2" in self.task_config["robot"]["robot_cfg"]:
            self.record_topic_list = [
                "/tf",
                "/joint_states",
                "/record/camera_rgb",
                "/record/head_front_camera_rgb",
                "/record/left_camera_rgb",
                "/record/right_camera_rgb",
                "/record/static_info",
            ]
        else:
            raise ValueError("Invalid robot cfg")

    def evaluate_episode(self, robot_cfg, single_te: TaskEvaluation):
        # Create agent to be evaluated
        observaion = self.env.reset()  # 1st frame
        start_time = time.time()
        (
            start_callbacks,
            step_callbacks,
            end_callbacks,
            data_callbacks,
        ) = get_hook_callbacks(self.policy)

        for callback in start_callbacks:  # before task
            callback(self.env, None)

        try:
            if not self.args.interactive:
                self.env.do_eval_action()
            while self.data_courier.loop_ok():
                if self.task_mode == "infer":
                    if self.args.interactive:
                        while not self.data_courier.get_infer_start():
                            logger.info("waiting for infer start")
                            time.sleep(1)
                            # if self.env.current_step != 0:
                            self.env.current_step = 0
                            observaion = self.env.reset()

                            self.policy.reset()
                        if self.data_courier.get_shuffle():
                            self.api_core.shuffle_scene()
                        single_instruction = [self.data_courier.get_instruction()]
                    else:
                        single_instruction = self.env.task.get_instruction()
                    action = self.policy.act(
                        observaion,
                        step_num=self.env.current_step,
                        task_instruction=single_instruction[0],
                        gen_config=self.env.get_camera_gen_config(),
                    )
                else:
                    action = self.policy.act(observaion, step_num=self.env.current_step)
                for callback in step_callbacks:  # during task
                    callback(self.env, action)

                self.env.set_infer_status(self.policy.need_infer())

                observaion, done, need_update, task_progress = self.env.step(action)
                logger.info(f"STEP {self.env.current_step}")
                if not self.args.interactive:
                    if need_update:
                        single_te.update_progress(task_progress)
                        self.data_courier.pub_dynamic_info_msg(self.evaluate_summary.make_temp_statistic())
                    if done:
                        self.env.current_step = 0
                        single_te.summarize_scores()
                        observaion = self.env.reset()
                        self.policy.reset()
                        break
                self.data_courier.sleep()

        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt during episode")
            single_te.assemble_expect_result()
            # Ensure recording is stopped on interrupt
            if self.args.record and hasattr(self, "env") and self.env is not None:
                try:
                    if hasattr(self.env, "api_core") and self.env.api_core is not None:
                        self.env.api_core.stop_all_recording()
                        logger.info("Recording stopped due to KeyboardInterrupt")
                except Exception as e:
                    logger.warning(f"Failed to stop recording on interrupt: {e}")

        for callback in end_callbacks:  # during task
            callback(self.env, action)

        end_time = time.time()
        single_te.update_from_dict(
            {
                "end_time": system_utils.get_local_time(),
                "duration": end_time - start_time,
            }
        )


def main(args=None, api_core=None):
    benchmark = TaskBenchmark(args, api_core)
    benchmark.evaluate_policy()  # Evaluate agent on the benchmark


if __name__ == "__main__":
    main()
