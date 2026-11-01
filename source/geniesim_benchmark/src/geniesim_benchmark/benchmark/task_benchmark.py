# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import argparse
from re import sub
import os
import sys
import glob
import shutil
import json
import copy
import math
import threading
from pathlib import Path

import numpy as np
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from geniesim_benchmark.plugins.output_system import TaskEvaluation, EvaluationSummary
from geniesim_benchmark.plugins.tgs import ObjectSampler, TaskGenerator
from geniesim_benchmark.utils.comm.retry import InferenceUnavailableError
from geniesim_benchmark.utils.data_courier import DataCourier
import geniesim_benchmark.utils.system_utils as system_utils
from geniesim_benchmark.utils.name_utils import robot_type_mapping
from geniesim_benchmark.benchmark.policy.demopolicy import DemoPolicy
from geniesim_benchmark.benchmark.hooks.task import TaskHook
from geniesim_benchmark.benchmark.hooks.pregrasp import PreGraspHook
from geniesim_benchmark.benchmark.subscene_override import resolve_sub_usd_path
from geniesim_benchmark.app.controllers.api_core import APICore
from geniesim_benchmark.utils.generalization_utils import update_init_env

system_utils.check_and_fix_env()


class _StartupValidationComplete(Exception):
    def __init__(self, result):
        super().__init__("G2_90d startup validation completed")
        self.result = result


def reject_free_base_vec(task_config, enable_vec):
    if int(enable_vec or 0) > 0 and bool(task_config.get("robot", {}).get("free_base", False)):
        raise ValueError("free_base is supported only in single-world mode; " "set benchmark.enable_vec=0")


def g2_90d_mobile_state_passed(profile, state):
    expected_drive = profile["chassis"]["velocity_drive"]
    transition = state.get("last_transition") or {}
    return (
        state["anchor"]["enabled"] is False
        and len(state["wheel_collisions"]) == 4
        and all(item["active"] is True for item in state["wheel_collisions"].values())
        and len(state["roll_drives"]) == 4
        and all(
            math.isclose(
                item["stiffness"],
                float(expected_drive["stiffness"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                item["damping"],
                float(expected_drive["damping"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for item in state["roll_drives"].values()
        )
        and int(state.get("transition_id", 0)) > 0
        and transition.get("helper") == "set_g2_90d_chassis_mode"
        and transition.get("anchored") is False
        and transition.get("transition_id") == state.get("transition_id")
    )


def get_hook_callbacks(policy):
    hooks = [
        TaskHook(policy),
        PreGraspHook(),
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
        self.control_mode = bool(getattr(args, "control_mode", False))
        self.policy = None
        self.env = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_reason = None
        self._control_shutdown_ready = False
        self._control_shutdown_invoked = False
        self.tasks = self.check_task()
        self.task_config = None
        # Set per task in _evaluate_policy_impl(); points at the in-progress
        # task's output. Used to clean up its result file on abnormal abort.
        self.evaluate_summary = None

        seed = getattr(args, "seed", 0)
        np.random.seed(seed)
        logger.info(f"Random seed set to: {seed}")

        self.data_courier = DataCourier(self.api_core, self.api_core.enable_ros, self.args.model_arc)

        # Shared WS client slot for serial-mode policies. Currently unused: policies
        # manage their own connection, so this stays None unless
        # _get_or_create_ws_client() is wired in; evaluate_policy() still closes it
        # defensively in its finally block.
        self._ws_client = None

    def request_shutdown(self, reason):
        """Stop this benchmark through its own control and transport seam."""

        with self._shutdown_lock:
            if self._shutdown_reason is None:
                self._shutdown_reason = str(reason)
            reason = self._shutdown_reason
            invoke_control = self.control_mode and self._control_shutdown_ready and not self._control_shutdown_invoked
            if invoke_control:
                self._control_shutdown_invoked = True

        result = {"reason": reason, "courier_stop": "requested"}
        self.data_courier.request_stop()
        if invoke_control:
            result["control_stop"] = self.policy.request_control_shutdown(reason)
        return result

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
        elif self.control_mode:
            raise ValueError(
                "control_mode requires an installed task configuration; " f"missing eval_tasks/{self.task_name}.json"
            )
        else:
            logger.warning("Use self-defined task_name: {}".format(self.task_name))
        return tasks

    def config_task(self):
        if self.control_mode:
            self._validate_control_mode()
            self.task_mode = "infer"
            return
        if self.args.model_arc == "corobot":
            self.task_mode = "infer"
        else:
            self.task_mode = "empty"

    def _validate_control_mode(self):
        args = self.args
        if args.model_arc != "corobot":
            raise ValueError("control_mode requires model_arc=corobot")
        if args.interactive:
            raise ValueError("control_mode uses the shared-control UI, not benchmark.interactive")
        if (getattr(args, "enable_vec", 0) or 0) != 0:
            raise ValueError("control_mode does not support vectorized environments")
        if args.preview or args.record or getattr(args, "dump_lerobot", False):
            raise ValueError("control_mode does not support preview, recording, or dataset export")
        if (
            getattr(args, "anno_bpt", False)
            or getattr(args, "anno_goalimg", False)
            or getattr(args, "scene_geometry", False)
        ):
            raise ValueError("control_mode forbids BPT, goal-image, and scene-geometry routing")
        if args.language_perturbation:
            raise ValueError("control_mode forbids benchmark instruction routing")
        if args.num_episode != 1 or args.num_instances != 1:
            raise ValueError("control_mode requires exactly one episode slot and one scene instance")
        user_text = getattr(args, "control_user_text", "")
        if not isinstance(user_text, str) or len(user_text) > 8192:
            raise ValueError("control_user_text must be a string of at most 8192 characters")

    def evaluate_policy(self):
        self.config_task()
        try:
            self._evaluate_policy_impl()
        except _StartupValidationComplete as completed:
            self.startup_validation_result = completed.result
            logger.info(
                "G2_90d startup validation finished with "
                f"status={completed.result['status']} "
                f"result={completed.result['result_path']}"
            )
        except BaseException:
            # Simulator aborted: an exception propagated out, or the inference
            # retry budget exhausted and called sys.exit (SystemExit). The
            # in-progress task's result file is partial and would be scored as
            # a bogus run, so remove it before the process unwinds.
            self._cleanup_task_output()
            raise
        finally:
            if self._ws_client is not None:
                try:
                    self._ws_client.close()
                except Exception:
                    pass
                self._ws_client = None

    def _cleanup_task_output(self):
        """Delete the in-progress task's evaluation result file.

        Only the result file this run created (``evaluate_summary.file_path``)
        is removed; result files from prior runs in the same output directory
        are left untouched.
        """
        summary = self.evaluate_summary
        result_file = getattr(summary, "file_path", None) if summary is not None else None
        if not result_file:
            return
        try:
            if os.path.isfile(result_file):
                os.remove(result_file)
                logger.info("Cleaned up task result file after abort: %s", result_file)
        except Exception as e:
            logger.warning("Failed to clean up task result file %s: %s", result_file, e)

    def _evaluate_policy_impl(self):
        for task in self.tasks:
            if self.data_courier.stop_requested():
                break
            # load task config
            task_config_file = os.path.join(system_utils.benchmark_conf_path(), "eval_tasks", task + ".json")
            self.task_config = system_utils.load_json(task_config_file)
            if self.control_mode:
                # The task config owns the base mode: fixed-base tasks stay
                # welded where the benchmark spawns them, and a task that
                # drives the chassis declares robot.free_base in its own config.
                reject_free_base_vec(
                    self.task_config,
                    getattr(self.args, "enable_vec", 0),
                )
            self.task_config["specific_task_name"] = task
            self.task_config["sub_task_name"] = self.args.sub_task_name
            gen_instruction = self.task_config.get("generalization", {}).get("instruction", {})
            self.task_config["language_perturbation"] = getattr(self.args, "language_perturbation", False) or bool(
                gen_instruction.get("enable", False)
            )
            self.task_config["language_perturbation_config"] = getattr(self.args, "language_perturbation_config", "")
            self.instruction = self.task_config.get("instruction", "")
            logger.info(f"sub_task_name: {self.args.sub_task_name}")

            if hasattr(self.api_core, "sub_task_name"):
                self.api_core.sub_task_name = self.args.sub_task_name
                if hasattr(self.api_core, "benchmark_ros_node") and self.api_core.benchmark_ros_node is not None:
                    if hasattr(self.api_core.benchmark_ros_node, "set_sub_task_name"):
                        self.api_core.benchmark_ros_node.set_sub_task_name(self.args.sub_task_name)
                        logger.info(f"Republished sub_task_name to ROS: {self.args.sub_task_name}")

            task_generator = TaskGenerator(self.task_config)
            if self.control_mode:
                control_task_root = os.environ.get("GENIESIM_CONTROL_TASK_DIR", "").strip()
                if not control_task_root:
                    raise RuntimeError("control_mode requires GENIESIM_CONTROL_TASK_DIR")
                task_folder = os.path.join(control_task_root, self.task_config["task"])
            else:
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

            instance_ids_arg = (getattr(self.args, "instance_ids", "") or "").strip()
            if instance_ids_arg:
                # Explicit override, in caller order; takes priority over
                # num_instances. Supervised control pins its scenario this way.
                scene_instance_ids = [int(x) for x in instance_ids_arg.split(",") if x.strip() != ""]
                logger.info(f"Explicit --benchmark.instance_ids override: {scene_instance_ids}")
            else:
                num_instances = getattr(self.args, "num_instances", 0) or 0
                if num_instances > 0 and num_instances < len(scene_instance_ids):
                    # Deterministic sub-sample based on the run's seed so the same
                    # config picks the same instances across machines / re-runs.
                    total = len(scene_instance_ids)
                    rng = np.random.RandomState(getattr(self.args, "seed", 0))
                    scene_instance_ids = sorted(
                        rng.choice(scene_instance_ids, size=num_instances, replace=False).tolist()
                    )
                    logger.info(
                        f"Sampled {num_instances} of {total} scene instances "
                        f"(seed={getattr(self.args, 'seed', 0)}): {scene_instance_ids}"
                    )

            if self.control_mode:
                self._evaluate_control_session(task_folder, scene_instance_ids[0])
                continue

            self.evaluate_summary = EvaluationSummary(
                os.path.join(system_utils.benchmark_output_path()),
                task,
                sub_task_name,
            )

            vec_batch_size = getattr(self.args, "enable_vec", 0) or 0
            if vec_batch_size > 1 and self.args.model_arc == "corobot":
                # Slice scene instances into batches of N cloned envs; clear the
                # vectorized stage between batches so VRAM is reclaimed.
                for batch_start in range(0, len(scene_instance_ids), vec_batch_size):
                    batch_ids = scene_instance_ids[batch_start : batch_start + vec_batch_size]
                    logger.info(
                        f"VecEnv batch [{batch_start}:{batch_start + len(batch_ids)}] / "
                        f"{len(scene_instance_ids)}: {batch_ids}"
                    )
                    self._evaluate_vectorized_sync(task, sub_task_name, robot_cfg, task_folder, batch_ids)
                    if batch_start + len(batch_ids) < len(scene_instance_ids):
                        logger.info("Clearing stage for next vectorized batch...")
                        self.api_core.clear_vectorized_stage()
                self.api_core.stop()
                try:
                    task_folder_abs = os.path.abspath(task_folder)
                    if os.path.isdir(task_folder_abs):
                        shutil.rmtree(task_folder_abs)
                        logger.info("Removed task folder: %s" % task_folder_abs)
                except Exception as e:
                    logger.warning("Failed to remove task folder %s: %s" % (task_folder, e))
                continue
            elif vec_batch_size > 1:
                logger.warning(
                    "enable_vec=%d requested but model_arc=%r is not 'corobot'; "
                    "running serially (vectorized eval supports corobot only).",
                    vec_batch_size,
                    self.args.model_arc,
                )

            for instance_id in scene_instance_ids:
                # A signal-driven request_shutdown stops the run here instead
                # of recording the remaining instances as 0-step failures.
                if self.data_courier.stop_requested():
                    break
                # one instance
                self.task_config["scene"]["scene_instance_id"] = instance_id
                specific_task_files = sorted(glob.glob(task_folder + "/*.json"))

                self.create_policy()
                self.create_env(specific_task_files[0], instance_id)
                if hasattr(self.policy, "_ikfk_solver") and hasattr(self.env, "ikfk_solver"):
                    self.policy._ikfk_solver = self.env.ikfk_solver
                time.sleep(0.5)
                self.api_core.collect_init_physics()

                seed = getattr(self.args, "seed", 0)
                for file_idx, episode_file in enumerate(specific_task_files):
                    if self.data_courier.stop_requested():
                        break
                    logger.info(f"EPISODE FILE: {episode_file}")
                    self.episode_content = system_utils.load_json(episode_file)
                    gen_seed = seed + instance_id * 1000 + file_idx
                    np.random.seed(gen_seed)
                    update_init_env(self.env, self.task_config, self.episode_content)
                    self.env.apply_generalization(self.api_core, self.task_config)

                    if self.args.record:
                        self.api_core.local_recorder.set_sub_task_name(self.args.sub_task_name or "")

                    episode_count = 1 if self.args.preview else len(self.env.task.instructions)
                    episode_seed = seed + instance_id
                    for episode_id in range(self.args.num_episode * episode_count):
                        if self.data_courier.stop_requested():
                            break
                        np.random.seed(episode_seed)
                        self.env.set_current_task(episode_id)
                        if self.instruction != "":
                            self.env.task.set_instruction(self.instruction)
                        current_instruction = self.env.task.get_instruction()
                        if self.args.preview:
                            self.policy.preview_instructions = [
                                it.get("instruction", "") if isinstance(it, dict) else str(it)
                                for it in self.env.task.instructions
                            ]
                        if self.args.record:
                            self.api_core.start_local_recording(
                                sub_task_name=self.args.sub_task_name or "",
                                episode_idx=episode_idx,
                                fps=self.args.fps,
                                output_root=os.path.dirname(self.evaluate_summary.out_dir),
                            )
                            if current_instruction:
                                self.api_core.local_recorder.update_instruction(current_instruction[0])
                        single_te = TaskEvaluation(
                            task_name=self.task_name,
                            sub_task_name=sub_task_name,
                            score_layout_name=getattr(self.env.task, "score_layout_name", None),
                        )
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
                        try:
                            self.evaluate_episode(robot_cfg, single_te)
                        except _StartupValidationComplete:
                            raise
                        except (InferenceUnavailableError, KeyboardInterrupt, SystemExit) as ep_err:
                            single_te.assemble_expect_result(ep_err)
                            # Server is down past the retry budget. Don't break
                            # to the next episode — propagate so the whole run
                            # exits and an external supervisor can react.
                            raise
                        except Exception as ep_err:
                            single_te.assemble_expect_result(ep_err)
                            # Physics is likely unrecoverable (e.g. after extreme joint
                            # positions corrupting PhysX transforms). Break the episode
                            # loop so GenieSim can shut down cleanly instead of hanging.
                            logger.error("Episode loop stopped due to fatal episode error: %s", ep_err)
                            break
                        finally:
                            if self.data_courier.stop_requested() and single_te.result.get("code", 0) == 0:
                                single_te.assemble_expect_result(RuntimeError("benchmark shutdown requested"))
                            if "scores" not in single_te.result:
                                single_te.update_progress(self.env.task.task_progress)
                            single_te.summarize_scores()
                            self.evaluate_summary.make_cache()
                        if self.data_courier.stop_requested():
                            # The interrupted attempt has already been saved above.
                            if self.args.record:
                                self.api_core.stop_local_recording(episode_idx=episode_idx)
                            logger.info("Shutdown requested; stopping before the next episode")
                            break
                        self.policy.set_episode_idx(episode_idx)
                        if self.args.record:
                            self.api_core.stop_local_recording(episode_idx=episode_idx)

                        episode_idx += 1

                self.env.stop()
            if self.args.record:
                self.api_core.concat_recordings(final_output_dir=getattr(self.args, "record_video_dir", None) or None)
            self.api_core.stop()
            try:
                task_folder_abs = os.path.abspath(task_folder)
                if os.path.isdir(task_folder_abs):
                    shutil.rmtree(task_folder_abs)
                    logger.info("Removed task folder: %s" % task_folder_abs)
            except Exception as e:
                logger.warning("Failed to remove task folder %s: %s" % (task_folder, e))

    def _evaluate_control_session(self, task_folder, instance_id):
        """Run one unscored CoRobot control session until its supervisor stops it.

        The selected task supplies the static scene and robot initialization. It
        does not supply an Agent instruction, a target, a score, or a terminal
        condition. The only text sent to shared control is the explicit
        operator-owned ``control_user_text`` value.
        """
        self.task_config["scene"]["scene_instance_id"] = instance_id
        specific_task_files = sorted(glob.glob(task_folder + "/*.json"))
        if not specific_task_files:
            raise RuntimeError(f"control_mode found no static task input in {task_folder}")

        self.create_policy()
        self.policy.control_runtime = True
        self.create_env(specific_task_files[0], instance_id)
        self.env.control_mode = True
        # Shared control publishes live camera and measured state throughout a
        # semantic trajectory. Keep rendering active without changing whether
        # the policy asks the server for a new command.
        self.env.force_fetch_images = True
        self.env.policy = self.policy
        if hasattr(self.policy, "_ikfk_solver") and hasattr(self.env, "ikfk_solver"):
            self.policy._ikfk_solver = self.env.ikfk_solver
        self.env.want_scene_geometry = False

        from geniesim_benchmark.control_evidence import write_control_scenario_identity

        write_control_scenario_identity(
            os.environ["GENIESIM_CONTROL_EVIDENCE_DIR"],
            source_root=Path(os.environ["GENIESIM_SOURCE_ROOT"]),
            task_name=self.task_name,
            sub_task_name=self.args.sub_task_name,
            instance_id=instance_id,
            task_config=Path(system_utils.benchmark_conf_path()) / "eval_tasks" / f"{self.task_name}.json",
            scene=Path(
                resolve_sub_usd_path(
                    scene_cfg=self.task_config.get("scene", {}),
                    benchmark_conf_path=system_utils.benchmark_conf_path(),
                    sub_task_name=self.args.sub_task_name,
                    instance_id=instance_id,
                )
            ),
            robot_config=Path(system_utils.app_root_path()) / "robot_cfg" / self.task_config["robot"]["robot_cfg"],
            background_scene=self.task_config["scene"]["scene_usd"],
            free_base=self.env._free_base,
            task_runtime=type(self.env.task).__name__,
            benchmark_seed=int(getattr(self.args, "seed", 0)),
        )

        time.sleep(0.5)
        self.api_core.collect_init_physics()
        self.episode_content = system_utils.load_json(specific_task_files[0])
        update_init_env(self.env, self.task_config, self.episode_content)
        self.env.apply_generalization(self.api_core, self.task_config)
        self.env.set_current_task(0)
        if hasattr(self.policy, "set_episode_idx"):
            self.policy.set_episode_idx(0)
        self.policy.reset()
        if hasattr(self.policy, "set_control_stop_callback") and hasattr(self.env, "request_control_hold"):
            self.policy.set_control_stop_callback(self.env.request_control_hold)
            self.env._direct_cutoff_handler = self.policy.cutoff_direct_segment
            self.env._direct_rejection_handler = self.policy.reject_direct_observation

        with self._shutdown_lock:
            self._control_shutdown_ready = True
            shutdown_pending = self._shutdown_reason is not None
        if shutdown_pending:
            self.request_shutdown(self._shutdown_reason)

        self.api_core.request_render()
        try:
            observation = self.env.reset()
        finally:
            self.api_core.release_render()

        domain_observer = None
        render_held = False

        def close_session():
            from contextlib import ExitStack

            nonlocal render_held
            with ExitStack() as cleanup:
                cleanup.callback(shutil.rmtree, task_folder, ignore_errors=True)
                if self.env is not None:
                    cleanup.callback(self.env.stop)
                if domain_observer is not None:
                    cleanup.callback(
                        close_domain_observer,
                        self.api_core,
                        domain_observer,
                        os.environ["GENIESIM_CONTROL_EVIDENCE_DIR"],
                    )
                if self.policy is not None:
                    cleanup.callback(self.policy.close)
                    evidence = getattr(self.policy, "control_evidence", None)
                    if evidence is not None:
                        cleanup.callback(evidence.close)
                if render_held:
                    render_held = False
                    cleanup.callback(self.api_core.release_render)

        # The cleanup owner exists before observer or evidence initialization.
        try:
            from geniesim_benchmark.domain.runtime import close_domain_observer, start_domain_observer

            if os.environ.get("GENIESIM_DOMAIN_OBSERVER_PORT"):
                domain_observer = start_domain_observer(
                    self.env,
                    source_root=Path(os.environ["GENIESIM_SOURCE_ROOT"]),
                    task_name=self.task_name,
                    sub_task_name=self.args.sub_task_name,
                    instance_id=instance_id,
                    evidence_dir=os.environ["GENIESIM_CONTROL_EVIDENCE_DIR"],
                    port=int(os.environ["GENIESIM_DOMAIN_OBSERVER_PORT"]),
                )

            def hold_render():
                nonlocal render_held
                if not render_held:
                    self.api_core.request_render()
                    render_held = True

            def drop_render():
                nonlocal render_held
                if render_held:
                    self.api_core.release_render()
                    render_held = False

            user_text = self.args.control_user_text
            from geniesim_benchmark.control_evidence import ControlEvidence
            from geniesim_benchmark.benchmark.policy.corobotpolicy import ControlShutdownRequested

            evidence_dir = os.environ.get("GENIESIM_CONTROL_EVIDENCE_DIR")
            self.policy.control_evidence = (
                ControlEvidence(evidence_dir, self.env.get_g2_90d_chassis_control_state) if evidence_dir else None
            )
            logger.info("Starting unscored G2_90d shared-control session")
            hold_render()
        except BaseException:
            close_session()
            raise
        try:
            while self.data_courier.loop_ok():
                should_pause = self.args.pause_physics_during_inference and self.policy.need_infer()
                if should_pause:
                    self.api_core.pause_physics()
                try:
                    try:
                        action = self.policy.act(
                            observation,
                            step_num=self.env.current_step,
                            task_instruction="",
                            gen_config=self.env.get_camera_gen_config(),
                            anno_bpt=False,
                            anno_goalimg=False,
                            send_scene_geometry=False,
                            user_text=user_text,
                        )
                    except Exception as error:
                        from geniesim_benchmark.benchmark.policy.corobotpolicy import (
                            ControlShutdownRequested,
                        )

                        if self.data_courier.stop_requested() and isinstance(
                            error,
                            ControlShutdownRequested,
                        ):
                            break
                        raise
                finally:
                    if should_pause:
                        self.api_core.resume_physics()
                if self.data_courier.stop_requested():
                    break
                if action is None:
                    raise RuntimeError("control_mode received no shared-control action")

                if hasattr(self.env, "set_policy_step_info"):
                    self.env.set_policy_step_info(
                        chunk_started=getattr(self.policy, "chunk_started_this_step", False),
                        chunk_len=getattr(self.policy, "chunk_len", None),
                        control_action_guard=getattr(self.policy, "control_action_guard", None),
                        control_write_token=getattr(self.policy, "returned_control_token", None),
                    )

                need_pixels = self.policy.need_infer()
                self.env.set_infer_status(need_pixels)
                set_depth_status = getattr(self.env, "set_depth_status", None)
                if callable(set_depth_status):
                    need_depth = getattr(self.policy, "need_depth", None)
                    set_depth_status(need_depth() if callable(need_depth) else False)
                settle_due = getattr(self.policy, "settle_due", None)
                self.env.settle_due = settle_due() if settle_due else need_pixels

                # PiEnv performs the guarded physical write and returns task
                # bookkeeping. Control mode intentionally ignores the latter:
                # no checker result can choose a prompt, end the session, or
                # advance a task stage.
                observation, _done, _need_update, _task_progress = self.env.step(action)
                finish_control_step = getattr(self.policy, "finish_control_step", None)
                if callable(finish_control_step):
                    observation = finish_control_step(observation, self.env)
                control_reset_requested = getattr(
                    self.policy,
                    "control_reset_requested",
                    None,
                )
                if (
                    callable(control_reset_requested)
                    and control_reset_requested()
                    and self.policy.acknowledge_control_reset(observation)
                ):
                    hold_render()
                    observation = self.env.reset()
                    self.policy.reset()
                self.data_courier.sleep()
        except ControlShutdownRequested:
            if not self.data_courier.stop_requested():
                raise
        finally:
            close_session()

    @staticmethod
    def _parse_infer_host(infer_host: str):
        """Parse infer_host in 'host:port' form to (host, port). Default port 8999 if omitted."""
        if ":" in infer_host:
            host, port_str = infer_host.rsplit(":", 1)
            return host.strip(), int(port_str.strip())
        return infer_host.strip(), None

    def _get_or_create_ws_client(self, host, port):
        # One WS client across all serial instances → one TCP/WS handshake total.
        if self._ws_client is None and not self.args.preview:
            from geniesim_benchmark.utils.comm.websocket_client import WebsocketClientPolicy

            self._ws_client = WebsocketClientPolicy(host=host, port=port)
        return self._ws_client

    def create_policy(self):
        # One policy (and one inference connection) is reused across all serial
        # instances. Rebuilding it per instance leaks the encode thread pool and
        # the dirt-mask cache; just clear its per-episode state.
        if self.policy is not None:
            self.policy.reset()
            return
        host, port = self._parse_infer_host(self.args.infer_host)
        logger.info(f"Infer service address: {host}" + (f":{port}" if port is not None else ""))
        if self.args.model_arc == "corobot":
            from geniesim_benchmark.benchmark.policy.corobotpolicy import CoRobotPolicy

            self.policy = CoRobotPolicy(
                task_name=self.args.task_name,
                host_ip=host,
                port=port,
                sub_task_name=self.args.sub_task_name,
                preview=self.args.preview,
                robot_cfg=self.task_config.get("robot_cfg", ""),
                realprobe=self.task_config.get("robot", {}).get("realprobe", False)
                or getattr(self.args, "realprobe", False),
                ik_backend=self.task_config.get("robot", {}).get("ik_backend", "ik_sdk"),
            )
        elif self.args.policy_class == "DemoPolicy":
            self.policy = DemoPolicy(
                task_name=self.args.task_name,
                sub_task_name=self.args.sub_task_name,
            )
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
            sub_usd_path = resolve_sub_usd_path(
                scene_cfg=self.task_config.get("scene", {}),
                benchmark_conf_path=system_utils.benchmark_conf_path(),
                sub_task_name=self.args.sub_task_name,
                instance_id=instance_id,
            )

        self.api_core.init_robot_cfg(
            robot_cfg,
            self.task_config["scene"]["scene_usd"],
            self.task_config["robot"]["robot_init_pose"]["position"],
            self.task_config["robot"]["robot_init_pose"]["quaternion"],
            sub_usd_path,
            free_base=bool(self.task_config.get("robot", {}).get("free_base", False)),
            collision_revision=(
                self.task_config["scene"].get("control_collision_revision") if self.control_mode else None
            ),
        )

        scene_info = None
        if self.args.sub_task_name == "":
            scene_info = self.gen_layouts(mode="instance")

        if self.args.model_arc == "corobot":
            from geniesim_benchmark.benchmark.envs.pi_env import PiEnv

            env_task_config = json.loads(json.dumps(self.task_config))
            env_task_config["control_mode"] = self.control_mode
            self.env = PiEnv(self.api_core, episode_file, env_task_config)
        elif self.args.model_arc == "" and self.args.env_class == "PiEnv":
            # Policy-free G2_90d profile diagnostics when explicitly requested.
            from geniesim_benchmark.benchmark.envs.pi_env import PiEnv

            self.env = PiEnv(self.api_core, episode_file, self.task_config)
        elif self.args.env_class == "DummyEnv":
            from geniesim_benchmark.benchmark.envs.dummy_env import DummyEnv

            self.env = DummyEnv(self.api_core, episode_file, self.task_config)
        else:
            raise ValueError("Invalid env_class {self.args.env_class}")

        self.env.set_data_courier(self.data_courier)
        self.env.set_scene_info(scene_info)

    def evaluate_episode(self, robot_cfg, single_te: TaskEvaluation):
        # Reset policy options before the first sensor capture of every episode.
        policy = getattr(self, "policy", None)
        if policy is not None:
            policy.reset()
        need_depth = getattr(policy, "need_depth", None)
        self.env.set_depth_status(need_depth() if callable(need_depth) else False)
        # Reset always needs a fresh observation, so hold a render request
        # across the first reset regardless of the on_demand_render flag.
        self.api_core.request_render()
        try:
            observaion = self.env.reset()  # 1st frame
        finally:
            self.api_core.release_render()
        startup_validation_seconds = float(getattr(self.args, "startup_validation_seconds", 0.0) or 0.0)
        if startup_validation_seconds > 0.0:
            try:
                evidence = self._run_profile_startup_validation(
                    observaion,
                    startup_validation_seconds,
                )
            except BaseException as error:
                self._write_startup_validation_result(
                    status="failed",
                    evidence=None,
                    error=error,
                )
                raise
            if not evidence["passed"]:
                error = RuntimeError(f"G2_90d startup validation failed: {evidence}")
                self._write_startup_validation_result(
                    status="failed",
                    evidence=evidence,
                    error=error,
                )
                raise error
            result = self._write_startup_validation_result(
                status="passed",
                evidence=evidence,
            )
            raise _StartupValidationComplete(result)
        start_time = time.time()
        (
            start_callbacks,
            step_callbacks,
            end_callbacks,
            data_callbacks,
        ) = get_hook_callbacks(self.policy)

        if (
            self.control_mode
            and hasattr(self.policy, "set_control_stop_callback")
            and hasattr(self.env, "request_control_hold")
        ):
            self.policy.set_control_stop_callback(self.env.request_control_hold)
            self.env._direct_cutoff_handler = self.policy.cutoff_direct_segment
            self.env._direct_rejection_handler = self.policy.reject_direct_observation

        for callback in start_callbacks:  # before task
            callback(self.env, None)
        single_te.start_policy_statistics(self.policy)
        self.env.policy_dispatch_count = getattr(self.policy, "policy_dispatch_count", None)

        # On-demand render bookkeeping: hold a single render-refcount slot
        # while the policy is in inference mode (need_infer == True). When
        # the policy is replaying an action chunk (need_infer == False) the
        # observation isn't consumed, so rendering can be skipped entirely.
        render_held = False

        def _hold_render():
            nonlocal render_held
            if not render_held:
                self.api_core.request_render()
                render_held = True

        def _drop_render():
            nonlocal render_held
            if render_held:
                self.api_core.release_render()
                render_held = False

        try:
            if not self.args.interactive:
                _hold_render()
                self.env.do_eval_action()
            while self.data_courier.loop_ok():
                should_pause = self.args.pause_physics_during_inference and self.policy.need_infer()
                if should_pause:
                    self.api_core.pause_physics()
                if self.task_mode == "infer":
                    if self.args.interactive:
                        while not self.data_courier.get_infer_start():
                            logger.info("waiting for infer start")
                            time.sleep(1)
                            # if self.env.current_step != 0:
                            self.env.current_step = 0
                            self.policy.reset()
                            need_depth = getattr(self.policy, "need_depth", None)
                            self.env.set_depth_status(need_depth() if callable(need_depth) else False)
                            _hold_render()
                            try:
                                observaion = self.env.reset()
                            finally:
                                _drop_render()

                        if self.data_courier.get_shuffle():
                            self.api_core.shuffle_scene()
                        single_instruction = [self.data_courier.get_instruction()]
                        instruction_segments = []
                    else:
                        single_instruction = self.env.task.get_instruction()
                        instruction_segments = self.env.task.get_instruction_segments()
                    # Record the instruction actually sent in the policy prompt.
                    # update_instruction deduplicates identical text.
                    if self.args.record and single_instruction and single_instruction[0]:
                        self.api_core.local_recorder.update_instruction(single_instruction[0])
                    action = self.policy.act(
                        observaion,
                        step_num=self.env.current_step,
                        task_instruction=single_instruction[0],
                        task_segments=instruction_segments,
                        gen_config=self.env.get_camera_gen_config(),
                    )
                else:
                    action = self.policy.act(observaion, step_num=self.env.current_step)
                single_te.update_policy_statistics(self.policy)
                self.env.policy_dispatch_count = getattr(self.policy, "policy_dispatch_count", None)
                if should_pause:
                    self.api_core.resume_physics()
                for callback in step_callbacks:  # during task
                    callback(self.env, action)

                if hasattr(self.env, "set_policy_step_info"):
                    self.env.set_policy_step_info(
                        chunk_started=getattr(self.policy, "chunk_started_this_step", False),
                        chunk_len=getattr(self.policy, "chunk_len", None),
                        control_action_guard=getattr(self.policy, "control_action_guard", None),
                        control_write_token=getattr(self.policy, "returned_control_token", None),
                    )

                need_pixels = self.policy.need_infer()
                self.env.set_infer_status(need_pixels)
                set_depth_status = getattr(self.env, "set_depth_status", None)
                if callable(set_depth_status):
                    need_depth = getattr(self.policy, "need_depth", None)
                    set_depth_status(need_depth() if callable(need_depth) else False)
                # settle_due marks the chunk's last step (the obs feeding the
                # next inference); history-capture render steps set need_pixels
                # without being settle_due, so they skip arm-settling.
                self.env.settle_due = (
                    self.policy.inference_due() if hasattr(self.policy, "inference_due") else need_pixels
                )
                if need_pixels:
                    _hold_render()
                else:
                    _drop_render()

                observaion, done, need_update, task_progress = self.env.step(action)
                # A policy backed by a replay/mock server can signal that it
                # has run out of recorded actions to give (e.g. corobot's
                # server_episode_done, set from the inference reply's
                # "episode_done" field) -- that ends the episode, regardless
                # of what the checker currently thinks, since there is
                # nothing further to feed it either way.
                #
                # server_episode_done latches True the INSTANT that reply
                # arrives, well before the sim has driven through the ~30
                # ticks of raw waypoints the chunk still queued -- acting on
                # it immediately cuts that last chunk short and resets one
                # chunk early, before the recording's true final pose is
                # ever reached. Gate on inference_due() (raw_action_buffer
                # and action_buffer both drained) so the episode only ends
                # once every queued waypoint from that final chunk has
                # actually been executed.
                if getattr(self.policy, "server_episode_done", False) and not done:
                    inference_due = getattr(self.policy, "inference_due", None)
                    if inference_due is None or inference_due():
                        logger.info(
                            "Policy server signaled episode_done and the last queued "
                            "chunk has fully executed; overriding checker and ending "
                            "episode now."
                        )
                        done = True
                if hasattr(self.policy, "update_task_status"):
                    self.policy.update_task_status(done, task_progress)
                if self.env.current_step % 30 == 0:
                    logger.info(f"STEP {self.env.current_step}")
                if not self.args.interactive:
                    if need_update:
                        single_te.update_progress(task_progress)
                        self.data_courier.pub_dynamic_info_msg(self.evaluate_summary.make_temp_statistic())
                    if done:
                        single_te.update_progress(task_progress)
                        self.env.current_step = 0
                        single_te.summarize_scores()
                        _hold_render()
                        try:
                            # In preview the main-loop act() already saved the
                            # frame; skip the extra act() so each layout = 1 image.
                            if not self.args.preview:
                                self.policy.act(
                                    observaion,
                                    task_instruction=single_instruction[0],
                                    task_segments=instruction_segments,
                                    gen_config=self.env.get_camera_gen_config(),
                                )
                            observaion = self.env.reset()
                        finally:
                            _drop_render()
                        self.policy.reset()
                        break
                self.data_courier.sleep()

        except KeyboardInterrupt as e:
            logger.info("Received KeyboardInterrupt during episode")
            single_te.assemble_expect_result(e)
            # Ensure recording is stopped on interrupt
            if self.args.record and hasattr(self, "env") and self.env is not None:
                try:
                    if hasattr(self.env, "api_core") and self.env.api_core is not None:
                        self.env.api_core.stop_all_recording()
                        logger.info("Recording stopped due to KeyboardInterrupt")
                except Exception as e:
                    logger.warning(f"Failed to stop recording on interrupt: {e}")
            raise
        except InferenceUnavailableError as e:
            # Inference server is unreachable after the configured retry budget.
            # Stop recording cleanly, then re-raise so the whole benchmark run
            # aborts — an external supervisor (shell / k8s / scheduler) decides
            # whether to restart. We do not skip the episode: a dead server
            # would otherwise turn every remaining episode into a silent
            # failure.
            logger.error("Inference unavailable, aborting benchmark: %s", e)
            single_te.assemble_expect_result(e)
            if self.args.record and hasattr(self, "env") and self.env is not None:
                try:
                    if hasattr(self.env, "api_core") and self.env.api_core is not None:
                        self.env.api_core.stop_all_recording()
                except Exception as stop_err:
                    logger.warning(f"Failed to stop recording after inference failure: {stop_err}")
            raise
        except Exception as e:
            # Unhandled exception (e.g. LinAlgError from SVD on NaN PhysX transforms).
            # Mark episode as failed and re-raise so evaluate_policy() can break the
            # episode loop — prevents the worker thread from crashing silently while
            # Isaac Sim physics keeps spinning at 99% CPU indefinitely.
            logger.error("Unhandled exception in episode: %s — marking failed and stopping loop", e)
            single_te.assemble_expect_result(e)
            raise
        finally:
            single_te.update_policy_statistics(self.policy)
            # Never leak a render refcount across episodes — otherwise the
            # main loop keeps rendering forever once on_demand_render is on.
            _drop_render()
            end_time = time.time()
            single_te.update_from_dict(
                {
                    "end_time": system_utils.get_local_time(),
                    "duration": end_time - start_time,
                }
            )
            if self.api_core.physics_paused:
                self.api_core.resume_physics()

        for callback in end_callbacks:  # during task
            callback(self.env, action)

    def _create_vec_policy(self, sub_task_name):
        """One CoRobotPolicy per env; each owns its own inference connection."""
        host, port = self._parse_infer_host(self.args.infer_host)
        from geniesim_benchmark.benchmark.policy.corobotpolicy import CoRobotPolicy

        policy = CoRobotPolicy(
            task_name=self.args.task_name,
            host_ip=host,
            port=port,
            sub_task_name=sub_task_name,
            preview=self.args.preview,
            robot_cfg=self.task_config.get("robot_cfg", ""),
            ik_backend=self.task_config.get("robot", {}).get("ik_backend", "ik_sdk"),
        )
        return policy

    def _write_startup_validation_result(self, *, status, evidence, error=None):
        configured = str(getattr(self.args, "startup_validation_result", "") or "")
        if configured:
            result_path = Path(configured).expanduser()
        else:
            output_dir = Path(str(getattr(self.args, "output_dir", "output") or "output"))
            result_path = output_dir / f"{self.task_name}-startup-validation.json"
        result_path = result_path.resolve()
        result = {
            "kind": "g2_90d_startup_validation",
            "status": status,
            "task": self.task_name,
            "result_path": str(result_path),
            "completed_at_unix": time.time(),
            "evidence": evidence,
        }
        if error is not None:
            result["error"] = f"{type(error).__name__}: {error}"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, result_path)
        self.api_core.startup_validation_status = status
        self.api_core.startup_validation_result = result
        return result

    def _run_profile_startup_validation(self, first_observation, duration_seconds):
        profile = getattr(self.api_core, "robot_profile", None) or {}
        if profile.get("name") != "G2_90d":
            raise RuntimeError("startup validation requires the G2_90d robot profile")
        if not self.task_config.get("robot", {}).get("free_base", False):
            raise RuntimeError("startup validation requires robot.free_base=true")
        validation = profile["runtime_validation"]
        expected_height = (
            float(validation["ground_plane_z_m"])
            + float(validation["wheel_radius_m"])
            - float(validation["wheel_center_from_base_z_m"])
        )
        position_tolerance = float(validation["wheel_radius_m"]) * 0.1
        tilt_tolerance = math.atan2(
            position_tolerance,
            float(profile["chassis"]["minimum_support_radius_m"]),
        )
        linear_speed_tolerance = position_tolerance / duration_seconds
        angular_speed_tolerance = tilt_tolerance / duration_seconds

        def camera_reference_pairs(observation):
            samples = (observation or {}).get("camera_samples") or {}
            pairs = {}
            for camera in ("head", "left_hand", "right_hand"):
                sample = samples.get(camera)
                if not isinstance(sample, dict):
                    continue
                numerator = sample.get("reference_time_numerator")
                denominator = sample.get("reference_time_denominator")
                if (
                    isinstance(numerator, int)
                    and not isinstance(numerator, bool)
                    and isinstance(denominator, int)
                    and not isinstance(denominator, bool)
                    and denominator > 0
                ):
                    pairs[camera] = (numerator, denominator)
            return pairs

        def pose_sample(
            label,
            position,
            quaternion,
            linear_velocity,
            angular_velocity,
        ):
            position = np.asarray(position, dtype=np.float64)
            quaternion = np.asarray(quaternion, dtype=np.float64)
            if position.shape != (3,) or quaternion.shape != (4,):
                raise RuntimeError(f"startup validation has malformed {label} base_link pose")
            w, x, y, z = quaternion
            roll = math.atan2(
                2.0 * (w * x + y * z),
                1.0 - 2.0 * (x * x + y * y),
            )
            pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
            return {
                "position_m": position.tolist(),
                "quaternion_wxyz": quaternion.tolist(),
                "height_error_m": abs(float(position[2]) - expected_height),
                "tilt_rad": max(abs(roll), abs(pitch)),
                "linear_speed_m_s": float(np.linalg.norm(linear_velocity)),
                "angular_speed_rad_s": float(np.linalg.norm(angular_velocity)),
            }

        def observation_sample(label, observation):
            pose = (observation or {}).get("base_pose")
            if pose is None:
                raise RuntimeError(f"startup validation has no {label} base_link pose")
            motion = self.api_core.get_robot_base_motion()
            return pose_sample(
                label,
                pose[0],
                pose[1],
                motion["linear_velocity_m_s"],
                motion["angular_velocity_rad_s"],
            )

        reset_samples = []
        immediate_observation = first_observation
        for reset_index in range(2):
            immediate = observation_sample("immediate", immediate_observation)
            time.sleep(duration_seconds)
            settled_observation = self.env.get_observation(fetch_images=False)
            settled = observation_sample("settled", settled_observation)
            reset_samples.append(
                {
                    "reset_index": reset_index,
                    "immediate": immediate,
                    "settled": settled,
                    "vertical_drift_m": abs(settled["position_m"][2] - immediate["position_m"][2]),
                    "xy_drift_m": float(
                        np.linalg.norm(np.asarray(settled["position_m"][:2]) - np.asarray(immediate["position_m"][:2]))
                    ),
                }
            )
            if reset_index == 0:
                immediate_observation = self.env.reset()

        chassis_before = self.api_core.get_g2_90d_chassis_state()
        probe = self.env.run_g2_90d_startup_chassis_probe(
            duration_seconds=duration_seconds,
            linear_speed_tolerance_m_s=linear_speed_tolerance,
            angular_speed_tolerance_rad_s=angular_speed_tolerance,
        )
        chassis_after = self.api_core.get_g2_90d_chassis_state()
        first_camera_pairs = camera_reference_pairs(first_observation)
        required_cameras = {"head", "left_hand", "right_hand"}
        if set(first_camera_pairs) != required_cameras:
            raise RuntimeError("startup validation first observation lacks three camera reference times")
        self.api_core.request_render()
        try:
            camera_deadline = time.time() + max(5.0, float(duration_seconds))
            while True:
                final_observation = self.env.get_observation(fetch_images=True)
                final_camera_pairs = camera_reference_pairs(final_observation)
                if set(final_camera_pairs) == required_cameras and all(
                    final_camera_pairs[camera] != first_camera_pairs[camera] for camera in required_cameras
                ):
                    break
                if time.time() >= camera_deadline:
                    raise RuntimeError("startup validation final camera reference times did not advance")
                time.sleep(0.02)
        finally:
            self.api_core.release_render()

        probe_settled = probe["settled"]["base"]
        probe_settled_pose = pose_sample(
            "probe settled",
            probe_settled["position_m"],
            probe_settled["quaternion_wxyz"],
            probe_settled["linear_velocity_m_s"],
            probe_settled["angular_velocity_rad_s"],
        )
        before_position = np.asarray(
            probe["before"]["base"]["position_m"],
            dtype=np.float64,
        )
        driven_position = np.asarray(
            probe["driven"]["base"]["position_m"],
            dtype=np.float64,
        )
        base_xy_motion = float(np.linalg.norm(driven_position[:2] - before_position[:2]))
        yaw_delta = probe["driven"]["base"]["yaw_rad"] - probe["before"]["base"]["yaw_rad"]
        base_yaw_motion = abs(math.atan2(math.sin(yaw_delta), math.cos(yaw_delta)))
        roll_motions = [abs(value) for value in probe["roll_delta_rad"].values()]
        probe_metrics = {
            "base_xy_motion_m": base_xy_motion,
            "base_yaw_motion_rad": base_yaw_motion,
            "maximum_steer_motion_rad": probe["maximum_steer_motion_rad"],
            "minimum_roll_motion_rad": min(roll_motions),
            "maximum_roll_motion_rad": max(roll_motions),
            "peak_path_deviation_m": probe["peak_path_deviation_m"],
            "peak_yaw_error_rad": probe["peak_yaw_error_rad"],
            "settled_pose": probe_settled_pose,
        }

        reset_passed = all(
            reset["vertical_drift_m"] <= position_tolerance
            and reset["xy_drift_m"] <= position_tolerance
            and reset["immediate"]["height_error_m"] <= position_tolerance
            and reset["immediate"]["tilt_rad"] <= tilt_tolerance
            and reset["settled"]["height_error_m"] <= position_tolerance
            and reset["settled"]["tilt_rad"] <= tilt_tolerance
            and reset["settled"]["linear_speed_m_s"] <= linear_speed_tolerance
            and reset["settled"]["angular_speed_rad_s"] <= angular_speed_tolerance
            for reset in reset_samples
        )
        chassis_motion_passed = (
            probe["production_helper"] == "PiEnv.apply_chassis_target"
            and probe["goal_completed"]
            and probe["stop_completed"]
            and probe_metrics["maximum_steer_motion_rad"] >= 0.01
            and probe_metrics["minimum_roll_motion_rad"] >= 0.02
            and (base_xy_motion >= 0.002 or base_yaw_motion >= 0.002)
            and probe["stop_stable_samples"] >= int(profile["completion"]["stable_samples"])
            and probe["stop_stable_duration_s"] + 1e-12 >= float(profile["completion"]["stable_duration_s"])
            and probe_metrics["peak_path_deviation_m"]
            <= float(profile["chassis"]["controller"]["maximum_path_deviation_m"])
            and probe_metrics["peak_yaw_error_rad"]
            <= float(profile["chassis"]["controller"]["maximum_transient_yaw_error_rad"])
            and probe_settled_pose["height_error_m"] <= position_tolerance
            and probe_settled_pose["tilt_rad"] <= tilt_tolerance
            and probe_settled_pose["linear_speed_m_s"] <= linear_speed_tolerance
            and probe_settled_pose["angular_speed_rad_s"] <= angular_speed_tolerance
        )
        evidence = {
            "profile": profile["name"],
            "task": self.task_name,
            "mode": "single_world",
            "free_base": True,
            "duration_seconds": duration_seconds,
            "expected_base_height_m": expected_height,
            "position_tolerance_m": position_tolerance,
            "tilt_tolerance_rad": tilt_tolerance,
            "linear_speed_tolerance_m_s": linear_speed_tolerance,
            "angular_speed_tolerance_rad_s": angular_speed_tolerance,
            "first_observation": {
                "measured": first_observation.get("measured"),
                "camera_samples": first_observation.get("camera_samples"),
            },
            "final_observation": {
                "measured": final_observation.get("measured"),
                "camera_samples": final_observation.get("camera_samples"),
            },
            "resets": reset_samples,
            "chassis_state_before_probe": chassis_before,
            "chassis_probe": probe,
            "chassis_probe_metrics": probe_metrics,
            "chassis_state_after_probe": chassis_after,
            "checks": {
                "reset_settle": reset_passed,
                "released_chassis_before_probe": (g2_90d_mobile_state_passed(profile, chassis_before)),
                "production_chassis_motion": chassis_motion_passed,
                "released_chassis_after_probe": (g2_90d_mobile_state_passed(profile, chassis_after)),
            },
        }
        evidence["passed"] = all(evidence["checks"].values())
        logger.info("[g2-90d-profile] startup validation " + json.dumps(evidence, sort_keys=True))
        return evidence

    def _evaluate_vectorized_sync(self, task, sub_task_name, robot_cfg, task_folder, scene_instance_ids):
        """Lock-step vectorized eval: N envs cloned on one stage, parallel inference."""
        from geniesim_benchmark.benchmark.envs.vec_env import BenchmarkVecEnv, VecPolicyWrapper
        from geniesim_benchmark.benchmark.envs.pi_env import PiEnv

        logger.info(f"VecEnv sync eval: {len(scene_instance_ids)} envs: {scene_instance_ids}")

        n_envs = len(scene_instance_ids)
        specific_task_files = sorted(glob.glob(task_folder + "/*.json"))
        scene_usd = self.task_config["scene"]["scene_usd"]
        init_position = self.task_config["robot"]["robot_init_pose"]["position"]
        init_rotation = self.task_config["robot"]["robot_init_pose"]["quaternion"]

        sub_usd_paths = []
        for iid in scene_instance_ids:
            if sub_task_name != "":
                p = resolve_sub_usd_path(
                    scene_cfg=self.task_config.get("scene", {}),
                    benchmark_conf_path=system_utils.benchmark_conf_path(),
                    sub_task_name=sub_task_name,
                    instance_id=iid,
                )
                sub_usd_paths.append(p if p and os.path.exists(p) else "")
            else:
                sub_usd_paths.append("")

        self.api_core.init_robot_cfg_multi(
            robot_cfg,
            scene_usd,
            init_position,
            init_rotation,
            sub_usd_paths,
            n_envs=n_envs,
        )
        time.sleep(0.5)
        self.api_core.collect_init_physics()

        seed = getattr(self.args, "seed", 0)

        envs = []
        policies = []
        data_couriers = []
        task_configs = []
        for slot, instance_id in enumerate(scene_instance_ids):
            api_view = self.api_core.fork_for_env(slot)
            task_config_copy = json.loads(json.dumps(self.task_config))
            task_config_copy["scene"]["scene_instance_id"] = instance_id
            task_configs.append(task_config_copy)

            # Shallow-copy the root data_courier so ROS/robot_cfg state is shared
            # but api_core points at the per-env view.
            env_data_courier = copy.copy(self.data_courier)
            env_data_courier.api_core = api_view
            data_couriers.append(env_data_courier)

            policy = self._create_vec_policy(sub_task_name)
            policy.set_data_courier(env_data_courier)
            policies.append(policy)

            env = PiEnv(api_view, specific_task_files[0], task_config_copy)
            env.set_data_courier(env_data_courier)
            # Keep parity with the serial path which seeds policy._ikfk_solver
            # from the env's solver (used for EEF_ABS post-processing).
            if hasattr(policy, "_ikfk_solver") and hasattr(env, "ikfk_solver"):
                policy._ikfk_solver = env.ikfk_solver
            envs.append(env)

        vec_env = BenchmarkVecEnv(self.api_core, envs, data_couriers)
        vec_policy = VecPolicyWrapper(policies)

        # Per-env episode counters spaced apart for unambiguous logs.
        episode_idxs = [slot * 10000 for slot in range(n_envs)]

        try:
            for file_idx, episode_file in enumerate(specific_task_files):
                logger.info(f"EPISODE FILE: {episode_file}")
                episode_content = system_utils.load_json(episode_file)

                for slot, instance_id in enumerate(scene_instance_ids):
                    gen_seed = seed + instance_id * 1000 + file_idx
                    np.random.seed(gen_seed)
                    update_init_env(envs[slot], task_configs[slot], episode_content)
                    envs[slot].apply_generalization(envs[slot].api_core, task_configs[slot])

                episode_count = 1 if self.args.preview else len(envs[0].task.instructions)
                for episode_id in range(self.args.num_episode * episode_count):
                    te_list = []
                    for slot, instance_id in enumerate(scene_instance_ids):
                        episode_seed = seed + instance_id
                        np.random.seed(episode_seed)
                        envs[slot].set_current_task(episode_id)
                        if self.instruction != "":
                            envs[slot].task.set_instruction(self.instruction)
                        current_instruction = envs[slot].task.get_instruction()
                        single_te = TaskEvaluation(
                            task_name=self.task_name,
                            sub_task_name=sub_task_name,
                            score_layout_name=getattr(envs[slot].task, "score_layout_name", None),
                        )
                        single_te.update_from_dict(
                            {
                                "task_name": task_configs[slot]["task"],
                                "task_type": "benchmark",
                                "robot_type": robot_cfg.split(".")[0],
                                "start_time": system_utils.get_local_time(),
                                "model_type": self.args.model_arc,
                                "task_instruction": current_instruction[0],
                            }
                        )
                        te_list.append(single_te)

                    try:
                        self._run_sync_episode(vec_env, vec_policy, te_list)
                    except BaseException as error:
                        for slot, evaluation in enumerate(te_list):
                            if "scores" not in evaluation.result:
                                evaluation.update_progress(envs[slot].task.task_progress)
                                evaluation.assemble_expect_result(error)
                                evaluation.summarize_scores()
                        raise
                    finally:
                        for evaluation in te_list:
                            self.evaluate_summary.update_current(evaluation)
                            self.evaluate_summary.make_cache()

                    for slot in range(n_envs):
                        policies[slot].set_episode_idx(episode_idxs[slot])
                        episode_idxs[slot] += 1
        finally:
            for env in envs:
                try:
                    env.stop()
                except Exception:
                    pass
            vec_policy.shutdown()

    def _run_sync_episode(self, vec_env, vec_policy, te_list):
        """Run one episode across all envs in lock-step."""
        n_envs = vec_env.n_envs
        start_time = time.time()

        vec_policy.reset_all()
        vec_env.set_depth_status(vec_policy.need_depth_list())
        obs_list = vec_env.reset_all()
        for i, evaluation in enumerate(te_list):
            evaluation.start_policy_statistics(vec_policy.get_policy(i))
            vec_env.get_env(i).policy_dispatch_count = getattr(vec_policy.get_policy(i), "policy_dispatch_count", None)
        vec_env.do_eval_actions()

        active = [True] * n_envs
        finish_times = [None] * n_envs

        while not vec_env.all_done():
            instructions = []
            instruction_segments = []
            gen_configs = []
            step_nums = []
            for i in range(n_envs):
                env = vec_env.get_env(i)
                if active[i]:
                    inst = env.task.get_instruction()
                    instructions.append(inst[0] if inst else "")
                    instruction_segments.append(env.task.get_instruction_segments())
                    gen_configs.append(env.get_camera_gen_config())
                    step_nums.append(env.current_step)
                else:
                    instructions.append("")
                    instruction_segments.append([])
                    gen_configs.append(None)
                    step_nums.append(0)

            actions = vec_policy.act_batch(
                obs_list,
                instructions,
                gen_configs,
                step_nums,
                active_mask=active,
                task_segments=instruction_segments,
            )
            for i, evaluation in enumerate(te_list):
                if active[i]:
                    evaluation.update_policy_statistics(vec_policy.get_policy(i))
                    vec_env.get_env(i).policy_dispatch_count = getattr(
                        vec_policy.get_policy(i),
                        "policy_dispatch_count",
                        None,
                    )

            need_infer = vec_policy.need_infer_list()
            vec_env.set_infer_status(need_infer)
            vec_env.set_depth_status(vec_policy.need_depth_list())

            obs_list, dones, need_updates, task_progresses = vec_env.step(actions)
            vec_policy.update_task_status_batch(dones, task_progresses)

            for i in range(n_envs):
                if not active[i]:
                    continue
                env = vec_env.get_env(i)
                if env.current_step % 30 == 0:
                    logger.info(f"[env{i}] STEP {env.current_step}")
                if need_updates[i]:
                    te_list[i].update_progress(task_progresses[i])
                if dones[i]:
                    te_list[i].update_progress(task_progresses[i])
                    active[i] = False
                    finish_times[i] = time.time()
                    env.current_step = 0
                    te_list[i].summarize_scores()
                    vec_policy.reset_policy(i)

        end_time = time.time()
        # Distribute batch wall-clock so Σ(durations) == parallel wall-clock,
        # weighted by per-env activity (the aggregator sums durations as timecost).
        batch_wall_clock = end_time - start_time
        per_env_active_secs = [
            ((finish_times[i] if finish_times[i] is not None else end_time) - start_time) for i in range(n_envs)
        ]
        total_active_secs = sum(per_env_active_secs)
        if total_active_secs > 0 and batch_wall_clock > 0:
            scale = batch_wall_clock / total_active_secs
            durations = [t * scale for t in per_env_active_secs]
        else:
            durations = [batch_wall_clock / max(n_envs, 1)] * n_envs

        for i in range(n_envs):
            te_list[i].update_from_dict(
                {
                    "end_time": system_utils.get_local_time(),
                    "duration": durations[i],
                }
            )


def main(args=None, api_core=None):
    benchmark = TaskBenchmark(args, api_core)
    benchmark.evaluate_policy()  # Evaluate agent on the benchmark


if __name__ == "__main__":
    main()
