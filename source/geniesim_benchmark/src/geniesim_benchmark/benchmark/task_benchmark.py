# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import os
import sys
import glob
import shutil
import json
import copy
from contextlib import contextmanager
import threading
from pathlib import Path

import numpy as np
import time

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()  # Create singleton instance

from geniesim_benchmark.plugins.output_system import TaskEvaluation, EvaluationSummary
from geniesim_benchmark.plugins.tgs import ObjectSampler, TaskGenerator
from geniesim_benchmark.utils.comm.retry import InferenceUnavailableError, InferenceServiceError
from geniesim_benchmark.plugins.output_system.checkpoint import (
    EvaluationCheckpoint,
    build_episode_plan,
    file_digest,
)
from geniesim_benchmark.utils.data_courier import DataCourier
import geniesim_benchmark.utils.system_utils as system_utils
from geniesim_benchmark.utils.name_utils import robot_type_mapping
from geniesim_benchmark.benchmark.policy.demopolicy import DemoPolicy
from geniesim_benchmark.benchmark.hooks.task import TaskHook
from geniesim_benchmark.benchmark.hooks.pregrasp import PreGraspHook
from geniesim_benchmark.benchmark.subscene_override import resolve_sub_usd_path
from geniesim_benchmark.app.controllers.api_core import APICore
from geniesim_benchmark.utils.generalization_utils import apply_hdr_texture_generalization, update_init_env

system_utils.check_and_fix_env()


def reject_free_base_vec(task_config, enable_vec):
    if int(enable_vec or 0) > 0 and bool(task_config.get("robot", {}).get("free_base", False)):
        raise ValueError("free_base is supported only in single-world mode; " "set benchmark.enable_vec=0")


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
        self.policy = None
        self.env = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_reason = None
        self.tasks = self.check_task()
        self.task_config = None
        # Set per task in _evaluate_policy_impl(); retains every started attempt,
        # including interrupted attempts, when the worker aborts.
        self.evaluate_summary = None
        self._checkpoint = None

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
        """Stop the benchmark worker and its transport."""
        with self._shutdown_lock:
            if self._shutdown_reason is None:
                self._shutdown_reason = str(reason)
            reason = self._shutdown_reason
        self.data_courier.request_stop()
        return {"reason": reason, "courier_stop": "requested"}

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
        logger.warning("Use self-defined task_name: {}".format(self.task_name))
        return tasks

    def config_task(self):
        if self.args.model_arc == "corobot":
            self.task_mode = "infer"
        else:
            self.task_mode = "empty"

    def evaluate_policy(self):
        self.config_task()
        try:
            self._evaluate_policy_impl()
        except BaseException:
            # Keep completed and explicitly invalid attempts. Remove only an
            # empty output from a run that aborted before recording an attempt.
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
        """Remove an empty task output without deleting recorded attempts."""
        summary = self.evaluate_summary
        if summary is not None and (summary.results or getattr(summary, "preserve_result", False)):
            return
        result_file = getattr(summary, "file_path", None) if summary is not None else None
        if not result_file:
            return
        try:
            if os.path.isfile(result_file):
                os.remove(result_file)
                logger.info("Cleaned up task result file after abort: %s", result_file)
        except Exception as e:
            logger.warning("Failed to clean up task result file %s: %s", result_file, e)

    def _cache_episode_result(self, evaluation, env, error=None):
        """Finalize one started attempt before control leaves its episode loop."""
        if error is None and self.data_courier.stop_requested() and "scores" not in evaluation.result:
            error = RuntimeError("benchmark shutdown requested")
        if error is not None:
            evaluation.assemble_expect_result(error)
        evaluation.update_progress(env.task.task_progress)
        evaluation.summarize_scores()
        self.evaluate_summary.update_current(evaluation)
        self.evaluate_summary.make_cache()

    def _evaluate_policy_impl(self):
        for task in self.tasks:
            if self.data_courier.stop_requested():
                break
            with self._checkpoint_session(task):
                self._evaluate_task(task)

    @contextmanager
    def _checkpoint_session(self, task):
        self.evaluate_summary = EvaluationSummary(system_utils.benchmark_output_path(), task, self.args.sub_task_name)
        supported = (
            self.args.model_arc == "corobot"
            and bool(self.args.sub_task_name)
            and not self.args.preview
            and not self.args.interactive
            and (getattr(self.args, "enable_vec", 0) or 0) <= 1
        )
        if not supported and getattr(self.args, "resume", False):
            logger.warning(
                "benchmark.resume requires a serial, non-interactive Corobot evaluation with a named subtask; ignoring resume"
            )
        # Before acquiring the lock, the generated result name may belong to
        # another worker that has not published its first result yet.
        self.evaluate_summary.preserve_result = True
        with EvaluationCheckpoint(
            self.evaluate_summary.out_dir, resume=supported and getattr(self.args, "resume", False)
        ) as checkpoint:
            self._checkpoint = checkpoint if supported else None
            self.evaluate_summary.preserve_result = checkpoint.state is not None
            self.evaluate_summary.file_path = str(checkpoint.result_path)
            try:
                yield
            finally:
                self._checkpoint = None

    def _checkpoint_identity(self, instance_ids):
        excluded = {"resume", "infer_host", "record_video_dir", "output_dir"}
        parameters = {k: v for k, v in vars(self.args).items() if k not in excluded}
        root = Path(system_utils.benchmark_conf_path())
        files = {}
        for iid in instance_ids:
            directory = root / "llm_task" / self.args.sub_task_name / str(iid)
            if self.args.sub_task_name and directory.is_dir():
                files.update(
                    {str(p.relative_to(root)): file_digest(p) for p in sorted(directory.rglob("*")) if p.is_file()}
                )
            override = self.task_config.get("scene", {}).get("sub_usd_override_root")
            if override:
                directory = Path(override) / str(iid)
                files.update(
                    {
                        f"override/{iid}/{p.relative_to(directory)}": file_digest(p)
                        for p in sorted(directory.rglob("*"))
                        if p.is_file()
                    }
                )
        perturbation = getattr(self.args, "language_perturbation_config", "")
        if not perturbation:
            perturbation = root / "instruction_perturbation.json"
        if Path(perturbation).is_file():
            files["instruction_perturbation"] = file_digest(perturbation)
        scoring_dir = Path(__file__).resolve().parent.parent / "plugins/output_system"
        if self.args.sub_task_name.startswith("hang_cups_on_cup_tree"):
            files["checker/hang.py"] = file_digest(scoring_dir.parent / "ader/action/custom/hang.py")
        if self.args.sub_task_name.startswith("pick_up_trash_wbc"):
            files["checker/wbc_trash_pickup.py"] = file_digest(scoring_dir.parent / "ader/action/wbc_trash_pickup.py")
        files.update(
            {f"scoring/{p.name}": file_digest(p) for p in sorted(scoring_dir.glob("*")) if p.suffix in (".py", ".json")}
        )
        return dict(
            parameters=parameters,
            task_config=copy.deepcopy(self.task_config),
            instances=instance_ids,
            inputs=files,
            app=getattr(self.api_core, "evaluation_app_config", {}),
        )

    def _finish_checkpoint_recordings(self):
        checkpoint = self._checkpoint
        if not self.args.record or not checkpoint.finished:
            return
        recorder = self.api_core.local_recorder
        if not checkpoint.recordings_ready:
            recorder.restore_checkpoint(checkpoint.run_id, checkpoint.recording_paths())
            artifacts = recorder.concat_all(preserve_bags=True)
            checkpoint.finalize_recordings(artifacts)
        recorder.publish_checkpoint_artifacts(
            [checkpoint.out_dir / item["name"] for item in checkpoint.state["artifacts"]] + [checkpoint.result_path],
            checkpoint.out_dir,
            getattr(self.args, "record_video_dir", None) or None,
        )
        checkpoint.cleanup_recordings(committed=True)

    def _evaluate_task(self, task):
        # load task config
        task_config_file = os.path.join(system_utils.benchmark_conf_path(), "eval_tasks", task + ".json")
        self.task_config = system_utils.load_json(task_config_file)
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

        episode_idx = 0
        scene_instance_ids = [0]
        sub_task_name = self.args.sub_task_name
        if sub_task_name != "":
            sub_task_path = os.path.join(system_utils.benchmark_conf_path(), "llm_task", sub_task_name)
            scene_instance_ids = sorted([int(name) for name in os.listdir(sub_task_path) if name.isdigit()])

        instance_ids_arg = (getattr(self.args, "instance_ids", "") or "").strip()
        if instance_ids_arg:
            # Explicit override, in caller order; takes priority over
            # num_instances and preserves caller order.
            scene_instance_ids = [int(x) for x in instance_ids_arg.split(",") if x.strip() != ""]
            logger.info(f"Explicit --benchmark.instance_ids override: {scene_instance_ids}")
        else:
            num_instances = getattr(self.args, "num_instances", 0) or 0
            if num_instances > 0 and num_instances < len(scene_instance_ids):
                # Deterministic sub-sample based on the run's seed so the same
                # config picks the same instances across machines / re-runs.
                total = len(scene_instance_ids)
                rng = np.random.RandomState(getattr(self.args, "seed", 0))
                scene_instance_ids = sorted(rng.choice(scene_instance_ids, size=num_instances, replace=False).tolist())
                logger.info(
                    f"Sampled {num_instances} of {total} scene instances "
                    f"(seed={getattr(self.args, 'seed', 0)}): {scene_instance_ids}"
                )

        checkpoint = self._checkpoint
        identity = self._checkpoint_identity(scene_instance_ids) if checkpoint is not None else None
        if checkpoint is not None and checkpoint.state is not None:
            restored = checkpoint.restore(identity)
            self.task_config = copy.deepcopy(checkpoint.state["task_config"])
            task_folder = str(checkpoint.task_dir)
            robot_cfg = self.task_config["robot"]["robot_cfg"]
            self.data_courier.set_robot_cfg(self.task_config["robot_cfg"])
            self.evaluate_summary.restore_results(restored)
        else:
            task_generator = TaskGenerator(self.task_config)
            task_folder = (
                str(checkpoint.task_dir)
                if checkpoint is not None
                else os.path.join(system_utils.benchmark_root_path(), "saved_task/%s" % self.task_config["task"])
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
            if checkpoint is not None:
                files = sorted(Path(task_folder).glob("*.json"))
                counts = {}
                for iid in scene_instance_ids:
                    path = (
                        Path(system_utils.benchmark_conf_path())
                        / "llm_task"
                        / sub_task_name
                        / str(iid)
                        / "instructions.json"
                    )
                    if sub_task_name:
                        try:
                            with path.open() as stream:
                                counts[iid] = len(json.load(stream)["instructions"])
                        except (FileNotFoundError, json.JSONDecodeError, KeyError):
                            counts[iid] = 1
                    else:
                        counts[iid] = 1
                plan = build_episode_plan(scene_instance_ids, files, counts, self.args.num_episode)
                checkpoint.prepare(identity, plan, files, self.task_config)
        if checkpoint is not None:
            self.evaluate_summary.checkpoint_progress = checkpoint.progress
            # Discard any diagnostic/uncommitted tail before recording another attempt.
            self.evaluate_summary.dump_eval_result()
            checkpoint.cleanup_recordings()
            if self.args.record:
                self.api_core.local_recorder.restore_checkpoint(checkpoint.run_id, checkpoint.recording_paths())
            if checkpoint.finished:
                self._finish_checkpoint_recordings()
                self.api_core.stop()
                return

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
            return
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
            if checkpoint is not None:
                instance_plan = [e for e in checkpoint.state["plan"] if e["instance_id"] == instance_id]
                if all(checkpoint.is_completed(e) or checkpoint.is_skipped(e) for e in instance_plan):
                    continue
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
                if checkpoint is not None:
                    planned_count = sum(
                        e["instance_id"] == instance_id and e["file_idx"] == file_idx for e in checkpoint.state["plan"]
                    )
                    if self.args.num_episode * episode_count != planned_count:
                        raise ValueError("Runtime instructions do not match the checkpoint plan")
                for episode_id in range(self.args.num_episode * episode_count):
                    if self.data_courier.stop_requested():
                        break
                    np.random.seed(episode_seed)
                    coord = dict(instance_id=instance_id, file_idx=file_idx, episode_id=episode_id)
                    if checkpoint is not None:
                        episode_idx = checkpoint.episode_index(coord)
                        if checkpoint.is_completed(coord) or checkpoint.is_skipped(coord):
                            continue
                    if self.task_config.get("generalization", {}).get("hdr_texture", {}).get("resample_per_episode"):
                        apply_hdr_texture_generalization(
                            self.api_core, self.task_config, seed=gen_seed, episode_id=episode_id, file_idx=file_idx
                        )
                    self.policy.set_episode_idx(episode_idx)
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
                    single_te.episode_coord = coord
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
                    episode_error = None
                    fatal_episode = False
                    try:
                        self.evaluate_episode(robot_cfg, single_te)
                        if checkpoint is not None and not single_te.episode_finished:
                            raise InferenceUnavailableError("Evaluation stopped before a terminal episode result")
                    except (InferenceUnavailableError, KeyboardInterrupt, SystemExit) as ep_err:
                        episode_error = ep_err
                        raise
                    except Exception as ep_err:
                        episode_error = ep_err
                        fatal_episode = True
                        if checkpoint is not None and self.args.resume and isinstance(ep_err, InferenceServiceError):
                            raise
                        logger.error("Episode loop stopped due to fatal episode error: %s", ep_err)
                    finally:
                        # A failed terminal notification cannot invalidate a sealed checker result.
                        sealed = checkpoint is not None and single_te.episode_finished
                        if sealed and single_te.terminal_result is not None:
                            single_te.result = copy.deepcopy(single_te.terminal_result)
                        self._cache_episode_result(single_te, self.env, None if sealed else episode_error)
                        bag = None
                        if self.args.record:
                            bag = self.api_core.stop_local_recording(episode_idx=episode_idx)
                        retryable = isinstance(
                            episode_error,
                            (InferenceUnavailableError, InferenceServiceError, KeyboardInterrupt, SystemExit),
                        )
                        if checkpoint is not None and (sealed or (fatal_episode and not retryable)):
                            if sealed or not self.data_courier.stop_requested():
                                if self.args.record and bag is None:
                                    raise RuntimeError("Cannot commit an episode without its requested recording")
                                checkpoint.commit(
                                    self.evaluate_summary.results[-1], bag, skip_remaining=fatal_episode and not sealed
                                )
                                self.evaluate_summary.checkpoint_progress = checkpoint.progress
                                self.evaluate_summary.dump_eval_result()
                    if self.data_courier.stop_requested() or fatal_episode:
                        break
                    episode_idx += 1

            self.env.stop()
        if checkpoint is not None:
            self._finish_checkpoint_recordings()
        elif self.args.record:
            self.api_core.concat_recordings(final_output_dir=getattr(self.args, "record_video_dir", None) or None)
        self.api_core.stop()
        try:
            task_folder_abs = os.path.abspath(task_folder)
            if checkpoint is None and os.path.isdir(task_folder_abs):
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
                ik_backend=self.task_config.get("robot", {}).get("ik_backend", "auto"),
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
        )

        scene_info = None
        if self.args.sub_task_name == "":
            scene_info = self.gen_layouts(mode="instance")

        if self.args.model_arc == "corobot":
            from geniesim_benchmark.benchmark.envs.pi_env import PiEnv

            env_task_config = json.loads(json.dumps(self.task_config))
            self.env = PiEnv(self.api_core, episode_file, env_task_config)
        elif self.args.model_arc == "" and self.args.env_class == "PiEnv":
            from geniesim_benchmark.benchmark.envs.pi_env import PiEnv

            self.env = PiEnv(self.api_core, episode_file, self.task_config)
        elif self.args.env_class == "DummyEnv":
            from geniesim_benchmark.benchmark.envs.dummy_env import DummyEnv

            self.env = DummyEnv(self.api_core, episode_file, self.task_config)
        else:
            raise ValueError("Invalid env_class {self.args.env_class}")

        self.env.set_data_courier(self.data_courier)
        self.env.set_scene_info(scene_info)

    def _finalize_no_action_episode(self, env, policy, evaluation):
        """Score the last executed state without issuing or counting another action."""
        termination = getattr(policy, "server_episode_termination", None)
        if termination is None:
            return False
        # Use the existing checker tree at the current step, including progress
        # since its last periodic update. Do not advance physics/action budgets,
        # cancel unfinished checkpoints, or infer success from the server flag.
        env.task.step(env)
        env.action_update()
        evaluation.update_policy_statistics(policy)
        evaluation.update_progress(env.task.task_progress)
        evaluation.result["termination"] = {
            "source": "policy",
            "native_step": env.current_step,
            "policy": copy.deepcopy(termination),
        }
        evaluation.summarize_scores()
        env.has_done = True
        policy.update_task_status(True, env.task.task_progress)
        logger.info("Policy terminated without an action; native scores finalized at step %s", env.current_step)
        return True

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
        start_time = time.time()
        (
            start_callbacks,
            step_callbacks,
            end_callbacks,
            data_callbacks,
        ) = get_hook_callbacks(self.policy)

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
                if not self.args.interactive and self._finalize_no_action_episode(self.env, self.policy, single_te):
                    break
                if should_pause:
                    self.api_core.resume_physics()
                for callback in step_callbacks:  # during task
                    callback(self.env, action)

                if hasattr(self.env, "set_policy_step_info"):
                    self.env.set_policy_step_info(
                        chunk_started=getattr(self.policy, "chunk_started_this_step", False),
                        chunk_len=getattr(self.policy, "chunk_len", None),
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
                        single_te.episode_finished = True
                        single_te.terminal_result = copy.deepcopy(single_te.result)
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
                            # The next evaluate_episode resets before its first
                            # observation. Resetting here can wait on a stopped
                            # physics loop and discard an already scored episode
                            # before the caller saves its result and recording.
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
            if self.api_core.physics_paused:
                self.api_core.resume_physics()
            single_te.update_from_dict(
                {
                    "end_time": system_utils.get_local_time(),
                    "duration": time.time() - start_time,
                }
            )

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
            ik_backend=self.task_config.get("robot", {}).get("ik_backend", "auto"),
        )
        return policy

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
                        if task_configs[slot].get("generalization", {}).get("hdr_texture", {}).get("resample_per_episode"):
                            apply_hdr_texture_generalization(
                                envs[slot].api_core, task_configs[slot],
                                seed=seed + instance_id * 1000 + file_idx,
                                episode_id=episode_id, file_idx=file_idx,
                            )
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

                    batch_error = None
                    try:
                        self._run_sync_episode(vec_env, vec_policy, te_list)
                    except BaseException as error:
                        batch_error = error
                        raise
                    finally:
                        for slot, evaluation in enumerate(te_list):
                            error = batch_error if "scores" not in evaluation.result else None
                            self._cache_episode_result(evaluation, envs[slot], error)

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
                    self._finalize_no_action_episode(vec_env.get_env(i), vec_policy.get_policy(i), evaluation)

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
