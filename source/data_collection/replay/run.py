import json
import os
import shutil
import sys
import time
before_start_simulation = time.perf_counter()
from dataclasses import fields

# Add the project root to the import path when replay runs as a script.
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_here)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from replay.cli import load_top_config, parse_cli_args

_args = parse_cli_args()
_top_config = load_top_config(_args.config_json)
_render_mode = _top_config.get("render_mode", "rtx2")

_extra_args = []
if _render_mode == "rtx2":
    _extra_args = [
        "--/rtx-transient/stableIds/enabled=true",
        "--/persistent/rtx/modes/rt/enabled=true",
        "--/persistent/rtx/modes/rt2/enabled=true",
        "--/persistent/rtx/rendermode=RealTimePathTracing",
        "--/rtx/rendermode=RealTimePathTracing",
        "--/app/hydraEngine/waitIdle=1",
        "--/app/updateOrder/checkForHydraRenderComplete=1000",
        "--/exts/isaacsim.ros2.bridge/publish_multithreading_disabled=1",
    ]   

from common.base_utils.logger import logger
from replay.config import FuzzingConfig, ReplayConfig, build_simulation_app_config
logger.info("render_mode=%s, extra_args=%s", _render_mode, _extra_args)
from isaacsim import SimulationApp
simulation_app = SimulationApp(build_simulation_app_config(_args.headless, _extra_args))
sim_launch_done = time.perf_counter()
simulation_app._carb_settings.set("/physics/cooking/ujitsoCollisionCooking", False)  # noqa: SLF001
simulation_app._carb_settings.set("/omni/replicator/asyncRendering", False)
simulation_app._carb_settings.set("/app/asyncRendering", False)
simulation_app._carb_settings.set("/rtx/raytracing/fractionalCutoutOpacity", True)

from replay.data_copy import copy_original_to_recording
from replay.fuzzer import Fuzzer
from replay.replayer import Replayer
from scripts.smooth_deformable import smooth_episode_deformable
from server.recording.sim_data_converter import SimDataConverter


_PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preset_configs")
_ROBOT_CONFIG_DIR = os.path.join(_project_root, "config", "robot_cfg")


def _robot_configs_by_name(config_dir: str):
    candidates = {}
    for filename in sorted(os.listdir(config_dir)):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(config_dir, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            name = (json.load(f).get("robot") or {}).get("robot_name", "").strip().upper()
        if name:
            candidates.setdefault(name, []).append(filename)
    return ({k: v[0] for k, v in candidates.items() if len(v) == 1},
            {k: v for k, v in candidates.items() if len(v) > 1})


def _default_robot_config_name(robot_name: str, filenames: list[str]) -> str | None:
    """Return the preferred config for known robot names with intentional variants."""
    preferred = {
        "G1_OMNIPICKER_FIXED": "G1_omnipicker_fixed_dual.json",
        "G2_CRSB_90D": "G2_crsB_90D.json",
        "G2_OMNIPICKER_FIXED": "G2_omnipicker_fixed_dual.json",
    }.get(robot_name)
    return preferred if preferred in filenames else None


def _resolve_robot_config_name(robot_name: str, unique, ambiguous) -> str:
    if robot_name in ambiguous:
        default = _default_robot_config_name(robot_name, ambiguous[robot_name])
        if default:
            return default
        raise ValueError(f"Robot name {robot_name!r} maps to multiple configs: {', '.join(ambiguous[robot_name])}; please specify robot_config explicitly")
    if robot_name in unique:
        return unique[robot_name]
    compatible = None
    if "G2_CRS_" in robot_name and "90D" in robot_name: compatible = "G2_CRS_90D"
    elif "G2" in robot_name and "90D" in robot_name: compatible = "G2_CRSB_90D"
    elif "G2" in robot_name: compatible = "G2_OMNIPICKER_FIXED"
    elif "G1" in robot_name: compatible = "G1_OMNIPICKER_FIXED"
    elif "ARX" in robot_name: compatible = "ARX"
    elif "ALOHA" in robot_name: compatible = "ALOHA"
    if compatible in ambiguous:
        default = _default_robot_config_name(compatible, ambiguous[compatible])
        if default:
            return default
        raise ValueError(f"Robot name {robot_name!r} maps to multiple configs: {', '.join(ambiguous[compatible])}; please specify robot_config explicitly")
    if compatible in unique:
        return unique[compatible]
    raise ValueError(f"Unknown robot name: {robot_name}")


def _robot_config_from_recording_info(input_dir: str) -> str:
    """Resolve the robot config filename from state.json.

    Prefer state.robot.name. For a link-pose recording without that field,
    infer the robot name from a ``<prefix>_base_link`` key in the first frame.
    """
    path = os.path.join(input_dir.rstrip("/"), "state.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"state.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        info = json.load(f)
    robot_name = (info.get("robot") or {}).get("name", "").upper()
    if not robot_name:
        frames = info.get("frames") or []
        if frames:
            for k in (frames[0].get("objects") or {}):
                if k.endswith("_base_link"):
                    robot_name = k[: -len("_base_link")].upper()
                    break
    unique, ambiguous = _robot_configs_by_name(_ROBOT_CONFIG_DIR)
    return _resolve_robot_config_name(robot_name, unique, ambiguous)


def _run_sim_data_converter(rec_path: str, robot_name: str, is_teleop: bool) -> bool:
    """Run SimDataConverter for one recording and report whether it succeeded."""
    try:
        state_path = os.path.join(rec_path, "state.json")
        metadata_path = os.path.join(rec_path, "metadata.yaml")
        rec_info_path = os.path.join(rec_path, "recording_info.json")
        if not os.path.isfile(state_path):
            logger.warning("SimDataConverter skip %s: no state.json", rec_path)
            return False
        if not os.path.isfile(metadata_path):
            logger.warning("SimDataConverter skip %s: no metadata.yaml", rec_path)
            return False
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        robot_type = (state.get("robot") or {}).get("name", robot_name) or robot_name
        gripper_names = []
        if os.path.isfile(rec_info_path):
            with open(rec_info_path, "r", encoding="utf-8") as f:
                rec_info = json.load(f)
            if rec_info.get("gripper_names"):
                gripper_names = rec_info["gripper_names"]
        converter = SimDataConverter(
            rec_path,
            rec_path,
            job_id=0,
            task_id=0,
            episode_id=0,
            gripper_names=gripper_names,
            robot_type=robot_type,
            skip_data_info=is_teleop,
        )
        converter.convert()
        logger.info("SimDataConverter done: %s", rec_path)
        return True
    except Exception as e:
        logger.warning("SimDataConverter failed %s: %s", rec_path, e)
        task_result_path = os.path.join(rec_path, "task_result.json")
        if os.path.isfile(task_result_path):
            try:
                os.remove(task_result_path)
                logger.info("Removed task_result.json due to convert failure: %s", task_result_path)
            except OSError as err:
                logger.warning("Failed to remove task_result.json %s: %s", task_result_path, err)
        return False


def _inject_camera_info_into_source_state(input_dir: str, first_rec_path: str) -> None:
    """Copy camera metadata from the first output into the source state.json.

    Later copies of the source recording inherit that metadata.
    """
    rec_info_path = os.path.join(first_rec_path, "recording_info.json")
    state_path = os.path.join(input_dir, "state.json")
    if not os.path.isfile(rec_info_path) or not os.path.isfile(state_path):
        return
    try:
        with open(rec_info_path, "r", encoding="utf-8") as f:
            rec_info = json.load(f)
        camera_info = rec_info.get("camera_info", {})
        frame_cameras_list = rec_info.get("frame_cameras", [])
        if not camera_info and not frame_cameras_list:
            return
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        if camera_info:
            state["cameras"] = camera_info
        if frame_cameras_list:
            frames = state.get("frames", [])
            for i, frame_cameras in enumerate(frame_cameras_list):
                if i >= len(frames):
                    break
                if "cameras" not in frames[i]:
                    frames[i]["cameras"] = {}
                for cam_name, cam_data in frame_cameras.items():
                    frames[i]["cameras"][cam_name] = dict(cam_data)
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        logger.info("Injected cameras into %s (will be copied to all rec_paths)", state_path)
    except Exception as e:
        logger.warning("Failed to inject camera_info into state.json: %s", e)


def _check_video_sizes(recording_dirs: list[str], failed_rec_paths: set) -> set:
    """Return recordings with missing videos or a video smaller than 5 KiB."""
    undersized: set = set()
    min_size_bytes = 5 * 1024
    for rec_path in recording_dirs:
        if rec_path in failed_rec_paths:
            continue
        video_dir = os.path.join(rec_path, "observations", "videos")
        if not os.path.isdir(video_dir):
            undersized.add(rec_path)
            continue
        has_video = False
        bad = False
        for fname in os.listdir(video_dir):
            fpath = os.path.join(video_dir, fname)
            if not os.path.isfile(fpath):
                continue
            has_video = True
            try:
                fsize = os.path.getsize(fpath)
            except OSError as e:
                logger.warning("Failed to stat video %s: %s", fpath, e)
                bad = True
                break
            if fsize < min_size_bytes:
                logger.warning(
                    "Video too small (<5KB): %s (%d bytes)", fpath, fsize,
                )
                bad = True
                break
        if bad or not has_video:
            undersized.add(rec_path)
    if undersized:
        logger.warning(
            "Detected %d recording dirs with missing or undersized videos: %s",
            len(undersized),
            [os.path.basename(p.rstrip("/")) for p in undersized],
        )
    else:
        logger.info("All recording dirs have valid videos.")
    return undersized


def _replay_config_from_dict(data: dict) -> ReplayConfig:
    """Build ReplayConfig using only fields declared on the dataclass."""
    allowed = {f.name for f in fields(ReplayConfig)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return ReplayConfig(**kwargs)


def _fuzzing_config_from_dict(data: dict) -> FuzzingConfig:
    """Build FuzzingConfig using only fields declared on the dataclass."""
    allowed = {f.name for f in fields(FuzzingConfig)}
    kwargs = {k: v for k, v in data.items() if k in allowed}
    return FuzzingConfig(**kwargs)


def main():
    args = _args
    top = _top_config

    preset_name = top.get("preset")
    if preset_name:
        preset_path = os.path.join(_PRESET_DIR, f"{preset_name}.json")
        if not os.path.isfile(preset_path):
            available = [f.removesuffix(".json") for f in os.listdir(_PRESET_DIR) if f.endswith(".json")]
            raise SystemExit(f"Unknown preset: '{preset_name}'; available: {available}")
        with open(preset_path, "r", encoding="utf-8") as f:
            preset = json.load(f)
        replay_config_raw = dict(preset.get("replay_config") or {})
        fuzzing_config_raw = dict(preset.get("fuzzing_config") or {})
        replay_config_raw.update(top.get("replay_config") or {})
        fuzzing_config_raw.update(top.get("fuzzing_config") or {})
    else:
        replay_config_raw = top.get("replay_config") or {}
        fuzzing_config_raw = top.get("fuzzing_config") or {}

    is_teleop = top.get("sub_mode") or False

    config = _replay_config_from_dict(replay_config_raw)
    if args.cloth_subdivision_levels is not None:
        config.cloth_subdivision_levels = args.cloth_subdivision_levels
    if type(config.cloth_subdivision_levels) is not int or config.cloth_subdivision_levels not in range(4):
        raise ValueError("replay_config.cloth_subdivision_levels must be an integer 0..3")
    config.recording_root = args.recording_root
    if not config.robot_config:
        config.robot_config = _robot_config_from_recording_info(args.input_dir)
    logger.info("Using robot_config: %s", config.robot_config)
    fuzzing_config = _fuzzing_config_from_dict(fuzzing_config_raw)
    if fuzzing_config.num_rounds is None:
        fuzzing_config.num_rounds = config.num_rounds

    replayer = Replayer(args.input_dir, config)
    replayer.load_payload()

    # Smooth PBD/FEM frame jitter before replay. The smoother backs up the
    # original NPZ files under deformable_raw and is a no-op without deformable/.
    # ClothReplayer reads the resulting NPZ files during replay.
    if (
        args.deformable_smoothing if args.deformable_smoothing is not None
        else not config.no_deformable_smooth
    ):
        try:
            smooth_episode_deformable(args.input_dir, log=logger.info)
        except Exception as e:
            logger.warning("smooth_episode_deformable failed for %s: %s", args.input_dir, e)
    else:
        logger.info("Deformable smoothing disabled; replay input NPZ unchanged: %s", args.input_dir)

    setup_start = time.perf_counter()
    replayer.setup(simulation_app)
    setup_done = time.perf_counter()

    sim_assets = os.environ.get("SIM_ASSETS", "")
    fuzzer = Fuzzer(
        input_dir=args.input_dir,
        config=fuzzing_config,
        sim_assets=sim_assets,
        objects_dir_name=replayer.objects_dir_name,
    )
    fuzzer.generate_all()
    if config.add_default_light:
        replayer.ensure_default_lights()

    round_index = 0
    replay_start = setup_done
    all_recording_dirs: list[str] = []
    frames_per_recording: int | None = None
    while simulation_app.is_running():
        record_env_ids = fuzzer.fuzz(replayer, replayer.num_clones)
        replayer.set_record_env_ids_for_round(record_env_ids)

        n_frames = replayer.run_one_round()
        if frames_per_recording is None:
            frames_per_recording = n_frames
        replayer.finish_round_recording(n_frames)
        dirs = replayer.get_recording_dirs_by_env()
        if dirs:
            all_recording_dirs.extend(dirs.values())
        replayer.log_resource_status()
        round_index += 1
        if not fuzzer.should_continue():
            break
        replayer.allocate_next_recording_dirs()
        # replayer.reset_to_frame0()

    total_pool_wait_sec = replayer.total_pool_wait_sec
    release_t0 = time.perf_counter()
    failed_rec_paths = replayer.release_writers()
    release_t1 = time.perf_counter()
    total_release_sec = release_t1 - release_t0
    total_video_wait_sec = total_pool_wait_sec + total_release_sec
    round_loop_done = time.perf_counter()

    total_task_count = len(all_recording_dirs)

    if not config.image_mode:
        undersized_rec_paths = _check_video_sizes(all_recording_dirs, failed_rec_paths)
        if undersized_rec_paths:
            failed_rec_paths.update(undersized_rec_paths)

    if failed_rec_paths:
        logger.warning(
            "Skipping %d recording dirs due to video encoding errors or empty video output: %s",
            len(failed_rec_paths),
            [os.path.basename(p.rstrip("/")) for p in failed_rec_paths],
        )
        all_recording_dirs = [d for d in all_recording_dirs if d not in failed_rec_paths]

    depth_convert_sec = 0.0
    record_depth_mode = config.record_depth.upper()
    depth_realtime = config.depth_realtime
    if record_depth_mode == "ALL" and not depth_realtime and not config.image_mode and all_recording_dirs:
        from replay.video_converter import VideoConverter
        depth_names = replayer.get_recording_camera_depth_image_names()
        if depth_names:
            logger.info(
                "Post-processing depth videos for %d dirs, depth cameras: %s",
                len(all_recording_dirs), depth_names,
            )
            depth_converter = VideoConverter(
                fps=config.video_fps,
                preset=config.video_preset,
                crf=config.video_crf,
                gop=config.video_gop,
                pix_fmt=config.video_pix_fmt,
                max_workers=4,
            )
            env_id_to_rec = {i: d for i, d in enumerate(all_recording_dirs)}
            depth_converter.submit(
                env_id_to_rec,
                camera_image_names=[],
                camera_depth_image_names=depth_names,
            )
            depth_converter.set_status_interval(10)
            depth_t0 = time.perf_counter()
            depth_converter.wait_all()
            depth_convert_sec = time.perf_counter() - depth_t0
            depth_ok_paths = depth_converter.get_ok_rec_paths()
            depth_all_paths = depth_converter.get_rec_paths()
            depth_failed_paths = depth_all_paths - depth_ok_paths
            if depth_failed_paths:
                logger.warning(
                    "Depth video conversion failed for %d dirs: %s",
                    len(depth_failed_paths),
                    [os.path.basename(p.rstrip("/")) for p in depth_failed_paths],
                )
                failed_rec_paths.update(depth_failed_paths)
                all_recording_dirs = [d for d in all_recording_dirs if d not in failed_rec_paths]
            d_total, d_ok, d_fail = depth_converter.get_stats()
            logger.info(
                "Depth video conversion done: total=%d ok=%d fail=%d in %.1fs",
                d_total, d_ok, d_fail, depth_convert_sec,
            )

    input_dir = replayer.input_dir.rstrip("/")
    if all_recording_dirs:
        _inject_camera_info_into_source_state(input_dir, all_recording_dirs[0])
    copy_success_count = 0
    copied_paths: list[str] = []
    copy_start = round_loop_done
    for rec_path in all_recording_dirs:
        copy_original_to_recording(input_dir, rec_path)
        try:
            shutil.copy2(args.config_json, os.path.join(rec_path, os.path.basename(args.config_json)))
        except Exception as e:
            logger.warning("Failed to copy config_json to %s: %s", rec_path, e)
        copy_success_count += 1
        copied_paths.append(rec_path)
    copy_done = time.perf_counter()

    converter_start = time.perf_counter()
    converter_ok = 0
    with open(os.path.join(input_dir, "state.json"), "r", encoding="utf-8") as f:
        robot_name = ((json.load(f) or {}).get("robot") or {}).get("name") or "G2"
    for rec_path in copied_paths:
        if _run_sim_data_converter(rec_path, robot_name, is_teleop):
            converter_ok += 1
    converter_done = time.perf_counter()

    fps = config.video_fps
    single_frames = frames_per_recording if frames_per_recording is not None else 0
    total_output_frames = single_frames * copy_success_count
    duration_sec = (total_output_frames / fps) if (fps and total_output_frames) else 0.0
    camera_rgb = replayer.get_recording_camera_image_names()
    camera_depth = replayer.get_recording_camera_depth_image_names()
    copy_fail_count = len(all_recording_dirs) - copy_success_count

    for rec_path in copied_paths:
        with open(os.path.join(rec_path, "replay_done"), "w") as f:
            f.write("replay_done")

    if config.clean_cameras:
        cleaned = 0
        for rec_path in copied_paths:
            cam_dir = os.path.join(rec_path, "camera")
            if os.path.isdir(cam_dir):
                try:
                    shutil.rmtree(cam_dir)
                    cleaned += 1
                except Exception as e:
                    logger.warning("Failed to clean camera dir %s: %s", cam_dir, e)
        if cleaned:
            logger.info("Cleaned camera dirs for %d recordings", cleaned)

    init_sec = setup_start - before_start_simulation
    sim_launch_sec = sim_launch_done - before_start_simulation
    config_load_sec = setup_start - sim_launch_done
    setup_sec = setup_done - setup_start
    replay_sec = round_loop_done - replay_start
    copy_sec = copy_done - copy_start
    converter_sec = converter_done - converter_start
    total_sec = converter_done - before_start_simulation
    sum_breakdown = init_sec + setup_sec + replay_sec + depth_convert_sec + copy_sec + converter_sec

    replay_pure_sec = replay_sec - total_video_wait_sec

    logger.info("========== Replay statistics ==========")
    logger.info("  Total time: %.2f s (sum of phases: %.2f s)", total_sec, sum_breakdown)
    logger.info(
        "  Timing: Isaac Sim startup %.2f s | configuration/load_payload %.2f s | setup %.2f s | "
        "replay (including encoding) %.2f s | depth postprocessing %.2f s | copy %.2f s | SimDataConverter %.2f s",
        sim_launch_sec, config_load_sec, setup_sec,
        replay_sec, depth_convert_sec, copy_sec, converter_sec,
    )
    logger.info(
        "  Replay breakdown: replay/capture %.2f s | waiting for video encoding %.2f s (between-round pool %.2f s + final release %.2f s)",
        replay_pure_sec, total_video_wait_sec, total_pool_wait_sec, total_release_sec,
    )
    logger.info("  Rounds: %d concurrency: %d tasks: %d succeeded: %d", round_index, config.max_concurrent_envs, total_task_count, len(all_recording_dirs))
    logger.info("  Frames per episode: %d  copies succeeded: %d -> total output frames: %d", single_frames, copy_success_count, total_output_frames)
    logger.info("  Total output duration: %.2f s (at %d fps)", duration_sec, fps)
    logger.info("  Rate: %.2f", duration_sec / total_sec)
    logger.info("  Robot configuration: %s", config.robot_config)
    logger.info(
        "  Video encoding: codec=libx265 preset=%s crf=%d gop=%d fps=%d",
        replayer.video_preset, config.video_crf, config.video_gop, config.video_fps,
    )
    logger.info("  Camera types RGB: %s  depth: %s", camera_rgb if camera_rgb else "[]", camera_depth if camera_depth else "[]")
    logger.info("  Depth mode: %s (realtime=%s)", record_depth_mode, depth_realtime)
    logger.info("  Copies succeeded: %d  copies failed: %d", copy_success_count, copy_fail_count)
    logger.info("  SimDataConverter: %d/%d succeeded", converter_ok, len(copied_paths))
    logger.info("==================================")
    simulation_app.close()


if __name__ == "__main__":
    main()
