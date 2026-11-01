"""
Record RGB/depth camera output, encode videos asynchronously, and write metadata.
This module receives a stage, environment paths, and recording configuration
without depending on KinematicReplay. It supports video and image-only output.
"""
import json
import os
import queue
import random
import subprocess
import threading
import time

import numpy as np
import psutil

from common.base_utils.logger import logger
from .camera_intrinsics import get_intrinsic_for_camera
from .camera_utils import get_camera_prim_name, get_camera_resolution_from_config

_SENTINEL = object()


class AsyncVideoWriter:
    """
    Encode queued frames through ffmpeg in a background thread.
    RGB uses H.265 and depth uses lossless gray16le PNG encoding.
    The capture thread only enqueues frames; ffmpeg uses one encoding thread.
    """

    def __init__(
        self,
        out_path: str,
        fps: int,
        frame_size: tuple[int, int],
        mode: str = "rgb",
        crf: int = 24,
        gop: int = 8,
        preset: str = "medium",
        maxsize: int = 0,
    ):
        import cv2
        self._cv2 = cv2
        w, h = frame_size
        self._out_path = out_path
        self._mode = mode
        self._frames_written = 0
        self._frame_size = (w, h)
        self._fps = fps
        self._crf = crf
        self._gop = gop
        self._preset = preset
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, name=f"vwriter-{os.path.basename(out_path)}", daemon=True,
        )
        self._thread.start()

    def _build_cmd(self) -> list[str]:
        w, h = self._frame_size
        if self._mode == "depth":
            return [
                "ffmpeg", "-y",
                "-f", "image2pipe",
                "-framerate", str(self._fps),
                "-i", "pipe:0",
                "-c:v", "png",
                "-pix_fmt", "gray16le",
                "-r", str(self._fps),
                "-movflags", "+faststart",
                self._out_path,
            ]
        return [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{w}x{h}",
            "-r", str(self._fps),
            "-i", "pipe:0",
            "-vcodec", "libx265",
            "-preset", self._preset,
            "-threads", "2",
            "-pix_fmt", "yuv420p",
            "-keyint_min", "8",
            "-sc_threshold", "0",
            "-vf", f"setpts=N/({self._fps}*TB)",
            "-bf", "0",
            "-crf", str(self._crf),
            "-g", str(self._gop),
            "-x265-params", "pools=1:frame-threads=1:lookahead-slices=1",
            self._out_path,
        ]

    def _run(self) -> None:
        proc = None
        # Write stderr to a separate file so its pipe cannot fill and block
        # ffmpeg input. The file also keeps encoding errors for inspection.
        stderr_log_path = self._out_path + ".ffmpeg.log"
        stderr_file = None
        try:
            cmd = self._build_cmd()
            stderr_file = open(stderr_log_path, "wb")
            proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_file,
            )
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    self._queue.task_done()
                    break
                if self._mode == "depth":
                    _, buf = self._cv2.imencode(".png", item)
                    proc.stdin.write(buf.tobytes())
                else:
                    bgr = self._cv2.cvtColor(item, self._cv2.COLOR_RGB2BGR)
                    proc.stdin.write(bgr.tobytes())
                self._frames_written += 1
                self._queue.task_done()
            proc.stdin.close()
            proc.wait()
            if proc.returncode != 0:
                self._error = RuntimeError(
                    f"ffmpeg failed (rc={proc.returncode}); see {stderr_log_path}"
                )
        except Exception as e:
            self._error = e
            logger.warning("AsyncVideoWriter thread error for %s: %s", self._out_path, e)
            if proc is not None:
                try:
                    proc.kill()
                    proc.wait()
                except Exception as kill_err:
                    logger.warning("AsyncVideoWriter failed to kill ffmpeg for %s: %s", self._out_path, kill_err)
        finally:
            if stderr_file is not None:
                try:
                    stderr_file.close()
                except Exception:
                    pass

    def write(self, frame: np.ndarray) -> None:
        self._queue.put(frame)

    def flush(self) -> None:
        self._queue.join()

    def release(self) -> None:
        self._queue.put(_SENTINEL)
        self._thread.join()
        if self._error:
            logger.warning("AsyncVideoWriter error for %s: %s", self._out_path, self._error)

    def release_async(self) -> None:
        self._queue.put(_SENTINEL)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def queue_maxsize(self) -> int:
        return self._queue.maxsize

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def out_path(self) -> str:
        return self._out_path


class AsyncIOWriter:
    """Write images and compressed arrays from a background thread pool."""

    def __init__(self, num_workers: int = 4, maxsize: int = 200):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._workers: list[threading.Thread] = []
        self._inflight = 0
        self._lock = threading.Lock()
        for i in range(num_workers):
            t = threading.Thread(target=self._worker, name=f"async-io-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def _worker(self):
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                break
            func, args, kwargs = item
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning("AsyncIOWriter task failed: %s", e)
            finally:
                with self._lock:
                    self._inflight -= 1
                self._queue.task_done()

    def submit(self, func, *args, **kwargs):
        with self._lock:
            self._inflight += 1
        self._queue.put((func, args, kwargs))

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def flush(self):
        self._queue.join()

    def shutdown(self):
        for _ in self._workers:
            self._queue.put(_SENTINEL)
        for t in self._workers:
            t.join(timeout=30)


class Recorder:
    """Manage camera output, video encoding, and recording metadata."""

    def __init__(
        self,
        record_config: dict,
        robot_name: str,
        source_input_dir: str | None = None,
    ):
        self._record_config = record_config
        self._robot_name = robot_name
        self._source_input_dir = (source_input_dir or "").rstrip("/")
        self._cameras: list[dict] = []
        self._output_dir_by_env: dict[int, str] = {}
        self._cv2 = None
        self._frame_cameras_accumulator: list[dict] = []
        self._env_path_for_origin: str = ""
        self._video_writers: dict[int, dict[str, AsyncVideoWriter]] = {}
        self._depth_writers: dict[int, dict[str, AsyncVideoWriter]] = {}
        self._retiring_writers: list[tuple[AsyncVideoWriter, str]] = []
        self._writers_per_round: int = 0
        self._max_pool_multiplier: int = 2
        self._total_pool_wait_sec: float = 0.0
        self._video_fps: int = record_config.get("video_fps", record_config.get("fps", 30))
        self._video_crf: int = record_config.get("video_crf", 24)
        self._video_gop: int = record_config.get("video_gop", 8)
        self._video_preset: str = record_config.get("video_preset", "medium")
        self._depth_realtime: bool = record_config.get("depth_realtime", False)
        self._image_mode: bool = record_config.get("image_mode", False)
        self._save_depth_in_uint16: bool = record_config.get("save_depth_in_uint16", True)
        self._save_frame_count: int = 0
        self._empty_capture_count_by_cam: dict[tuple[int, str], int] = {}
        self._process = psutil.Process()
        self._io_writer = AsyncIOWriter(num_workers=4, maxsize=200)

    def set_output_dirs(self, env_id_to_path: dict) -> None:
        env_id_to_path = dict(env_id_to_path)
        if env_id_to_path == self._output_dir_by_env:
            if (
                not self._image_mode
                and not self._video_writers
                and not self._depth_writers
            ):
                self._init_video_writers()
            return
        self._io_writer.flush()
        self._retire_current_writers()
        self._wait_for_pool_capacity()
        self._output_dir_by_env = env_id_to_path
        self._frame_cameras_accumulator.clear()
        self._save_frame_count = 0
        for rec_path in self._output_dir_by_env.values():
            os.makedirs(rec_path, exist_ok=True)
        self._init_video_writers()

    def _retire_current_writers(self) -> None:
        for env_id, env_writers in self._video_writers.items():
            rec_path = self._output_dir_by_env.get(env_id, "")
            for w in env_writers.values():
                w.release_async()
                self._retiring_writers.append((w, rec_path))
        for env_id, env_writers in self._depth_writers.items():
            rec_path = self._output_dir_by_env.get(env_id, "")
            for w in env_writers.values():
                w.release_async()
                self._retiring_writers.append((w, rec_path))
        self._video_writers = {}
        self._depth_writers = {}

    def _reap_finished_writers(self) -> int:
        before = len(self._retiring_writers)
        still_alive = []
        for w, rec_path in self._retiring_writers:
            if w.is_alive:
                still_alive.append((w, rec_path))
            else:
                if w._error:
                    logger.warning("Retiring writer error for %s: %s", w.out_path, w._error)
        self._retiring_writers = still_alive
        return before - len(still_alive)

    def _wait_for_pool_capacity(self) -> None:
        if self._writers_per_round <= 0:
            self._reap_finished_writers()
            return
        max_retiring = self._writers_per_round * (self._max_pool_multiplier - 1)
        self._reap_finished_writers()
        if len(self._retiring_writers) <= max_retiring:
            logger.info(
                "Recorder: pool has capacity (%d retiring <= %d max), starting next round immediately",
                len(self._retiring_writers), max_retiring,
            )
            return
        logger.info(
            "Recorder: pool full (%d retiring > %d max), waiting for %d writers to finish ...",
            len(self._retiring_writers), max_retiring,
            len(self._retiring_writers) - max_retiring,
        )
        t0 = time.perf_counter()
        while len(self._retiring_writers) > max_retiring:
            for w, _ in self._retiring_writers:
                if not w.is_alive:
                    break
            else:
                time.sleep(0.1)
            self._reap_finished_writers()
        dt = time.perf_counter() - t0
        self._total_pool_wait_sec += dt
        logger.info(
            "Recorder: pool has capacity now (%d retiring), waited %.1fs (total pool wait: %.1fs)",
            len(self._retiring_writers), dt, self._total_pool_wait_sec,
        )


    def init_cameras(
        self,
        stage,
        env_paths: list,
        robot_cfg: dict,
    ) -> None:
        """Create render products and RGB annotators for each environment camera."""
        import omni.replicator.core as rep
        raw_list = self._record_config.get("camera_prim_list", [])
        additional = self._record_config.get("additional_cam_parameters") or {}
        noised_probability = additional.get("noised_probability", 0.0)
        noise_parameters = additional.get("noise_parameters", {})
        noised_camera_indices = additional.get(
            "noised_camera_indices",
            list(range(len(raw_list))),
        )
        noised_camera_set = set(noised_camera_indices)
        record_depth = (self._record_config.get("record_depth") or "NONE").upper()
        self._record_depth_mode = record_depth
        self._cameras = []
        self._env_path_for_origin = (env_paths[0] if env_paths else "").rstrip("/")

        for env_id, env_path in enumerate(env_paths):
            for idx, raw_path in enumerate(raw_list):
                prim_path = env_path + raw_path
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    logger.error("Skip recording camera (prim not found): %s", prim_path)
                    continue
                res = get_camera_resolution_from_config(prim_path, robot_cfg)
                rp = rep.create.render_product(prim_path, (int(res[0]), int(res[1])))
                annotator = rep.AnnotatorRegistry.get_annotator("rgb")
                annotator.attach(rp)
                prim_name = get_camera_prim_name(prim_path, self._robot_name, rename=self._record_config.get("rename_camera", True))
                noised = idx in noised_camera_set and (np.random.uniform() < noised_probability)
                merged_noise = {}
                if noised:
                    try:
                        from server.ros_publisher.camera_noiser import get_random_parameters
                        noise_type = noise_parameters.get("noise_type", None)
                        if noise_type is None:
                            noise_type = np.random.choice(
                                ["gaussian", "poisson", "salt_pepper", "speckle", "quantization"]
                            )
                        merged_noise = get_random_parameters(noise_type)
                        merged_noise.update(
                            {k: v for k, v in noise_parameters.items() if k != "noise_type"}
                        )
                        merged_noise["noise_type"] = noise_type
                        merged_noise["seed"] = random.randint(0, 99999)
                    except Exception as e:
                        logger.error("Noise params init failed: %s", e)
                record_depth_this = False
                if record_depth == "ALL":
                    if self._record_config.get("record_all_camera_depth", False):
                        record_depth_this = True
                    else:
                        if "head_right_Camera" in prim_path or "head_left_Camera" in prim_path:
                            pass
                        elif "Fisheye" in prim_path or "Top" in prim_path:
                            pass
                        else:
                            record_depth_this = True
                depth_annotator = None
                if record_depth_this:
                    rp_depth = rep.create.render_product(prim_path, (int(res[0]), int(res[1])))
                    depth_type = "distance_to_image_plane"
                    depth_annotator = rep.AnnotatorRegistry.get_annotator(depth_type)
                    depth_annotator.attach(rp_depth)
                self._cameras.append({
                    "env_id": env_id,
                    "env_path": env_path,
                    "prim_path": prim_path,
                    "prim_name": prim_name,
                    "resolution": [int(res[0]), int(res[1])],
                    "annotator": annotator,
                    "noised": noised,
                    "noise_parameters": merged_noise,
                    "depth_annotator": depth_annotator,
                })

        if not self._cameras:
            return
        import cv2  # noqa: F401
        self._cv2 = __import__("cv2")


        #self._scene_pcd = np.array([[0.0, 0.0, 0.0]])


    def get_record_depth_mode(self) -> str:
        """NONE / ALL。"""
        return getattr(self, "_record_depth_mode", "NONE")

    @property
    def depth_realtime(self) -> bool:
        return self._depth_realtime

    def _capture_frame_cameras_for_env0(self) -> dict:
        """Capture environment-zero camera poses in the recording convention."""
        import omni.usd
        from omni.usd import get_world_transform_matrix

        from common.base_utils.transform_utils import pose2mat, quat_wxyz_to_xyzw

        rotation_x_180 = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        frame_cameras = {}
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return frame_cameras
        env_path = self._env_path_for_origin
        if not env_path:
            return frame_cameras
        env_prim = stage.GetPrimAtPath(env_path)
        if not env_prim or not env_prim.IsValid():
            return frame_cameras
        try:
            env_world = get_world_transform_matrix(env_prim)
            env_origin = np.array(env_world.ExtractTranslation())
        except Exception as e:
            logger.error("Capture env origin failed: %s", e)
            return frame_cameras

        for item in self._cameras:
            if item.get("env_id", 0) != 0:
                continue
            prim_path = item.get("prim_path", "")
            prim_name = item.get("prim_name", "")
            if not prim_path or not prim_name:
                continue
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue
            try:
                cam_world = get_world_transform_matrix(prim)
                cam_pos_world = np.array(cam_world.ExtractTranslation())
                cam_pos_env = cam_pos_world - env_origin

                orient = cam_world.ExtractRotationQuat()
                quat_tf_xyzw = np.array([
                    orient.real,
                    orient.imaginary[2],
                    -orient.imaginary[1],
                    -orient.imaginary[0],
                ], dtype=np.float64)
                quat_tf_wxyz = np.array([
                    quat_tf_xyzw[3], quat_tf_xyzw[0], quat_tf_xyzw[1], quat_tf_xyzw[2]
                ])
                quat_xyzw = quat_wxyz_to_xyzw(quat_tf_wxyz)

                pose_mat = pose2mat((cam_pos_env, quat_xyzw)) @ rotation_x_180
                frame_cameras[prim_name] = {"pose": pose_mat.tolist()}
            except Exception as e:
                logger.error("Capture camera pose %s failed: %s", prim_name, e)
        return frame_cameras

    def _init_video_writers(self) -> None:
        """Create RGB and optional real-time depth writers unless in image mode."""
        self._video_writers = {}
        self._depth_writers = {}
        if self._image_mode:
            logger.info("Recorder: image_mode=True, skipping video writer creation")
            return
        for env_id, rec_path in self._output_dir_by_env.items():
            out_dir = os.path.join(rec_path, "observations", "videos")
            os.makedirs(out_dir, exist_ok=True)
            env_rgb_writers: dict[str, AsyncVideoWriter] = {}
            env_depth_writers: dict[str, AsyncVideoWriter] = {}
            for item in self._cameras:
                if item.get("env_id", 0) != env_id:
                    continue
                prim_name = item["prim_name"]
                res = item.get("resolution", [640, 480])
                w_res, h_res = int(res[0]), int(res[1])
                rgb_key = f"{prim_name}_color"
                if rgb_key not in env_rgb_writers:
                    out_path = os.path.join(out_dir, f"{rgb_key}.mp4")
                    try:
                        env_rgb_writers[rgb_key] = AsyncVideoWriter(
                            out_path, self._video_fps, (w_res, h_res),
                            mode="rgb", crf=self._video_crf, gop=self._video_gop,
                            preset=self._video_preset,
                        )
                    except Exception as e:
                        logger.warning("Failed to open RGB AsyncVideoWriter: %s", e)
                if item.get("depth_annotator") is not None and self._depth_realtime:
                    depth_key = f"{prim_name}_depth"
                    if depth_key not in env_depth_writers:
                        out_path = os.path.join(out_dir, f"{depth_key}.mp4")
                        try:
                            env_depth_writers[depth_key] = AsyncVideoWriter(
                                out_path, self._video_fps, (w_res, h_res),
                                mode="depth",
                            )
                        except Exception as e:
                            logger.warning("Failed to open depth AsyncVideoWriter: %s", e)
            if env_rgb_writers:
                self._video_writers[env_id] = env_rgb_writers
            if env_depth_writers:
                self._depth_writers[env_id] = env_depth_writers
        n_rgb = sum(len(w) for w in self._video_writers.values())
        n_depth = sum(len(w) for w in self._depth_writers.values())
        n_ffmpeg = n_rgb + n_depth
        self._writers_per_round = n_ffmpeg
        max_pool = n_ffmpeg * self._max_pool_multiplier
        logger.info(
            "Recorder: initialized %d AsyncVideoWriters (%d rgb + %d depth), "
            "pool=%d/%d retiring=%d, fps=%d crf=%d gop=%d preset=%s",
            n_ffmpeg, n_rgb, n_depth,
            n_ffmpeg + len(self._retiring_writers), max_pool,
            len(self._retiring_writers),
            self._video_fps, self._video_crf, self._video_gop, self._video_preset,
        )

    def save_frame(self, frame_id: int) -> tuple[float, float, int]:
        """Capture RGB/depth and return the legacy timing tuple."""
        from server.ros_publisher.camera_noiser import apply_noise_to_image

        if not self._cameras or not self._output_dir_by_env:
            return 0.0, 0.0, 0

        t0 = time.perf_counter()
        t_cameras_start = t0
        for item in self._cameras:
            env_id = item.get("env_id", 0)
            if env_id not in self._output_dir_by_env:
                continue
            prim_name = item["prim_name"]
            noised = item.get("noised", False)
            noise_params = dict(item.get("noise_parameters") or {})
            _t_cam_start = time.perf_counter()
            try:
                _t_get = time.perf_counter()
                data = item["annotator"].get_data(device="cuda" if noised else "cpu")
                _get_dt = (time.perf_counter() - _t_get) * 1000
                if data is None or len(data) == 0:
                    cam_key = (env_id, prim_name)
                    seen = self._empty_capture_count_by_cam.get(cam_key, 0)
                    self._empty_capture_count_by_cam[cam_key] = seen + 1
                    if seen == 0 or (seen + 1) % 100 == 0:
                        logger.warning(
                            "Capture frame camera %s (env %s) returned empty data "
                            "(type=%s) at frame %d; total empty so far for this camera=%d",
                            prim_name, env_id, type(data).__name__, frame_id, seen + 1,
                        )
                    continue
                _t_proc = time.perf_counter()
                if noised:
                    rgb = apply_noise_to_image(data[:, :, :3], **noise_params)
                else:
                    rgb = np.array(data)
                if rgb.ndim == 3 and rgb.shape[2] >= 3:
                    rgb = rgb[:, :, :3]
                if rgb.dtype != np.uint8:
                    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                _proc_dt = (time.perf_counter() - _t_proc) * 1000
                if self._image_mode:
                    frame_dir = os.path.join(
                        self._output_dir_by_env[env_id], "camera", str(frame_id),
                    )
                    os.makedirs(frame_dir, exist_ok=True)
                    img_path = os.path.join(frame_dir, f"{prim_name}_color.jpg")
                    _t = time.perf_counter()
                    bgr = rgb[:, :, ::-1].copy()
                    self._io_writer.submit(self._cv2.imwrite, img_path, bgr)
                    _rgb_dt = (time.perf_counter() - _t) * 1000
                else:
                    _rgb_dt = 0.0
                    rgb_key = f"{prim_name}_color"
                    env_writers = self._video_writers.get(env_id, {})
                    writer = env_writers.get(rgb_key)
                    if writer is not None:
                        writer.write(rgb)
                _t_depth = time.perf_counter()
                self._save_depth_frame(item, env_id, prim_name, frame_id)
                _depth_dt = (time.perf_counter() - _t_depth) * 1000
                _cam_total = (time.perf_counter() - _t_cam_start) * 1000
                logger.debug(
                    "IO cam %s: total=%.1fms (get=%.1f proc=%.1f rgb=%.1f depth=%.1f)",
                    prim_name, _cam_total, _get_dt, _proc_dt, _rgb_dt, _depth_dt,
                )
            except Exception as e:
                logger.warning(
                    "Capture frame camera %s (env %s) failed: %s",
                    prim_name,
                    item.get("env_id"),
                    e,
                )

        t_cameras_end = time.perf_counter()
        fc = self._capture_frame_cameras_for_env0()
        if fc:
            self._frame_cameras_accumulator.append(fc)
        capture_dt = time.perf_counter() - t0
        self._save_frame_count += 1
        logger.info(
            "save_frame breakdown: cameras=%.1fms total=%.1fms",
            (t_cameras_end - t_cameras_start) * 1000,
            capture_dt * 1000,
        )
        if self._save_frame_count % 5 == 0:
            self.log_resource_status()
        return capture_dt, 0.0, 0

    def _save_depth_frame(self, item: dict, env_id: int, prim_name: str, frame_id: int) -> None:
        depth_annotator = item.get("depth_annotator")
        if depth_annotator is None:
            return
        try:
            _t_get = time.perf_counter()
            depth_data = depth_annotator.get_data(device="cpu")
            _get_dt = (time.perf_counter() - _t_get) * 1000
            if depth_data is None or depth_data.size == 0:
                return
            depth = np.array(depth_data, dtype=np.float32)
            if self._save_depth_in_uint16:
                if "head" in prim_name.lower():
                    depth = depth * 1000.0
                else:
                    depth = depth * 10000.0
                depth_mm = (np.clip(depth, 0, 65535)).astype(np.uint16)
                if self._image_mode or not self._depth_realtime:
                    frame_dir = os.path.join(
                        self._output_dir_by_env[env_id], "camera", str(frame_id),
                    )
                    os.makedirs(frame_dir, exist_ok=True)
                    depth_path = os.path.join(frame_dir, f"{prim_name}_depth.png")
                    self._io_writer.submit(self._cv2.imwrite, depth_path, depth_mm)
                    logger.debug("IO depth %s: submit (get_data=%.1fms)", prim_name, _get_dt)
                else:
                    depth_key = f"{prim_name}_depth"
                    env_depth_writers = self._depth_writers.get(env_id, {})
                    depth_writer = env_depth_writers.get(depth_key)
                    if depth_writer is not None:
                        depth_writer.write(depth_mm)
            else:
                frame_dir = os.path.join(
                    self._output_dir_by_env[env_id], "camera", str(frame_id),
                )
                os.makedirs(frame_dir, exist_ok=True)
                depth_path = os.path.join(frame_dir, f"{prim_name}_depth.npz")
                self._io_writer.submit(np.savez_compressed, depth_path, depth=depth)
                logger.debug("IO depth %s: submit (get_data=%.1fms, npz_compressed)", prim_name, _get_dt)
        except Exception as e:
            logger.warning("Capture depth %s (env %s) failed: %s", prim_name, env_id, e)


    def release_writers(self) -> set[str]:
        self._retire_current_writers()
        if not self._retiring_writers:
            return set()
        logger.info("Recorder: releasing all %d retiring writers ...", len(self._retiring_writers))
        t0 = time.perf_counter()
        failed_paths: set[str] = set()
        empty_video_paths: set[str] = set()
        for i, (w, rec_path) in enumerate(self._retiring_writers):
            if w.is_alive:
                logger.info(
                    "Recorder: >>> waiting [%d/%d] %s qsize=%d frames=%d ...",
                    i + 1, len(self._retiring_writers), os.path.basename(w.out_path),
                    w.queue_size, w.frames_written,
                )
                w._thread.join()
            if w._error:
                logger.warning("Recorder: writer error %s: %s", w.out_path, w._error)
                if rec_path:
                    failed_paths.add(rec_path)
                continue
            if w.frames_written == 0:
                logger.warning(
                    "Recorder: writer wrote 0 frames (empty video): %s -> mark rec_path failed",
                    w.out_path,
                )
                if rec_path:
                    failed_paths.add(rec_path)
                    empty_video_paths.add(rec_path)
        dt = time.perf_counter() - t0
        n_total = len(self._retiring_writers)
        self._retiring_writers.clear()
        if failed_paths:
            logger.warning(
                "Recorder: %d/%d writers had errors or empty output, %d rec_paths failed "
                "(empty=%d): %s",
                len(failed_paths), n_total, len(failed_paths), len(empty_video_paths),
                [os.path.basename(p.rstrip("/")) for p in failed_paths],
            )
        else:
            logger.info("Recorder: all %d writers released successfully in %.1fs", n_total, dt)
        return failed_paths

    def drain(self) -> None:
        all_w = []
        for env_id, env_writers in self._video_writers.items():
            for name, w in env_writers.items():
                all_w.append((env_id, name, "rgb", w))
        for env_id, env_writers in self._depth_writers.items():
            for name, w in env_writers.items():
                all_w.append((env_id, name, "depth", w))
        if not all_w:
            return
        logger.info(
            "Recorder.drain: waiting %d writers, queue status: %s",
            len(all_w),
            " | ".join(f"env{eid}/{n}({m}) q={w.queue_size}" for eid, n, m, w in all_w),
        )
        t0 = time.perf_counter()
        for i, (env_id, name, mode, w) in enumerate(all_w):
            if w.queue_size > 0:
                logger.info(
                    "Recorder.drain: >>> flushing [%d/%d] env=%d %s (%s) qsize=%d ...",
                    i + 1, len(all_w), env_id, name, mode, w.queue_size,
                )
            tw0 = time.perf_counter()
            try:
                w.flush()
            except Exception as e:
                logger.warning("Recorder.drain: flush failed for env=%d %s: %s", env_id, name, e)
            tw1 = time.perf_counter()
            if tw1 - tw0 > 0.5:
                logger.info(
                    "Recorder.drain: <<< flushed [%d/%d] env=%d %s in %.1fs",
                    i + 1, len(all_w), env_id, name, tw1 - tw0,
                )
        dt = time.perf_counter() - t0
        logger.info("Recorder.drain: all %d writers drained in %.1fs", len(all_w), dt)

    @property
    def total_pool_wait_sec(self) -> float:
        return self._total_pool_wait_sec

    @property
    def video_preset(self) -> str:
        return self._video_preset

    def log_resource_status(self) -> None:
        """Log ffmpeg processes, queue size, written frames, memory, and CPU."""
        rgb_info = []
        for env_id, env_writers in self._video_writers.items():
            for name, w in env_writers.items():
                rgb_info.append((env_id, name, w.queue_size, w.queue_maxsize, w.frames_written))
        depth_info = []
        for env_id, env_writers in self._depth_writers.items():
            for name, w in env_writers.items():
                depth_info.append((env_id, name, w.queue_size, w.queue_maxsize, w.frames_written))
        n_total = len(rgb_info) + len(depth_info)
        max_q = max((q for _, _, q, _, _ in rgb_info + depth_info), default=0)
        total_frames = sum(f for _, _, _, _, f in rgb_info + depth_info)

        try:
            proc_mem = self._process.memory_info()
            proc_rss_mb = proc_mem.rss / (1024 * 1024)
            proc_cpu = self._process.cpu_percent(interval=0)
        except Exception:
            proc_rss_mb = 0.0
            proc_cpu = 0.0

        ffmpeg_rss_mb = 0.0
        try:
            for child in self._process.children(recursive=True):
                if "ffmpeg" in (child.name() or ""):
                    ffmpeg_rss_mb += child.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        vm = psutil.virtual_memory()
        sys_mem_used_mb = vm.used / (1024 * 1024)
        sys_mem_total_mb = vm.total / (1024 * 1024)
        sys_mem_pct = vm.percent
        sys_cpu_pct = psutil.cpu_percent(interval=0)

        logger.info(
            "Recorder status [frame %d]: %d ffmpeg procs (%d rgb + %d depth), "
            "max_queue=%d, total_frames_written=%d | "
            "io_writer inflight=%d qsize=%d | "
            "proc RSS=%.0fMB cpu=%.1f%% | ffmpeg RSS=%.0fMB | "
            "sys mem=%.0f/%.0fMB(%.1f%%) cpu=%.1f%%",
            self._save_frame_count,
            n_total, len(rgb_info), len(depth_info),
            max_q, total_frames,
            self._io_writer.inflight, self._io_writer.queue_size,
            proc_rss_mb, proc_cpu,
            ffmpeg_rss_mb,
            sys_mem_used_mb, sys_mem_total_mb, sys_mem_pct,
            sys_cpu_pct,
        )

    def write_recording_info(self, num_frames: int) -> None:
        """Update recording_info.json and recording_env_paths.json per environment."""
        source_path = os.path.join(self._source_input_dir, "recording_info.json") if self._source_input_dir else ""
        base_info: dict = {}
        if source_path and os.path.isfile(source_path):
            try:
                with open(source_path, "r", encoding="utf-8") as f:
                    base_info = json.load(f)
            except Exception as e:
                logger.warning("Failed to load source recording_info %s: %s", source_path, e)

        cameras_by_env = {}
        for item in self._cameras:
            env_id = item.get("env_id", 0)
            if env_id not in cameras_by_env:
                cameras_by_env[env_id] = []
            cameras_by_env[env_id].append(item)
        for env_id, rec_path in self._output_dir_by_env.items():
            camera_info = {}
            for item in cameras_by_env.get(env_id, []):
                name = item["prim_name"]
                if self._image_mode:
                    out = {
                        "rgb": "camera/{frame_num}/" + f"{name}_color.jpg",
                    }
                else:
                    out = {
                        "rgb": "camera/{frame_num}/" + f"{name}_color.jpg",
                        "video": f"{name}_color.mp4",
                    }
                if item.get("depth_annotator") is not None:
                    depth_ext = "png" if self._save_depth_in_uint16 else "npz"
                    out["depth"] = "camera/{frame_num}/" + f"{name}_depth.{depth_ext}"
                    if not self._image_mode and self._depth_realtime:
                        out["depth_video"] = f"{name}_depth.mp4"
                intrinsic = get_intrinsic_for_camera(self._robot_name, name)
                camera_info[name] = {"intrinsic": intrinsic, "output": out}
            info = dict(base_info)
            for removed_key in ("lidar_info", "mask_prim_list", "record_semantic", "record_bbox"):
                info.pop(removed_key, None)
            info.update({
                "task_name": self._record_config.get("task_name", ""),
                "fps": self._record_config["fps"],
                "camera_info": camera_info,
                "frame_count": num_frames,
                "env_id": env_id,
                "input_dir": self._source_input_dir,
            })
            if self._frame_cameras_accumulator:
                info["frame_cameras"] = self._frame_cameras_accumulator
            path = os.path.join(rec_path, "recording_info.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=4)
            logger.info("Wrote %s", path)
        if self._output_dir_by_env:
            first_path = self._output_dir_by_env.get(0)
            if first_path:
                index_path = os.path.join(first_path, "recording_env_paths.json")
                with open(index_path, "w", encoding="utf-8") as f:
                    json.dump(self._output_dir_by_env, f, indent=2)
                logger.info("Wrote %s", index_path)

    def get_camera_image_names(self) -> list[str]:
        """Return RGB image names used by ffmpeg, such as head_color."""
        names = list(dict.fromkeys(item.get("prim_name", "") for item in self._cameras))
        return [f"{n}_color" for n in names if n]

    def get_camera_depth_image_names(self) -> list[str]:
        """Return depth camera names used for video conversion."""
        seen = set()
        out = []
        for item in self._cameras:
            if item.get("depth_annotator") is None:
                continue
            name = item.get("prim_name", "")
            if name and name not in seen:
                seen.add(name)
                out.append(f"{name}_depth")
        return out

    def shutdown(self) -> None:
        """Release recording resources."""
        self._io_writer.flush()
        self._io_writer.shutdown()
        self.release_writers()
