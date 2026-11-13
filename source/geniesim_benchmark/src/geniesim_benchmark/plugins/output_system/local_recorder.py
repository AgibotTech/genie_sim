# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""In-process video recorder.

Replaces the older ROS rosbag → external extractor pipeline. Camera frames are
pulled directly from the simulator's image cache, optionally annotated with the
active task instruction, and piped into per-camera ffmpeg subprocesses (rawvideo
bgr24 → libvpx-vp9 webm). On the same sampled ticks a state sampler
(robot base pose / velocity, joint state) is appended to ``state.jsonl`` so a
recording can be replayed, chassis included.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import cv2
import numpy as np

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


DEFAULT_FPS = 30
DEFAULT_FFMPEG_THREADS = 2
DEFAULT_QUEUE_DEPTH = 128
STATE_SCHEMA = "geniesim.local-recorder-state.v1"
INSTRUCTION_OVERLAY_CAMERA = "head"


def _overlay_instruction_text(cv_image, text, font_scale=0.6, thickness=1, margin=10):
    """Render instruction text on the top of the image with a semi-transparent background.

    Bakes the task-instruction overlay into head.webm at record time.
    """
    if not text:
        return cv_image

    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = cv_image.shape[:2]
    max_text_width = img_w - 2 * margin

    (char_w, _), _ = cv2.getTextSize("A", font, font_scale, thickness)
    max_chars_per_line = max(1, int(max_text_width / max(char_w, 1)))

    lines = [text[i : i + max_chars_per_line] for i in range(0, len(text), max_chars_per_line)]

    line_heights = []
    for line in lines:
        (_, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        line_heights.append(th + baseline)

    total_text_height = sum(line_heights) + margin * (len(lines) + 1)
    bg_height = min(total_text_height, img_h // 3)

    overlay = cv_image.copy()
    cv2.rectangle(overlay, (0, 0), (img_w, bg_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, cv_image, 0.4, 0, cv_image)

    y = margin
    for i, line in enumerate(lines):
        y += line_heights[i]
        if y > bg_height - margin:
            break
        cv2.putText(cv_image, line, (margin, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += margin // 2

    return cv_image


def _overlay_episode_index(cv_image, episode_idx, font_scale=0.6, thickness=1, margin=10):
    """Render the recording's zero-based episode index in the bottom-right corner."""
    text = f"Episode: {episode_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = cv_image.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    if text_w > max(1, img_w - 2 * margin):
        font_scale *= max(1, img_w - 2 * margin) / text_w
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, img_w - margin - text_w)
    y = max(text_h, img_h - margin - baseline)
    padding = margin // 2

    overlay = cv_image.copy()
    cv2.rectangle(
        overlay,
        (max(0, x - padding), max(0, y - text_h - padding)),
        (min(img_w - 1, x + text_w + padding), min(img_h - 1, y + baseline + padding)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, cv_image, 0.4, 0, cv_image)
    cv2.putText(cv_image, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return cv_image


def resolve_camera_name(camera_id: str, prim_path: str = "", world_camera: bool = False) -> str | None:
    """Map a simulator camera identifier to the canonical recording name.

    Returns None for cameras that should not be recorded (depth, semantic, …).
    Naming policy follows api_core.process_camera_info_list so output filenames
    stay consistent.
    """
    if not camera_id:
        return None

    name = camera_id.lower()
    prim_lower = (prim_path or "").lower()

    # Skip non-RGB streams.
    if "depth" in name or "semantic" in name or "fisheye" in prim_lower or "fisheye" in name:
        return None

    if "head_front" in name or "head_front" in prim_lower:
        return "head"
    if "head_left" in name or "head_right" in name:
        # Stereo heads are not part of the canonical 4-camera contract.
        return None
    if "head" in name:
        return "head"
    if "right" in name:
        return "hand_right"
    if "left" in name:
        return "hand_left"
    if "top" in name:
        # Top-of-scene world camera.
        return "world_img"
    if "world" in name:
        return "world_img"
    if world_camera and prim_lower.startswith("/world/") and prim_lower.count("/") == 2:
        # Third-person camera placed at the stage root by the background
        # (e.g. /World/Camera, registered with camera_id "camera").
        return "world_img"
    if name == "camera_rgb" or "camera_rgb" in name:
        return "world_img"
    return None


class _CameraEncoder:
    """Owns one ffmpeg subprocess and the worker thread feeding it."""

    def __init__(self, camera_name: str, output_path: str, width: int, height: int, fps: int, ffmpeg_threads: int):
        self.camera_name = camera_name
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.queue: Queue = Queue(maxsize=DEFAULT_QUEUE_DEPTH)
        # VP9 realtime: -deadline realtime + -cpu-used 8 is the fastest mode;
        # -row-mt + -tile-columns parallelize within one encoder; CRF needs -b:v 0.
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-vsync",
                "0",
                "-c:v",
                "libvpx-vp9",
                "-deadline",
                "realtime",
                "-cpu-used",
                "8",
                "-row-mt",
                "1",
                "-tile-columns",
                "2",
                "-b:v",
                "0",
                "-crf",
                "32",
                "-threads",
                str(ffmpeg_threads),
                "-an",
                "-loglevel",
                "error",
                output_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            # Discard stderr: ffmpeg runs with `-loglevel error`, and an undrained
            # PIPE can fill the OS buffer on a long encode and deadlock the writer.
            stderr=subprocess.DEVNULL,
        )
        self._stop = threading.Event()
        self.frames_written = 0
        self.frames_dropped = 0
        self._worker = threading.Thread(target=self._drain, name=f"local-recorder-{camera_name}", daemon=True)
        self._worker.start()

    def submit(self, bgr_frame: np.ndarray) -> None:
        if self._stop.is_set():
            return
        try:
            self.queue.put_nowait(bgr_frame)
        except Exception:
            # Drop the frame if the encoder can't keep up — better than blocking sim.
            # Log once per camera; total drops surfaced in close().
            if self.frames_dropped == 0:
                logger.warning(f"[local_recorder] queue full for {self.camera_name}, dropping frames")
            self.frames_dropped += 1

    def _drain(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frame = self.queue.get(timeout=0.2)
                except Empty:
                    continue
                if frame is None:
                    break
                try:
                    self._proc.stdin.write(frame.tobytes())
                    self.frames_written += 1
                except (BrokenPipeError, ValueError):
                    logger.warning(f"[local_recorder] ffmpeg pipe broken for {self.camera_name}")
                    break
        except Exception as exc:
            logger.error(f"[local_recorder] worker error for {self.camera_name}: {exc}")

    def close(self) -> None:
        self._stop.set()
        # Drain remaining queued frames.
        while True:
            try:
                frame = self.queue.get_nowait()
            except Empty:
                break
            if frame is None:
                continue
            try:
                self._proc.stdin.write(frame.tobytes())
                self.frames_written += 1
            except Exception:
                break
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            logger.warning(f"[local_recorder] ffmpeg for {self.camera_name} did not exit, killing")
            self._proc.kill()
            self._proc.wait()
        if self._worker.is_alive():
            self._worker.join(timeout=5)


class LocalRecorder:
    """In-process recorder for benchmark / teleop episodes.

    Lifecycle:
      r = LocalRecorder()
      r.set_sub_task_name(sub_task_name)
      r.start(output_root=..., episode_idx=...)         # opens ffmpeg pipes
      r.update_instruction(text)                        # any time during recording
      r.write_frames(robot_interface, current_step_index, state_sampler)  # called per sim tick
      r.stop(episode_idx=...)                           # closes pipes, writes instructions_raw.json
      r.concat_all(final_output_dir=...)                # cross-episode concat at the end of the run

    state.jsonl: a header line ``{"schema", "episode_idx", "fps", "every_n_frame"}``
    followed by one line per sampled tick ``{"frame_index", "step", "sim_time", **sample}``.
    ``frame_index`` counts sampled ticks from 0, i.e. the video frame index when
    every camera delivered that tick.
    """

    def __init__(
        self,
        fps: int = DEFAULT_FPS,
        ffmpeg_threads: int = DEFAULT_FFMPEG_THREADS,
        record_state: bool = True,
        record_world_camera: bool = False,
    ):
        self._fps = fps
        self._ffmpeg_threads = ffmpeg_threads
        # Replay state is enabled by default; world-camera recording is opt-in.
        self._record_state = record_state
        self._record_world_camera = record_world_camera

        self._lock = threading.RLock()
        self._is_recording = False
        self._encoders: dict[str, _CameraEncoder] = {}
        self._camera_id_to_name: dict[str, str] = {}
        self._record_every_n_frame = 1
        self._step_counter = 0

        self._sub_task_name = ""
        self._current_instruction = ""
        self._instruction_segments: list[dict] = []
        self._recording_start_time: float | None = None

        self._state_file = None
        self._state_error_logged = False

        self._current_bag_path: str | None = None
        self._current_episode_idx: int = 0
        self._processed_bags: list[str] = []
        self._checkpoint_run_id: str | None = None
        self._last_stopped_bag = None

    def restore_checkpoint(self, run_id, bag_paths):
        """Install ordered, committed segments before starting new episodes."""
        with self._lock:
            if self._is_recording:
                raise RuntimeError("Cannot restore an active recording")
            self._checkpoint_run_id = run_id
            self._processed_bags = list(bag_paths)
            self._last_stopped_bag = None

    # ----- Configuration --------------------------------------------------

    def set_sub_task_name(self, name: str) -> None:
        with self._lock:
            self._sub_task_name = name or ""

    def update_instruction(self, text: str) -> None:
        text = (text or "").strip()
        with self._lock:
            if text == self._current_instruction:
                return
            now = time.time()
            if self._instruction_segments and self._instruction_segments[-1].get("end_time") is None:
                self._instruction_segments[-1]["end_time"] = now
            if text:
                self._instruction_segments.append({"instruction": text, "start_time": now, "end_time": None})
            self._current_instruction = text

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def current_bag_path(self) -> str | None:
        return self._current_bag_path

    # ----- Episode lifecycle ---------------------------------------------

    def start(
        self,
        output_root: str,
        episode_idx: int,
        camera_specs: list[dict],
        sub_task_name: str | None = None,
        fps: int | None = None,
    ) -> str | None:
        """Open ffmpeg pipes for one episode.

        camera_specs: [{"camera_id": str, "prim_path": str, "width": int, "height": int,
                       "every_n_frame": int}, ...]  — usually built from
        ``robot_interface.parameters``.
        Returns the bag directory path, or None if no recordable cameras.
        """
        with self._lock:
            if self._is_recording:
                logger.warning("[local_recorder] start() called while already recording")
                return self._current_bag_path

            if sub_task_name is not None:
                self._sub_task_name = sub_task_name
            if fps is not None:
                self._fps = fps

            sub_task = self._sub_task_name or "unknown"
            recording_dir = os.path.join(output_root, sub_task)
            os.makedirs(recording_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            short_uuid = uuid.uuid4().hex[:8]
            bag_name = f"recording_{episode_idx:04d}_{timestamp}_{short_uuid}"
            bag_path = os.path.join(recording_dir, bag_name)
            video_dir = os.path.join(bag_path, "video")
            os.makedirs(video_dir, exist_ok=True)
            if self._checkpoint_run_id is not None:
                from .checkpoint import atomic_write_json

                atomic_write_json(Path(bag_path) / "checkpoint_run.json", {"run_id": self._checkpoint_run_id})

            # Resolve canonical camera names + dedupe (multiple sim cams may
            # map to the same canonical name; first one wins).
            self._encoders.clear()
            self._camera_id_to_name.clear()
            every_ns: list[int] = []
            for spec in camera_specs:
                cam_id = spec.get("camera_id", "")
                prim = spec.get("prim_path", "")
                cname = resolve_camera_name(cam_id, prim, self._record_world_camera)
                if cname is None:
                    continue
                if cname in self._encoders:
                    continue
                width = int(spec["width"])
                height = int(spec["height"])
                output_path = os.path.join(video_dir, f"{cname}.webm")
                encoder = _CameraEncoder(
                    camera_name=cname,
                    output_path=output_path,
                    width=width,
                    height=height,
                    fps=self._fps,
                    ffmpeg_threads=self._ffmpeg_threads,
                )
                self._encoders[cname] = encoder
                self._camera_id_to_name[cam_id] = cname
                every_ns.append(max(1, int(spec.get("every_n_frame", 1))))

            if not self._encoders:
                logger.warning("[local_recorder] no recordable cameras, skipping episode")
                shutil.rmtree(bag_path, ignore_errors=True)
                return None

            self._record_every_n_frame = min(every_ns) if every_ns else 1
            self._step_counter = 0
            self._state_error_logged = False
            self._state_file = None
            if self._record_state:
                self._state_file = open(os.path.join(bag_path, "state.jsonl"), "w", encoding="utf-8")
                self._state_file.write(
                    json.dumps(
                        {
                            "schema": STATE_SCHEMA,
                            "episode_idx": episode_idx,
                            "fps": self._fps,
                            "every_n_frame": self._record_every_n_frame,
                        }
                    )
                    + "\n"
                )
            self._current_bag_path = bag_path
            self._current_episode_idx = episode_idx
            self._recording_start_time = time.time()

            # Reset instruction segment timeline; if there's an active
            # instruction, open its first segment at the recording start.
            if self._current_instruction:
                self._instruction_segments = [
                    {
                        "instruction": self._current_instruction,
                        "start_time": self._recording_start_time,
                        "end_time": None,
                    }
                ]
            else:
                self._instruction_segments = []

            self._is_recording = True
            logger.info(
                f"[local_recorder] started episode {episode_idx} → {bag_path} "
                f"({list(self._encoders)}, every_n={self._record_every_n_frame})"
            )
            return bag_path

    def write_frames(self, robot_interface, current_step_index: int, state_sampler=None, sim_time=None) -> None:
        """Pull the latest frames out of robot_interface and queue them for encoding.

        ``robot_interface._img_data_cache[cam_id]`` is RGBA uint8 — convert to
        BGR before submitting to ffmpeg. The head camera also gets the active
        instruction and episode index overlaid. ``state_sampler`` is called only on sampled ticks
        and must return a JSON-serialisable dict.
        """
        if not self._is_recording:
            return
        if current_step_index % self._record_every_n_frame != 0:
            return

        cache = getattr(robot_interface, "_img_data_cache", None)
        if not cache:
            return

        with self._lock:
            instruction = self._current_instruction
            episode_idx = self._current_episode_idx
            cam_map = dict(self._camera_id_to_name)
            encoders = dict(self._encoders)

        for cam_id, cam_name in cam_map.items():
            encoder = encoders.get(cam_name)
            if encoder is None:
                continue
            rgba = cache.get(cam_id)
            if rgba is None or rgba.size == 0:
                continue
            # RGBA → BGR (drop alpha, swap R/B). copy() to detach from sim
            # buffer that gets overwritten next tick.
            bgr = np.ascontiguousarray(rgba[..., 2::-1])
            if cam_name == INSTRUCTION_OVERLAY_CAMERA:
                _overlay_instruction_text(bgr, instruction)
                _overlay_episode_index(bgr, episode_idx)
            encoder.submit(bgr)

        if state_sampler is not None:
            self._write_state(state_sampler, current_step_index, sim_time)

        self._step_counter += 1

    def _write_state(self, state_sampler, current_step_index, sim_time) -> None:
        with self._lock:
            state_file = self._state_file
        if state_file is None:
            return
        try:
            record = {"frame_index": self._step_counter, "step": current_step_index, "sim_time": sim_time}
            record.update(state_sampler())
            state_file.write(json.dumps(record) + "\n")
        except Exception as exc:
            # A missing state sample must not stop the video recording.
            if not self._state_error_logged:
                logger.warning(f"[local_recorder] state sample failed, skipping: {exc}")
                self._state_error_logged = True

    def stop(self, episode_idx: int | None = None, discard: bool = False) -> str | None:
        """Close ffmpeg pipes and write instructions_raw.json. Returns bag_path."""
        with self._lock:
            if not self._is_recording:
                if self._checkpoint_run_id is not None and self._last_stopped_bag is not None:
                    index, path = self._last_stopped_bag
                    if episode_idx == index and not discard:
                        return path
                return None
            encoders = list(self._encoders.values())
            self._encoders.clear()
            cam_map = dict(self._camera_id_to_name)
            self._camera_id_to_name.clear()
            state_file, self._state_file = self._state_file, None
            bag_path = self._current_bag_path
            recording_start = self._recording_start_time
            stop_time = time.time()
            if self._instruction_segments and self._instruction_segments[-1].get("end_time") is None:
                self._instruction_segments[-1]["end_time"] = stop_time
            instruction_segments = list(self._instruction_segments)
            self._instruction_segments = []
            self._is_recording = False
            self._current_bag_path = None
            self._recording_start_time = None
            if episode_idx is not None:
                self._current_episode_idx = episode_idx

        failures = []
        for encoder in encoders:
            try:
                encoder.close()
                if self._checkpoint_run_id is not None:
                    if encoder._proc.returncode != 0 or encoder.frames_written == 0:
                        failures.append(f"{encoder.camera_name}: encoder failed or produced no frames")
            except Exception as exc:
                failures.append(f"{encoder.camera_name}: {exc}")
                logger.error(f"[local_recorder] failed to close encoder for {encoder.camera_name}: {exc}")
        if state_file is not None:
            state_file.close()
        if failures and self._checkpoint_run_id is not None and not discard:
            raise RuntimeError("Episode recording failed: " + "; ".join(failures))

        if bag_path is None:
            return None

        if discard:
            shutil.rmtree(bag_path, ignore_errors=True)
            logger.info(f"[local_recorder] discarded episode {episode_idx} → {bag_path}")
            return None

        if instruction_segments and recording_start is not None:
            self._save_instructions_raw(bag_path, instruction_segments, recording_start, stop_time)

        with self._lock:
            self._processed_bags.append(bag_path)
            self._last_stopped_bag = (self._current_episode_idx, bag_path)

        total_frames = sum(getattr(e, "frames_written", 0) for e in encoders)
        total_dropped = sum(getattr(e, "frames_dropped", 0) for e in encoders)
        drop_suffix = f", dropped={total_dropped}" if total_dropped else ""
        logger.info(
            f"[local_recorder] stopped episode {episode_idx} → {bag_path} "
            f"({total_frames} frames across {len(encoders)} cameras, "
            f"duration={stop_time - (recording_start or stop_time):.2f}s{drop_suffix})"
        )
        del cam_map
        return bag_path

    @staticmethod
    def _save_instructions_raw(bag_path, instruction_segments, recording_start, recording_stop):
        recording_duration = recording_stop - recording_start
        segments = []
        for seg in instruction_segments:
            start_ratio = (seg["start_time"] - recording_start) / recording_duration if recording_duration > 0 else 0.0
            end_ratio = (
                (seg["end_time"] - recording_start) / recording_duration
                if (seg.get("end_time") is not None and recording_duration > 0)
                else None
            )
            segments.append(
                {
                    "instruction": seg["instruction"],
                    "start_ratio": max(0.0, round(start_ratio, 6)),
                    "end_ratio": min(1.0, round(end_ratio, 6)) if end_ratio is not None else None,
                }
            )
        os.makedirs(bag_path, exist_ok=True)
        json_path = os.path.join(bag_path, "instructions_raw.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"segments": segments, "recording_duration": round(recording_duration, 3)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.info(f"[local_recorder] saved {json_path}")

    # ----- Cross-episode concat ------------------------------------------

    def concat_all(self, final_output_dir: str | None = None, *, preserve_bags=False):
        """Finalize per-episode webms into the parent recording directory.

        For every camera:
          - if there are 2+ episode segments, ffmpeg-concat them into
            ``<parent>/<camera>.webm``;
          - if there is exactly 1 segment, just move it to
            ``<parent>/<camera>.webm`` (no re-encode).
        Per-episode bag directories are removed afterwards so the final layout
        matches the contract regardless of whether concat actually ran.
        """
        if preserve_bags:
            return self._concat_checkpoint_recordings()
        with self._lock:
            bag_paths = list(self._processed_bags)

        if not bag_paths:
            logger.info("[local_recorder] concat_all: no processed bags")
            return

        output_dir = Path(bag_paths[0]).parent

        camera_videos: dict[str, list[str]] = {}
        for bag_path in bag_paths:
            video_dir = Path(bag_path) / "video"
            if not video_dir.is_dir():
                continue
            for vf in sorted(video_dir.glob("*.webm")):
                camera_videos.setdefault(vf.stem, []).append(str(vf))

        for camera_name, video_list in camera_videos.items():
            concat_output = output_dir / f"{camera_name}.webm"
            if len(video_list) == 1:
                # Single-episode case: just move the segment up one level.
                try:
                    if concat_output.exists():
                        concat_output.unlink()
                    shutil.move(video_list[0], concat_output)
                    logger.info(f"[local_recorder] moved single segment → {concat_output}")
                except Exception as exc:
                    logger.error(f"[local_recorder] move failed for {camera_name}: {exc}")
                continue

            concat_file = output_dir / f"{camera_name}_concat_list.txt"
            concat_file.write_text("\n".join(f"file '{v}'" for v in video_list))
            try:
                logger.info(f"[local_recorder] concat {len(video_list)} → {concat_output}")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_file),
                        "-c",
                        "copy",
                        str(concat_output),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                for v in video_list:
                    Path(v).unlink(missing_ok=True)
                concat_file.unlink(missing_ok=True)
            except subprocess.CalledProcessError as exc:
                logger.error(
                    f"[local_recorder] concat failed for {camera_name}: "
                    f"{exc.stderr.decode() if exc.stderr else exc}"
                )
            except Exception as exc:
                logger.error(f"[local_recorder] concat error for {camera_name}: {exc}")

        # Per-episode state streams are not concatenated: each keeps its bag
        # name (episode index + timestamp + uuid) so episodes stay separable.
        for bag_path in bag_paths:
            state_path = Path(bag_path) / "state.jsonl"
            if state_path.is_file():
                try:
                    shutil.move(str(state_path), output_dir / f"{Path(bag_path).name}.state.jsonl")
                except Exception as exc:
                    logger.error(f"[local_recorder] could not keep {state_path}: {exc}")

        for bag_path in bag_paths:
            try:
                shutil.rmtree(bag_path)
                logger.info(f"[local_recorder] removed {bag_path}")
            except Exception as exc:
                logger.warning(f"[local_recorder] could not remove {bag_path}: {exc}")

        with self._lock:
            self._processed_bags.clear()

        if final_output_dir:
            self._move_to_final(output_dir, final_output_dir)

    @staticmethod
    def _publish_artifact(temporary, destination, mode):
        os.chmod(temporary, mode & 0o777)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory = os.open(Path(destination).parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    @staticmethod
    def _copy_artifact(source, destination):
        destination = Path(destination)
        if Path(source).resolve() == destination.resolve():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.stem}.", suffix=destination.suffix, dir=destination.parent
        )
        os.close(fd)
        try:
            shutil.copyfile(source, temporary)
            LocalRecorder._publish_artifact(temporary, destination, Path(source).stat().st_mode)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def publish_checkpoint_artifacts(artifacts, output_dir, final_output_dir):
        """Export media without moving the result file or checkpoint directory."""
        if final_output_dir:
            target = Path(final_output_dir) / Path(output_dir).name
            for path in artifacts:
                LocalRecorder._copy_artifact(path, target / Path(path).name)

    def _concat_checkpoint_recordings(self):
        """Publish each final video atomically; keep sources until manifest commit."""
        with self._lock:
            bags = [Path(p) for p in self._processed_bags]
        if not bags:
            return []
        output_dir = bags[0].parent
        videos = {}
        expected = None
        for bag in bags:
            camera_files = {p.stem: p for p in (bag / "video").glob("*.webm") if p.stat().st_size > 0}
            if not camera_files or (expected is not None and camera_files.keys() != expected):
                raise RuntimeError(f"Incomplete committed recording: {bag}")
            expected = camera_files.keys()
            for name, path in sorted(camera_files.items()):
                videos.setdefault(name, []).append(path)
        artifacts = []
        for name, segments in videos.items():
            destination = output_dir / f"{name}.webm"
            if len(segments) == 1:
                self._copy_artifact(segments[0], destination)
            else:
                fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".webm", dir=output_dir)
                os.close(fd)
                listing = Path(temporary).with_suffix(".txt")
                try:
                    # ffconcat's single-quote escaping differs from JSON/shell quoting.
                    listing.write_text("\n".join("file '" + str(p).replace("'", "'\\''") + "'" for p in segments))
                    subprocess.run(
                        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", temporary],
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                    )
                    self._publish_artifact(temporary, destination, segments[0].stat().st_mode)
                finally:
                    listing.unlink(missing_ok=True)
                    Path(temporary).unlink(missing_ok=True)
            artifacts.append(destination)
        for bag in bags:
            state = bag / "state.jsonl"
            if state.is_file():
                destination = output_dir / f"{bag.name}.state.jsonl"
                self._copy_artifact(state, destination)
                artifacts.append(destination)
        return artifacts

    def _move_to_final(self, src_dir: Path, final_output_dir: str) -> None:
        try:
            os.makedirs(final_output_dir, exist_ok=True)
            sub_task = src_dir.name
            dst_dir = os.path.join(final_output_dir, sub_task)
            if os.path.exists(dst_dir):
                for entry in os.scandir(str(src_dir)):
                    dst_entry = os.path.join(dst_dir, entry.name)
                    if os.path.exists(dst_entry):
                        continue
                    shutil.move(entry.path, dst_entry)
                shutil.rmtree(str(src_dir), ignore_errors=True)
            else:
                shutil.move(str(src_dir), dst_dir)
            logger.info(f"[local_recorder] moved → {dst_dir}")
        except Exception as exc:
            logger.error(f"[local_recorder] move-to-final failed: {exc}")
