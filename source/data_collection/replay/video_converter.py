"""
Submit ffmpeg conversion jobs asynchronously and wait for them before exit.
A background reporter can print job status even while conversion continues.
"""
import os
import subprocess
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

import numpy as np

from common.base_utils.logger import logger

DEFAULT_STATUS_INTERVAL_SEC = 10


def _cmd_to_shell(cmd: list[str]) -> str:
    """Format argv as a shell command with quoted arguments."""
    parts: list[str] = []
    for a in cmd:
        if not a or any(c in a for c in " \n\t\"$\\`*?[]|;&<>"):
            a = "'" + a.replace("'", "'\"'\"'") + "'"
        parts.append(a)
    return " ".join(parts)


def _run_ffmpeg_rgb(
    rec_path: str,
    image_name: str,
    fps: int = 30,
    preset: str = "medium",
    crf: int = 24,
    gop: int = 8,
    pix_fmt: str = "yuv420p",
) -> None:
    """Convert one RGB image sequence to MP4 with image2 input."""
    rec_name = os.path.basename(rec_path.rstrip("/")) or rec_path
    logger.info("VideoConverter: [start] %s/%s", rec_name, image_name)
    camera_type = "jpg"
    input_pattern = os.path.join(rec_path, "camera", "%d", f"{image_name}.{camera_type}")
    out_dir = os.path.join(rec_path, "observations", "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{image_name}.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "image2",
        "-threads", "4",
        "-r", str(fps),
        "-i", input_pattern,
        "-vcodec", "libx265",
        "-pix_fmt", pix_fmt,
        "-keyint_min", "8",
        "-sc_threshold", "0",
        "-vf", "setpts=N/(30*TB)",
        "-bf", "0",
        "-crf", str(crf),
        "-g", str(gop),
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command {_cmd_to_shell(e.cmd)} failed with exit code {e.returncode}"
        ) from e
    logger.info("VideoConverter: [done] %s/%s -> %s", rec_name, image_name, out_path)


def _run_ffmpeg_depth(
    rec_path: str,
    image_name: str,
    fps: int = 30,
) -> None:
    """Convert one depth PNG sequence to MP4 using PNG encoding."""
    rec_name = os.path.basename(rec_path.rstrip("/")) or rec_path
    logger.info("VideoConverter: [start depth] %s/%s", rec_name, image_name)
    input_pattern = os.path.join(rec_path, "camera", "%d", f"{image_name}.png")
    out_dir = os.path.join(rec_path, "observations", "videos")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{image_name}.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", input_pattern,
        "-c:v", "png",
        "-pix_fmt", "gray16le",
        "-r", str(fps),
        "-movflags", "+faststart",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Command {_cmd_to_shell(e.cmd)} failed with exit code {e.returncode}"
        ) from e
    logger.info("VideoConverter: [done depth] %s/%s -> %s", rec_name, image_name, out_path)


def _encode_depth_from_memory(
    frames: list[np.ndarray],
    out_path: str,
    fps: int = 30,
) -> None:
    """Pipe in-memory depth frames to ffmpeg for lossless gray16le MP4."""
    import cv2

    if not frames:
        raise ValueError("No depth frames to encode")
    h, w = frames[0].shape[:2]
    rec_name = os.path.basename(os.path.dirname(os.path.dirname(out_path)))
    image_name = os.path.splitext(os.path.basename(out_path))[0]
    logger.info("VideoConverter: [start depth memory] %s/%s (%d frames %dx%d)", rec_name, image_name, len(frames), w, h)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "image2pipe",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-c:v", "png",
        "-pix_fmt", "gray16le",
        "-r", str(fps),
        "-movflags", "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            _, buf = cv2.imencode(".png", frame)
            proc.stdin.write(buf.tobytes())
        proc.stdin.close()
        _, stderr = proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg depth failed (rc={proc.returncode}): {stderr.decode('utf-8', errors='replace')[:200]}"
            )
    except Exception:
        proc.kill()
        proc.wait()
        raise
    logger.info("VideoConverter: [done depth memory] %s/%s -> %s", rec_name, image_name, out_path)


class VideoConverter:
    """
    Submit ffmpeg jobs from all rounds into one task list.
    Each camera in each environment gets one job per batch.
    """

    def __init__(
        self,
        fps: int = 30,
        preset: str = "medium",
        crf: int = 24,
        gop: int = 8,
        pix_fmt: str = "yuv420p",
        max_workers: int = 2,
    ):
        self.fps = fps
        self.preset = preset
        self.crf = crf
        self.gop = gop
        self.pix_fmt = pix_fmt
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="video-convert")
        self._futures: list[Any] = []
        self._tasks: list[tuple[Any, str, str, str]] = []
        self._rec_paths: list[str] = []
        self._expected_per_path: dict[str, int] = {}
        self._status_interval_sec = DEFAULT_STATUS_INTERVAL_SEC
        self._total = 0
        self._done = 0
        self._fail = 0
        self._completed_ok: list[tuple[str, str]] = []
        self._completed_fail: list[tuple[str, str, str]] = []
        self._completed_ok_full: list[tuple[str, str]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._status_thread: threading.Thread | None = None

    def _status_loop(self) -> None:
        """Periodically report job status from the background."""
        while not self._stop_event.is_set():
            interval = self._status_interval_sec
            if interval <= 0:
                self._stop_event.wait(timeout=1.0)
                continue
            self._stop_event.wait(timeout=interval)
            if self._stop_event.is_set():
                break
            with self._lock:
                total = self._total
                done = self._done
                fail = self._fail
                completed_ok = list(self._completed_ok)
                completed_fail = list(self._completed_fail)
                tasks = list(self._tasks)
            running = []
            queued = []
            for fut, rec_path, image_name, _task_type in tasks:
                rec_name = os.path.basename(rec_path.rstrip("/")) if rec_path else "?"
                label = f"{rec_name}/{image_name}"
                if fut.running():
                    running.append(label)
                elif not fut.done():
                    queued.append(label)
            lines = [
                "VideoConverter: [status]",
                f"  Total: {total} | Completed: {done} (ok) + {fail} (fail) | Remaining: {len(running) + len(queued)}",
            ]
            if running:
                lines.append(f"  Running: {', '.join(running)}")
            if queued:
                n_show = 5
                shown = queued[:n_show]
                more = f" ...+{len(queued)-n_show}" if len(queued) > n_show else ""
                lines.append(f"  Queued: {', '.join(shown)}{more}")
            if completed_ok:
                n_show = 5
                shown = [f"{r}/{i}" for r, i in completed_ok[-n_show:]]
                more = f" (+{len(completed_ok)-n_show} more)" if len(completed_ok) > n_show else ""
                lines.append(f"  Completed: {', '.join(shown)}{more}")
            if completed_fail:
                for rec_name, image_name, err in completed_fail[-3:]:
                    lines.append(f"  Failed: {rec_name}/{image_name}: {err}")
                if len(completed_fail) > 3:
                    lines.append(f"    ... (+{len(completed_fail)-3} more failed)")
            logger.info("\n".join(lines))

    def set_status_interval(self, sec: float) -> None:
        """Set the status-report interval in seconds; zero disables it."""
        self._status_interval_sec = max(0.0, float(sec))
        if sec > 0 and self._status_thread is None:
            self._status_thread = threading.Thread(
                target=self._status_loop,
                name="video-convert-status",
                daemon=True,
            )
            self._status_thread.start()

    def submit(
        self,
        env_id_to_rec_path: dict[int, str],
        camera_image_names: list[str],
        camera_depth_image_names: list[str] | None = None,
        depth_only_env_ids: list[int] | None = None,
    ) -> None:
        """
        Submit RGB and optional depth conversion jobs for each environment.
        camera_image_names and camera_depth_image_names identify image sequences.
        depth_only_env_ids limits depth jobs to the listed environments.
        """
        depth_names = camera_depth_image_names or []
        depth_env_set = set(depth_only_env_ids) if depth_only_env_ids is not None else None
        n_new_tasks = 0
        for env_id, rec_path in env_id_to_rec_path.items():
            if rec_path not in self._expected_per_path:
                self._expected_per_path[rec_path] = 0
            if rec_path not in self._rec_paths:
                self._rec_paths.append(rec_path)
            for image_name in camera_image_names:
                fut = self._executor.submit(
                    _run_ffmpeg_rgb,
                    rec_path,
                    image_name,
                    fps=self.fps,
                    preset=self.preset,
                    crf=self.crf,
                    gop=self.gop,
                    pix_fmt=self.pix_fmt,
                )
                self._futures.append(fut)
                self._tasks.append((fut, rec_path, image_name, "rgb"))
                self._expected_per_path[rec_path] += 1
                n_new_tasks += 1
            if depth_names and (depth_env_set is None or env_id in depth_env_set):
                for image_name in depth_names:
                    fut = self._executor.submit(
                        _run_ffmpeg_depth,
                        rec_path,
                        image_name,
                        fps=self.fps,
                    )
                    self._futures.append(fut)
                    self._tasks.append((fut, rec_path, image_name, "depth"))
                    self._expected_per_path[rec_path] += 1
                    n_new_tasks += 1
        with self._lock:
            self._total = len(self._futures)
        rec_name = os.path.basename(list(env_id_to_rec_path.values())[0].rstrip("/")) if env_id_to_rec_path else "?"
        logger.info(
            "VideoConverter: [submit] +%d tasks (total %d), envs=%d (e.g. %s)",
            n_new_tasks,
            len(self._futures),
            len(env_id_to_rec_path),
            rec_name,
        )

    def submit_depth_from_memory(
        self,
        env_id_to_rec_path: dict[int, str],
        depth_frame_buffers: dict[int, dict[str, list[np.ndarray]]],
        depth_only_env_ids: list[int] | None = None,
    ) -> None:
        """
        Submit in-memory depth frame buffers for ffmpeg pipe encoding.
        The recorder writes RGB video separately.
        depth_frame_buffers: {env_id: {image_name: [frame0, ...]}} uint16
        """
        depth_env_set = set(depth_only_env_ids) if depth_only_env_ids is not None else None
        n_new_tasks = 0
        for env_id, rec_path in env_id_to_rec_path.items():
            if depth_env_set is not None and env_id not in depth_env_set:
                continue
            if rec_path not in self._expected_per_path:
                self._expected_per_path[rec_path] = 0
            if rec_path not in self._rec_paths:
                self._rec_paths.append(rec_path)
            env_depth = depth_frame_buffers.get(env_id, {})
            for image_name, frames in env_depth.items():
                if not frames:
                    continue
                out_dir = os.path.join(rec_path, "observations", "videos")
                out_path = os.path.join(out_dir, f"{image_name}.mp4")
                fut = self._executor.submit(
                    _encode_depth_from_memory,
                    frames,
                    out_path,
                    fps=self.fps,
                )
                self._futures.append(fut)
                self._tasks.append((fut, rec_path, image_name, "depth_memory"))
                self._expected_per_path[rec_path] += 1
                n_new_tasks += 1
        with self._lock:
            self._total = len(self._futures)
        rec_name = os.path.basename(list(env_id_to_rec_path.values())[0].rstrip("/")) if env_id_to_rec_path else "?"
        logger.info(
            "VideoConverter: [submit_depth_from_memory] +%d tasks (total %d), envs=%d (e.g. %s)",
            n_new_tasks,
            len(self._futures),
            len(env_id_to_rec_path),
            rec_name,
        )

    MAX_RETRIES = 5

    def _drain(self) -> None:
        """Wait for all jobs and retry failures up to MAX_RETRIES times."""
        if not self._futures:
            return
        n_total = len(self._futures)
        with self._lock:
            self._done = 0
            self._fail = 0
            self._completed_ok = []
            self._completed_fail = []
            self._completed_ok_full = []
        n_done, n_fail, failed_to_retry = self._wait_futures(
            self._futures,
            self._tasks,
            n_total,
        )
        self._futures = []
        self._tasks = []
        for attempt in range(1, self.MAX_RETRIES + 1):
            if not failed_to_retry:
                break
            logger.info(
                "VideoConverter: [retry %d/%d] resubmitting %d failed tasks",
                attempt, self.MAX_RETRIES, len(failed_to_retry),
            )
            with self._lock:
                retry_set = {
                    (os.path.basename(rec_path.rstrip("/")) or rec_path, image_name)
                    for rec_path, image_name, _tt in failed_to_retry
                }
                self._completed_fail[:] = [
                    (r, i, e) for r, i, e in self._completed_fail
                    if (r, i) not in retry_set
                ]
            retry_futures: list[Any] = []
            retry_tasks: list[tuple[Any, str, str, str]] = []
            for rec_path, image_name, task_type in failed_to_retry:
                if task_type == "depth":
                    fut = self._executor.submit(
                        _run_ffmpeg_depth,
                        rec_path,
                        image_name,
                        fps=self.fps,
                    )
                else:
                    fut = self._executor.submit(
                        _run_ffmpeg_rgb,
                        rec_path,
                        image_name,
                        fps=self.fps,
                        preset=self.preset,
                        crf=self.crf,
                        gop=self.gop,
                        pix_fmt=self.pix_fmt,
                    )
                retry_futures.append(fut)
                retry_tasks.append((fut, rec_path, image_name, task_type))
            n_retry = len(retry_futures)
            rd, rf, failed_to_retry = self._wait_futures(
                retry_futures, retry_tasks, n_total,
                base_done=n_done, base_fail=n_fail,
                progress_total=n_retry,
            )
            n_done += rd
            n_fail = rf
            with self._lock:
                self._fail = n_fail
        logger.info(
            "VideoConverter: [drain done] total=%d ok=%d fail=%d",
            n_total, n_done, n_fail,
        )

    def _wait_futures(
        self,
        futures: list[Any],
        tasks: list[tuple[Any, str, str, str]],
        n_total: int,
        base_done: int = 0,
        base_fail: int = 0,
        progress_total: int | None = None,
    ) -> tuple[int, int, list[tuple[str, str, str]]]:
        """Wait for futures and return completed, failed, and retryable jobs."""
        n_done, n_fail = 0, 0
        failed_to_retry: list[tuple[str, str, str]] = []
        remaining = list(futures)
        remaining_tasks = list(tasks)
        display_total = progress_total if progress_total is not None else n_total
        while remaining:
            done, _ = wait(
                remaining,
                timeout=None,
                return_when=FIRST_COMPLETED,
            )
            for fut in done:
                remaining.remove(fut)
                task_info = next((t for t in remaining_tasks if t[0] is fut), (None, "", "", "rgb"))
                _, rec_path, image_name, task_type = task_info
                rec_name = os.path.basename(rec_path.rstrip("/")) if rec_path else "?"
                try:
                    fut.result()
                    n_done += 1
                    with self._lock:
                        self._done = base_done + n_done
                        self._fail = base_fail + n_fail
                        self._completed_ok.append((rec_name, image_name))
                        self._completed_ok_full.append((rec_path, image_name))
                    logger.info(
                        "VideoConverter: [progress] %d/%d done (%s/%s)",
                        n_done + n_fail, display_total, rec_name, image_name,
                    )
                except Exception as e:
                    n_fail += 1
                    err_msg = str(e)[:80]
                    failed_to_retry.append((rec_path, image_name, task_type))
                    with self._lock:
                        self._done = base_done + n_done
                        self._fail = base_fail + n_fail
                        self._completed_fail.append((rec_name, image_name, err_msg))
                    logger.warning(
                        "VideoConverter: [failed] %s/%s: %s",
                        rec_name, image_name, e,
                    )
        return n_done, n_fail, failed_to_retry

    def wait_all(self) -> None:
        """Wait for submitted jobs and shut down the thread pool."""
        logger.info("VideoConverter: [wait_all] draining tasks...")
        self._stop_event.set()
        self._drain()
        self._executor.shutdown(wait=True)
        logger.info("VideoConverter: [shutdown] executor closed")

    def had_failures(self) -> bool:
        """Report whether any conversion failed after wait_all."""
        with self._lock:
            return self._fail > 0

    def get_stats(self) -> tuple[int, int, int]:
        """Return total, successful, and failed job counts after wait_all."""
        with self._lock:
            return (self._total, self._done, self._fail)

    def get_ok_rec_paths(self) -> set[str]:
        """
        Return recording paths for which every required video succeeded.
        Return an empty set before any job is submitted.
        """
        with self._lock:
            if not self._expected_per_path:
                return set()
            ok = set()
            for rec_path, expected in self._expected_per_path.items():
                count = sum(1 for p, _ in self._completed_ok_full if p == rec_path)
                if expected > 0 and count == expected:
                    ok.add(rec_path)
            return ok

    def get_rec_paths(self) -> set[str]:
        """Return all recording paths submitted for conversion."""
        with self._lock:
            return set(self._rec_paths)
