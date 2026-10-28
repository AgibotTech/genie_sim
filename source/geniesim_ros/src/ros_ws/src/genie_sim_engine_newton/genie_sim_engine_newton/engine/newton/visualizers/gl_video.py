# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Optional observer-view MP4 recording, sampled on the simulation clock."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GlVideoConfig:
    path: Path
    fps: float

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> GlVideoConfig | None:
        raw_path = params.get("newton_video_path", "")
        if raw_path == "":
            return None
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("newton_video_path must be a nonempty MP4 path or an empty string to disable recording")
        raw_fps = params.get("newton_video_fps", 30)
        if isinstance(raw_fps, bool):
            raise ValueError("newton_video_fps must be finite and positive")
        try:
            fps = float(raw_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError("newton_video_fps must be finite and positive") from exc
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("newton_video_fps must be finite and positive")
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() != ".mp4":
            raise ValueError("newton_video_path must end in .mp4")
        return cls(path, fps)


class GlVideoRecorder:
    """Feed real completed observer frames to one owned CPU ffmpeg process.

    Only the entry point's normal render hook calls ``record``. Mounted sensor
    renders never do, and a paused simulation adds no wall-clock filler frames.
    """

    def __init__(self, config: GlVideoConfig, *, physics_hz: float) -> None:
        if config.fps > physics_hz:
            raise ValueError("newton_video_fps must not exceed physics_hz; duplicate frames are not synthesized")
        self.config = config
        self._ffmpeg = shutil.which("ffmpeg")
        if self._ffmpeg is None:
            raise RuntimeError("newton_video_path requires ffmpeg on PATH")
        if config.path.exists():
            raise FileExistsError(f"refusing to overwrite video: {config.path}")
        self._process = None
        self._log = None
        self._shape = None
        self._next_sample = None
        self._last_frame_time = None
        self._closed = False
        self.frame_count = 0

    def due(self, sim_time: float) -> bool:
        if not math.isfinite(sim_time):
            raise ValueError("video simulation time must be finite")
        if self._closed:
            raise RuntimeError("video recorder is closed")
        if self._last_frame_time is not None and sim_time < self._last_frame_time - 1e-9:
            # Episode reset: concatenate the next episode without inserting a
            # wall-clock gap or waiting for the previous episode's sim_time.
            return True
        return self._next_sample is None or sim_time + 1e-9 >= self._next_sample

    def _start(self, shape: tuple[int, ...]) -> None:
        height, width, _ = shape
        if width % 2 or height % 2:
            raise ValueError("MP4 recording requires an even framebuffer width and height")
        self.config.path.parent.mkdir(parents=True, exist_ok=True)
        # Reserve exclusively, then let our encoder populate this owned file.
        # A run must choose a new artifact path rather than replace old results.
        with self.config.path.open("xb"):
            pass
        try:
            self._log = self.config.path.with_suffix(".ffmpeg.log").open("x")
            self._process = subprocess.Popen(
                [
                    self._ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-nostdin",
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    f"{width}x{height}",
                    "-framerate",
                    f"{self.config.fps:g}",
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+frag_keyframe+empty_moov+default_base_moof",
                    str(self.config.path),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._log,
                start_new_session=True,
            )
            self._shape = shape
        except BaseException:
            self.close()
            raise

    def record(self, viewer: Any, sim_time: float) -> bool:
        """Record a due, already-rendered observer frame; return whether written."""
        if not self.due(sim_time):
            return False
        if not viewer.is_running():
            raise RuntimeError("Newton GL window closed before video readback")
        image = np.asarray(viewer.get_frame(render_ui=False).numpy())
        return self.record_rgb(image, sim_time)

    def record_rgb(self, image: np.ndarray, sim_time: float) -> bool:
        """Encode a completed real RGB readback from GL or Isaac Sim."""
        if not self.due(sim_time):
            return False
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("video requires a real uint8 HxWx3 RGB framebuffer")
        if self._process is None:
            self._start(image.shape)
        if image.shape != self._shape:
            raise RuntimeError("observer framebuffer was resized during video recording")
        if self._process.poll() is not None:
            raise RuntimeError(f"video encoder exited; see {self.config.path.with_suffix('.ffmpeg.log')}")
        pixels = np.ascontiguousarray(image).tobytes()
        try:
            written = self._process.stdin.write(pixels)
            if written != len(pixels):
                raise OSError("incomplete raw RGB frame write")
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"video encoding failed; see {self.config.path.with_suffix('.ffmpeg.log')}") from exc
        period = 1 / self.config.fps
        if self._next_sample is None or (self._last_frame_time is not None and sim_time < self._last_frame_time - 1e-9):
            self._next_sample = sim_time + period
        else:
            # Advance the fixed simulation-clock phase, without duplicating a
            # late frame for missed sample times. Ordinarily 60 Hz / 30 fps
            # calls this exactly once per two physics ticks.
            skipped = max(1, math.floor((sim_time - self._next_sample + 1e-9) / period) + 1)
            self._next_sample += skipped * period
        self._last_frame_time = sim_time
        self.frame_count += 1
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process, self._process = self._process, None
        failure = None
        try:
            if process is not None:
                try:
                    process.stdin.close()
                except (BrokenPipeError, OSError) as exc:
                    failure = exc
                try:
                    code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    raise RuntimeError("video encoder did not finalize within 15 seconds")
                if code != 0 or failure is not None:
                    raise RuntimeError(
                        f"video encoder failed ({code}); see {self.config.path.with_suffix('.ffmpeg.log')}"
                    )
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
