# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Record synchronized IPC observations through the common video/state recorder."""

from pathlib import Path
from types import SimpleNamespace
import uuid

import cv2
import numpy as np

from .adapter import _compose_fixed_pose, wire_images
from geniesim_benchmark.plugins.output_system.local_recorder import LocalRecorder


class EpisodeRecording:
    """One frame per observed control tick; native physical archives remain separate."""

    def __init__(self, output, episode_idx, instruction, *, fps, record_state, fixed_joints,
                 frame_bindings=None, run_id=None):
        self.output = Path(output)
        self.episode_idx, self.fps = episode_idx, fps
        self.fixed_joints, self.frame_bindings = fixed_joints, frame_bindings
        self.recorder = LocalRecorder(fps=fps, record_state=record_state)
        self.recorder.restore_checkpoint(run_id or uuid.uuid4().hex, [])
        self.recorder.update_instruction(instruction)
        self.last_step = None

    def capture(self, packet, step):
        if step == self.last_step:
            return
        images = wire_images(packet["images"])
        cache = {}
        for name, row in images.items():
            image = cv2.imdecode(np.frombuffer(row["image_data"], dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.shape[:2] != (row["height"], row["width"]):
                raise ValueError("Recording requires a decodable camera image with declared dimensions")
            cache[name] = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        if not self.recorder.is_recording:
            specs = [{"camera_id": name, "width": row["width"], "height": row["height"]}
                     for name, row in images.items()]
            self.recorder.start(str(self.output.parent), self.episode_idx, specs,
                                sub_task_name=self.output.name)

        def state():
            snapshot = packet["snapshot"]
            joints = {**snapshot["joints"], **self.fixed_joints}
            if self.frame_bindings is not None:
                binding = self.frame_bindings["frames"]["base_link"]
                pose = _compose_fixed_pose(snapshot["bodies"][binding["body_path"]],
                                           binding["body_from_frame_pose"])
            else:
                matches = [pose for path, pose in snapshot["bodies"].items() if path.rsplit("/", 1)[-1] == "base_link"]
                if len(matches) != 1:
                    raise ValueError("Recording requires an unambiguous base_link pose")
                pose = matches[0]
            return {"joints": {"name": list(joints), "position": list(joints.values())},
                    "base_pose": {"position": list(pose[:3]), "orientation_xyzw": [*pose[4:], pose[3]]}}

        sample = state()  # Required replay state must not be silently skipped.
        self.recorder.write_frames(SimpleNamespace(_img_data_cache=cache), step, lambda: sample,
                                   sim_time=packet["snapshot"]["sim_time"])
        self.last_step = step

    def close(self):
        return self.recorder.stop(self.episode_idx)
