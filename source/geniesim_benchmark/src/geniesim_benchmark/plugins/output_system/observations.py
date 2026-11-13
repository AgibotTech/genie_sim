# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Record native RGB observations using the same video/state format as Isaac."""

from pathlib import Path
from types import SimpleNamespace

from .local_recorder import LocalRecorder


class ObservationRecording:
    def __init__(self, output, episode_idx, instruction, *, fps=30, record_state=True):
        self.output = Path(output)
        self.index = episode_idx
        self.recorder = LocalRecorder(fps=fps, record_state=record_state)
        self.recorder.update_instruction(instruction)
        self.last_step = None

    def capture(self, frames, step, state=None, sim_time=None):
        if not frames or step == self.last_step:
            return
        if not self.recorder.is_recording:
            specs = [
                dict(camera_id=name, width=frame.shape[1], height=frame.shape[0]) for name, frame in frames.items()
            ]
            self.recorder.start(str(self.output.parent), self.index, specs, sub_task_name=self.output.name)
        self.recorder.write_frames(
            SimpleNamespace(_img_data_cache=frames),
            step,
            (lambda: state) if state is not None else None,
            sim_time=sim_time,
        )
        self.last_step = step

    def close(self):
        return self.recorder.stop(self.index)


def finalize_recordings(output, bags, final_output=None, reports=()):
    """Keep episode sources while publishing the three aggregate camera files."""
    if not bags:
        return []
    recorder = LocalRecorder()
    recorder.restore_checkpoint("native-evaluation", list(map(str, bags)))
    artifacts = recorder.concat_all(preserve_bags=True)
    recorder.publish_checkpoint_artifacts([*artifacts, *reports], output, final_output)
    return artifacts


def snapshot_state(snapshot):
    indices = [i for i, path in enumerate(snapshot.schema.body_paths) if path.rsplit("/", 1)[-1] == "base_link"]
    if len(indices) != 1:
        raise ValueError("recording requires one published base_link pose")
    pose = snapshot.body_poses[0, indices[0]]
    return {
        "joints": {
            "name": list(snapshot.schema.joint_names),
            "position": list(map(float, snapshot.joint_positions[0])),
        },
        "base_pose": {"position": pose[:3].tolist(), "orientation_xyzw": [*map(float, pose[4:]), float(pose[3])]},
    }
