"""
Replayer wraps KinematicReplay for multiple fuzzed rounds and output directories.
"""
import os

from common.base_utils.logger import logger
from .config import ReplayConfig
from .kinematic_replay import KinematicReplay

# Resolve output paths from the project root when importing replay as a package.
_replay_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_replay_dir)


class Replayer:
    """
    Set up cloned environments once, then fuzz, replay, and record each round.
    Reset to the first frame and allocate new recording directories between rounds.
    """

    def __init__(self, input_dir: str, config: ReplayConfig):
        self.input_dir = input_dir.rstrip("/")
        self.config = config
        self._replay: KinematicReplay | None = None
        recording_root = config.recording_root
        if not os.path.isabs(recording_root):
            recording_root = os.path.join(_project_root, recording_root)
        self._replay = KinematicReplay(
            input_dir=self.input_dir,
            robot_config=config.robot_config,
            num_clones=config.max_concurrent_envs,
            cloth_subdivision_levels=config.cloth_subdivision_levels,
            clone_spacing=config.clone_spacing,
            backgrounds=config.backgrounds,
            enable_object_compensation=config.enable_object_compensation,
            filter_unmoved_objects=config.filter_unmoved_objects,
            enable_unmoved_object_disappear=config.enable_unmoved_object_disappear,
            enable_recording=config.enable_recording,
            recording_root=recording_root,
            record_depth=config.record_depth,
            depth_realtime=config.depth_realtime,
            image_mode=config.image_mode,
            render_hz=config.render_hz,
            video_fps=config.video_fps,
            video_crf=config.video_crf,
            video_gop=config.video_gop,
            video_preset=config.video_preset,
            rename_camera=config.rename_camera,
            save_depth_in_uint16=config.save_depth_in_uint16,
            record_all_camera_depth=config.record_all_camera_depth,
            add_default_ground_plane=config.add_default_ground_plane,
        )

    @property
    def num_clones(self) -> int:
        return self.config.max_concurrent_envs

    @property
    def objects_dir_name(self) -> str:
        if self._replay:
            return self._replay._objects_dir_name
        return "Objects"

    def load_payload(self) -> None:
        """Select the scene payload before setting up replay environments."""
        if self._replay:
            self._replay._load_payload()

    def setup(self, simulation_app) -> None:
        """Load the scene, clones, and recorder after SimulationApp starts."""
        if self._replay:
            self._replay.setup(simulation_app)

    def get_env_paths(self) -> list[str]:
        """Return environment paths ordered by environment ID for Fuzzer."""
        if not self._replay:
            return []
        return self._replay.get_env_paths()

    def ensure_default_lights(self) -> None:
        if self._replay:
            self._replay.ensure_default_lights()

    def set_record_env_ids_for_round(self, record_env_ids: list[int]) -> None:
        """Record only the selected environments in this round."""
        if self._replay:
            self._replay.set_record_env_ids_for_round(record_env_ids)

    def run_one_round(self) -> int:
        """Replay from the current frame to the end and return frame count."""
        if not self._replay:
            return 0
        return self._replay.run_one_round()

    def finish_round_recording(self, num_frames: int) -> None:
        """Flush output and write recording_info."""
        if self._replay:
            self._replay.finish_round_recording(num_frames)

    def allocate_next_recording_dirs(self) -> dict:
        """Allocate recording directories for the next round."""
        if not self._replay:
            return {}
        return self._replay.allocate_next_recording_dirs()

    def reset_to_frame0(self) -> None:
        """Reset replay to the first frame."""
        if self._replay:
            self._replay.reset_to_frame0()

    def get_recording_dirs_by_env(self) -> dict:
        """Return current recording paths by environment ID."""
        if not self._replay:
            return {}
        return self._replay.get_recording_dirs_by_env()

    def get_recording_camera_image_names(self) -> list[str]:
        """Return RGB camera image names used by ffmpeg."""
        if not self._replay:
            return []
        return self._replay.get_recording_camera_image_names()

    def get_recording_camera_depth_image_names(self) -> list:
        """Return depth camera names used for video conversion."""
        if not self._replay:
            return []
        return self._replay.get_recording_camera_depth_image_names()

    def release_writers(self) -> set[str]:
        if self._replay:
            return self._replay.release_writers()
        return set()

    def log_resource_status(self) -> None:
        if self._replay:
            self._replay.log_resource_status()

    @property
    def total_pool_wait_sec(self) -> float:
        if self._replay:
            return self._replay.total_pool_wait_sec
        return 0.0

    @property
    def video_preset(self) -> str:
        if self._replay:
            return self._replay.video_preset
        return "medium"

    def get_record_depth_mode(self) -> str:
        """NONE / ALL。"""
        if not self._replay:
            return "NONE"
        return self._replay.get_record_depth_mode()

    @property
    def depth_realtime(self) -> bool:
        if self._replay:
            return self._replay.depth_realtime
        return False

    def get_object_asset_dict(self) -> dict:
        """Map prim paths to USD assets for Fuzzer's exclusion list."""
        if not self._replay:
            return {}
        return self._replay.get_object_asset_dict()
