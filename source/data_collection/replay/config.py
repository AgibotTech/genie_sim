"""Replay and fuzzing configuration that can be built from JSON."""
from dataclasses import dataclass, field
from typing import Optional

from common.base_utils.logger import logger


def build_simulation_app_config(headless: bool, extra_args: list[str]) -> dict:
    """Build SimulationApp settings without disabling the viewport in GUI mode."""
    config = {
        "headless": headless,
        "renderer": "RealTimePathTracing",
        "hide_ui": headless,
        "extra_args": extra_args,
    }
    if headless:
        config["disable_viewport_updates"] = True
    return config


@dataclass
class FuzzingConfig:
    """Options for lighting, backgrounds, materials, rounds, and random seeds."""

    enable_light_fuzz: bool = True
    enable_background_fuzz: bool = True
    enable_material_fuzz: bool = True
    # Material modes: scan scene peers, read material_config.json, or use both.
    material_fuzz_mode: str = "both"
    no_fuzzing_object_id_list: list[str] = field(default_factory=list)
    num_rounds: Optional[int] = None
    seed: Optional[int] = None
    backgrounds: list[dict] = field(default_factory=list)


@dataclass
class ReplayConfig:
    """Options for replaying and recording multiple fuzzed rounds."""

    # Maximum number of cloned environments recorded in one round.
    max_concurrent_envs: int = 3
    # Root directory for replay output.
    recording_root: str = "recording_data"
    # Maximum number of rounds.
    num_rounds: Optional[int] = 1
    # Spacing between clones, in metres.
    clone_spacing: float = 3.0
    # Robot config filename; an empty value is resolved from input state.json.
    robot_config: str = ""
    # Enable output recording.
    enable_recording: bool = True
    # Skip temporal cloth smoothing and replay the input NPZ files directly.
    no_deformable_smooth: bool = False
    # Loop subdivision level for cloth rendering; 0 disables subdivision.
    cloth_subdivision_levels: int = 2
    # Enable object centre compensation and hiding of unmoved objects.
    enable_object_compensation: bool = False
    # Candidate backgrounds.
    backgrounds: list[dict] = field(default_factory=lambda: [])
    # backgrounds: list[dict] = field(default_factory=lambda: [{
    #     "usd": "background/room/room_1/background.usda",
    #     "position": [0, 0, 0],
    #     "rotation": [1, 0, 0, 0],
    # }])
    filter_unmoved_objects: bool = True
    enable_unmoved_object_disappear: bool = False

    # Depth modes: NONE disables depth; ALL records depth per environment.
    record_depth: str = "NONE"
    # Include stereo and fisheye cameras when recording depth.
    record_all_camera_depth: bool = False
    # Encode depth during replay, or save images for later VideoConverter processing.
    depth_realtime: bool = False

    # Image mode stores images without encoding videos.
    image_mode: bool = False
    # Map camera prim names to short names; otherwise keep the original names.
    rename_camera: bool = True
    # Store scaled uint16 PNG depth; otherwise store float32 NPZ depth.
    save_depth_in_uint16: bool = True
    # Remove the camera image directory after conversion to save space.
    clean_cameras: bool = True

    # Isaac Sim stepping frequency during replay; does not set dataset timing.
    render_hz: int = 30

    # Add the default dome and distant lights.
    add_default_light: bool = True

    # Add a collision plane at Z=0 when the scene USD lacks a ground plane.
    # Some dynamic bodies otherwise shift during world.reset(). This remains
    # disabled by default; individual presets can enable it when needed.
    add_default_ground_plane: bool = False

    # Fixed video encoding settings; conflicting user values are logged and replaced.
    _FIXED_VIDEO_PARAMS = {
        "video_fps": 30,
        "video_preset": "ultrafast",
        "video_crf": 24,
        "video_gop": 8,
        "video_pix_fmt": "yuv420p",
    }

    video_fps: int = 30
    video_preset: str = "ultrafast"
    video_crf: int = 24
    video_gop: int = 8
    video_pix_fmt: str = "yuv420p"

    def __post_init__(self):
        for name, fixed_val in self._FIXED_VIDEO_PARAMS.items():
            user_val = getattr(self, name)
            if user_val != fixed_val:
                logger.error(
                    "ReplayConfig: %s=%s differs from fixed setting %s; overriding with %s",
                    name, user_val, fixed_val, fixed_val,
                )
                object.__setattr__(self, name, fixed_val)
