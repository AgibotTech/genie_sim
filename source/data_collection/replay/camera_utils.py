"""Camera names and resolutions used by recordings and annotators."""


def get_camera_prim_name(prim_path: str, robot_name: str, rename: bool = True) -> str:
    """Convert a prim path to a filename; retain its name when rename is false."""
    name = prim_path.split("/")[-1]
    if not rename:
        return name
    if "G1" in robot_name:
        if "head" in name.lower():
            return "head"
        if "right" in name.lower():
            return "hand_right"
        if "left" in name.lower():
            return "hand_left"
        if "top" in name.lower():
            return "head_front_fisheye"
    if "G2" in robot_name:
        if "head_front" in name.lower():
            return "head"
        if "head_right" in name.lower():
            return "head_stereo_right"
        if "head_left" in name.lower():
            return "head_stereo_left"
        if "left_camera" in name.lower() and "head" not in name.lower():
            return "hand_left"
        if "right_camera" in name.lower() and "head" not in name.lower():
            return "hand_right"
    return name


def get_camera_resolution_from_config(prim_path: str, robot_cfg: dict) -> list:
    """Read camera resolution from robot_cfg['camera'].

    Match the full prim path first, including the robot root.
    Fall back to legacy G1/G2 prefix matching where necessary.
    """
    camera_cfg = robot_cfg.get("camera") or {}
    # Cloned environments can add a /World/envs/env_N prefix to the prim path.
    for cfg_key, res in camera_cfg.items():
        if prim_path == cfg_key or prim_path.endswith(cfg_key):
            return list(res)
    for key in ("/G2/", "/G1/"):
        if key in prim_path:
            config_key = key + prim_path.split(key)[-1]
            if config_key in camera_cfg:
                return list(camera_cfg[config_key])
    return [640, 480]
