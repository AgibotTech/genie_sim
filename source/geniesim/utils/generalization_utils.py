# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Generalization utilities for benchmark tasks.

This module provides functions for applying various generalization settings
such as joint PD control, camera parameters, and material variations.
"""

from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation as R

from geniesim.plugins.logger import Logger
from geniesim.utils.name_utils import G1_DUAL_ARM_JOINT_NAMES, G2_DUAL_ARM_JOINT_NAMES

logger = Logger()


def _sample_from_list(choices):
    """Sample a single value from a list of choices using np.random."""
    if not choices:
        return None
    return float(np.random.choice(choices))


def _dynamic_joint_pd_sample(task_config):
    """Dynamically sample joint PD parameters at runtime when not pre-generated.

    Args:
        task_config: Task configuration dictionary.
    """
    gen_config = task_config.get("generalization", {})
    joint_pd = gen_config.get("joint_pd", {})
    if not joint_pd.get("enable", False):
        return {}

    kp_range = joint_pd.get("kp", [])
    kd_range = joint_pd.get("kd", [])
    if not kp_range or not kd_range:
        return {}

    kp = _sample_from_list(kp_range)
    kd = _sample_from_list(kd_range)
    if kp is not None and kd is not None:
        return {"enable": True, "kp": kp, "kd": kd}
    return {}


def _dynamic_camera_noise_sample(task_config):
    """Dynamically sample camera noise parameters at runtime when not pre-generated.

    Args:
        task_config: Task configuration dictionary.
    """
    gen_config = task_config.get("generalization", {})
    camera = gen_config.get("camera", {})
    noise = camera.get("noise", {})
    if not noise.get("enable", False):
        return {}

    noise_types = noise.get("types", [])
    if not noise_types:
        return {}

    noise_type = str(np.random.choice(noise_types))
    noise_params = {"enable": True, "type": noise_type}

    if noise_type == "gaussian":
        std_range = noise.get("gaussian", {}).get("std_range", [])
        noise_params["std"] = _sample_from_list(std_range)
    elif noise_type == "uniform":
        noise_params["low"] = float(noise.get("uniform", {}).get("low", -0.1))
        noise_params["high"] = float(noise.get("uniform", {}).get("high", 0.1))
    elif noise_type == "salt_pepper":
        amount_range = noise.get("salt_pepper", {}).get("amount_range", [])
        noise_params["salt_vs_pepper"] = float(noise.get("salt_pepper", {}).get("salt_vs_pepper", 0.5))
        noise_params["amount"] = _sample_from_list(amount_range)
    elif noise_type == "exponential":
        scale_range = noise.get("exponential", {}).get("scale_range", [])
        noise_params["scale"] = _sample_from_list(scale_range)

    return noise_params


def _dynamic_camera_drop_frame_sample(task_config):
    """Dynamically sample camera drop frame parameters at runtime when not pre-generated.

    Args:
        task_config: Task configuration dictionary.
    """
    gen_config = task_config.get("generalization", {})
    camera = gen_config.get("camera", {})
    drop_frame = camera.get("drop_frame", {})
    if not drop_frame.get("enable", False):
        return {}

    drop_prob_range = drop_frame.get("drop_prob_range", [])
    drop_prob = _sample_from_list(drop_prob_range)
    if drop_prob is not None:
        return {"enable": True, "drop_prob": drop_prob}
    return {}


def _dynamic_camera_occlusion_sample(task_config):
    """Dynamically sample camera occlusion parameters at runtime when not pre-generated.

    Args:
        task_config: Task configuration dictionary.
    """
    gen_config = task_config.get("generalization", {})
    camera = gen_config.get("camera", {})
    occlusion = camera.get("occlusion", {})
    if not occlusion.get("enable", False):
        return {}

    ratio_range = occlusion.get("ratio_range", [])
    if not ratio_range:
        return {}

    ratio = _sample_from_list(ratio_range)
    if ratio is not None:
        return {"enable": True, "ratio": ratio}
    return {}


def _dynamic_camera_position_sample(task_config):
    """Dynamically sample camera position perturbation at runtime when not pre-generated.

    Args:
        task_config: Task configuration dictionary.
    """
    gen_config = task_config.get("generalization", {})
    camera = gen_config.get("camera", {})
    position = camera.get("position", {})
    if not position.get("enable", False):
        return {}

    threshold = position.get("threshold", {})
    if not threshold:
        return {}

    perturbations = {"enable": True}
    for axis in ["x", "y", "z", "roll", "pitch", "yaw"]:
        thresh = threshold.get(axis, 0.0)
        perturbations[axis] = float(np.random.uniform(-thresh, thresh))

    return perturbations


def apply_joint_pd_generalization(api_core, task_config, gen_config):
    """Apply joint PD control parameters generalization.

    Args:
        api_core: The API core instance for controlling robot parameters.
        task_config: Task configuration dictionary containing robot_cfg.
        gen_config: Generalization configuration containing joint_pd settings.
    """
    joint_pd = gen_config.get("joint_pd", {})
    if not joint_pd:
        return

    enable = joint_pd.get("enable", False)
    if not enable:
        return

    kp = joint_pd.get("kp")
    kd = joint_pd.get("kd")
    if kp is None or kd is None:
        return

    robot_cfg = task_config.get("robot_cfg", "")

    # Determine which joint names to use based on robot type
    if "G1" in robot_cfg:
        joint_names = G1_DUAL_ARM_JOINT_NAMES
    elif "G2" in robot_cfg:
        joint_names = G2_DUAL_ARM_JOINT_NAMES
    else:
        logger.warning(f"apply_joint_pd_generalization: Unknown robot type {robot_cfg}")
        return

    # Create joint gains dict: joint_name -> (kp, kd)
    joint_gains = {name: (kp, kd) for name in joint_names}

    api_core.set_robot_joint_drive_gains(joint_gains)
    logger.info(f"Applied joint PD generalization: kp={kp}, kd={kd} for {len(joint_names)} joints")


def _apply_camera_position_perturbation(api_core, camera_list, perturbations):
    """Apply position perturbations to camera prims (in local space).

    Args:
        api_core: The API core instance for setting camera poses.
        camera_list: List of camera prim paths to perturb.
        perturbations: Dictionary containing x, y, z offsets and roll, pitch, yaw angles.
    """
    if not camera_list or not perturbations:
        return

    for camera_prim_path in camera_list:
        # Get original local pose using the new interface
        original_pos, original_quat = api_core.get_prim_local_pose(camera_prim_path)

        # Position perturbation - add delta to original position
        new_pos = [
            original_pos[0] + perturbations.get("x", 0.0),
            original_pos[1] + perturbations.get("y", 0.0),
            original_pos[2] + perturbations.get("z", 0.0) + 0.01,
        ]

        # Rotation perturbation (convert degrees to radians)
        roll = np.deg2rad(perturbations.get("roll", 0.0))
        pitch = np.deg2rad(perturbations.get("pitch", 0.0))
        yaw = np.deg2rad(perturbations.get("yaw", 0.0))

        # Create rotation from perturbations
        rot_perturb = R.from_euler("xyz", [roll, pitch, yaw])
        # Convert original quaternion to rotation (wxyz -> xyzw)
        rot_original = R.from_quat([original_quat[1], original_quat[2], original_quat[3], original_quat[0]])
        # Apply perturbation
        rot_new = rot_perturb * rot_original
        new_quat = rot_new.as_quat()
        # Convert back to wxyz format
        new_quat_wxyz = [new_quat[3], new_quat[0], new_quat[1], new_quat[2]]

        # Use the new interface to set local pose
        api_core.set_prim_local_pose(
            camera_prim_path,
            position=new_pos,
            orientation=new_quat_wxyz,
        )
        logger.info(f"Applied camera perturbation to {camera_prim_path}: new_pos={new_pos}, quat={new_quat_wxyz}")


def _sample_camera_generalization(env, task_config, gen_config):
    """Sample camera generalization configs and store into env.

    Args:
        env: Environment instance to store camera configs into.
        task_config: Task configuration dictionary.
        gen_config: Episode-specific generalization configuration.
    """
    camera_config = task_config.get("generalization", {}).get("camera", {})
    camera_gen_config = {}

    camera_noise = gen_config.get("camera_noise")
    if not camera_noise and camera_config.get("noise", {}).get("enable", False):
        camera_noise = _dynamic_camera_noise_sample(task_config)
    if camera_noise and camera_noise.get("enable", False):
        logger.info(f"Camera noise generalization: {camera_noise}")
        camera_gen_config["camera_noise"] = camera_noise

    camera_drop_frame = gen_config.get("camera_drop_frame")
    if not camera_drop_frame and camera_config.get("drop_frame", {}).get("enable", False):
        camera_drop_frame = _dynamic_camera_drop_frame_sample(task_config)
    if camera_drop_frame and camera_drop_frame.get("enable", False):
        logger.info(f"Camera drop frame generalization: {camera_drop_frame}")
        camera_gen_config["camera_drop_frame"] = camera_drop_frame

    camera_occlusion = gen_config.get("camera_occlusion")
    if not camera_occlusion and camera_config.get("occlusion", {}).get("enable", False):
        camera_occlusion = _dynamic_camera_occlusion_sample(task_config)
    if camera_occlusion and camera_occlusion.get("enable", False):
        logger.info(f"Camera occlusion generalization: {camera_occlusion}")
        camera_gen_config["camera_occlusion"] = camera_occlusion

    camera_position = gen_config.get("camera_position")
    if not camera_position and camera_config.get("position", {}).get("enable", False):
        camera_position = _dynamic_camera_position_sample(task_config)
    if camera_position and camera_position.get("enable", False):
        logger.info(f"Camera position perturbation: {camera_position}")
        camera_gen_config["camera_position"] = camera_position

    env.set_camera_gen_config(camera_gen_config)


def _apply_camera_generalization_from_env(api_core, task_config, camera_gen_config):
    """Apply camera generalization configs via api_core.

    Args:
        api_core: The API core instance.
        task_config: Task configuration dictionary.
        camera_gen_config: Camera generalization config dict stored in env.
    """
    camera_config = task_config.get("generalization", {}).get("camera", {})
    camera_list = camera_config.get("camera_list", [])

    camera_position = camera_gen_config.get("camera_position")
    if camera_position:
        _apply_camera_position_perturbation(api_core, camera_list, camera_position)


def apply_material_generalization(api_core, task_config):
    """Apply material randomization generalization.

    This function supports two modes:
    1. Pre-generated mode: Uses pre-selected materials
    2. Dynamic mode: Randomly selects materials at runtime

    Args:
        api_core: The API core instance for changing materials.
        task_config: Task configuration dictionary containing generalization settings.
    """
    gen_config = task_config.get("generalization", {})
    material_config = gen_config.get("material", {})

    enable = material_config.get("enable", False)
    if not enable:
        return

    material_info = api_core.collect_material_info()
    material_change_list = ["door", "table"]
    for mesh_path, material_list in material_info.items():
        if not any(keyword in mesh_path for keyword in material_change_list):
            continue
        material_path = np.random.choice(sorted(material_list))
        logger.info(f"Material replaced: mesh='{mesh_path}' -> material='{material_path}'")
        api_core.change_material(mesh_path, material_path)


def apply_hdr_texture_generalization(api_core, task_config):
    """Apply HDR texture randomization generalization.

    Args:
        api_core: The API core instance for randomizing HDR textures.
        task_config: Task configuration dictionary containing generalization settings.
    """
    gen_config = task_config.get("generalization", {})
    hdr_texture_config = gen_config.get("hdr_texture", {})

    enable = hdr_texture_config.get("enable", False)
    if not enable:
        return

    api_core.randomize_hdr_textures()
    logger.info("Applied HDR texture randomization")


def update_init_env(env, task_config, episode_content):
    """Sample and store generalization configs into env.

    This function supports two modes:
    1. Pre-generated mode: Uses pre-computed values from episode_content
    2. Dynamic mode: Samples values at runtime when episode_content values are missing

    Args:
        env: The environment instance to store generalization configs into.
        task_config: Task configuration dictionary containing generalization settings.
        episode_content: Episode configuration containing generalization_config.
    """
    gen_config = episode_content.get("generalization_config", {})
    gen_task_config = task_config.get("generalization", {})

    # Handle robot_init_pose (init_base generalization)
    robot_init_pose = gen_config.get("robot_init_pose")
    if robot_init_pose is None and gen_task_config.get("init_base", {}).get("enable", False):
        # Dynamic sampling when not pre-generated
        init_base = gen_task_config.get("init_base", {})
        x_thresh = init_base.get("x_thresh", 0.1)
        y_thresh = init_base.get("y_thresh", 0.1)
        robot_init_pose = {"x_thresh": x_thresh, "y_thresh": y_thresh}
        logger.info(f"Dynamically sampled robot base offsets: x={x_thresh}, y={y_thresh}")

    if robot_init_pose is not None:
        env.set_robot_init_pose(robot_init_pose)

    # Handle init_joint generalization
    rand_init_arm = gen_config.get("rand_init_arm")
    if rand_init_arm is None and gen_task_config.get("init_joint", {}).get("enable", False):
        # Dynamic sampling when not pre-generated
        init_joint = gen_task_config.get("init_joint", {})
        joint_thresh = init_joint.get("thresh", 0.1)
        rand_init_arm = [np.random.uniform(-joint_thresh, joint_thresh) for _ in range(14)]
        logger.info(f"Dynamically sampled rand_init_arm: {rand_init_arm}")

    if rand_init_arm is not None:
        env.set_rand_init_arm(rand_init_arm)

    # Handle light_config
    light_config = gen_config.get("light_config", {})
    if not light_config and gen_task_config.get("lights", {}).get("enable", False):
        # Dynamic light sampling
        lights = gen_task_config.get("lights", {})
        dynamic_light_config = {}
        temperature = lights.get("temperature", [])
        if temperature:
            dynamic_light_config["temperature"] = int(np.random.choice(temperature))
        intensity = lights.get("intensity", [])
        if intensity:
            dynamic_light_config["intensity"] = int(np.random.choice(intensity))
        if dynamic_light_config:
            light_config = dynamic_light_config
            logger.info(f"Dynamically sampled light config: {dynamic_light_config}")

    if light_config:
        env.set_light_config(light_config)

    # Handle joint_pd generalization
    joint_pd = gen_config.get("joint_pd")
    if not joint_pd and gen_task_config.get("joint_pd", {}).get("enable", False):
        joint_pd = _dynamic_joint_pd_sample(task_config)
    if joint_pd:
        env.set_joint_pd(joint_pd)

    # Handle camera generalization (supports dynamic sampling)
    env.set_camera_gen_config({})
    _sample_camera_generalization(env, task_config, gen_config)


# =============================================================================
# Image Augmentation Utilities for Camera Generalization
# =============================================================================


def _apply_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to an image.

    Args:
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        std: Standard deviation of Gaussian noise.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noise = np.random.normal(0.0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0)


def _apply_uniform_noise(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Add uniform noise to an image.

    Args:
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        low: Lower bound of uniform noise.
        high: Upper bound of uniform noise.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noise = np.random.uniform(low, high, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0)


def _apply_salt_pepper_noise(image: np.ndarray, amount: float, salt_vs_pepper: float) -> np.ndarray:
    """Add salt-and-pepper noise to an image.

    Args:
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        amount: Fraction of pixels to be noise (0 to 1).
        salt_vs_pepper: Ratio of salt vs pepper; 1.0 = all salt, 0.0 = all pepper.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = int(h * w * amount)
    for _ in range(num_pixels):
        y = np.random.randint(0, h)
        x = np.random.randint(0, w)
        noisy[y, x] = 1.0 if np.random.random() > salt_vs_pepper else 0.0
    return noisy


def _apply_exponential_noise(image: np.ndarray, scale: float) -> np.ndarray:
    """Add exponential noise to an image.

    Args:
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        scale: Scale parameter (lambda = 1/scale) of exponential distribution.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noise = np.random.exponential(scale, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0)


def _generate_value_noise(h: int, w: int, grid_size: int = 16) -> np.ndarray:
    """Generate smooth value noise by interpolating random control points.

    Uses vectorized operations for performance.

    Args:
        h: Image height
        w: Image width
        grid_size: Size of the noise grid (smaller = finer details)

    Returns:
        Normalized noise map in range [0, 1]
    """
    from scipy.ndimage import zoom

    # Create grid of random values at lower resolution
    grid_h = max(2, h // grid_size)
    grid_w = max(2, w // grid_size)
    grid = np.random.rand(grid_h, grid_w)

    # Upsample using scipy zoom (bilinear interpolation)
    zoom_factor = (h / grid_h, w / grid_w)
    noise = zoom(grid, zoom_factor, order=1)

    # Crop to exact size if needed
    if noise.shape[0] > h:
        noise = noise[:h, :]
    if noise.shape[1] > w:
        noise = noise[:, :w]

    return noise


def _generate_dirt_mask(h: int, w: int, density: float) -> np.ndarray:
    """Generate organic dirt pattern using interpolated value noise.

    Args:
        h: Image height
        w: Image width
        density: Density of dirt (0.0-1.0)

    Returns:
        Dirt mask in range [0, 1]
    """
    from scipy.ndimage import gaussian_filter

    # Generate base value noise with multiple scales
    noise1 = _generate_value_noise(h, w, grid_size=20)
    noise2 = _generate_value_noise(h, w, grid_size=40)

    # Combine noise layers
    combined = 0.7 * noise1 + 0.3 * noise2

    # Apply Gaussian blur for smooth edges
    combined = gaussian_filter(combined, sigma=4)

    # Normalize to [0, 1]
    combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

    threshold = 1.0 - density * 0.5
    mask = np.clip((combined - threshold) / (1.0 - threshold + 1e-8), 0, 1)
    mask = np.power(mask, 0.8)

    return mask


def _generate_dust_spots(h: int, w: int, density: float) -> np.ndarray:
    """Generate small dust particle spots.

    Args:
        h: Image height
        w: Image width
        density: Density of dust spots (0.0-1.0)

    Returns:
        Dust spots mask in range [0, 1]
    """
    from scipy.ndimage import gaussian_filter

    num_spots = int(h * w * density * 0.001)
    num_spots = max(10, min(num_spots, 400))

    if num_spots == 0:
        return np.zeros((h, w), dtype=np.float32)

    cy = np.random.randint(0, h, num_spots)
    cx = np.random.randint(0, w, num_spots)
    radius = np.random.randint(2, 6, num_spots)
    intensity = np.random.uniform(0.5, 1.0, num_spots)

    y, x = np.ogrid[:h, :w]
    mask = np.zeros((h, w), dtype=np.float32)

    for i in range(num_spots):
        dist_sq = (y - cy[i]) ** 2 + (x - cx[i]) ** 2
        spot = np.exp(-dist_sq / (2 * radius[i] ** 2)) * intensity[i]
        mask += spot

    # Clip and smooth
    mask = np.clip(mask, 0, 1)
    mask = gaussian_filter(mask, sigma=1.0)

    return mask


def _generate_smudges(h: int, w: int, density: float) -> np.ndarray:
    """Generate larger smudge patterns with irregular shapes.

    Args:
        h: Image height
        w: Image width
        density: Density of smudges (0.0-1.0)

    Returns:
        Smudge mask in range [0, 1]
    """
    from scipy.ndimage import gaussian_filter

    num_smudges = int(density * 4)
    num_smudges = max(1, min(num_smudges, 8))

    if num_smudges == 0:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]

    for _ in range(num_smudges):
        cy = np.random.randint(h // 4, 3 * h // 4)
        cx = np.random.randint(w // 4, 3 * w // 4)
        size_x = np.random.randint(30, 70)
        size_y = np.random.randint(30, 70)

        dist = np.sqrt(((y - cy) / size_y) ** 2 + ((x - cx) / size_x) ** 2)
        smudge = np.exp(-(dist**2) / 2)
        mask += smudge * np.random.uniform(0.5, 0.9)

    mask = np.clip(mask, 0, 1)
    mask = gaussian_filter(mask, sigma=5)

    return mask


def _apply_camera_dirt(
    cache_dict: Dict[tuple, np.ndarray], camera_name: str, image: np.ndarray, ratio: float
) -> np.ndarray:
    """Apply realistic camera lens dirt/dust occlusion.

    Uses multiple noise layers to simulate:
    1. Dust particles: small random spots
    2. Smudges: organic shapes from value noise
    3. Stains: larger irregular patches with soft edges

    Args:
        cache_dict: Dictionary containing cached dirt masks.
        camera_name: Name of the camera (e.g., "head", "left_hand", "right_hand").
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        ratio: Dirt coverage ratio (0.0-1.0). Controls overall density.

    Returns:
        Image with dirt occlusion applied, same shape and dtype as input.
    """
    h, w = image.shape[:2]
    result = image.copy()

    # Create cache key with camera_name to ensure each camera has unique dirt pattern
    cache_key = (camera_name, h, w, ratio)

    # Check cache first
    if cache_key in cache_dict:
        logger.debug(f"Camera dirt cache HIT: {cache_key}")
        combined_mask = cache_dict[cache_key]
    else:
        logger.debug(f"Camera dirt cache MISS: {cache_key}")
        # Generate masks
        dirt_mask = _generate_dirt_mask(h, w, ratio * 0.8)
        dust_mask = _generate_dust_spots(h, w, ratio * 0.4)
        smudge_mask = _generate_smudges(h, w, ratio * 0.25)

        # Combine masks
        combined_mask = np.maximum.reduce([dirt_mask, dust_mask, smudge_mask])

        # Cache the result
        cache_dict[cache_key] = combined_mask

    # Ensure mask is 2D for broadcasting
    if combined_mask.ndim == 2:
        combined_mask = combined_mask[:, :, np.newaxis]

    # Apply dirt
    result = result * (1.0 - combined_mask * 0.55)

    return result


def apply_camera_image_augmentation(
    cache_dict: Dict[tuple, np.ndarray],
    images: dict,
    camera_gen_config: dict,
) -> dict:
    """Apply camera image augmentations to a dict of RGB images.

    Supports the following augmentation types:
    - Gaussian / uniform / salt_pepper / exponential noise (pixel-level)
    - Camera dirt occlusion (realistic lens dust/smudges with soft edges)
    - Drop frame (full black image)

    Args:
        cache_dict: Dictionary containing cached dirt masks.
        images: Dict mapping camera name -> RGB image (H, W, 3), uint8 [0, 255].
        camera_gen_config: Camera generalization config dict,
            containing camera_noise, camera_occlusion, etc.

    Returns:
        Augmented images dict with same keys as input, uint8 [0, 255].
    """
    if not images:
        return images

    camera_noise = camera_gen_config.get("camera_noise", {})
    if not camera_noise or not camera_noise.get("enable", False):
        noise_type = None
    else:
        noise_type = camera_noise.get("type", "gaussian")

    camera_occlusion = camera_gen_config.get("camera_occlusion", {})
    occlusion_enabled = camera_occlusion.get("enable", False)
    occlusion_ratio = camera_occlusion.get("ratio", 0.1)
    occlusion_prob = camera_occlusion.get("prob", 1.0)

    camera_drop_frame = camera_gen_config.get("camera_drop_frame", {})
    drop_frame_enabled = camera_drop_frame.get("enable", False)
    drop_prob = camera_drop_frame.get("drop_prob", 0.0) if drop_frame_enabled else 0.0

    result = {}
    cam_names = list(images.keys())

    drop_decisions = {}
    for cam_name in cam_names:
        if drop_frame_enabled and np.random.random() < drop_prob:
            drop_decisions[cam_name] = True
        else:
            drop_decisions[cam_name] = False

    if all(drop_decisions.values()):
        # All cameras were going to be dropped, randomly keep one
        kept_cam = np.random.choice(cam_names)
        drop_decisions[kept_cam] = False
        logger.info(f"Prevented all-camera drop, keeping: {kept_cam}")

    for cam_name, image in images.items():
        img = image.astype(np.float32) / 255.0

        if drop_decisions[cam_name]:
            logger.info(f"Camera frame dropped: {cam_name}")
            result[cam_name] = np.zeros_like(image)
            continue

        # --- Camera dirt occlusion: apply realistic lens dirt/dust ---
        if occlusion_enabled and np.random.random() < occlusion_prob:
            img = _apply_camera_dirt(cache_dict, cam_name, img, occlusion_ratio)

        # --- Pixel-level noise ---
        if noise_type == "gaussian":
            std = camera_noise.get("std", 0.01)
            img = _apply_gaussian_noise(img, std)
        elif noise_type == "uniform":
            low = camera_noise.get("low", -0.1)
            high = camera_noise.get("high", 0.1)
            img = _apply_uniform_noise(img, low, high)
        elif noise_type == "salt_pepper":
            amount = camera_noise.get("amount", 0.05)
            salt_vs_pepper = camera_noise.get("salt_vs_pepper", 0.5)
            img = _apply_salt_pepper_noise(img, amount, salt_vs_pepper)
        elif noise_type == "exponential":
            scale = camera_noise.get("scale", 0.1)
            img = _apply_exponential_noise(img, scale)

        result[cam_name] = (img * 255.0).clip(0, 255).astype(np.uint8)

    return result
