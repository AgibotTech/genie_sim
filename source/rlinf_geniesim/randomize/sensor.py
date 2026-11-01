# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Per-environment image augmentation adapted from geniesim_benchmark.utils.generalization_utils.

Noise, dirt occlusion and dropped frames use explicit numpy Generators so other
environments cannot perturb their seeded streams. The all-cameras-dropped
fallback is evaluated separately for each environment."""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Pixel-level noise (benchmark :424-:490, verbatim except explicit rng)
# ---------------------------------------------------------------------------


def _apply_gaussian_noise(rng: np.random.Generator, image: np.ndarray, std: float) -> np.ndarray:
    """Add Gaussian noise to an image.

    Args:
        rng: Explicit random generator (see module docstring point 1).
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        std: Standard deviation of Gaussian noise.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noise = rng.normal(0.0, std, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0)


def _apply_uniform_noise(rng: np.random.Generator, image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Add uniform noise to an image.

    Args:
        rng: Explicit random generator.
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        low: Lower bound of uniform noise.
        high: Upper bound of uniform noise.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noise = rng.uniform(low, high, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0)


def _apply_salt_pepper_noise(
    rng: np.random.Generator, image: np.ndarray, amount: float, pepper_vs_salt: float
) -> np.ndarray:
    """Add salt-and-pepper noise to an image.

    Args:
        rng: Explicit random generator.
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        amount: Fraction of pixels to be noise (0 to 1).
        pepper_vs_salt: Fraction of the noisy pixels set to pepper (black, 0.0);
            the rest become salt (white, 1.0). 1.0 = all pepper, 0.0 = all salt,
            0.5 = even split.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noisy = image.copy()
    h, w = image.shape[:2]
    num_pixels = int(h * w * amount)
    for _ in range(num_pixels):
        y = rng.integers(0, h)
        x = rng.integers(0, w)
        # rand() > pepper_vs_salt -> salt (1.0); otherwise pepper (0.0)
        noisy[y, x] = 1.0 if rng.random() > pepper_vs_salt else 0.0
    return noisy


def _apply_exponential_noise(rng: np.random.Generator, image: np.ndarray, scale: float) -> np.ndarray:
    """Add exponential noise to an image.

    Args:
        rng: Explicit random generator.
        image: Input image, shape (H, W, 3), dtype float [0, 1].
        scale: Scale parameter (lambda = 1/scale) of exponential distribution.

    Returns:
        Noisy image, same shape and dtype as input.
    """
    noise = rng.exponential(scale, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Camera dirt occlusion (benchmark :492-:664, verbatim except explicit rng)
# ---------------------------------------------------------------------------


def _generate_value_noise(rng: np.random.Generator, h: int, w: int, grid_size: int = 16) -> np.ndarray:
    """Generate smooth value noise by interpolating random control points.

    Uses vectorized operations for performance.

    Args:
        rng: Explicit random generator.
        h: Image height
        w: Image width
        grid_size: Size of the noise grid (smaller = finer details)

    Returns:
        Normalized noise map in range [0, 1]
    """
    from scipy.ndimage import zoom  # noqa: PLC0415

    # Create grid of random values at lower resolution
    grid_h = max(2, h // grid_size)
    grid_w = max(2, w // grid_size)
    grid = rng.random((grid_h, grid_w))

    # Upsample using scipy zoom (bilinear interpolation)
    zoom_factor = (h / grid_h, w / grid_w)
    noise = zoom(grid, zoom_factor, order=1)

    # Crop to exact size if needed
    if noise.shape[0] > h:
        noise = noise[:h, :]
    if noise.shape[1] > w:
        noise = noise[:, :w]

    return noise


def _generate_dirt_mask(rng: np.random.Generator, h: int, w: int, density: float) -> np.ndarray:
    """Generate organic dirt pattern using interpolated value noise.

    Args:
        rng: Explicit random generator.
        h: Image height
        w: Image width
        density: Density of dirt (0.0-1.0)

    Returns:
        Dirt mask in range [0, 1]
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    # Generate base value noise with multiple scales
    noise1 = _generate_value_noise(rng, h, w, grid_size=20)
    noise2 = _generate_value_noise(rng, h, w, grid_size=40)

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


def _generate_dust_spots(rng: np.random.Generator, h: int, w: int, density: float) -> np.ndarray:
    """Generate small dust particle spots.

    Args:
        rng: Explicit random generator.
        h: Image height
        w: Image width
        density: Density of dust spots (0.0-1.0)

    Returns:
        Dust spots mask in range [0, 1]
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    num_spots = int(h * w * density * 0.001)
    num_spots = max(10, min(num_spots, 400))

    if num_spots == 0:
        return np.zeros((h, w), dtype=np.float32)

    cy = rng.integers(0, h, num_spots)
    cx = rng.integers(0, w, num_spots)
    radius = rng.integers(2, 6, num_spots)
    intensity = rng.uniform(0.5, 1.0, num_spots)

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


def _generate_smudges(rng: np.random.Generator, h: int, w: int, density: float) -> np.ndarray:
    """Generate larger smudge patterns with irregular shapes.

    Args:
        rng: Explicit random generator.
        h: Image height
        w: Image width
        density: Density of smudges (0.0-1.0)

    Returns:
        Smudge mask in range [0, 1]
    """
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    num_smudges = int(density * 4)
    num_smudges = max(1, min(num_smudges, 8))

    if num_smudges == 0:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    y, x = np.ogrid[:h, :w]

    for _ in range(num_smudges):
        cy = rng.integers(h // 4, 3 * h // 4)
        cx = rng.integers(w // 4, 3 * w // 4)
        size_x = rng.integers(30, 70)
        size_y = rng.integers(30, 70)

        dist = np.sqrt(((y - cy) / size_y) ** 2 + ((x - cx) / size_x) ** 2)
        smudge = np.exp(-(dist**2) / 2)
        mask += smudge * rng.uniform(0.5, 0.9)

    mask = np.clip(mask, 0, 1)
    mask = gaussian_filter(mask, sigma=5)

    return mask


def _apply_camera_dirt(
    rng: np.random.Generator,
    cache_dict: dict[tuple, np.ndarray],
    camera_name: str,
    image: np.ndarray,
    ratio: float,
) -> np.ndarray:
    """Apply realistic camera lens dirt/dust occlusion.

    Uses multiple noise layers to simulate:
    1. Dust particles: small random spots
    2. Smudges: organic shapes from value noise
    3. Stains: larger irregular patches with soft edges

    Args:
        rng: Explicit random generator (only consumed on a cache MISS --
            a cache HIT reuses the previously generated mask verbatim and
            draws nothing, exactly like benchmark's original cache).
        cache_dict: Dictionary containing cached dirt masks.
        camera_name: Name of the camera (e.g., "main_images", "wrist_images").
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
        combined_mask = cache_dict[cache_key]
    else:
        # Generate masks
        dirt_mask = _generate_dirt_mask(rng, h, w, ratio * 0.8)
        dust_mask = _generate_dust_spots(rng, h, w, ratio * 0.4)
        smudge_mask = _generate_smudges(rng, h, w, ratio * 0.25)

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


# ---------------------------------------------------------------------------
# Main entry (benchmark :689-:760, adapted for the (N, H, W, 3) env axis --
# see module docstring point 2)
# ---------------------------------------------------------------------------


def _apply_camera_image_augmentation_one_env(
    rng: np.random.Generator,
    cache_dict: dict[tuple, np.ndarray],
    images: dict[str, np.ndarray],
    camera_gen_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """One env's worth of camera images -> augmented images.

    Corresponds verbatim to benchmark's ``apply_camera_image_augmentation``
    (the single-env version, where ``images`` is
    ``{camera name: (H, W, 3) uint8}``), only with the global ``np.random``
    swapped for an explicit ``rng``. The drop-frame fallback
    (``all(drop_decisions)``) naturally only ever sees THIS ONE env's set of
    cameras -- which is exactly why the main entry point
    ``apply_camera_image_augmentation`` loops over the env dimension on the
    outside and calls this function as the processing unit for each env.

    Args:
        rng: Explicit random generator for this env.
        cache_dict: Shared dirt-mask cache (see ``_apply_camera_dirt``).
        images: camera name -> RGB image (H, W, 3), uint8 [0, 255].
        camera_gen_config: This env's concrete generalization config
            (``camera_noise`` / ``camera_occlusion`` / ``camera_drop_frame``
            -- same shape ``sampler.Sampler.sample`` fills
            ``RandomizationPlan.sensor`` with).

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
        if drop_frame_enabled and rng.random() < drop_prob:
            drop_decisions[cam_name] = True
        else:
            drop_decisions[cam_name] = False

    if all(drop_decisions.values()):
        # All cameras of THIS env were going to be dropped -- randomly keep
        # one so this env is never fully blacked out (per-env fallback; see
        # module docstring point 2 for why this must not be judged across envs).
        kept_cam = rng.choice(cam_names)
        drop_decisions[kept_cam] = False

    for cam_name, image in images.items():
        img = image.astype(np.float32) / 255.0

        if drop_decisions[cam_name]:
            result[cam_name] = np.zeros_like(image)
            continue

        # --- Camera dirt occlusion: apply realistic lens dirt/dust ---
        if occlusion_enabled and rng.random() < occlusion_prob:
            img = _apply_camera_dirt(rng, cache_dict, cam_name, img, occlusion_ratio)

        # --- Pixel-level noise ---
        if noise_type == "gaussian":
            std = camera_noise.get("std", 0.01)
            img = _apply_gaussian_noise(rng, img, std)
        elif noise_type == "uniform":
            low = camera_noise.get("low", -0.1)
            high = camera_noise.get("high", 0.1)
            img = _apply_uniform_noise(rng, img, low, high)
        elif noise_type == "salt_pepper":
            amount = camera_noise.get("amount", 0.05)
            # The config key is historically named "salt_vs_pepper" but its
            # value is actually the pepper fraction (see _apply_salt_pepper_noise).
            pepper_vs_salt = camera_noise.get("salt_vs_pepper", 0.5)
            img = _apply_salt_pepper_noise(rng, img, amount, pepper_vs_salt)
        elif noise_type == "exponential":
            scale = camera_noise.get("scale", 0.1)
            img = _apply_exponential_noise(rng, img, scale)

        result[cam_name] = (img * 255.0).clip(0, 255).astype(np.uint8)

    return result


def apply_camera_image_augmentation(
    rngs: Sequence[Optional[np.random.Generator]],
    cache_dict: dict[tuple, np.ndarray],
    images: dict[str, np.ndarray],
    camera_gen_configs: Sequence[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Augment camera-name -> (N, H, W, 3) uint8 RGB arrays.

    rngs and camera_gen_configs have one entry per environment. A missing RNG or
    empty configuration leaves that environment unchanged. Apply noise, dirt and
    frame dropping per environment, including the all-cameras-dropped fallback.
    cache_dict shares dirt masks keyed by camera name, dimensions and ratio.
    Return the same camera keys and array shapes."""
    if not images:
        return images

    cam_names = list(images.keys())
    n_envs = next(iter(images.values())).shape[0]
    result = {cam: np.array(arr, copy=True) for cam, arr in images.items()}

    for e in range(n_envs):
        cfg_e = camera_gen_configs[e] if e < len(camera_gen_configs) else {}
        rng_e = rngs[e] if e < len(rngs) else None
        if not cfg_e or rng_e is None:
            continue  # already byte-copied into `result` above
        single_env_images = {cam: images[cam][e] for cam in cam_names}
        augmented = _apply_camera_image_augmentation_one_env(rng_e, cache_dict, single_env_images, cfg_e)
        for cam in cam_names:
            result[cam][e] = augmented[cam]

    return result


__all__ = ["apply_camera_image_augmentation"]
