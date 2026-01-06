# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
1. GAUSSIAN NOISE
   - Description: Additive white noise with normal distribution, affects all pixels independently
   - Parameters:
        sigma: Standard deviation (0.01-0.3, typical: 0.05)
   - Use Case: General purpose noise simulation, sensor read noise

2. SALT AND PEPPER NOISE
   - Description: Random white (salt) and black (pepper) pixels simulating sensor defects
   - Parameters:
        salt_prob: Probability of white pixels (0.001-0.05)
        pepper_prob: Probability of black pixels (0.001-0.05)
   - Use Case: Dead pixels, sensor faults, transmission errors

3. POISSON NOISE (SHOT NOISE)
   - Description: Signal-dependent noise from photon counting statistics, variance ∝ signal
   - Parameters:
        scale: Noise intensity multiplier (0.5-3.0, typical: 1.2)
   - Use Case: Low-light conditions, photon-limited imaging

4. SPECKLE NOISE
   - Description: Multiplicative noise common in coherent imaging systems
   - Parameters:
        sigma: Standard deviation of multiplicative factor (0.05-0.3)
   - Use Case: Ultrasound, SAR, laser imaging, medical imaging

5. QUANTIZATION NOISE
   - Description: Artifacts from reduced bit depth, creates banding effects
   - Parameters:
        bits: Number of quantization bits (2-7, typical: 4)
   - Use Case: Low-bit-depth sensors, compression artifacts

6. SENSOR NOISE (PHYSICAL MODEL) (NOT USED)
   - Description: Combined shot noise (signal-dependent) and read noise (signal-independent)
   - Parameters:
        shot_noise: Photon shot noise intensity (0.01-0.05)
        read_noise: Sensor readout noise (0.005-0.03)
   - Use Case: Realistic camera noise simulation, physical modeling

7. BROWNIAN NOISE (FRACTAL NOISE) (NOT USED)
   - Description: Correlated noise with natural texture, multiple frequency octaves
   - Parameters:
        intensity: Overall noise strength (0.05-0.2)
   - Use Case: Natural-looking noise, film grain simulation
"""
import numpy as np
import warp as wp

def truncated_absolute_normal(
    mean=0.15,
    std=0.1,
    lower=0.1,
    upper=0.4,
    max_attempts=10000,
    fallback=np.random.uniform,
):
    attempts = 0
    while attempts < max_attempts:
        sample = np.random.normal(loc=mean, scale=std)
        abs_sample = np.abs(sample)
        if lower <= abs_sample <= upper:
            return abs_sample
    return np.abs(fallback(lower, upper))


def get_random_parameters(noise_type):
    if noise_type == "gaussian":
        return {
            "sigma": truncated_absolute_normal(
                mean=0.15, std=0.08, lower=0.1, upper=0.4, max_attempts=10000
            )
        }
    elif noise_type == "salt_pepper":
        prob = np.random.uniform(0.002, 0.02)
        return {"salt_prob": prob, "pepper_prob": prob}
    elif noise_type == "poisson":
        return {"scale": np.random.uniform(0.05, 0.3)}
    elif noise_type == "speckle":
        return {"sigma": np.random.uniform(0.05, 0.2)}
    elif noise_type == "quantization":
        return {"bits": np.random.randint(4, 7)}
    else:
        raise ValueError("Invalid noise type")


@wp.kernel
def image_gaussian_noise_warp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    seed: int,
    sigma: float = 0.5,
):
    i, j = wp.tid()
    dim_i = data_out.shape[0]
    dim_j = data_out.shape[1]
    pixel_id = i * dim_i + j
    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    data_out[i, j, 0] = wp.uint8(float(data_in[i, j, 0]) + (255.0 * sigma * wp.randn(state_r)))
    data_out[i, j, 1] = wp.uint8(float(data_in[i, j, 1]) + (255.0 * sigma * wp.randn(state_g)))
    data_out[i, j, 2] = wp.uint8(float(data_in[i, j, 2]) + (255.0 * sigma * wp.randn(state_b)))


@wp.kernel
def image_salt_pepper_noise_warp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    seed: int,
    salt_prob: float = 0.01,
    pepper_prob: float = 0.01,
):
    i, j = wp.tid()
    dim_i = data_out.shape[0]
    data_out.shape[1]
    pixel_id = i * dim_i + j

    state = wp.rand_init(seed, pixel_id)
    rand_val = wp.randf(state)

    if rand_val < salt_prob:
        # Salt noise (white pixels)
        data_out[i, j, 0] = wp.uint8(255)
        data_out[i, j, 1] = wp.uint8(255)
        data_out[i, j, 2] = wp.uint8(255)
    elif rand_val < salt_prob + pepper_prob:
        # Pepper noise (black pixels)
        data_out[i, j, 0] = wp.uint8(0)
        data_out[i, j, 1] = wp.uint8(0)
        data_out[i, j, 2] = wp.uint8(0)
    else:
        # No noise
        data_out[i, j, 0] = data_in[i, j, 0]
        data_out[i, j, 1] = data_in[i, j, 1]
        data_out[i, j, 2] = data_in[i, j, 2]


@wp.kernel
def image_poisson_noise_warp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    seed: int,
    scale: float = 1.0,
):
    i, j = wp.tid()
    dim_i = data_out.shape[0]
    dim_j = data_out.shape[1]
    pixel_id = i * dim_i + j

    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Convert to float and normalize
    val_r = float(data_in[i, j, 0]) / 255.0
    val_g = float(data_in[i, j, 1]) / 255.0
    val_b = float(data_in[i, j, 2]) / 255.0

    # Poisson noise approximation using Gaussian (valid for large counts)
    # For Poisson distribution, variance = mean
    noise_r = wp.randn(state_r) * wp.sqrt(val_r) * scale
    noise_g = wp.randn(state_g) * wp.sqrt(val_g) * scale
    noise_b = wp.randn(state_b) * wp.sqrt(val_b) * scale

    # Apply noise and convert back
    data_out[i, j, 0] = wp.uint8(wp.clamp((val_r + noise_r) * 255.0, 0.0, 255.0))
    data_out[i, j, 1] = wp.uint8(wp.clamp((val_g + noise_g) * 255.0, 0.0, 255.0))
    data_out[i, j, 2] = wp.uint8(wp.clamp((val_b + noise_b) * 255.0, 0.0, 255.0))


@wp.kernel
def image_speckle_noise_warp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    seed: int,
    sigma: float = 0.1,
):
    i, j = wp.tid()
    dim_i = data_out.shape[0]
    dim_j = data_out.shape[1]
    pixel_id = i * dim_i + j

    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Convert to float
    val_r = float(data_in[i, j, 0])
    val_g = float(data_in[i, j, 1])
    val_b = float(data_in[i, j, 2])

    # Speckle noise: multiplicative noise
    speckle_r = 1.0 + sigma * wp.randn(state_r)
    speckle_g = 1.0 + sigma * wp.randn(state_g)
    speckle_b = 1.0 + sigma * wp.randn(state_b)

    data_out[i, j, 0] = wp.uint8(wp.clamp(val_r * speckle_r, 0.0, 255.0))
    data_out[i, j, 1] = wp.uint8(wp.clamp(val_g * speckle_g, 0.0, 255.0))
    data_out[i, j, 2] = wp.uint8(wp.clamp(val_b * speckle_b, 0.0, 255.0))


@wp.kernel
def image_quantization_noise_warp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    seed: int,
    bits: int = 4,
):
    i, j = wp.tid()

    # Convert to float
    val_r = float(data_in[i, j, 0])
    val_g = float(data_in[i, j, 1])
    val_b = float(data_in[i, j, 2])

    # Quantize to lower bit depth
    levels = wp.pow(2.0, float(bits))
    quant_step = 255.0 / (levels - 1.0)

    data_out[i, j, 0] = wp.uint8(wp.round(val_r / quant_step) * quant_step)
    data_out[i, j, 1] = wp.uint8(wp.round(val_g / quant_step) * quant_step)
    data_out[i, j, 2] = wp.uint8(wp.round(val_b / quant_step) * quant_step)


@wp.kernel
def image_sensor_noise_warp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    seed: int,
    shot_noise: float = 0.01,
    read_noise: float = 0.02,
):
    i, j = wp.tid()
    dim_i = data_out.shape[0]
    dim_j = data_out.shape[1]
    pixel_id = i * dim_i + j

    state_r = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    state_g = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    state_b = wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Convert to float and normalize
    val_r = float(data_in[i, j, 0]) / 255.0
    val_g = float(data_in[i, j, 1]) / 255.0
    val_b = float(data_in[i, j, 2]) / 255.0

    # Shot noise (signal-dependent) - approximated as Gaussian
    shot_noise_r = wp.randn(state_r) * wp.sqrt(val_r) * shot_noise
    shot_noise_g = wp.randn(state_g) * wp.sqrt(val_g) * shot_noise
    shot_noise_b = wp.randn(state_b) * wp.sqrt(val_b) * shot_noise

    # Read noise (signal-independent)
    state_r2 = wp.rand_init(seed + 1000, pixel_id + (dim_i * dim_j * 0))
    state_g2 = wp.rand_init(seed + 1000, pixel_id + (dim_i * dim_j * 1))
    state_b2 = wp.rand_init(seed + 1000, pixel_id + (dim_i * dim_j * 2))

    read_noise_r = wp.randn(state_r2) * read_noise
    read_noise_g = wp.randn(state_g2) * read_noise
    read_noise_b = wp.randn(state_b2) * read_noise

    # Combine both noise sources
    data_out[i, j, 0] = wp.uint8(
        wp.clamp((val_r + shot_noise_r + read_noise_r) * 255.0, 0.0, 255.0)
    )
    data_out[i, j, 1] = wp.uint8(
        wp.clamp((val_g + shot_noise_g + read_noise_g) * 255.0, 0.0, 255.0)
    )
    data_out[i, j, 2] = wp.uint8(
        wp.clamp((val_b + shot_noise_b + read_noise_b) * 255.0, 0.0, 255.0)
    )


@wp.kernel
def image_brownian_noise_warp(
    data_in: wp.array3d(dtype=wp.uint8),
    data_out: wp.array3d(dtype=wp.uint8),
    seed: int,
    intensity: float = 0.1,
):
    i, j = wp.tid()
    dim_i = data_out.shape[0]
    dim_j = data_out.shape[1]
    pixel_id = i * dim_i + j

    wp.rand_init(seed, pixel_id + (dim_i * dim_j * 0))
    wp.rand_init(seed, pixel_id + (dim_i * dim_j * 1))
    wp.rand_init(seed, pixel_id + (dim_i * dim_j * 2))

    # Brownian noise (fractal/Brownian motion type noise)
    # This creates correlated noise that looks more natural
    brownian_r = 0.0
    brownian_g = 0.0
    brownian_b = 0.0

    # Simple approximation using multiple octaves
    for octave in range(4):
        freq = wp.pow(2.0, float(octave))
        weight = 1.0 / freq

        state_oct_r = wp.rand_init(seed + octave, pixel_id + (dim_i * dim_j * 0))
        state_oct_g = wp.rand_init(seed + octave, pixel_id + (dim_i * dim_j * 1))
        state_oct_b = wp.rand_init(seed + octave, pixel_id + (dim_i * dim_j * 2))

        brownian_r += weight * wp.randn(state_oct_r)
        brownian_g += weight * wp.randn(state_oct_g)
        brownian_b += weight * wp.randn(state_oct_b)

    # Normalize and apply
    brownian_scale = 1.0 / 1.875  # Approximate normalization for 4 octaves

    val_r = float(data_in[i, j, 0]) + 255.0 * intensity * brownian_r * brownian_scale
    val_g = float(data_in[i, j, 1]) + 255.0 * intensity * brownian_g * brownian_scale
    val_b = float(data_in[i, j, 2]) + 255.0 * intensity * brownian_b * brownian_scale

    data_out[i, j, 0] = wp.uint8(wp.clamp(val_r, 0.0, 255.0))
    data_out[i, j, 1] = wp.uint8(wp.clamp(val_g, 0.0, 255.0))
    data_out[i, j, 2] = wp.uint8(wp.clamp(val_b, 0.0, 255.0))


# Example usage function
def apply_noise_to_image(image, noise_type="gaussian", seed=0, **kwargs):
    """
    Apply specified type of noise to image
    """
    # Initialize Warp
    wp.init()

    # Copy image data to Warp array
    height, width, channels = image.shape
    data_in = wp.array(image, dtype=wp.uint8)
    data_out = wp.array(shape=(height, width, channels), dtype=wp.uint8)

    # Call corresponding kernel based on noise type
    if noise_type == "gaussian":
        sigma = kwargs.get("sigma")
        wp.launch(
            kernel=image_gaussian_noise_warp,
            dim=(height, width),
            inputs=[data_in, data_out, seed, sigma],
        )
    elif noise_type == "salt_pepper":
        salt_prob = kwargs.get("salt_prob")
        pepper_prob = kwargs.get("pepper_prob")
        wp.launch(
            kernel=image_salt_pepper_noise_warp,
            dim=(height, width),
            inputs=[data_in, data_out, seed, salt_prob, pepper_prob],
        )
    elif noise_type == "poisson":
        scale = kwargs.get("scale")
        wp.launch(
            kernel=image_poisson_noise_warp,
            dim=(height, width),
            inputs=[data_in, data_out, seed, scale],
        )
    elif noise_type == "speckle":
        sigma = kwargs.get("sigma")
        wp.launch(
            kernel=image_speckle_noise_warp,
            dim=(height, width),
            inputs=[data_in, data_out, seed, sigma],
        )
    elif noise_type == "quantization":
        bits = kwargs.get("bits")
        wp.launch(
            kernel=image_quantization_noise_warp,
            dim=(height, width),
            inputs=[data_in, data_out, seed, bits],
        )
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    wp.synchronize()
    # Copy result back to numpy array
    result = data_out.numpy()
    return result
