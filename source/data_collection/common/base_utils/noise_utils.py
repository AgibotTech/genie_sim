# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import re
from typing import Any, Dict

import numpy as np

from common.base_utils.logger import logger


def add_noise_with_regex(
    data_dict: Dict[str, float], noise_config: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Supported Noise Types and Their Parameters:

    1. Gaussian Noise ('gaussian')
    - Description: Adds Gaussian (normal) distributed noise
    - Parameters:
        - mean (float): Mean of the Gaussian distribution (default: 0)
        - std (float): Standard deviation of the Gaussian distribution (default: 0.1)

    2. Uniform Noise ('uniform')
    - Description: Adds uniformly distributed noise
    - Parameters:
        - low (float): Lower bound of the uniform distribution (default: -0.1)
        - high (float): Upper bound of the uniform distribution (default: 0.1)

    3. Salt and Pepper Noise ('salt_pepper')
    - Description: Adds salt (high values) and pepper (low values) noise
    - Parameters:
        - amount (float): Proportion of pixels to be affected by noise (default: 0.05)
        - salt_vs_pepper (float): Ratio of salt noise to pepper noise (default: 0.5)

    4. Poisson Noise ('poisson')
    - Description: Adds Poisson distributed noise (commonly used for count data)
    - Parameters: None

    5. Exponential Noise ('exponential')
    - Description: Adds exponentially distributed noise
    - Parameters:
        - scale (float): Scale parameter (inverse of rate) (default: 1.0)

    Usage Example:
    data_dict = {'temperature': 25.5, 'pressure': 1013.2, 'humidity': 65.2}
    noise_config = {
        'temp.*': {'noise_type': 'gaussian', 'std': 0.5},
        'pres.*': {'noise_type': 'uniform', 'low': -2, 'high': 2},
        '.*': {'noise_type': 'gaussian', 'std': 0.1}  # Default for all other keys
    }
    """
    # Create a copy of the original data to avoid modifying it
    noisy_data = data_dict.copy()

    # Pre-compile all regex patterns for efficiency
    compiled_patterns = {re.compile(pattern): params for pattern, params in noise_config.items()}

    # Process each key in the data dictionary
    for key in noisy_data.keys():
        if noisy_data[key] is None:
            continue
        # Find the first matching pattern for this key
        matched_params = None
        for pattern, params in compiled_patterns.items():
            if pattern.match(key):
                matched_params = params
                break

        # If no pattern matched, skip this key
        if matched_params is None:
            continue

        # Add noise based on the matched parameters
        noisy_data[key] = _add_noise_to_scalar(noisy_data[key], matched_params)

    return noisy_data


def _add_noise_to_scalar(scalar: float, params: Dict[str, Any]) -> float:
    """Add noise to a scalar value based on noise parameters"""
    noise_type = params.get("noise_type", "gaussian")

    if noise_type == "gaussian":
        mean = params.get("mean", 0)
        std = params.get("std", 0.1)
        return scalar + np.random.normal(mean, std)

    elif noise_type == "uniform":
        low = params.get("low", -0.1)
        high = params.get("high", 0.1)
        return scalar + np.random.uniform(low, high)

    elif noise_type == "salt_pepper":
        amount = params.get("amount", 0.05)
        salt_vs_pepper = params.get("salt_vs_pepper", 0.5)

        if np.random.random() < amount:
            if np.random.random() < salt_vs_pepper:
                return scalar * (1 + np.random.uniform(0.5, 1.0))  # Salt noise
            else:
                return scalar * np.random.uniform(0, 0.5)  # Pepper noise
        return scalar

    elif noise_type == "poisson":
        return np.random.poisson(max(scalar, 0))

    elif noise_type == "exponential":
        scale = params.get("scale", 1.0)
        return scalar + np.random.exponential(scale)

    else:
        raise ValueError(f"Unsupported noise type: {noise_type}")


# Example usage
if __name__ == "__main__":
    # Sample data dictionary
    data_dict = {
        "temperature": 25.5,
        "pressure": 1013.2,
        "humidity": 65.2,
        "wind_speed": 12.3,
        "temp_outside": 18.7,
        "pressure_inside": None,
    }

    # Noise configuration with regex patterns
    noise_config = {
        "temp.*": {  # Matches keys starting with 'temp'
            "noise_type": "gaussian",
            "std": 0.5,
        },
        "pressure.*": {  # Matches keys starting with 'pressure'
            "noise_type": "uniform",
            "low": -2,
            "high": 2,
        },
        "humidity": {  # Exact match for 'humidity'
            "noise_type": "salt_pepper",
            "amount": 0.5,
            "salt_vs_pepper": 0.5,
        },
        ".*": {  # Default pattern for all other keys
            "noise_type": "gaussian",
            "std": 0.1,
        },
    }

    # Add noise based on regex patterns
    noisy_data = add_noise_with_regex(data_dict, noise_config)

    logger.info("Original data:")
    for key, value in data_dict.items():
        logger.info(f"  {key}: {value}")

    logger.info("\nNoisy data:")
    for key, value in noisy_data.items():
        logger.info(f"  {key}: {value} (original: {data_dict[key]})")
