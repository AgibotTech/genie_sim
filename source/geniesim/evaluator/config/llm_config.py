# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""
Unified LLM and VLM configuration loader
"""

import os
import yaml
from geniesim.plugins.logger import Logger

logger = Logger()


def load_llm_config(config_type="llm"):
    """
    Load LLM or VLM configuration from the centralized config file.

    Args:
        config_type: "llm" for language models or "vlm" for vision-language models

    Returns:
        tuple: (api_key, base_url, model)
    """
    # Get config path relative to this file (in evaluator directory)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "llm_config.yaml")

    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    # Load YAML config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get the appropriate config section
    if config_type not in config:
        raise ValueError(f"Config section '{config_type}' not found in config file")

    model_config = config[config_type]

    # Read from environment variables first, then fall back to config file
    api_key = os.environ.get("API_KEY", model_config.get("api_key"))
    base_url = os.environ.get("BASE_URL", model_config.get("base_url"))

    # Use different env var names for model based on config_type
    if config_type == "vlm":
        model = os.environ.get("VL_MODEL", model_config.get("model"))
    else:
        model = os.environ.get("MODEL", model_config.get("model"))

    # Validate required fields
    if not api_key:
        logger.warning(
            f"API key not provided: please set {config_type}.api_key in config file or environment variable API_KEY"
        )

    return api_key, base_url, model
