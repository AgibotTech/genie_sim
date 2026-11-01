"""Load an OpenPI checkpoint without importing its training data loader.

OpenPI's policy_config.create_trained_policy imports training.checkpoints,
which imports LeRobot's dataset loader at module import time. Serving only
needs the checkpoint's normalization assets and the public Policy transforms.
"""

from __future__ import annotations

import logging
from pathlib import Path

import jax.numpy as jnp

from openpi import transforms
from openpi.models import model as model_module
from openpi.policies import policy as policy_module
from openpi.shared import download, normalize


def load_policy(config, checkpoint: str | Path, *, default_prompt: str, device: str | None):
    checkpoint_dir = download.maybe_download(str(checkpoint))
    weight_path = checkpoint_dir / "model.safetensors"
    is_pytorch = weight_path.is_file()
    if is_pytorch:
        model = config.model.load_pytorch(config, str(weight_path))
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = config.model.load(
            model_module.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        )

    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load checkpoint normalization stats")
    stats_dir = checkpoint_dir / "assets" / data_config.asset_id
    norm_stats = normalize.load(stats_dir)
    logging.info("Loaded OpenPI norm stats from %s", stats_dir)

    if is_pytorch and device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    return policy_module.Policy(
        model,
        transforms=[
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=device if is_pytorch else None,
    )
