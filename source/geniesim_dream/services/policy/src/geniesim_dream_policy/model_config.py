"""OpenPI pi0.5 serving transforms for Dream's 16D policy state and actions."""

from __future__ import annotations

import dataclasses
import logging
import pathlib

import numpy as np
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi import transforms as _transforms

from . import delta_actions

DEFAULT_PROMPT = "Complete the task shown in the images."

# Raw observation keys sent by the Dream / OpenPI policy client (dot-separated, flat).
HEAD_KEY = "observation.images.head"
LEFT_KEY = "observation.images.hand_left"
RIGHT_KEY = "observation.images.hand_right"
STATE_KEY = "observation.state"

REAL_DIM = 16   # meaningful action/state dims; the model is padded to 32
MODEL_DIM = 32


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce one camera image to uint8 HWC RGB."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[2] != 3:
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _square_resize_224(img_hwc_uint8: np.ndarray) -> np.ndarray:
    """Resize to the 224x224 input expected by the released policy."""
    from PIL import Image

    im = Image.fromarray(img_hwc_uint8, mode="RGB").resize((224, 224), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class DreamInputs(_transforms.DataTransformFn):
    """Map a Dream observation to the openpi pi05 model input dict."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _square_resize_224(_to_hwc_uint8(data[HEAD_KEY])),
            "left_wrist_0_rgb": _square_resize_224(_to_hwc_uint8(data[LEFT_KEY])),
            "right_wrist_0_rgb": _square_resize_224(_to_hwc_uint8(data[RIGHT_KEY])),
        }
        image_mask = {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }

        # The model consumes a 32D state; the last 16 coordinates are unused.
        state = np.asarray(data[STATE_KEY], dtype=np.float32).reshape(-1)
        state = _transforms.pad_to_dim(state, MODEL_DIM)
        state[REAL_DIM:] = 0.0

        inputs: dict = {"image": images, "image_mask": image_mask, "state": state}

        prompt = data.get("prompt")
        if prompt is not None:
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class DreamOutputs(_transforms.DataTransformFn):
    """Slice to 16 dims and reorder to Dream WM layout [L7, L_grip, R7, R_grip]."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[:, :REAL_DIM]  # (horizon, 16) in [L7, R7, L_grip, R_grip]
        out = np.empty_like(actions)
        out[:, 0:7] = actions[:, 0:7]     # L arm
        out[:, 7] = actions[:, 14]        # L gripper
        out[:, 8:15] = actions[:, 7:14]   # R arm
        out[:, 15] = actions[:, 15]       # R gripper
        return {"actions": out}


def make_config(
    asset_id: str = "dream",
    action_horizon: int = 50,
    compile_mode: str | None = None,
    delta_actions_enabled: bool = False,
) -> _config.TrainConfig:
    """Build OpenPI's serving configuration with the checkpoint's action space."""
    def data_transforms(model):
        group = _transforms.Group(
            inputs=[DreamInputs()],
            outputs=[DreamOutputs()],
        )
        return delta_actions.install(group) if delta_actions_enabled else group

    return _config.TrainConfig(
        name="pi05_dream",
        exp_name="serve",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=MODEL_DIM,
            action_horizon=action_horizon,
            pytorch_compile_mode=compile_mode,
        ),
        data=_config.SimpleDataConfig(
            assets=_config.AssetsConfig(asset_id=asset_id),
            data_transforms=data_transforms,
        ),
    )


def resolve_action_space(checkpoint: str | pathlib.Path, asset_id: str, requested: str) -> bool:
    """Read the checkpoint's action-space marker; older assets use absolute actions."""
    recorded = delta_actions.read_marker(pathlib.Path(checkpoint) / "assets", asset_id)
    if requested == "auto":
        if recorded is None:
            logging.info("action space: absolute (no marker in the checkpoint; legacy default)")
            return False
        logging.info(
            "action space: %s (from the checkpoint marker)",
            delta_actions.space_name(recorded),
        )
        return recorded
    wanted = requested == delta_actions.DELTA
    if recorded is not None and recorded != wanted:
        raise SystemExit(
            f"--action-space {requested} contradicts the checkpoint, which is marked "
            f"{delta_actions.space_name(recorded)}. Drop the flag to use the marker."
        )
    logging.info("action space: %s (forced by --action-space)", delta_actions.space_name(wanted))
    return wanted
