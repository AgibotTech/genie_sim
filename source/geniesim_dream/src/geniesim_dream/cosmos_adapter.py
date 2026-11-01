"""Dream's G1/G2 action input adapter for the unmodified Cosmos Framework.

The Framework's prebuilt-sample hook lets Dream supply its own five-item K4
batch. Only the action mRoPE position needs a model-instance adjustment: the
public packer anchors actions to the first vision item, while this checkpoint
anchors them to the current (last) item.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DOMAIN_IDS = {"agibotworld_g1": 15, "agibotworld_g2": 17}


def build_sample_spec(
    *,
    caption: str,
    action_path: Path,
    vision_path: Path,
    fps: int,
    num_steps: int,
    guidance: float,
    shift: float,
    seed: int,
    image_size: int,
    action_chunk_size: int,
) -> dict[str, Any]:
    """Use only fields understood by NVIDIA's ed8287f sample schema."""
    return {
        "name": "generation",
        "prompt": caption,
        "model_mode": "forward_dynamics",
        "action_chunk_size": action_chunk_size,
        "action_path": str(action_path.resolve()),
        "domain_name": "agibotworld",
        "view_point": "concat_view",
        "vision_path": str(vision_path.resolve()),
        "image_size": image_size,
        "num_frames": action_chunk_size + 1,
        "fps": fps,
        "num_steps": num_steps,
        "guidance": guidance,
        "shift": shift,
        "seed": seed,
    }


def build_action_data(
    *,
    spec: dict[str, Any],
    history_paths: list[Path],
    domain_id: int,
    input_video_key: str,
    max_action_dim: int,
    device: str = "cuda",
) -> dict[str, Any]:
    """Build a flat [K4 history, current window] batch with public helpers."""
    import torch

    from cosmos_framework.data.generator.action.action_processing import pad_action_to_max_dim
    from cosmos_framework.data.generator.action.transforms import reflection_pad_to_target
    from cosmos_framework.inference.action import build_action_batch
    from cosmos_framework.inference.args import ModelMode
    from cosmos_framework.inference.vision import read_media_frames

    if len(history_paths) != 4:
        raise ValueError(f"K4 requires four history images, got {len(history_paths)}")
    if domain_id not in DOMAIN_IDS.values():
        raise ValueError(f"Unsupported action domain ID: {domain_id}")
    raw = torch.tensor(json.loads(Path(spec["action_path"]).read_text()), dtype=torch.float32)
    if raw.shape != (spec["action_chunk_size"], 29):
        raise ValueError(f"Expected [{spec['action_chunk_size']}, 29] actions, got {tuple(raw.shape)}")
    current, _ = read_media_frames(Path(spec["vision_path"]), max_frames=spec["action_chunk_size"] + 1)
    batch = build_action_batch(
        video=current,
        action=pad_action_to_max_dim(raw, max_action_dim),
        raw_action_dim=29,
        prompt=spec["prompt"],
        view_point=spec["view_point"],
        domain_name="agibotworld",
        model_mode=ModelMode.FORWARD_DYNAMICS,
        action_chunk_size=spec["action_chunk_size"],
        fps=spec["fps"],
        resolution=str(spec["image_size"]),
        input_video_key=input_video_key,
        device=device,
    )
    current_video = batch[input_video_key][0][0]
    current_size = batch["image_size"]
    target_h, target_w = current_video.shape[-2:]
    videos = []
    sizes = []
    for path in history_paths:
        history, _ = read_media_frames(path, max_frames=1)
        padded = {"video": history}
        reflection_pad_to_target(
            padded, ["video"], keep_aspect_ratio=True, target_w=target_w, target_h=target_h
        )
        videos.append(padded["video"])
        sizes.append(padded["image_size"].unsqueeze(0).to(device=device))
    batch[input_video_key] = [*videos, current_video]
    batch["image_size"] = [*sizes, current_size]
    batch["num_vision_items_per_sample"] = [5]
    batch["domain_id"] = [torch.tensor(domain_id, dtype=torch.long)]
    return batch


def attach_sample_data(sample: Any, batch: dict[str, Any]) -> Any:
    """Supply the prebuilt batch through Framework's SampleArgs.get_data hook."""
    from pydantic import PrivateAttr

    class DreamSampleArgs(type(sample)):
        _dream_data: dict[str, Any] = PrivateAttr()

        def get_data(self, *, device: str | int = "cpu") -> dict[str, Any]:
            return self._dream_data

    wrapped = DreamSampleArgs.model_validate(sample.model_dump())
    wrapped._dream_data = batch
    return wrapped


def install_action_alignment(model: Any, *, alignment: str) -> None:
    """Align action positions to the current item on this model instance only."""
    if alignment == "first_vision":
        return
    if alignment != "current":
        raise ValueError(f"Unsupported action temporal alignment: {alignment}")
    original_pack = model._pack_input_sequence

    def pack_with_current_action_position(*args: Any, **kwargs: Any) -> Any:
        packed = original_pack(*args, **kwargs)
        plans = args[0] if args else kwargs["sequence_plans"]
        data = args[2] if len(args) > 2 else kwargs["gen_data_clean"]
        return align_packed_action_to_current(packed, plans, data)

    model._pack_input_sequence = pack_with_current_action_position


def align_packed_action_to_current(packed: Any, plans: list[Any], data: Any) -> Any:
    """Shift only action mRoPE positions from history start to current start."""
    counts = data.num_vision_items_per_sample
    if len(plans) != 1 or counts != [5] or packed.vision is None or packed.action is None:
        raise ValueError("Dream action alignment requires one five-item action sample")
    starts = {
        span.payload_index: span.sequence_start
        for span in packed.vision.spans
        if span.payload_start == 0
    }
    if set(starts) != set(range(5)):
        raise ValueError("Unexpected packed K4 vision layout")
    delta = packed.position_ids[0, starts[4]] - packed.position_ids[0, starts[0]]
    packed.position_ids[0, packed.action.sequence_indexes] += delta
    return packed
