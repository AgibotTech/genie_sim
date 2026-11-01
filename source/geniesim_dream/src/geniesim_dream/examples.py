from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .config import DreamConfig
from .contracts import ActionChunk
from .vision import CONCAT_LAYOUT, HEAD_SIZE, VIEW_ROLES, compose_concat_view


@dataclass
class Example:
    example_id: str
    title: str
    domain: str
    caption: str
    first_frame: Path | None
    action_paths: list[Path]
    outputs: list[Path | None]
    spec: dict
    manifest: Path
    views: dict[str, Path] | None = None
    action_array: Any | None = None
    normalized: bool = True
    stats_sha256: str | None = None
    external_only: bool = False
    policy_profile: str | None = None
    policy_checkpoint: str | None = None
    chunk_length: int = 32
    source_kind: str | None = None

    @property
    def chunk_count(self) -> int:
        if self.action_array is not None:
            return int(self.action_array.shape[0])
        return len(self.action_paths)

    def public(self):
        return {
            "id": self.example_id,
            "title": self.title,
            "domain": self.domain,
            "caption": self.caption,
            "chunks": self.chunk_count,
            "chunk_length": self.chunk_length,
            "fps": 10,
            "policy": "external" if self.external_only else "recorded_action_replay",
            "external_only": self.external_only,
            "policy_profile": self.policy_profile,
            "policy_checkpoint": self.policy_checkpoint,
            "source_kind": self.source_kind,
            "recorded_outputs": bool(self.outputs)
            and all(path and path.is_file() for path in self.outputs),
            "stats_sha256": self.stats_sha256,
        }

    def action(self, step: int) -> ActionChunk:
        if step < 0 or step >= self.chunk_count:
            raise ValueError("Replay actions exhausted")
        if self.action_array is not None:
            action = self.action_array[step].tolist()
        else:
            action = json.loads(self.action_paths[step].read_text())
        return ActionChunk(action=action, normalized=self.normalized)

    def conditioning_image(self) -> Image.Image:
        if self.views is not None:
            images = {}
            for role, path in self.views.items():
                with Image.open(path) as image:
                    images[role] = image.convert("RGB")
            return compose_concat_view(images)
        if self.first_frame is None:
            raise ValueError("Example has no conditioning image")
        with Image.open(self.first_frame) as image:
            return image.convert("RGB")


def _asset_path(sample_dir: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Asset path must be a non-empty relative path")
    path = (sample_dir / relative_path).resolve()
    if not path.is_relative_to(sample_dir.resolve()):
        raise ValueError("Asset path escapes the example directory")
    return path


def _same_checkpoint(path: str | Path, checkpoint: Path | None) -> bool:
    if checkpoint is None:
        return False
    candidate = Path(path)
    try:
        return candidate.resolve() == checkpoint.resolve() or candidate.samefile(checkpoint)
    except OSError:
        return False


def _resolve_seed_fields(payload: dict) -> tuple[dict | None, str | None]:
    """Return (initial_state, policy_checkpoint), unwrapping nested seed_state packing."""
    state = payload.get("initial_state")
    policy_checkpoint = payload.get("policy_checkpoint")
    if not isinstance(state, dict):
        return None, policy_checkpoint if isinstance(policy_checkpoint, str) else None
    # Packed seed_state.json: {initial_state: {...}, policy_checkpoint, state_source, ...}
    if "policy_profile" not in state and isinstance(state.get("initial_state"), dict):
        if policy_checkpoint is None and isinstance(state.get("policy_checkpoint"), str):
            policy_checkpoint = state["policy_checkpoint"]
        state = state["initial_state"]
    if not isinstance(policy_checkpoint, str):
        policy_checkpoint = None
    return state, policy_checkpoint


def _closed_loop_id(payload: dict, open_id: str) -> str:
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        closed_id = provenance.get("closed_loop_id")
        if isinstance(closed_id, str) and closed_id:
            return closed_id
    return f"pi05-{open_id}"


def _source_kind(payload: dict) -> str | None:
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        kind = provenance.get("source_kind")
        if isinstance(kind, str) and kind:
            return kind
    return None


def _is_dual_use(payload: dict, action_info: Any, seed_state: dict | None) -> bool:
    """One directory with both replay actions and an explicit closed-loop seed id/file."""
    if not isinstance(action_info, dict):
        return False
    has_seed = isinstance(seed_state, dict) and bool(seed_state.get("policy_profile"))
    if not has_seed:
        return False
    provenance = payload.get("provenance")
    has_closed_id = isinstance(provenance, dict) and bool(provenance.get("closed_loop_id"))
    has_closed_seed = isinstance(payload.get("closed_loop_seed"), dict)
    # Require an explicit closed-loop marker so accidental actions on a pure
    # external_only seed still reject (see test_task_seeds).
    return bool(has_closed_id or has_closed_seed)


def _recorded_output_paths(
    manifest: Path, payload: dict, n_chunks: int
) -> list[Path | None]:
    recorded = payload.get("recorded_outputs") or {}
    if not isinstance(recorded, dict):
        recorded = {}
    recorded_paths = recorded.get("paths", [])
    if recorded_paths and len(recorded_paths) != n_chunks:
        raise ValueError("Recorded output list must have one entry per action chunk")
    output_paths: list[Path | None] = []
    for relative_path in recorded_paths or [None] * n_chunks:
        if relative_path is None:
            output_paths.append(None)
            continue
        output_path = _asset_path(manifest.parent, relative_path)
        output_paths.append(output_path if output_path.is_file() else None)
    return output_paths


def _load_action_array(manifest: Path, action_info: dict) -> Any:
    import numpy as np

    actions_path = _asset_path(manifest.parent, action_info["path"])
    if not actions_path.is_file():
        raise ValueError(f"Missing action array: {actions_path}")
    actions = np.load(actions_path, allow_pickle=False)
    expected_shape = tuple(action_info["shape"])
    if actions.shape != expected_shape or actions.ndim != 3:
        raise ValueError("Action array shape does not match the manifest")
    if actions.dtype != np.float32:
        raise ValueError("Actions must be float32")
    if actions.shape[2] != 29 or actions.shape[1] not in (16, 32):
        raise ValueError("Actions must have shape [chunks,16|32,29]")
    if not np.isfinite(actions).all():
        raise ValueError("Actions contain non-finite values")
    if action_info.get("sha256"):
        action_sha = hashlib.sha256(actions_path.read_bytes()).hexdigest()
        if action_sha != action_info["sha256"]:
            raise ValueError("Action array SHA256 mismatch")
    return actions


def _discover_compact_example(manifest: Path, config: DreamConfig) -> list[Example]:
    """Load one compact manifest into one or two catalog entries (dual open+closed)."""
    from PIL import Image

    payload = json.loads(manifest.read_text())
    example_format = payload.get("format")
    if example_format not in {"geniesim-dream-example-v1", "geniesim-dream-example-v2"}:
        raise ValueError("Unsupported Dream example format")
    example_id = payload["id"]
    domain = "agibotworld_g1" if payload["domain"] == "agibotworld" else payload["domain"]
    external_only = payload.get("external_only", False)
    if not isinstance(external_only, bool):
        raise ValueError("external_only must be a boolean")
    if domain not in config.domains and config.backend != "mock":
        raise ValueError("Example domain is not configured")

    seed_state, policy_checkpoint = _resolve_seed_fields(payload)
    action_info = payload.get("actions")
    dual = _is_dual_use(payload, action_info, seed_state)

    # Images and measured joint states have no WM normalization. Only examples
    # containing replay actions are tied to the stats used to encode those actions.
    needs_stats = (not external_only) or dual
    if domain in config.domains and needs_stats:
        expected_sha = config.domains[domain].stats_sha256
        if payload.get("domain_stats_sha256") != expected_sha:
            raise ValueError("Example stats do not match configured domain stats")
    if config.backend == "recorded" and not _same_checkpoint(
        payload["checkpoint"], config.checkpoint
    ):
        raise ValueError("Example checkpoint does not match selected release")

    contract = payload["contract"]
    if contract["chunk_length"] not in (16, 32) or contract["fps"] != 10:
        raise ValueError("Example must have chunk_length 16 or 32 at 10 FPS")
    if contract["history_bucket_frames"] != [1, 1, 1, 1]:
        raise ValueError("Example is not K4 history 1,1,1,1")
    if contract["history_sampling"] != "endpoints_fixed":
        raise ValueError("Example history sampling mismatch")
    if contract["num_conditional_frames"] != 1:
        raise ValueError("Example must have one conditioning frame")

    observation = payload["observation"]
    image_path = None
    view_paths = None
    if example_format == "geniesim-dream-example-v1":
        if observation["layout"] != "concat_view":
            raise ValueError("Example observation must use concat_view")
        if observation.get("native_views_available"):
            raise ValueError("Legacy examples must provide a prepared concat_view PNG")
        image_path = _asset_path(manifest.parent, observation["path"])
        if not image_path.is_file():
            raise ValueError(f"Missing conditioning image: {image_path}")
        with Image.open(image_path) as image:
            if image.size != (observation["width"], observation["height"]):
                raise ValueError("Conditioning image dimensions do not match the manifest")
        image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if observation.get("sha256") and image_sha != observation["sha256"]:
            raise ValueError("Conditioning image SHA256 mismatch")
    else:
        if observation["layout"] != "three_views":
            raise ValueError("Example observation must contain three separate views")
        composition = observation["concat"]
        if (
            composition["width"],
            composition["height"],
            composition["layout"],
        ) != (640, 600, CONCAT_LAYOUT):
            raise ValueError("Three-view composition does not match the documented layout")
        view_specs = observation["views"]
        if set(view_specs) != set(VIEW_ROLES):
            raise ValueError("Exactly head, left_wrist, and right_wrist views are required")
        view_paths = {}
        for role in VIEW_ROLES:
            view_spec = view_specs[role]
            view_path = _asset_path(manifest.parent, view_spec["path"])
            if not view_path.is_file():
                raise ValueError(f"Missing {role} view: {view_path}")
            with Image.open(view_path) as image:
                if image.size != (view_spec["width"], view_spec["height"]):
                    raise ValueError(f"{role} view dimensions do not match the manifest")
                if role == "head" and image.size != HEAD_SIZE:
                    raise ValueError("Head view dimensions do not match the documented layout")
                if image.width * image.height > config.max_image_pixels:
                    raise ValueError(f"{role} view exceeds the pixel limit")
            view_sha = hashlib.sha256(view_path.read_bytes()).hexdigest()
            if not view_spec.get("sha256") or view_sha != view_spec["sha256"]:
                raise ValueError(f"{role} view SHA256 mismatch")
            view_paths[role] = view_path

    title = payload["title"]
    caption = payload.get("caption", "")
    source_kind = _source_kind(payload)

    if dual:
        if config.backend == "recorded":
            raise ValueError("Dual-use seeds require a fresh-inference backend for closed-loop")
        if not isinstance(seed_state, dict) or not seed_state.get("policy_profile"):
            raise ValueError("Dual-use examples require initial_state with policy_profile")
        actions = _load_action_array(manifest, action_info)
        if int(contract["chunk_length"]) != int(actions.shape[1]):
            raise ValueError("contract.chunk_length must match actions.shape[1]")
        output_paths = _recorded_output_paths(manifest, payload, actions.shape[0])
        closed_id = _closed_loop_id(payload, example_id)
        chunk_length = int(actions.shape[1])
        replay = Example(
            example_id=example_id,
            title=title,
            domain=domain,
            caption=caption,
            first_frame=image_path,
            action_paths=[],
            outputs=output_paths,
            spec=contract,
            manifest=manifest,
            views=view_paths,
            action_array=actions,
            normalized=action_info["normalized"],
            stats_sha256=payload["domain_stats_sha256"],
            policy_profile=seed_state.get("policy_profile"),
            policy_checkpoint=policy_checkpoint,
            chunk_length=chunk_length,
            source_kind=source_kind,
        )
        closed = Example(
            example_id=closed_id,
            title=title,
            domain=domain,
            caption=caption,
            first_frame=image_path,
            action_paths=[],
            outputs=[],
            spec=contract,
            manifest=manifest,
            views=view_paths,
            stats_sha256=None,
            external_only=True,
            policy_profile=seed_state["policy_profile"],
            policy_checkpoint=policy_checkpoint,
            chunk_length=chunk_length,
            source_kind=source_kind,
        )
        return [replay, closed]

    if external_only:
        if action_info is not None or payload.get("recorded_outputs"):
            raise ValueError(
                "External-only seeds cannot contain replay actions or recorded outputs"
            )
        if config.backend == "recorded":
            raise ValueError("External-only seeds require a fresh-inference backend")
        if not isinstance(seed_state, dict) or not seed_state.get("policy_profile"):
            raise ValueError("External-only seeds require initial_state with policy_profile")
        return [
            Example(
                example_id=example_id,
                title=title,
                domain=domain,
                caption=caption,
                first_frame=image_path,
                action_paths=[],
                outputs=[],
                spec=contract,
                manifest=manifest,
                views=view_paths,
                stats_sha256=None,
                external_only=True,
                policy_profile=seed_state["policy_profile"],
                policy_checkpoint=policy_checkpoint,
                source_kind=source_kind,
            )
        ]

    if not isinstance(action_info, dict):
        raise ValueError(
            "Replay example requires actions; seed-only examples must set external_only"
        )
    actions = _load_action_array(manifest, action_info)
    if int(contract["chunk_length"]) != int(actions.shape[1]):
        raise ValueError("contract.chunk_length must match actions.shape[1]")
    output_paths = _recorded_output_paths(manifest, payload, actions.shape[0])
    return [
        Example(
            example_id=example_id,
            title=title,
            domain=domain,
            caption=caption,
            first_frame=image_path,
            action_paths=[],
            outputs=output_paths,
            spec=contract,
            manifest=manifest,
            views=view_paths,
            action_array=actions,
            normalized=action_info["normalized"],
            stats_sha256=payload["domain_stats_sha256"],
            policy_profile=(seed_state or {}).get("policy_profile") if seed_state else None,
            policy_checkpoint=policy_checkpoint,
            chunk_length=int(actions.shape[1]),
            source_kind=source_kind,
        )
    ]


def discover_examples(config: DreamConfig) -> tuple[dict[str, Example], list[dict]]:
    examples, rejected = {}, []
    excluded = set(config.exclude_example_ids)
    for root in config.example_roots:
        if not root.is_dir():
            rejected.append({"source": str(root), "error": "example root missing"})
            continue
        for manifest in sorted(root.rglob("manifest.json")):
            try:
                discovered = _discover_compact_example(manifest, config)
                for example in discovered:
                    if example.example_id in excluded:
                        continue
                    previous = examples.get(example.example_id)
                    if previous and previous.outputs:
                        example.outputs = previous.outputs
                    examples[example.example_id] = example
            except (ValueError, KeyError, OSError, TypeError) as error:
                rejected.append({"source": str(manifest), "error": str(error)})
    return examples, rejected
