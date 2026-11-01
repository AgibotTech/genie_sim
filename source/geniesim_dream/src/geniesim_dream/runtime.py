from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .cosmos_adapter import (
    attach_sample_data,
    build_action_data,
    build_sample_spec,
    install_action_alignment,
)
from .config import DreamConfig, framework_root
from .contracts import ActionChunk
from .examples import Example
from .inference_runner import make_pipe, verify_checkpoint_stats_pin

logger = logging.getLogger(__name__)


class Runtime(Protocol):
    def generate(
        self,
        *,
        image: Path,
        history: list[Path],
        chunk: ActionChunk,
        domain: str,
        caption: str,
        step: int,
        workdir: Path,
        example: Example | None,
        num_steps: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


def history_indices(frame_count: int) -> list[int]:
    if frame_count < 2:
        return [0, 0, 0, 0]
    return [int(round(position)) for position in np.linspace(0, frame_count - 2, 4)]


class MockRuntime:
    def generate(self, *, image, chunk, workdir=None, **kwargs):
        initial = np.array(Image.open(image).convert("RGB"))
        frames = [initial]
        for index in range(len(chunk.action)):
            frame = initial.copy()
            brightness = int(np.clip(np.mean(chunk.action[index]) * 12, -20, 20))
            frame = np.clip(frame.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
            frames.append(frame)
        # Mirror CosmosRuntime so finish can assemble session actions.npy in tests.
        if workdir is not None:
            action = np.asarray(chunk.action, dtype=np.float32)
            Path(workdir).mkdir(parents=True, exist_ok=True)
            (Path(workdir) / "action.normalized.json").write_text(
                json.dumps(action.tolist(), allow_nan=False)
            )
        return np.stack(frames)

    def close(self):
        pass


class RecordedRuntime:
    def generate(self, *, chunk, example, step, **kwargs):
        from safetensors.numpy import load_file

        if example is None:
            raise ValueError("recorded backend requires an existing example")
        expected = example.action(step)
        if not chunk.normalized or not np.array_equal(chunk.action, expected.action):
            raise ValueError("Recorded backend only accepts the exact recorded normalized actions")
        output = example.outputs[step]
        if output is None or not output.is_file():
            raise ValueError("Recorded output is missing; choose cosmos for fresh inference")
        vision = load_file(str(output))["vision"]
        return np.rint(np.clip(vision.transpose(1, 2, 3, 0), 0, 1) * 255).astype(np.uint8)

    def close(self):
        pass


class CosmosRuntime:
    def __init__(self, config: DreamConfig):
        if not os.environ.get("_G01_NVIDIA_LIBS_READY"):
            raise RuntimeError("Launch through geniesim-dream serve to prepare NVIDIA libraries")
        self.config = config
        framework = framework_root(config.cosmos_repo)
        if str(framework) not in sys.path:
            sys.path.insert(0, str(framework))
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if config.verify_checkpoint_stats_pin:
            verify_checkpoint_stats_pin(
                config.checkpoint,
                {name: domain.stats_path for name, domain in config.domains.items()},
            )
        else:
            logger.warning(
                "Checkpoint stats pin comparison is disabled; using configured domain stats"
            )
        from cosmos_framework.data.generator.action.action_normalization import (
            load_action_stats,
            normalize_action,
        )
        import torch

        self.torch = torch
        self.normalize = normalize_action
        self.stats = {}
        self.methods = {}
        for name, domain in config.domains.items():
            metadata = json.loads(domain.stats_path.read_text(encoding="utf-8"))
            if int(metadata.get("action_dim", 29)) != 29:
                raise ValueError(f"{name} stats are not 29D")
            self.methods[name] = metadata.get("method", "quantile")
            self.stats[name] = {
                key: torch.from_numpy(value)
                for key, value in load_action_stats(str(domain.stats_path)).items()
            }
        original_cwd = Path.cwd()
        model_dir = config.output_dir / "_models" / config.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chdir(framework)
            self.pipe, self.setup = make_pipe(
                config.checkpoint,
                model_dir,
                inference_config=config.inference_config,
                torch_compile=config.torch_compile,
            )
            install_action_alignment(self.pipe.model, alignment=config.action_temporal_alignment)
        finally:
            os.chdir(original_cwd)
        self.overrides = self.setup.get_sample_overrides_cls()

    def generate(
        self,
        *,
        image,
        history,
        chunk,
        domain,
        caption,
        step,
        workdir,
        example,
        num_steps=None,
        seed=None,
    ):
        from safetensors.torch import load_file

        if domain not in self.stats:
            raise ValueError("Unknown inference domain")
        # The session chose its chunk length; the action's own row count IS
        # that choice.  Follow it faithfully: 16 rows -> 16-frame generation,
        # 32 rows -> 32-frame generation, anything else is a contract break.
        rows = len(chunk.action)
        if rows not in (16, 32):
            raise ValueError(
                f"action has {rows} rows; the contract allows 16 or 32 "
                "(SessionRequest.chunk_length selects which)"
            )
        action = self.torch.tensor(chunk.action, dtype=self.torch.float32)
        if not chunk.normalized:
            action = self.normalize(action, self.methods[domain], self.stats[domain])
        if not self.torch.isfinite(action).all():
            raise ValueError("Normalized action contains non-finite values")
        action_path = workdir / "action.normalized.json"
        action_path.write_text(json.dumps(action.tolist(), allow_nan=False))
        sample_domain = domain
        if self.config.inference_domain_id_override is not None:
            sample_domain = next(
                name
                for name, domain_config in self.config.domains.items()
                if domain_config.domain_id == self.config.inference_domain_id_override
            )
        spec = build_sample_spec(
            caption=caption,
            action_path=action_path,
            vision_path=image,
            fps=10,
            num_steps=self.config.num_steps if num_steps is None else num_steps,
            guidance=self.config.guidance,
            shift=self.config.shift,
            seed=self.config.seed if seed is None else seed,
            image_size=self.config.image_size,
            action_chunk_size=rows,
        )
        if len(history) < 2:
            anchors = [image] * 4
        else:
            anchors = [history[index] for index in history_indices(len(history))]
        bucket_paths = []
        for index, anchor in enumerate(anchors):
            bucket = workdir / f"history_b{index}.png"
            frame = np.array(Image.open(anchor).convert("RGB"))
            Image.fromarray(frame).save(bucket, format="PNG")
            bucket_paths.append(str(bucket))
        batch = build_action_data(
            spec=spec,
            history_paths=[Path(path) for path in bucket_paths],
            domain_id=self.config.domains[sample_domain].domain_id,
            input_video_key=self.pipe.model_config.input_video_key,
            max_action_dim=self.pipe.model_config.max_action_dim,
        )
        spec_path = workdir / "generation.spec.json"
        spec_path.write_text(json.dumps(spec))
        override = self.overrides.from_files([spec_path], overrides=self.setup.sample_overrides)[0]
        override.output_dir = workdir / "generation"
        override.download(override.output_dir / "inputs")
        sample = attach_sample_data(
            override.build_sample(model_config=self.pipe.model_config), batch
        )
        self.pipe.generate([sample])
        vision = load_file(str(override.output_dir / "output.safetensors"))["vision"]
        return (
            (vision.permute(1, 2, 3, 0).clamp(0, 1) * 255)
            .round()
            .to("cpu")
            .numpy()
            .astype(np.uint8)
        )

    def close(self):
        self.pipe = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def create_runtime(config: DreamConfig) -> Runtime:
    config.verify_release()
    if config.backend == "cosmos":
        return CosmosRuntime(config)
    if config.backend == "recorded":
        return RecordedRuntime()
    return MockRuntime()
