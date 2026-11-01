"""The small inference interface needed from Cosmos Framework by Dream."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint_stats_pin(checkpoint: Path, stats_paths: dict[str, Path]) -> dict[str, str]:
    """Bind every configured domain to the stats used with this checkpoint."""
    pin_path = checkpoint / "world_model_stats_pin.json"
    try:
        payload = json.loads(pin_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read checkpoint stats pin: {pin_path}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Invalid checkpoint stats pin: {pin_path}")
    pins = payload.get("domains")
    if not isinstance(pins, dict) or set(pins) != set(stats_paths):
        raise ValueError(f"Checkpoint stats pin domains differ from configuration: {pin_path}")
    for name, path in stats_paths.items():
        expected = pins[name]
        if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError(f"Invalid checkpoint stats SHA256 for {name}: {pin_path}")
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"Checkpoint stats SHA256 mismatch for {name}: {path}")
    return pins


def prepare_inference_config(checkpoint: Path, scratch_dir: Path) -> Path:
    """Resolve portable checkpoint paths for Framework's current process."""
    source = checkpoint / "config.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    model = payload["model"]["config"]
    # These inactive fields belonged to the local Framework fork. Keep an
    # existing checkpoint config readable by NVIDIA's unmodified schema.
    flow = model.get("rectified_flow_inference_config", {})
    for key in (
        "bridge_condition_frame_index",
        "bridge_condition_inference",
        "bridge_condition_inference_fixed_sigma",
        "bridge_condition_inference_sigma",
        "bridge_condition_inference_timestep_sigma",
    ):
        flow.pop(key, None)
    fields = (
        (model["tokenizer"], "vae_path", "file"),
        (model["vlm_config"]["tokenizer"], "pretrained_model_name", "directory"),
    )
    for node, key, kind in fields:
        configured = Path(node[key]).expanduser()
        resolved = (configured if configured.is_absolute() else checkpoint / configured).resolve()
        if not (resolved.is_file() if kind == "file" else resolved.is_dir()):
            raise FileNotFoundError(f"World-model {key} is missing: {resolved}")
        node[key] = str(resolved)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = scratch_dir / "_runtime_model_config.json"
    runtime_config.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return runtime_config


def make_pipe(
    checkpoint: Path,
    scratch_dir: Path,
    *,
    inference_config: Path | None = None,
    torch_compile: bool = False,
):
    """Create one Framework inference pipeline for all Dream sessions."""
    from cosmos_framework.inference.args import OmniSetupOverrides
    from cosmos_framework.inference.common.init import init_output_dir, init_script

    scratch_dir.mkdir(parents=True, exist_ok=True)
    if inference_config is None:
        inference_config = prepare_inference_config(checkpoint, scratch_dir)
    if not inference_config.is_file():
        raise ValueError(f"Inference config is missing: {inference_config}")
    init_script()
    setup = OmniSetupOverrides(
        checkpoint_path=str(checkpoint),
        config_file=str(inference_config),
        output_dir=scratch_dir,
        parallelism_preset="latency",
        guardrails=False,
        debug=True,
        use_torch_compile=torch_compile,
    ).build_setup()
    init_output_dir(setup.output_dir)
    return setup.get_inference_cls().create(setup), setup
