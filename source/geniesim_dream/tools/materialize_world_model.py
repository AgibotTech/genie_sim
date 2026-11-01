"""Install local inference paths and a stats pin beside downloaded WM weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PACKAGE_ROOT / "configs/world_model.template.json"
WAN_REPO = "Wan-AI/Wan2.2-TI2V-5B"
WAN_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
QWEN_REPO = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_missing_public_assets(vae: Path, tokenizer: Path) -> None:
    """Fetch only absent VAE/tokenizer files from their pinned publishers."""
    missing_tokenizer = [name for name in TOKENIZER_FILES if not (tokenizer / name).is_file()]
    if vae.is_file() and not missing_tokenizer:
        return
    from huggingface_hub import hf_hub_download

    if not vae.is_file():
        if vae.name != "Wan2.2_VAE.pth":
            raise ValueError("Automatic VAE download requires the Wan2.2_VAE.pth filename")
        vae.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(WAN_REPO, vae.name, revision=WAN_REVISION, local_dir=vae.parent)
        print(f"Downloaded {WAN_REPO}/{vae.name} to {vae}", file=sys.stderr)
    if missing_tokenizer:
        tokenizer.mkdir(parents=True, exist_ok=True)
        for name in missing_tokenizer:
            hf_hub_download(QWEN_REPO, name, revision=QWEN_REVISION, local_dir=tokenizer)
            print(f"Downloaded {QWEN_REPO}/{name} to {tokenizer}", file=sys.stderr)


def materialize(
    checkpoint: Path,
    vae: Path,
    tokenizer: Path,
    g1_stats: Path,
    g2_stats: Path,
    *,
    replace: bool = False,
    download_missing: bool = False,
) -> dict:
    checkpoint = checkpoint.expanduser().absolute()
    # Keep dependencies relative to the checkpoint. Dream resolves these paths
    # before handing the config to Framework.
    vae = vae.expanduser().absolute()
    tokenizer = tokenizer.expanduser().absolute()
    stats = {
        "agibotworld_g1": g1_stats.expanduser().resolve(),
        "agibotworld_g2": g2_stats.expanduser().resolve(),
    }
    if not checkpoint.is_dir() or not (checkpoint / "model.safetensors.index.json").is_file():
        raise ValueError("Checkpoint needs a model.safetensors.index.json and weight shards")
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(name for name in set(weight_map.values()) if not (checkpoint / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing weight shards: {missing}")
    if download_missing:
        download_missing_public_assets(vae, tokenizer)
    if not vae.is_file() or any(not (tokenizer / name).is_file() for name in TOKENIZER_FILES):
        raise FileNotFoundError("VAE and five Qwen tokenizer files are required; pass --download-missing")
    if any(not path.is_file() for path in stats.values()):
        raise FileNotFoundError("Both public stats files are required")

    config = json.loads(TEMPLATE.read_text())
    model_config = config["model"]["config"]
    if model_config["tokenizer"]["vae_path"] != "__DREAM_VAE_PATH__":
        raise ValueError("Unexpected VAE marker in template")
    vlm_tokenizer = model_config["vlm_config"]["tokenizer"]
    if vlm_tokenizer["pretrained_model_name"] != "__DREAM_TOKENIZER_PATH__":
        raise ValueError("Unexpected tokenizer marker in template")
    model_config["tokenizer"]["vae_path"] = os.path.relpath(vae, checkpoint)
    vlm_tokenizer["pretrained_model_name"] = os.path.relpath(tokenizer, checkpoint)

    config_bytes = (json.dumps(config, indent=2, sort_keys=True) + "\n").encode()
    pin = {
        "schema_version": 1,
        "domains": {name: sha256(path) for name, path in stats.items()},
    }
    pin_bytes = (json.dumps(pin, indent=2) + "\n").encode()
    destinations = {
        checkpoint / "config.json": config_bytes,
        checkpoint / "world_model_stats_pin.json": pin_bytes,
    }
    conflicting = [
        str(path)
        for path, contents in destinations.items()
        if path.exists() and path.read_bytes() != contents
    ]
    if conflicting and not replace:
        raise FileExistsError(f"Existing asset metadata differs; pass --replace: {conflicting}")
    for path, contents in destinations.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_bytes() != contents:
            path.write_bytes(contents)
    return {
        "checkpoint": str(checkpoint),
        "stats_sha256": pin["domains"],
        "config_sha256": sha256(checkpoint / "config.json"),
        "weight_shards": len(set(weight_map.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--g1-stats", type=Path, required=True)
    parser.add_argument("--g2-stats", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.checkpoint,
                args.vae,
                args.tokenizer,
                args.g1_stats,
                args.g2_stats,
                replace=args.replace,
                download_missing=args.download_missing,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
