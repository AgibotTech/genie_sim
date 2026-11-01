from __future__ import annotations

import hashlib
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    domain_id: Literal[15, 17]
    stats_path: Path
    stats_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    label: str = Field(min_length=1, max_length=160)
    label_en: str | None = Field(default=None, max_length=160)
    checkpoint: Path | None = None
    supported_domains: list[str] = Field(min_length=1)
    inference_domain_id_override: Literal[15, 17] | None = None
    verify_checkpoint_stats_pin: bool = False
    domains: dict[str, DomainConfig] = Field(default_factory=dict)
    inference_config: Path | None = None
    num_steps: int | None = Field(default=None, ge=1, le=200)
    action_temporal_alignment: Literal["current", "first_vision"] = "current"


class DreamConfig(BaseModel):
    _model_defaults: dict | None = PrivateAttr(default=None)
    model_config = ConfigDict(extra="forbid")
    backend: Literal["cosmos", "recorded", "mock"] = "cosmos"
    model_id: str = "g1-g2-mixed-ar-iter80000"
    checkpoint: Path | None = None
    model_profiles: list[ModelProfile] = Field(default_factory=list)
    default_model_id: str | None = None
    inference_domain_id_override: Literal[15, 17] | None = None
    verify_checkpoint_stats_pin: bool = True
    inference_config: Path | None = None
    cosmos_repo: Path | None = None
    example_roots: list[Path] = Field(default_factory=list)
    exclude_example_ids: list[str] = Field(default_factory=list)
    domains: dict[str, DomainConfig] = Field(default_factory=dict)
    output_dir: Path = Path("outputs/dream")
    # 32 = the original contract (demo/openloop).  16 = the G2 cloth
    # closed-loop configuration: the pi05 policy emits 50 actions per query,
    # and only 16 WM frames x 3 actions/frame fit that horizon.  Everything
    # downstream (runtime spec, contracts validation, platform frame check,
    # web display) reads this field; default 32 keeps the original behaviour.
    chunk_length: Literal[16, 32] = 32
    fps: Literal[10] = 10
    history_bucket_frames: tuple[Literal[1], Literal[1], Literal[1], Literal[1]] = (1, 1, 1, 1)
    history_sampling: Literal["endpoints_fixed"] = "endpoints_fixed"
    num_steps: int = Field(default=10, ge=1, le=200)
    action_temporal_alignment: Literal["current", "first_vision"] = "current"
    guidance: float = Field(default=1.0, ge=0)
    shift: float = Field(default=10.0, gt=0)
    seed: int = 42
    image_size: Literal[480] = 480
    torch_compile: bool = False
    max_sessions: int = Field(default=4, ge=1, le=100)
    max_chunks: int = Field(default=64, ge=1, le=1000)
    queue_size: int = Field(default=8, ge=1, le=100)
    idle_ttl_seconds: int = Field(default=3600, ge=60)
    max_request_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)
    max_image_pixels: int = Field(default=4_000_000, ge=1024)
    api_token_env: str = "DREAM_API_TOKEN"
    policy_url: str | None = None
    policy_token_env: str = "DREAM_POLICY_TOKEN"
    evaluator_url: str | None = None
    evaluator_token_env: str = "DREAM_EVALUATOR_TOKEN"
    evaluator_health_url: str | None = None
    evaluator_timeout_seconds: float = Field(default=600, gt=0, le=1800)
    vlm_models_config: Path | None = None
    vlm_default_model_key: str = "qwen3_5"
    remote_timeout_seconds: float = Field(default=120, gt=0, le=1800)

    @model_validator(mode="after")
    def validate_domains(self):
        expected = {"agibotworld_g1": 15, "agibotworld_g2": 17}
        for name, domain in self.domains.items():
            if expected.get(name) != domain.domain_id:
                raise ValueError(f"Unsupported domain contract: {name} / {domain.domain_id}")
        profile_ids = [profile.id for profile in self.model_profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("Model profile ids must be unique")
        if self.default_model_id and self.default_model_id not in profile_ids:
            raise ValueError("default_model_id must identify a configured model profile")
        if self.model_profiles and (self.default_model_id or self.model_id) not in profile_ids:
            raise ValueError("model_id must identify a configured model profile")
        for profile in self.model_profiles:
            for name, domain in profile.domains.items():
                if expected.get(name) != domain.domain_id or name not in profile.supported_domains:
                    raise ValueError("Model stats overrides must identify a supported domain")
            if any(domain not in expected for domain in profile.supported_domains):
                raise ValueError("Unsupported model domain")
            if self.backend != "mock" and any(
                domain not in self.domains for domain in profile.supported_domains
            ):
                raise ValueError("Every model domain must have configured stats")
            if profile.inference_domain_id_override is not None and not any(
                domain.domain_id == profile.inference_domain_id_override
                for domain in self.domains.values()
            ):
                raise ValueError("Inference domain id override must have configured domain stats")
        if self.inference_domain_id_override is not None and not any(
            domain.domain_id == self.inference_domain_id_override
            for domain in self.domains.values()
        ):
            raise ValueError("Inference domain id override must have configured domain stats")
        return self

    def for_model(self, model_id: str) -> DreamConfig:
        if not self.model_profiles:
            if model_id != self.model_id:
                raise ValueError(f"Unknown model profile: {model_id}")
            return self.model_copy(deep=True)
        profile = next((item for item in self.model_profiles if item.id == model_id), None)
        if profile is None:
            raise ValueError(f"Unknown model profile: {model_id}")
        defaults = self._model_defaults
        if defaults is None:
            defaults = {
                name: deepcopy(getattr(self, name))
                for name in ("checkpoint", "domains", "inference_config", "num_steps")
            }
        selected = self.model_copy(
            deep=True,
            update={
                "model_id": profile.id,
                "checkpoint": profile.checkpoint or defaults["checkpoint"],
                "inference_domain_id_override": profile.inference_domain_id_override,
                "verify_checkpoint_stats_pin": profile.verify_checkpoint_stats_pin,
                "domains": {**deepcopy(defaults["domains"]), **deepcopy(profile.domains)},
                "inference_config": profile.inference_config or defaults["inference_config"],
                "num_steps": profile.num_steps
                if profile.num_steps is not None
                else defaults["num_steps"],
                "action_temporal_alignment": profile.action_temporal_alignment,
            },
        )

        selected._model_defaults = deepcopy(defaults)
        return selected

    def verify_release(self) -> dict:
        if self.backend == "mock":
            return {"backend": "mock", "verified": True, "synthetic": True}
        if not self.checkpoint or not self.checkpoint.is_dir():
            raise ValueError("checkpoint must be an existing exported model directory")
        for name, domain in self.domains.items():
            if not domain.stats_path.is_file():
                raise ValueError(f"Missing {name} stats: {domain.stats_path}")
            if sha256_file(domain.stats_path) != domain.stats_sha256:
                raise ValueError(f"Stats SHA256 mismatch for {name}")
            stats = json.loads(domain.stats_path.read_text())
            metadata = stats.get("metadata", {})
            if int(stats.get("action_dim", metadata.get("action_dim", 29))) != 29:
                raise ValueError(f"{name} stats must describe 29D actions")
            if stats.get("action_source", metadata.get("action_source", "stateend")) != "stateend":
                raise ValueError(f"{name} stats must describe stateend actions")
        if not self.domains:
            raise ValueError("No domain stats configured")
        if self.backend == "cosmos":
            from .inference_runner import verify_checkpoint_stats_pin

            framework_root(self.cosmos_repo)
            if self.inference_config and not self.inference_config.is_file():
                raise ValueError("Missing inference_config")
            index = self.checkpoint / "model.safetensors.index.json"
            if not index.is_file():
                raise ValueError("Missing exported model.safetensors.index.json")
            weights = json.loads(index.read_text())["weight_map"]
            for shard in set(weights.values()):
                if not (self.checkpoint / shard).is_file():
                    raise ValueError(f"Missing checkpoint shard: {shard}")
            if self.verify_checkpoint_stats_pin:
                verify_checkpoint_stats_pin(
                    self.checkpoint,
                    {name: domain.stats_path for name, domain in self.domains.items()},
                )
        return {"backend": self.backend, "verified": True, "model_id": self.model_id}


def load_config(path: Path) -> DreamConfig:
    text = os.path.expandvars(path.read_text())
    if "${" in text:
        raise ValueError("Deployment configuration contains unresolved environment variables")
    payload = json.loads(text)
    config = DreamConfig.model_validate(payload)
    for field in (
        "checkpoint",
        "inference_config",
        "cosmos_repo",
        "output_dir",
        "vlm_models_config",
    ):
        value = getattr(config, field)
        if value is not None:
            setattr(config, field, value.expanduser().resolve())
    config.model_profiles = [
        profile.model_copy(
            update={
                "checkpoint": profile.checkpoint.expanduser().resolve()
                if profile.checkpoint is not None
                else None,
                "inference_config": profile.inference_config.expanduser().resolve()
                if profile.inference_config is not None
                else None,
                "domains": {
                    name: domain.model_copy(
                        update={"stats_path": domain.stats_path.expanduser().resolve()}
                    )
                    for name, domain in profile.domains.items()
                },
            }
        )
        for profile in config.model_profiles
    ]
    config.example_roots = [root.expanduser().resolve() for root in config.example_roots]
    for domain in config.domains.values():
        domain.stats_path = domain.stats_path.expanduser().resolve()
    return config.for_model(config.default_model_id or config.model_id)


def framework_root(cosmos_repo: Path | None) -> Path:
    """Accept either the standalone Framework checkout or the development tree."""
    if cosmos_repo is None:
        raise ValueError("cosmos_repo must point to a Cosmos Framework checkout")
    for candidate in (cosmos_repo, cosmos_repo / "codes/cosmos-framework"):
        if (candidate / "cosmos_framework").is_dir():
            return candidate
    raise ValueError("cosmos_repo must point to a Cosmos Framework checkout")
