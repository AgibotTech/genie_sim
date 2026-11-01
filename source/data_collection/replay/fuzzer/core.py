"""
Fuzzer supports two material modes:
  - scene_scan: discover peer materials in the scene and choose randomly
  - config_file: cycle through combinations from material_config.json
It also randomizes light values and cycles through candidate backgrounds.
The replay object supplies environment paths; fuzzing logic stays here.
"""
import json
import os
import random
from typing import Any

import omni.usd

from common.base_utils.logger import logger
from replay.config import FuzzingConfig
from .parse import collect_material_config, material_choices_from_config, shuffle_material_config
from .set import (
    set_lights_random,
    set_materials_by_choices,
    set_background,
    collect_scene_material_info,
    randomize_scene_materials,
)


class Fuzzer:
    """
    Apply scene-scan or config-file material variants to replay clones.

    Distant lights vary in angle and intensity; other lights vary in intensity.
    Candidate backgrounds are reused cyclically.
    """

    def __init__(
        self,
        input_dir: str,
        config: FuzzingConfig | None = None,
        sim_assets: str | None = None,
        objects_dir_name: str = "Objects",
    ):
        cfg = config or FuzzingConfig()
        self._input_dir = (input_dir or "").rstrip("/")
        self._sim_assets = sim_assets or os.environ.get("SIM_ASSETS", "")
        self._enable_light_fuzz = cfg.enable_light_fuzz
        self._enable_background_fuzz = cfg.enable_background_fuzz
        self._enable_material_fuzz = cfg.enable_material_fuzz
        self._material_fuzz_mode = cfg.material_fuzz_mode or "scene_scan"
        self._no_fuzzing_set = set(cfg.no_fuzzing_object_id_list or [])
        self._num_rounds = cfg.num_rounds
        self._rng = random.Random(cfg.seed)
        self._background_lists = cfg.backgrounds if cfg.backgrounds else None
        self._round_count = 0
        self._background_list_shuffled: list[dict[str, Any]] = []
        self._background_cursor = 0
        self._objects_dir_name = objects_dir_name
        self._scene_material_info: dict[str, list[str]] = {}
        self._object_asset_dict: dict = {}
        self._material_config: dict = {}
        self._shuffled_material_config: dict | None = None
        self._all_configs: list[dict[str, Any]] = []
        self._material_cursor = 0

    def _load_object_asset_dict(self) -> dict:
        path = os.path.join(self._input_dir, "recording_info.json")
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("object_asset_dict", {}) or {}
        except Exception as e:
            logger.warning("Fuzzer: failed to load object_asset_dict from %s: %s", path, e)
            return {}

    def generate_all(self) -> int:
        """Prepare material combinations and backgrounds for the selected mode."""
        self._scene_material_info = {}
        self._round_count = 0
        self._material_cursor = 0

        if self._material_fuzz_mode in ("config_file", "both") and self._enable_material_fuzz:
            self._object_asset_dict = self._load_object_asset_dict()
            if self._object_asset_dict and self._sim_assets:
                self._material_config = collect_material_config(
                    self._sim_assets,
                    self._object_asset_dict,
                    no_fuzzing_set=self._no_fuzzing_set,
                    objects_dir_name=self._objects_dir_name,
                )
            self._shuffled_material_config = shuffle_material_config(self._material_config, self._rng)
            mat_combos = material_choices_from_config(self._shuffled_material_config)
            if not mat_combos:
                mat_combos = [{}]
            self._all_configs = [{"material": m} for m in mat_combos]
            logger.info("Fuzzer: config_file material configs: %d", len(self._all_configs))
        else:
            self._all_configs = []
            self._shuffled_material_config = {}

        if self._background_lists:
            self._background_list_shuffled = list(self._background_lists)
            self._rng.shuffle(self._background_list_shuffled)
        else:
            self._background_list_shuffled = []
        self._background_cursor = 0
        logger.info(
            "Fuzzer: generate_all done (material=%s mode=%s, light=%s, background=%s)",
            self._enable_material_fuzz, self._material_fuzz_mode,
            self._enable_light_fuzz, self._enable_background_fuzz,
        )
        return max(len(self._all_configs), 1)

    def _get_next_backgrounds_for_round(self, n: int) -> list[dict[str, Any]]:
        """Take n backgrounds from the shared list, wrapping at the end."""
        lst = getattr(self, "_background_list_shuffled", []) or []
        if not lst:
            return []
        L = len(lst)
        out = []
        for i in range(n):
            out.append(lst[(self._background_cursor + i) % L])
        self._background_cursor = (self._background_cursor + n) % L
        return out

    def sample_for_envs(self, n: int) -> list[dict[str, Any]]:
        """Select n configs, cycling combinations or using empty scene-scan configs."""
        self._round_count += 1
        if self._material_fuzz_mode in ("config_file", "both") and self._all_configs:
            L = len(self._all_configs)
            configs = []
            for i in range(n):
                idx = (self._material_cursor + i) % L
                configs.append(dict(self._all_configs[idx]))
            self._material_cursor = (self._material_cursor + n) % L
            logger.info(
                "Fuzzer: round %d, %d envs, config_file cursor %d/%d",
                self._round_count, n, self._material_cursor, L,
            )
            return configs
        logger.info("Fuzzer: round %d, %d envs (scene_scan)", self._round_count, n)
        return [{} for _ in range(n)]

    def exhausted(self) -> bool:
        """Material combinations cycle, so they are never exhausted."""
        return False

    def should_continue(self) -> bool:
        """Continue until the configured round limit, regardless of materials."""
        if self._num_rounds is not None and self._round_count >= self._num_rounds:
            return False
        return True

    def remaining_count(self) -> int:
        """Return the combination count, or one for scene-scan mode."""
        return max(len(self._all_configs), 1)

    def fuzz(self, replayer, n: int) -> list[int]:
        """
        Select n configurations cyclically and apply them to scene objects.
        Read only environment paths from replayer. Return the environment
        indices to record in this round.
        """
        configs = self.sample_for_envs(n)
        num_sampled = n
        env_paths = replayer.get_env_paths() if hasattr(replayer, "get_env_paths") else []
        if not env_paths or len(env_paths) < n:
            return list(range(num_sampled)) if num_sampled or self._enable_light_fuzz else []
        stage = omni.usd.get_context().get_stage()
        if self._enable_background_fuzz:
            bg_for_round = self._get_next_backgrounds_for_round(n)
            if bg_for_round:
                sim_assets = os.environ.get("SIM_ASSETS", "")
                set_background(stage, env_paths, bg_for_round, sim_assets, log=logger, shuffle_list=False)
        if self._enable_material_fuzz:
            use_config_file = self._material_fuzz_mode in ("config_file", "both")
            use_scene_scan = self._material_fuzz_mode in ("scene_scan", "both")
            logger.info(f"Fuzzer: round {self._round_count}, use_config_file={use_config_file}, use_scene_scan={use_scene_scan}")
            if use_scene_scan:
                self._scene_material_info = collect_scene_material_info(
                    stage, env_paths,
                )
                if self._scene_material_info:
                    randomize_scene_materials(stage, self._scene_material_info, self._rng, logger)
            if use_config_file:
                env_id_to_mat = {i: configs[i].get("material", {}) for i in range(len(configs))}
                mat_config = self._shuffled_material_config or self._material_config
                if mat_config and any(env_id_to_mat.values()):
                    set_materials_by_choices(
                        stage, env_paths, mat_config, env_id_to_mat, logger,
                        objects_dir_name=self._objects_dir_name,
                    )
        if self._enable_light_fuzz:
            set_lights_random(stage, env_paths, self._rng, logger)
        record_env_ids = list(range(num_sampled)) if num_sampled > 0 else []
        return record_env_ids

    def apply_defaults(self, replayer, n: int) -> list[int]:
        """
        Apply the default variants without sampling and record all n environments.
        """
        env_paths = replayer.get_env_paths() if hasattr(replayer, "get_env_paths") else []
        if env_paths and len(env_paths) >= n:
            stage = omni.usd.get_context().get_stage()
            if self._enable_background_fuzz:
                bg_for_round = self._get_next_backgrounds_for_round(n)
                if bg_for_round:
                    sim_assets = os.environ.get("SIM_ASSETS", "")
                    set_background(stage, env_paths, bg_for_round, sim_assets, log=logger, shuffle_list=False)
            if self._enable_material_fuzz:
                use_config_file = self._material_fuzz_mode in ("config_file", "both")
                use_scene_scan = self._material_fuzz_mode in ("scene_scan", "both")
                if use_scene_scan:
                    self._scene_material_info = collect_scene_material_info(
                        stage, env_paths,
                    )
                    if self._scene_material_info:
                        randomize_scene_materials(stage, self._scene_material_info, self._rng, logger)
                if use_config_file:
                    env_id_to_mat = {i: {} for i in range(n)}
                    mat_config = self._material_config
                    if mat_config:
                        set_materials_by_choices(
                            stage, env_paths, mat_config, env_id_to_mat, logger,
                            objects_dir_name=self._objects_dir_name,
                        )
            if self._enable_light_fuzz:
                set_lights_random(stage, env_paths, self._rng, logger)
        return list(range(n))
