# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Rubik episode commits reuse the common single-writer checkpoint contract."""

from pathlib import Path

import mujoco
import numpy as np

from geniesim_benchmark.plugins.output_system.checkpoint import (
    CheckpointRejected,
    EvaluationCheckpoint,
    file_digest,
)
from geniesim_benchmark.plugins.output_system.dexhand2 import EVALUATION_FILENAME, RUBIK_TASK

from .config import config_dict
from .native import load_package
from .scenes import SceneCatalog, episode_scenes


class RubikCheckpoint(EvaluationCheckpoint):
    def __init__(self, directory, cfg):
        super().__init__(directory, resume=cfg.benchmark.resume)
        self.cfg = cfg

    def initialize(self):
        cfg = self.cfg
        _, _, manifest = load_package(cfg.rubik.package_root)
        catalog = SceneCatalog(cfg.rubik.scene_archive) if cfg.rubik.scene_archive else None
        self.scenes = episode_scenes(
            catalog, cfg.benchmark.task_name, cfg.benchmark.seed, cfg.benchmark.num_instances, cfg.benchmark.num_episode
        )
        self.plan = [
            dict(
                episode_idx=i,
                instance_id=i // cfg.benchmark.num_episode,
                file_idx=0,
                episode_id=i % cfg.benchmark.num_episode,
            )
            for i in range(cfg.benchmark.num_instances * cfg.benchmark.num_episode)
        ]
        self.tokens = np.random.default_rng(cfg.benchmark.seed).permutation(
            [face + suffix for face in "URFDLB" for suffix in ("", "'")]
        )
        settings = config_dict(cfg)
        for key in ("resume", "output_dir", "python", "record_video_dir"):
            settings["benchmark"].pop(key, None)
        # Scheduler identity can outlive a gateway/session address.
        if cfg.benchmark.resume_id:
            settings["benchmark"].pop("infer_host", None)
        settings["rubik"]["scene_archive"] = catalog.archive_sha256 if catalog else None
        settings["app"].pop("headless", None)
        package = Path(__file__).resolve().parents[3]
        sources = [
            *Path(__file__).parent.rglob("*.py"),
            *(
                package / "plugins/output_system" / name
                for name in (
                    "checkpoint.py",
                    "observations.py",
                    "local_recorder.py",
                    "dexhand2.py",
                    "policy_budget.py",
                    "weighted_scoring.py",
                    "dexhand2_score_weights.json",
                )
            ),
            *(package / "utils" / name for name in ("hard_ik.py", "ikfk_utils.py", "eef_control.py")),
        ]
        identity = dict(
            runtime="mujoco-rubik",
            settings=settings,
            package_manifest=manifest,
            scenes=[scene.receipt() for scene in self.scenes] if self.scenes else None,
            sources={str(p.relative_to(package)): file_digest(p) for p in sources},
            mujoco=mujoco.__version__,
            numpy=np.__version__,
        )
        if self.state is None:
            self._result_file = (
                EVALUATION_FILENAME if cfg.benchmark.task_name == RUBIK_TASK else "evaluate_ret_rubik.json"
            )
            if self.result_path.exists():
                raise CheckpointRejected(
                    "Rubik results exist without a reusable checkpoint; choose a new output directory"
                )
            self.prepare(identity, self.plan, [], config_dict(cfg))
            return []
        rows = self.restore(identity)
        if self.state["plan"] != self.plan:
            raise CheckpointRejected("Rubik episode plan changed")
        self.recording_paths()
        for row in rows:
            artifacts = row.get("episode_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise CheckpointRejected("Rubik checkpoint has no native episode artifacts")
            for entry in artifacts:
                path = (self.out_dir / entry["path"]).resolve()
                if not path.is_relative_to(self.out_dir) or not path.is_file() or file_digest(path) != entry["sha256"]:
                    raise CheckpointRejected(f"Committed Rubik artifact is missing or changed: {entry['path']}")
        return rows

    def scramble(self, index):
        return self.cfg.rubik.scramble or str(self.tokens[index % len(self.tokens)])

    def artifacts(self, directory):
        names = ["result.json", "samples.json", "actions.json"]
        if self.cfg.benchmark.record:
            names.append("raw-1000hz.npz")
        return [
            dict(path=str((directory / name).relative_to(self.out_dir)), sha256=file_digest(directory / name))
            for name in names
        ]
