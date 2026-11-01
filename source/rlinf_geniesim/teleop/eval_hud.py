# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pure-function helpers for the opt-in Pico teleop eval-mode HUD:
instruction sampling from the benchmark's llm_task/ tree, a reset-consistency
checker, and a PIL frame overlay. Numpy + PIL + stdlib only -- no sim/comm
dependency, so it is fully host-testable. The only benchmark-package coupling
(task_loader._benchmark_config_root, task_loader.resolve_instance_group) is
imported lazily inside InstructionSampler._load_table and resolve_gen_pool
respectively, so this module itself imports bare."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import ImageDraw, ImageFont


class InstructionSampler:
    """Loads ``llm_task/<sub_task_name>/<instance_id>/instructions.json`` and
    samples uniformly (seeded rng) over the available ``(id, text)`` pairs.

    Diverges from ``vec_env._instr_table`` on purpose: a missing file or an
    unknown task name raises ``FileNotFoundError`` (listing the available
    task names) instead of silently falling back to a synthetic entry.

    Multi-instance (--gen): tables are cached per instance_id (same lazy-load
    pattern as ``vec_env._instr_table``) in ``self._tables``, sharing one rng
    across every instance so a --gen run gets one continuous seeded stream
    regardless of which instance a given reset lands on. ``.table``/``.sample()``
    keep referring to the instance the sampler was constructed with; only
    ``sample_for(instance_id)`` reaches into other instances' tables.
    """

    def __init__(self, sub_task_name: str, instance_id: int, seed: int = 0):
        self._sub_task_name = sub_task_name
        self._instance_id = instance_id
        self._rng = np.random.default_rng(seed)
        self._tables: dict[int, list] = {}
        self._table = self._load_table(instance_id)
        self._tables[instance_id] = self._table

    def _load_table(self, instance_id: int) -> list:
        from rlinf_geniesim.task_loader import _benchmark_config_root  # noqa: PLC0415

        llm_task_root = _benchmark_config_root() / "llm_task"
        instructions_path = llm_task_root / self._sub_task_name / str(instance_id) / "instructions.json"
        if not instructions_path.is_file():
            if llm_task_root.is_dir():
                available = sorted(p.name for p in llm_task_root.iterdir() if p.is_dir())
            else:
                available = []
            raise FileNotFoundError(
                f"instructions.json not found at {instructions_path} "
                f"(sub_task_name={self._sub_task_name!r}, instance_id={instance_id!r}); "
                f"available tasks under {llm_task_root}: {available}"
            )

        data = json.loads(instructions_path.read_text())
        # Schema tolerance: pick_block_color-style entries carry an explicit
        # "id"; open_door's real instructions.json has none (just
        # {"instruction": "..."}, alongside a "subtask_steps" block this
        # sampler doesn't consume). Entries missing "id" get 1-based
        # enumeration ids in file order; entries WITH "id" keep them verbatim.
        # ASSUMPTION: a single file never mixes both schemas (all 20 real
        # open_door instances are single no-id entries; pick_block files are
        # all-id) — a mixed file could collide an enumerated id with an
        # explicit one. Harmless for sampling (draws by list index, ids are
        # display-only) but the id would be ambiguous in logs.
        return [
            (int(entry["id"]) if "id" in entry else i, str(entry["instruction"]))
            for i, entry in enumerate(data["instructions"], start=1)
        ]

    @property
    def table(self) -> list:
        """[(id, text), ...] in instructions.json order, for the instance this
        sampler was constructed with."""
        return self._table

    def sample_for(self, instance_id: int) -> tuple:
        """Uniform draw over ``instance_id``'s table -> ``(int(id), text)``.

        Loads-and-caches that instance's table on first use (same error
        semantics as the constructor: missing file -> FileNotFoundError
        listing available task names); a repeat call for an already-seen
        instance_id reads from the cache, not the filesystem. Shares this
        sampler's rng, so draws for different instances come from one
        continuous seeded stream.
        """
        table = self._tables.get(instance_id)
        if table is None:
            table = self._load_table(instance_id)
            self._tables[instance_id] = table
        idx = int(self._rng.choice(len(table)))
        instr_id, text = table[idx]
        return int(instr_id), text

    def sample(self) -> tuple:
        """Uniform draw over ``table`` -> ``(int(id), text)``. Equivalent to
        ``sample_for(instance_id)`` for the instance this sampler was
        constructed with."""
        return self.sample_for(self._instance_id)


def resolve_gen_pool(sub_task_name: str, instance_id: int, superset: "list[int] | None" = None) -> list:
    """Resolve the pool of instance ids a --gen run may reset into.

    ``superset`` non-empty -> returned as-is, deduped order-preserving (the
    launcher's explicit ``--superset=a,b,c`` list wins outright). Otherwise
    lazy-imports ``rlinf_geniesim.task_loader.resolve_instance_group`` and
    returns its ``"group_instances"`` (the topology group ``instance_id``
    belongs to). Exceptions from ``resolve_instance_group`` (unknown task
    name/instance) propagate unchanged -- loud by design, same as
    ``InstructionSampler``'s missing-file contract.
    """
    if superset:
        seen: list = []
        for inst in superset:
            if inst not in seen:
                seen.append(inst)
        return seen

    from rlinf_geniesim.task_loader import resolve_instance_group  # noqa: PLC0415

    group = resolve_instance_group(sub_task_name, instance_id)
    return group["group_instances"]


class ResetChecker:
    """Flags drift between a reference snapshot (first ``observe()`` call, or
    right after ``reset_reference()``) and later snapshots -- a sanity check
    that the human actually reset the scene between episodes."""

    def __init__(self, joint_tol_rad: float = 1e-3, pos_tol_mm: float = 1.0, ang_tol_deg: float = 0.5):
        self.joint_tol_rad = joint_tol_rad
        self.pos_tol_mm = pos_tol_mm
        self.ang_tol_deg = ang_tol_deg
        self._ref_qpos: Optional[np.ndarray] = None
        self._ref_object_poses: Optional[np.ndarray] = None

    def reset_reference(self) -> None:
        """Drop the stored reference; the next observe() re-baselines
        instead of reporting a diff (used with --vary-seed, where the reset
        pose legitimately differs episode to episode)."""
        self._ref_qpos = None
        self._ref_object_poses = None

    def observe(self, qpos: np.ndarray, object_poses: np.ndarray) -> Optional[str]:
        qpos = np.asarray(qpos, dtype=float)
        object_poses = np.asarray(object_poses, dtype=float)

        if self._ref_qpos is None:
            self._ref_qpos = qpos.copy()
            self._ref_object_poses = object_poses.copy()
            return None

        joint_diff = float(np.max(np.abs(qpos - self._ref_qpos))) if qpos.size else 0.0

        if object_poses.shape[0]:
            pos_delta = object_poses[:, :3] - self._ref_object_poses[:, :3]
            pos_diff_mm = float(np.max(np.linalg.norm(pos_delta, axis=1))) * 1000.0

            q1 = object_poses[:, 3:7]
            q2 = self._ref_object_poses[:, 3:7]
            dots = np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0.0, 1.0)
            ang_diff_deg = float(np.max(2.0 * np.arccos(dots))) * 180.0 / np.pi
        else:
            pos_diff_mm = 0.0
            ang_diff_deg = 0.0

        warn = joint_diff > self.joint_tol_rad or pos_diff_mm > self.pos_tol_mm or ang_diff_deg > self.ang_tol_deg
        prefix = "WARN " if warn else ""
        return (
            f"{prefix}[reset-check] joint max|Δ|={joint_diff:.6f} rad, "
            f"obj max|Δpos|={pos_diff_mm:.6f} mm, "
            f"|Δang|={ang_diff_deg:.6f} deg"
        )


def draw_hud(frame, *, instruction: str, episode: int, step: int, reward: float, success: Optional[bool]) -> None:
    """Overlay eval-mode HUD text in place on a PIL frame.

    - top: ``instruction``, word-wrapped past 60 characters
    - bottom-left: episode/step/reward
    - ``success is True``  -> green banner reading "SUCCESS"
    - ``success is None``  -> red "EVAL OFF" text (eval mode disabled)
    - ``success is False`` -> no banner (episode in progress, not yet judged)
    """
    draw = ImageDraw.Draw(frame)
    font = ImageFont.load_default()
    width, height = frame.size

    wrapped = textwrap.fill(instruction, width=60)
    draw.multiline_text((2, 2), wrapped, fill=(255, 255, 255), font=font)

    stats = f"ep {episode} | step {step} | reward {reward:.3f}"
    stats_bbox = draw.textbbox((0, 0), stats, font=font)
    stats_h = stats_bbox[3] - stats_bbox[1] + 4
    draw.text((2, height - stats_h - 2), stats, fill=(255, 255, 0), font=font)

    if success is True:
        banner_h = min(12, height)
        draw.rectangle([0, 0, width - 1, banner_h - 1], fill=(0, 160, 0))
        draw.text((2, 1), "SUCCESS", fill=(255, 255, 255), font=font)
    elif success is None:
        draw.text((max(0, width - 52), 1), "EVAL OFF", fill=(255, 0, 0), font=font)
