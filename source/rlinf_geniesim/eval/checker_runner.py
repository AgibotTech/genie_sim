# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Run benchmark success checkers for one simulation environment.

ApiCoreShim supplies backend state; AderEnv and parse_action build and tick
the benchmark behavior tree. Score progress through score_layout_steps;
success requires the final scored task step to reach SCORE == 1.

ScissorsHandleInserted and ScissorsHandleActuated require articulation mesh
geometry that ApiCoreShim does not expose, so those checkers return score 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rlinf_geniesim.eval.api_core_shim import ApiCoreShim

# Structural / timeout acts carry no task SCORE (benchmark ignores them when
# extracting per-step scores).
_IGNORED_ACTS = {
    "ActionList",
    "ActionSetWaitAny",
    "ActionSetWaitAll",
    "ActionSetWaitSome",
    "ActionWaitForTime",
    "StepOut",
    "Timeout",
    "TimeOut",
}


def _tree_success(task_progress: list, sub_steps: list | None = None) -> bool:
    """Task succeeded iff the LAST scored step reached SCORE == 1 (mirrors
    benchmark ``eval_utils.summarize_scores`` + its E2E ``STEPS[-1].score ==
    1`` rule).

    With the task's registered ``TASK_STEPS`` layout, scoring runs through the
    benchmark's own ``score_layout_steps`` so a run of same-class rows scores
    best-of — on either-hand tasks (``pick_common_sense`` and the two
    ``pick_object_*_position`` tasks, whose ``ActionSetWaitAny`` holds a
    left/right pair aimed at the same object) an episode solved with either
    arm counts. Taking ``steps[-1]`` of the raw leaf list instead scored the
    LAST-listed gripper only, halving the recorded rate on those tasks and
    disagreeing with the OR-aware ``completed_steps()`` that feeds reward.

    Without a layout (task absent from ``TASK_STEPS``), fall back to the last
    non-structural checker."""
    if sub_steps:
        from geniesim_benchmark.plugins.output_system.eval_utils import (
            score_layout_steps,
        )

        steps = score_layout_steps(task_progress, sub_steps)
        return bool(steps) and steps[-1]["score"] == 1.0
    steps = [it for it in task_progress if it.get("class_name") not in _IGNORED_ACTS]
    if not steps:
        return False
    prog = steps[-1].get("progress") or {}
    return isinstance(prog, dict) and prog.get("SCORE", 0) == 1


def _stage_scores(task_progress: list) -> list[dict]:
    """Snapshot of every non-structural checker's frozen SCORE, in tree order
    (same ``_IGNORED_ACTS`` filter as ``_tree_success``, but keeps every stage
    instead of only the last — the per-stage counterpart of the E2E rule)."""
    steps = [it for it in task_progress if it.get("class_name") not in _IGNORED_ACTS]
    return [{"name": it.get("class_name"), "score": float((it.get("progress") or {}).get("SCORE", 0))} for it in steps]


# ---------------------------------------------------------------------------
# Staged reward: completed_steps() / total_steps, latched.
#
# The denominator is deliberately not the number of leaf checkers: an OR
# branch such as "either hand counts" puts several leaves under one logical
# step and would inflate the count. The real step count is the number of
# direct children of the top-level ``ActionList``, which the benchmark
# executes in order.
#
# A problem whose Acts[0] is a single step with no enclosing ActionList still
# fits (total_steps == 1). A problem with a nested ``ActionList`` does not:
# there is no single flat step sequence to divide by, so total_steps is None
# and the caller falls back to sparse 0/1 reward.
_STEP_CONTAINER_KEYS = ("ActionList", "ActionSetWaitAny", "ActionSetWaitAll")


def _has_nested_action_list(node: dict, *, top: bool) -> bool:
    """Return whether an ActionList occurs below the problem's top level.

    Nested sequences are excluded from flat staged-reward counting.
    """
    for key, value in node.items():
        if key == "ActionList" and not top:
            return True
        if (key in _STEP_CONTAINER_KEYS or key.startswith("ActionSetWaitSome")) and value:
            for item in value:
                if isinstance(item, dict) and _has_nested_action_list(item, top=False):
                    return True
    return False


def _count_leaf_checks(node: dict) -> int:
    """Count the non-structural (SCORE-bearing) checker leaves under
    ``node``, in the same DFS pre-order ``parse_action`` will later emit
    into ``task_progress`` and the same ``_IGNORED_ACTS`` filter
    ``_stage_scores`` applies — so ``completed_steps()`` can carve the flat
    ``_stage_scores()`` list back into per-step groups (one group per
    top-level step) without touching the benchmark's internal action-object
    tree."""
    total = 0
    for key, value in node.items():
        if key in _STEP_CONTAINER_KEYS or key.startswith("ActionSetWaitSome"):
            for item in value or []:
                if isinstance(item, dict):
                    total += _count_leaf_checks(item)
        elif key in _IGNORED_ACTS:
            continue  # StepOut/Timeout/... — times out, never carries a SCORE
        else:
            total += 1
    return total


def _compute_total_steps(problem: dict, *, label: str = "") -> tuple:
    """Real step count + per-step leaf-check counts for ``problem``.

    Returns ``(total_steps, step_leaf_counts)``. ``total_steps`` is the
    direct-child count of ``Acts[0]["ActionList"]`` (or ``1`` when Acts[0]
    itself is the sole step, unwrapped). Returns ``(None, [])`` — printing a
    one-line explanation — when the problem's Acts[0] can't be expressed as
    a flat ordered list of steps; callers must retreat to sparse 0/1 reward
    rather than silently produce a misleading fraction."""
    acts = problem.get("Acts") or []
    if not acts or not isinstance(acts[0], dict) or not acts[0]:
        print(
            f"[checker_runner] {label}: no Acts[0] to count steps from; " f"retreating to sparse 0/1 reward", flush=True
        )
        return None, []
    acts0 = acts[0]
    if _has_nested_action_list(acts0, top=True):
        print(
            f"[checker_runner] {label}: Acts[0] has a nested/repeated "
            f"ActionList (not a flat N-step problem); retreating to "
            f"sparse 0/1 reward",
            flush=True,
        )
        return None, []
    steps = acts0["ActionList"] if "ActionList" in acts0 else [acts0]
    leaf_counts = [_count_leaf_checks(s) for s in steps]
    return len(steps), leaf_counts


def _make_ader_env(api_core, sub_task_name: str, robot_cfg: str, task_progress: list):
    """Build a benchmark ``AderEnv`` subclass carrying the extra attributes the
    checkers read off ``env`` (kept here so the import of the benchmark eval
    framework is lazy — it must NOT happen at rlinf import time)."""
    from rlinf_geniesim.eval._kit_stubs import install_kit_stubs  # noqa: PLC0415

    install_kit_stubs(logger=lambda m: print(m, flush=True))
    from geniesim_benchmark.plugins.ader.ader_base import AderEnv, AderParams

    class _RlinfAderEnv(AderEnv):
        def __init__(self):
            super().__init__(api_core, AderParams())
            # _analyze_obj_name picks /Workspace/Objects/<id> when sub_task is
            # set; _get_gripper_position picks robot_base from robot_cfg.
            self.init_task_config = {"sub_task_name": sub_task_name, "robot_cfg": robot_cfg}
            self.current_step = 0
            self._followed_objects: set = set()
            self._picked_objects: set = set()
            self.task = _DuckTask(task_progress)

    return _RlinfAderEnv()


class _DuckTask:
    """Minimal stand-in for benchmark ``AderTask``: holds ``task_progress`` and
    routes ``update_progress`` (the only bits the action tree touches)."""

    def __init__(self, task_progress: list) -> None:
        self.task_progress = task_progress

    def update_progress(self, act_id, progress) -> None:
        for item in self.task_progress:
            if item["id"] == act_id:
                item["progress"] = progress
                break


def _resolve_problems_path(task_name: str, sub_task_name: str, instance_id: int) -> Path:
    """problems.json sits next to the task's scene.usda
    (``.../llm_task/<sub>/<instance>/problems.json``)."""
    from rlinf_geniesim.task_loader import resolve_benchmark_task

    r = resolve_benchmark_task(task_name, sub_task_name, instance_id)
    return Path(r["scene_usda"]).resolve().parent / "problems.json"


def _kinematic_alias_keys(task_name: str, sub_task_name: str, instance_id: int, alias) -> set:
    """Return member-prim aliases known to refer to kinematic objects.

    Read the current instance's scene because alias keys use its prim names.
    These names suppress expected warnings for static colliders that have no
    engine body entry. On lookup failure, log a note and return an empty set.
    """
    if not alias:
        return set()
    try:
        from rlinf_geniesim.task_loader import _scene_kinematic_prim_names, resolve_benchmark_task  # noqa: PLC0415

        r = resolve_benchmark_task(task_name, sub_task_name, instance_id)
        kinematic = _scene_kinematic_prim_names(r["scene_usda"])
        return {k for k in alias if k in kinematic}
    except Exception as exc:  # noqa: BLE001
        print(
            f"[checker_runner] kinematic_alias_keys unavailable ({exc!r}); "
            f"name_alias WARN threshold may still include known-kinematic "
            f"misses",
            flush=True,
        )
        return set()


class CheckerRunner:
    """Runs the success-checker tree for ONE env. Tick once per sim step."""

    def __init__(
        self,
        backend,
        env_id: int,
        task_name: str,
        sub_task_name: str,
        instance_id: int = 0,
        problem_key: str = "problem1",
        robot_cfg: str = "G2_omnipicker",
        problems_path: Any = None,
        alias: Any = None,
    ) -> None:
        # Benchmark modules import Kit APIs even when checkers use only the
        # backend shim. Install import-only stubs before loading that chain.
        from rlinf_geniesim.eval._kit_stubs import install_kit_stubs  # noqa: PLC0415

        install_kit_stubs(logger=lambda m: print(m, flush=True))
        from geniesim_benchmark.plugins.ader.action.action_parsing import (
            parse_action,
        )

        self._backend = backend
        self._env_id = int(env_id)
        # Keep prefix compatibility for older scenes; alias maps this instance
        # to the built topology's prim names or shared asset slots.
        self.shim = ApiCoreShim(
            backend,
            env_id=env_id,
            obj_prefix=f"i{int(instance_id)}_",
            alias=alias,
            kinematic_alias_keys=_kinematic_alias_keys(task_name, sub_task_name, instance_id, alias),
        )

        if problems_path is None:
            problems_path = _resolve_problems_path(task_name, sub_task_name, instance_id)
        try:
            with open(problems_path, "r") as f:
                problems = json.load(f)
        except FileNotFoundError:
            # e.g. geniesim_place_workpiece: the task has no problems.json at
            # all, so there is no flat step sequence (or any checker tree) to
            # build — not applicable to staged reward. Print BEFORE re-raising
            # so this reads as a deliberate retreat, not a silent crash; the
            # caller (sim_handler._ensure_runners) already catches this and
            # disables eval (reward stays 0), same as any other build failure.
            print(
                f"[checker_runner] {sub_task_name} instance={instance_id}: "
                f"no problems.json at {problems_path}; this task is not "
                f"applicable to staged reward, retreating to sparse 0/1",
                flush=True,
            )
            raise
        if problem_key not in problems:
            # The official chain selects a problem by FILE POSITION, never by
            # instruction id. The two agree for almost every instance, but a
            # few carry an instruction id with no matching problem key, where
            # selecting by id would KeyError and leave the episode silently
            # unscored. Match the official semantics: fall back to the id's
            # 1-based position in instructions.json, printing loudly, and
            # raise only when neither resolves.
            fb_key = None
            try:
                instr_path = Path(problems_path).parent / "instructions.json"
                entries = json.loads(instr_path.read_text())["instructions"]
                want = int(str(problem_key).replace("problem", ""))
                for pos, e in enumerate(entries, 1):
                    if int(e.get("id", pos)) == want:
                        fb_key = f"problem{pos}"
                        break
            except Exception:  # noqa: BLE001 — fallback resolution is best-effort
                fb_key = None
            if fb_key is not None and fb_key in problems:
                print(
                    f"[checker] {problem_key} missing in {problems_path}; "
                    f"falling back to positional {fb_key} (official-chain "
                    f"semantics)",
                    flush=True,
                )
                problem_key = fb_key
        problem = problems[problem_key]

        # Staged reward derived purely from the problem JSON, with no engine
        # or tree dependency, so it degrades independently of the tree.
        self.total_steps, self._step_leaf_counts = _compute_total_steps(
            problem, label=f"{sub_task_name} instance={instance_id} " f"{problem_key}"
        )

        # Scoring layout for _tree_success. Resolved once here (not per step)
        # and kept alongside the benchmark's other lazy imports so importing
        # this module doesn't drag in the eval stack. Empty list = task not in
        # TASK_STEPS; _tree_success then falls back to its leaf-order rule.
        try:
            from geniesim_benchmark.plugins.output_system.eval_utils import (
                get_task_steps,
            )

            self._sub_steps = list(get_task_steps(sub_task_name) or [])
        except Exception as exc:  # noqa: BLE001 — degrade, don't crash eval
            print(
                f"[checker_runner] {sub_task_name}: cannot resolve TASK_STEPS "
                f"layout ({exc!r}); success falls back to last-leaf rule",
                flush=True,
            )
            self._sub_steps = []

        self._task_progress: list = []
        self.env = _make_ader_env(self.shim, sub_task_name, robot_cfg, self._task_progress)
        # parse_action fills _task_progress (record_act_obj) and returns the
        # root; env.task must already be set (start() ticks update_progress).
        self.tree = parse_action(problem["Acts"][0], self._task_progress, self.env)
        self.env.execute_action = self.tree
        self.env.do_eval_action()  # start the tree in the "eval" slot

    def step(self, current_step: int) -> tuple[bool, bool]:
        """Tick this environment's tree and return the shared done/success properties."""
        self.env.current_step = int(current_step)
        self.env.action_update()
        return self.done, self.success

    @property
    def done(self) -> bool:
        return bool(self.env.has_done)

    @property
    def success(self) -> bool:
        return _tree_success(self._task_progress, self._sub_steps)

    def stage_scores(self) -> list[dict]:
        """Snapshot of every stage's frozen SCORE (see ``_stage_scores``)."""
        return _stage_scores(self._task_progress)

    def completed_steps(self) -> int:
        """Return the latched length of the achieved top-level step prefix.

        A step is achieved when any leaf in its group reaches ``SCORE == 1``.
        The count never decreases. Return 0 for an unsupported layout; callers
        must inspect ``total_steps`` to distinguish that case from no progress.
        """
        if not self.total_steps:
            return 0
        scores = [s["score"] for s in _stage_scores(self._task_progress)]
        idx = 0
        done = 0
        for cnt in self._step_leaf_counts:
            group = scores[idx : idx + cnt]
            idx += cnt
            if any(v == 1.0 for v in group):
                done += 1
            else:
                break
        return done


def build_runners(
    backend, task_name: str, sub_task_name: str, num_envs: int, instance_id: int = 0, robot_cfg: str = "G2_omnipicker"
) -> list:
    """One CheckerRunner per env (each tree keeps its own state, scoped to its
    env via a per-env ApiCoreShim)."""
    return [
        CheckerRunner(
            backend,
            env_id=i,
            task_name=task_name,
            sub_task_name=sub_task_name,
            instance_id=instance_id,
            robot_cfg=robot_cfg,
        )
        for i in range(int(num_envs))
    ]


__all__ = ["CheckerRunner", "build_runners"]
