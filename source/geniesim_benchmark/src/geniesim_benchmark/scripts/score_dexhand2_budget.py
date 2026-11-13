"""Derive budget scores from a finished DexHand 2 episode's native records."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

from geniesim_benchmark.newton.progress import TaskProgress
from geniesim_benchmark.plugins.output_system.dexhand2 import (
    EVALUATION_FILENAME,
    RUBIK_STAGES,
    RUBIK_TASK,
    WOK_STAGES,
    WOK_TASK,
    attach_episode_budget,
)

SOFT_TASKS = ("fold_cloth", "cloth_hanger", "drawstring_bag")


@dataclass(frozen=True)
class Episode:
    task: str
    stages: list[dict]
    e2e: int
    dispatches: list[int]
    completed_at: list[int | None]
    clock: str


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _events(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _soft_evaluation(root: Path) -> dict:
    direct = root / EVALUATION_FILENAME
    if direct.is_file():
        details = _json(direct)["details"]
        _require(len(details) == 1, "standalone libuipc report must contain one episode")
        return details[0]
    legacy = root / "evaluation.json"
    if legacy.is_file():
        return _json(legacy)
    prefix, _, number = root.name.partition("_")
    _require(prefix == "episode" and number.isdecimal(), "invalid libuipc episode directory")
    details = _json(root.parent / EVALUATION_FILENAME)["details"]
    index = int(number) - 1
    _require(0 <= index < len(details), "libuipc episode missing from its run report")
    return details[index]


def _soft_episode(root: Path, native: dict) -> Episode:
    _require(
        native["execution_status"] == "completed"
        and native["scoring_status"] == "scored",
        "libuipc episode must have completed with native scoring",
    )
    verified = _json(root / "recording_verified.json")
    _require(
        all(
            verified[k] is True
            for k in (
                "recording_clean",
                "recording_closed",
                "solver_shutdown_clean",
                "all_physical_frames_verified",
            )
        ),
        "libuipc episode requires a verified physical archive",
    )
    evaluation = _soft_evaluation(root)
    stages = native["stages"]
    names = [stage["name"] for stage in stages]
    _require(
        bool(names) and len(set(names)) == len(names), "invalid libuipc stage layout"
    )
    layout = list(enumerate(names))
    scores = evaluation["result"]["scores"]
    _require(
        evaluation["task_name"] == native["task_name"]
        and scores["STEPS"] == stages
        and evaluation["policy_dispatch_count"] == native["policy_dispatch_count"],
        "libuipc evaluation and result disagree",
    )

    dispatches = []
    dispatch_path = root / "policy_dispatches.jsonl"
    action_path = root / "action_execution.jsonl"
    dispatch_log = (
        _events(dispatch_path)
        if native["policy_dispatch_count"] or dispatch_path.is_file()
        else iter(())
    )
    request = -1
    offset = -1
    last_tick = 0
    steps = 0
    for action in (
        _events(action_path) if native["steps"] or action_path.is_file() else ()
    ):
        steps += 1
        ack = action["ack"]
        start = ack["tick"] - ack["ipc_steps"]
        _require(
            action["step"] == steps
            and ack["status"] == "completed"
            and ack["ipc_steps"] > 0
            and start >= last_tick,
            "invalid libuipc action execution timeline",
        )
        if action["request_index"] != request:
            _require(
                action["request_index"] > request, "libuipc request order regressed"
            )
            request, offset = action["request_index"], 0
            event = next(dispatch_log, None)
            _require(
                event is not None
                and event["request_index"] == request
                and event["steps"] == steps - 1,
                "missing libuipc dispatch for the first action of a batch",
            )
            accepted = event["ack"]
            _require(
                accepted["status"] in ("accepted", "completed")
                and all(
                    accepted[k] == ack[k]
                    for k in ("request_id", "source_id", "epoch", "ipc_steps")
                ),
                "libuipc dispatch and completion ACKs do not match",
            )
            dispatches.append(start)
        else:
            offset += 1
        _require(
            action["chunk_offset"] == offset, "libuipc action offsets are incomplete"
        )
        last_tick = ack["tick"]
    _require(
        steps == native["steps"]
        and len(dispatches) == native["policy_dispatch_count"]
        and next(dispatch_log, None) is None,
        "libuipc action or dispatch log is incomplete",
    )

    completed_at = [None] * len(names)
    last_stages = None
    frame_count = 0
    first_tick = last_frame_tick = None
    for frame in _events(root / "scoring/checker_frames.jsonl"):
        tick = frame["tick"]
        if first_tick is None:
            first_tick = tick
        _require(
            type(tick) is int and tick == first_tick + frame_count,
            "missing libuipc checker tick",
        )
        last_stages = frame["stages"]
        _require(
            [(r["step"], r["name"]) for r in last_stages] == layout,
            "libuipc checker layout changed during episode",
        )
        for i, row in enumerate(last_stages):
            if row["score"] == 1 and completed_at[i] is None:
                completed_at[i] = tick
            _require(
                completed_at[i] is None or row["score"] == 1,
                "libuipc stage completion regressed",
            )
        last_frame_tick = tick
        frame_count += 1
    _require(
        frame_count == native["observed_frames"]
        and last_stages == stages
        and (
            not dispatches
            or first_tick <= dispatches[0] <= last_tick <= last_frame_tick
        ),
        "libuipc checker frames do not cover the native result and all actions",
    )
    return Episode(
        native["task_name"],
        stages,
        scores["E2E"],
        dispatches,
        completed_at,
        "libuipc_physics_tick",
    )


def _rubik_episode(root: Path, native: dict) -> Episode:
    _require(
        native["valid"] is True
        and native["scoring_contract"]["name"] == "rubik_stages",
        "Rubik episode requires a valid result with the supported scoring contract",
    )
    raw = native["result"]
    terminal_tick = raw["step"]
    _require(
        math.isclose(
            native["simulation_seconds"], terminal_tick * 0.001, abs_tol=1e-7, rel_tol=0
        ),
        "Rubik terminal time does not match the native clock",
    )
    names = RUBIK_STAGES
    stages = [
        {"step": i, "name": name, "score": raw["scores"][name]}
        for i, name in enumerate(names)
    ]
    _require(
        raw["scores"]["E2E"] == stages[-1]["score"], "Rubik Solve and E2E disagree"
    )
    events = native["stage_events"]
    turn = events["first_turn"]
    times = (
        events["pickup_time"],
        turn["to_time"] if turn is not None else None,
        events["solve_time"],
    )
    completed_at = []
    for row, time in zip(stages, times, strict=True):
        _require(
            row["score"] in (0, 1) and (time is not None) == (row["score"] == 1),
            "Rubik stage event and native score disagree",
        )
        tick = round(time / 0.001) if time is not None else None
        _require(
            tick is None
            or (
                0 <= tick <= terminal_tick
                and math.isclose(time, tick * 0.001, abs_tol=1e-7, rel_tol=0)
            ),
            "Rubik stage time does not match the native clock",
        )
        completed_at.append(tick)

    dispatches = []
    expected_tick = 0
    previous_reply = 0
    offset = -1
    for action in _json(root / "actions.json"):
        start = action["at_step"]
        reply = action["reply_index"]
        planned = action["planned_physics_ticks"]
        _require(
            0 <= start == expected_tick < terminal_tick
            and type(planned) is int
            and planned > 0
            and 1 <= reply <= native["inference_count"]
            and reply >= previous_reply,
            "Rubik action log does not match the native trajectory",
        )
        if reply != previous_reply:
            dispatches.append(start)
            previous_reply, offset = reply, 0
        else:
            offset += 1
        _require(
            action["reply_row"] == offset, "Rubik reply action offsets are incomplete"
        )
        expected_tick = min(terminal_tick, start + planned)
    _require(
        expected_tick == terminal_tick,
        "Rubik action log ends before the native trajectory",
    )
    return Episode(
        RUBIK_TASK,
        stages,
        int(raw["scores"]["E2E"]),
        dispatches,
        completed_at,
        "mujoco_physics_tick",
    )


def _wok_episode(root: Path) -> Episode:
    episodes = _json(root / "results.json")
    _require(
        isinstance(episodes, list) and len(episodes) == 1,
        "Newton budget export requires one episode per output directory",
    )
    native = episodes[0]
    final = TaskProgress.from_message(native["evaluation"])
    _require(
        native["milestones"] == final.milestones
        and native["score"] == final.score
        and native["task_success"] is final.task_success,
        "Newton result and final task progress disagree",
    )

    dispatches = []
    completed_at = [None] * len(WOK_STAGES)
    latest = None
    latest_step = -1
    steps = requests = remaining = 0
    awaiting_reply = first_action = False
    for event in _events(root / "events.jsonl"):
        kind = event["event"]
        if kind == "task_progress":
            progress = TaskProgress.from_message(event)
            _require(
                event["steps"] == steps
                and (progress.session_id, progress.reset_epoch)
                == (final.session_id, final.reset_epoch)
                and (latest is None or progress.tick >= latest.tick),
                "Newton progress is not on the episode's action timeline",
            )
            for i, name in enumerate(WOK_STAGES):
                if progress.milestones[name] and completed_at[i] is None:
                    completed_at[i] = steps
                _require(
                    completed_at[i] is None or progress.milestones[name],
                    "Newton milestone regressed",
                )
            latest, latest_step = progress, steps
        elif kind == "inference_start":
            _require(
                not awaiting_reply
                and remaining == 0
                and event["steps"] == latest_step == steps,
                "Newton inference start has no matching observation",
            )
            awaiting_reply = True
        elif kind == "inference_end":
            remaining = event["chunk_steps"]
            _require(
                awaiting_reply
                and event["steps"] == steps
                and type(remaining) is int
                and remaining >= 0,
                "Newton inference end has no matching request",
            )
            awaiting_reply, first_action = False, True
            requests += 1
        elif kind == "action_issued":
            _require(
                not awaiting_reply
                and remaining > 0
                and event["steps"] == steps + 1
                and latest_step == steps,
                "Newton action has no matching chunk or task progress",
            )
            if first_action:
                dispatches.append(steps)
                first_action = False
            remaining -= 1
            steps += 1
        else:
            raise ValueError(f"unexpected event in completed Newton episode: {kind}")
    _require(
        not awaiting_reply
        and requests == native["inference_requests"]
        and steps == latest_step == native["issued_steps"]
        and latest == final,
        "Newton event log does not match the final native result",
    )
    stages = [
        {"step": i, "name": name, "score": int(final.milestones[name])}
        for i, name in enumerate(WOK_STAGES)
    ]
    return Episode(
        WOK_TASK,
        stages,
        int(final.task_success),
        dispatches,
        completed_at,
        "newton_confirmed_action_step",
    )


def _episode_step_out(root: Path, task: str) -> tuple[int, Path]:
    if task == WOK_TASK:
        import yaml

        path = root / "evaluation.yaml"
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"invalid archived run config: {path}") from exc
        _require(
            config["task_name"] == task, "Newton config does not match the episode"
        )
        return config["max_steps"], path

    path = root.parent / "config.json"
    config = _json(path)
    _require(
        config["benchmark"]["task_name"] == task, "config does not match the episode"
    )
    if task == RUBIK_TASK:
        rubik = config["rubik"]
        return math.floor(rubik["timeout_seconds"] * rubik["control_hz"]), path
    return config["libuipc"]["max_steps"], path


def score_episode(root: Path, step_out: int | None = None) -> dict:
    """Derive stage scores using a single budget for the whole episode."""
    if (root / "results.json").is_file():
        episode = _wok_episode(root)
    else:
        native = _json(root / "result.json")
        task = native.get("task_name")
        if task in SOFT_TASKS and native.get("runtime") == "libuipc":
            episode = _soft_episode(root, native)
        elif task == RUBIK_TASK and native.get("backend") == "mujoco":
            episode = _rubik_episode(root, native)
        else:
            raise ValueError(f"unsupported DexHand 2 task: {task!r}")

    try:
        configured_step_out, config = _episode_step_out(root, episode.task)
    except FileNotFoundError as exc:
        if step_out is None:
            raise ValueError(
                f"archived run config missing: {exc.filename}; supply the episode's --step-out"
            ) from exc
        budget_source = "explicit_step_out"
    else:
        _require(
            step_out is None or step_out == configured_step_out,
            f"--step-out differs from the archived run config: {configured_step_out}",
        )
        step_out = configured_step_out
        budget_source = str(config)
    result = {
        "task_name": episode.task,
        "scores": {"STEPS": episode.stages, "E2E": episode.e2e},
    }
    attach_episode_budget(
        result,
        episode.task,
        count=len(episode.dispatches),
        step_out=step_out,
        source="finished_episode_artifacts",
    )
    result["budget_evidence"].update(
        clock=episode.clock,
        step_out_source=budget_source,
        stage_completed_at={
            row["name"]: tick
            for row, tick in zip(episode.stages, episode.completed_at, strict=True)
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_directory", type=Path)
    parser.add_argument(
        "--step-out",
        type=int,
        metavar="REFERENCE_STEPS",
        help="Total reference StepOut when the effective run config is unavailable.",
    )
    args = parser.parse_args(argv)
    try:
        report = score_episode(args.episode_directory, args.step_out)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
