# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Measured Rubik stage points, separate from geometric solve success."""

from dataclasses import dataclass, field
import math

from .turns import quarter_turn_neighbors

SOLVED = "".join(face * 9 for face in "URFDLB")
STAGE_POINTS = {"PickUp": 20.0, "Turn": 30.0, "Solve": 50.0}


@dataclass
class RubikScorer:
    hold_seconds: float = 0.0
    minimum_support_fingers: int = 2
    pickup_clearance: float = 0.02
    initial_height: float = 0.0
    pickup_hold_seconds: float = 1.0
    picked_up: bool = False
    turned: bool = False
    solved_seen: bool = False
    maximum_hold_seconds: float = 0.0
    success: bool = False
    previous_time: float | None = None
    pickup_time: float | None = None
    turn_time: float | None = None
    solve_time: float | None = None
    first_turn: dict | None = None
    _pickup_since: float | None = field(default=None, repr=False)
    _previous_state: str | None = field(default=None, repr=False)
    _previous_state_time: float | None = field(default=None, repr=False)
    _turn_neighbors: dict = field(default_factory=dict, repr=False)

    def update(self, sample):
        now = float(sample["time"])
        if not math.isfinite(now) or (self.previous_time is not None and now <= self.previous_time):
            raise ValueError("Scoring timestamps must be finite and strictly increasing")
        checker = sample["cube_checker"]
        if checker["simulation_time"] != now or checker["hold_seconds"] != self.hold_seconds:
            raise ValueError("Checker sample timestamp or hold configuration does not match the scorer")
        self.previous_time = now
        supported = (
            sample["support_fingers"] >= self.minimum_support_fingers
            and sample["height"] >= self.initial_height + self.pickup_clearance
        )
        if supported:
            if self._pickup_since is None:
                self._pickup_since = now
            if not self.picked_up and now - self._pickup_since + 1e-12 >= self.pickup_hold_seconds:
                self.picked_up, self.pickup_time = True, now
        else:
            self._pickup_since = None
        self._update_turn(sample, checker, now)
        self.solved_seen |= checker["solved"]
        self.maximum_hold_seconds = max(self.maximum_hold_seconds, checker["held_seconds"])
        if checker["success"] and not self.success:
            self.success, self.solve_time = True, now
        return self.success

    def _update_turn(self, sample, checker, now):
        # Only assembled, aligned, legal measured states can be endpoints.
        # Keep the last endpoint while the layer is between its detents.
        if self.turned or not (checker.get("assembled") and checker.get("aligned") and sample["legal"] is True):
            return
        state = checker["facelets"]
        if state == self._previous_state:
            self._previous_state_time = now
            return
        move = self._turn_neighbors.get(state)
        if move is not None:
            self.turned, self.turn_time = True, now
            self.first_turn = {
                "move": move,
                "from_time": self._previous_state_time,
                "to_time": now,
                "from_facelets": self._previous_state,
                "to_facelets": state,
            }
        self._previous_state, self._previous_state_time = state, now
        self._turn_neighbors = {} if self.turned else quarter_turn_neighbors(state)

    def scores(self):
        hold = min(1.0, self.maximum_hold_seconds / self.hold_seconds) if self.hold_seconds else float(self.success)
        return {
            "PickUp": float(self.picked_up),
            "Turn": float(self.turned),
            "Solve": float(self.success),
            "SolvedObserved": float(self.solved_seen),
            "Hold": hold,
            "E2E": float(self.success),
        }

    def stage_points(self):
        scores = self.scores()
        return {name: points * scores[name] for name, points in STAGE_POINTS.items()}

    @property
    def score(self):
        return sum(self.stage_points().values())

    def stage_events(self):
        return {"pickup_time": self.pickup_time, "first_turn": self.first_turn, "solve_time": self.solve_time}

    def contract(self):
        return {
            "name": "rubik_stages",
            "stage_points": dict(STAGE_POINTS),
            "maximum_score": 100.0,
            "pickup_clearance_meters": self.pickup_clearance,
            "pickup_minimum_support_fingers": self.minimum_support_fingers,
            "pickup_hold_seconds": self.pickup_hold_seconds,
            "turn": "one assembled, aligned, legal outer-layer quarter turn between measured states",
            "turn_tolerances": "checker_contract",
            "solve": "checker success, including optional solved hold",
            "stages_latched_once": True,
            "requires_stage_order": False,
            "requires_hand_sequence": False,
            "solved_overrides_other_stages": False,
            "aggregate_score": "mean stage points over valid episodes",
        }


def summarize(episodes):
    valid = [e for e in episodes if e["status"] in ("success", "task_failure")]
    successes = sum(e["status"] == "success" for e in valid)
    return {
        "total_episodes": len(episodes),
        "valid_episodes": len(valid),
        "invalid_episodes": len(episodes) - len(valid),
        "successful_episodes": successes,
        "average_e2e": successes / len(valid) if valid else None,
        "success_rate": 100 * successes / len(valid) if valid else None,
        "score": sum(e["score"] for e in valid) / len(valid) if valid else None,
        "average_stage_points": {
            name: sum(e["stage_points"][name] for e in valid) / len(valid) if valid else None
            for name in STAGE_POINTS
        },
        "stage_success_rates": {
            name: 100 * sum(e["result"]["scores"][name] for e in valid) / len(valid) if valid else None
            for name in STAGE_POINTS
        },
        "failure_counts": {
            reason: sum(e["reason"] == reason for e in episodes)
            for reason in sorted({e["reason"] for e in episodes if e["status"] != "success"})
        },
    }
