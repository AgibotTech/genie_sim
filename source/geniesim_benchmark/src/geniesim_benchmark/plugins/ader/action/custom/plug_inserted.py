# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class PlugInserted(EvaluateAction):
    """Evaluate whether an attractor has attached an inserted object.

    Success is read off the FixedJoint that the ``phone_plug_attractor.py``
    BehaviorScript (mounted on ``<plug>/entity/body``) creates the moment the
    plug tip lands within its per-axis threshold of the phone hole anchor. That
    joint IS the simulation's definition of "plugged in", so this checker needs
    no pose tolerances of its own — and, because it reads a USD attribute rather
    than a rigid-body transform, it is unaffected by the stale-xform behaviour
    that GPU dynamics introduces for pose reads.

    The attractor never detaches once attached, so the joint prim survives on
    the stage. To avoid scoring a leftover joint from an earlier episode,
    success requires having observed the joint in a not-attached state first --
    a joint that reads enabled from the very first tick and stays that way is
    treated as stale and never scores. This also tolerates the attractor's
    ``on_play`` detach landing a tick or two after this action starts.

    Args:
        env: The simulation environment.
        plug_name: Inserted object id or prim path, e.g. "benchmark_plug_29d5cc20".
            May also be a placeholder, e.g. "{@placeholder_str1}".
        joint_name: Child prim name of the attach joint (default
            "PhoneAttachJoint", matching the phone attractor script). The pen
            cap attractor creates "PenAttachJoint" on the entity rigid body.
        hold_ticks: Consecutive checker updates the joint must read as enabled
            before success is declared. Note these are eval ticks (the eval tree
            is updated once per 30 policy steps), not simulation frames.

    Supported parameter formats:
        - "benchmark_plug_29d5cc20"
        - "benchmark_plug_29d5cc20|PhoneAttachJoint"
        - "benchmark_plug_29d5cc20|PhoneAttachJoint|2"
    """

    _JOINT_ENABLED_ATTR = "physics:jointEnabled"
    _BODY_SUBPATH = "/entity/body"

    def __init__(self, env, plug_name, joint_name="PhoneAttachJoint", hold_ticks=1):
        super().__init__(env)
        self._holder_name, self._plug_name = self.placeholder_sparser(plug_name)
        self.joint_name = joint_name or "PhoneAttachJoint"
        self.hold_ticks = max(1, int(hold_ticks))

        self._done_flag = False
        self._seen_unattached = False  # gate against a stale joint from a prior episode
        self._stale_warned = False
        self._enabled_ticks = 0

    @property
    def plug_name(self) -> str:
        if self._holder_name:
            return getattr(self, self._plug_name)
        return self._plug_name

    def joint_path(self) -> str:
        """Absolute prim path of the attractor's attach joint."""
        plug_path = self._analyze_obj_name(self.plug_name)
        if plug_path.endswith(self._BODY_SUBPATH):
            body_path = plug_path
        elif plug_path.endswith("/entity"):
            body_path = plug_path + "/body"
        else:
            body_path = plug_path + self._BODY_SUBPATH

        # Phone cables author their joint on /entity/body; the pen cap's
        # rigid body and PenAttachJoint are on /entity. Probe for the authored
        # attribute so either asset layout uses its actual attach contract.
        candidates = [
            f"{body_path}/{self.joint_name}",
            f"{body_path.rsplit('/body', 1)[0]}/{self.joint_name}",
        ]
        for path in candidates:
            if self.api_core.get_prim_attribute(path, self._JOINT_ENABLED_ATTR, default=None) is not None:
                return path
        return candidates[0]

    def is_attached(self) -> bool:
        """True iff the attach joint exists and is currently enabled."""
        enabled = self.api_core.get_prim_attribute(
            self.joint_path(), self._JOINT_ENABLED_ATTR, default=None
        )
        return bool(enabled)

    def update(self, delta_time: float) -> float:
        if self._done_flag:
            return super().update(delta_time)

        attached = self.is_attached()

        if not attached:
            # Baseline established: any later attach is this episode's doing.
            self._seen_unattached = True
        elif not self._seen_unattached and not self._stale_warned:
            self._stale_warned = True
            logger.warning(
                f"[PlugInserted] {self.plug_name}: attach joint already enabled at start "
                f"({self.joint_path()}); treating as stale until it is seen detached."
            )

        if attached and self._seen_unattached:
            self._enabled_ticks += 1
            if self._enabled_ticks >= self.hold_ticks:
                self._done_flag = True
                logger.info(
                    f"[PlugInserted] SUCCESS — {self.plug_name} attached via "
                    f"{self.joint_path()} (held {self._enabled_ticks}/{self.hold_ticks} ticks)"
                )
        else:
            self._enabled_ticks = 0

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [PlugInserted] plug: {self.plug_name}, evt: {event.value}")

        if event == ActionEvent.FINISHED:
            if self._done_flag:
                self.progress_info["SCORE"] = 1
                logger.info(f"[PlugInserted] Task completed successfully for {self.plug_name}")
            else:
                self.progress_info["SCORE"] = 0
                logger.info(
                    f"[PlugInserted] FAILED — {self.plug_name} never attached "
                    f"(joint {self.joint_path()})"
                )
