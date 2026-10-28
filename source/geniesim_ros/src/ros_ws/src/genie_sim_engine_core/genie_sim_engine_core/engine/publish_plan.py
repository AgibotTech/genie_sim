# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""``PublishPlan`` — the v2 wire-protocol routing decision, as one value.

Which joint lands on which topic, and which body-pose channel is live, is a
decision made ONCE at bootstrap by :class:`~genie_sim_engine_core.common.session.EngineSession`
out of the manifest and the backend's advertised topology. The publish loop
then executes that decision every tick without re-deriving it.

Before this existed the decision was five attributes stamped onto the backend
object (``sim._v2_robot_joint_indices`` and friends) and read back with
``getattr`` defaults on the other side. That coupled ``genie_sim_engine_core``
to private attribute *names* on packages it does not import, and nothing
checked the coupling — the five ``# type: ignore[attr-defined]`` at the write
site suppressed exactly the check that would have caught a typo or a
duplicate assignment. A frozen value passed down the call chain makes both
failure modes unrepresentable.

The plan is deliberately dumb: no methods that consult the backend, no lazy
derivation. Everything in it is known before the first tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(frozen=True)
class PublishPlan:
    """Per-run routing for the state-publish path.

    Attributes
    ----------
    robot_joint_indices
        Indices into ``sim.get_joint_states()`` that go on ``/joint_states``
        (robot only — a colocated ``robot_state_publisher`` fans them onto
        ``/tf``).

        ``None`` is the LEGACY sentinel: "nobody planned this run". The loop
        then publishes the backend's whole joint vector on ``/joint_states``
        and leaves the scene channel quiet, which is exactly v1 behaviour.
        A caller that wired :class:`~genie_sim_engine_core.common.loop.EngineRunLoop`
        directly, without going through ``EngineSession``, gets this.

        An EMPTY tuple is a different statement — "planned, and no robot DoF
        belongs on the wire" — and publishes an empty ``JointState``.
    scene_joint_names
        Names for ``/joint_states_scene``, in publish order. Bare joint names,
        ``env_<w>/``-prefixed at ``num_envs > 1`` like every other multi-env
        label.
    scene_joint_indices
        Indices into ``sim.get_joint_states()`` for the scene DoFs — the
        SLICED sourcing mode. Empty when ``scene_joint_direct`` is set,
        because no index into that vector could reach a joint the backend
        keeps out of it.
    scene_joint_direct
        DIRECT sourcing mode: read the scene DoFs from
        ``sim.get_scene_joint_states()`` instead of slicing. Set for backends
        that keep scene articulations off the controllable surface entirely
        (newton-standalone — a door hinge must not receive a PD drive or an RL
        action slot), so the manifest subtraction would find nothing to slice.

        Wins over :attr:`scene_joint_indices` when both are somehow set; see
        ``session.py``'s bootstrap for why only one can legitimately be.
    use_tf_render
        Body-pose channel selector. ``False`` → ``/tf_scene``, scene bodies
        only (robot links ride ``/tf`` via ``robot_state_publisher``).
        ``True`` → ``/tf_render``, every body including robot links, for
        deployments with no RSP. Mutually exclusive: ``init_ros`` only built
        the selected publisher, and the other one is a null pointer whose
        publish entry point is a silent C++ no-op.
    """

    robot_joint_indices: Optional[Tuple[int, ...]] = None
    scene_joint_names: Tuple[str, ...] = field(default_factory=tuple)
    scene_joint_indices: Tuple[int, ...] = field(default_factory=tuple)
    scene_joint_direct: bool = False
    use_tf_render: bool = False

    @classmethod
    def legacy(cls) -> "PublishPlan":
        """The unplanned run: everything on ``/joint_states``, ``/tf_scene``.

        Named rather than spelled ``PublishPlan()`` at call sites so the
        intent is legible — a bare constructor call reads like an oversight.
        """
        return cls()

    @property
    def planned(self) -> bool:
        """True when a session computed the joint split for this run."""
        return self.robot_joint_indices is not None
