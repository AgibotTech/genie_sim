# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""genie_sim_engine's concrete EngineSession subclass.

Binds the abstract USD-helper hooks declared on
:class:`genie_sim_engine_core.common.session.EngineSession` to
``genie_sim_engine_isaacsim.kit.stage``.  Importing this module also imports
the in-tree backends so each one self-registers with
:class:`genie_sim_engine_core.engine.base.PhysicsEngine`.
"""

from __future__ import annotations

from typing import Any

from genie_sim_engine_core.common.session import EngineSession, SimpleLogger  # noqa: F401  (re-export)


def _register_backends() -> None:
    """Import the in-tree backends so the ``@PhysicsEngine.register``
    decorator on each one runs and populates the registry.

    Wrapped in try/except so a build that ships only a subset of
    backends (e.g. no Kit) does not fail to import this module — the
    factory will raise a clear error later if the missing backend is
    asked for.
    """
    # isaac_physx and isaac_newton pull in omni.* on import and only work
    # inside a live SimulationApp.  ``genie_sim_engine_isaacsim_node.py``
    # constructs the SimulationApp before importing this module, so
    # those imports succeed.  ``genie_sim_engine_newton_node.py`` does not
    # construct one — there, the kit.* imports raise and we silently
    # skip them.
    try:
        from genie_sim_engine_isaacsim.kit import isaac_physx  # noqa: F401
    except Exception:
        pass
    try:
        from genie_sim_engine_isaacsim.kit import isaac_newton  # noqa: F401
    except Exception:
        pass
    try:
        from genie_sim_engine_newton.engine import newton  # noqa: F401
    except Exception:
        pass


_register_backends()


class GenieSimEngineSession(EngineSession):
    """:class:`EngineSession` bound to ``genie_sim_engine_isaacsim.kit.stage``.

    Used by both Kit-based and Kit-free entry points in this package.
    ``_wait_for_manifest`` inherits the pure-stdlib default from
    :class:`EngineSession`; only ``_snapshot_body_transforms`` needs a
    backend-specific override (pxr.Usd reads via kit.stage).
    """

    def _snapshot_body_transforms(self, stage: Any, body_paths: list) -> tuple:
        from genie_sim_engine_isaacsim.kit.stage import snapshot_body_transforms

        return snapshot_body_transforms(stage, body_paths)

    def bind_newton_body_topology(self) -> None:
        """Register the complete built model; called only by the Newton node."""
        import numpy as np

        poses, frames = self.sim.get_body_transforms()
        frames = list(frames)
        # Core's initial USD snapshot skips runtime plugin bodies without prims
        # (e.g. chef_wok). Its C++ publisher rejects a pose array whose length
        # differs from that table. Rebind through the existing topology API to
        # the complete model ordering used by every subsequent Newton readback.
        poses = np.asarray(poses)
        if frames != list(self.sim.body_paths) or poses.shape != (len(frames), 7) or not np.isfinite(poses).all():
            raise RuntimeError("Newton body readback does not match the complete model body topology")
        robot_prefix = self.sim._robot_prefix_str
        robot_root = f"/{robot_prefix}/"
        body_is_robot = [name == f"/{robot_prefix}" or name.startswith(robot_root) for name in frames]
        worlds = int(self.sim.num_envs)
        if worlds > 1 and frames:
            if len(frames) % worlds:
                raise RuntimeError("Newton body topology has nonuniform world sizes")
            per_world = len(frames) // worlds
            frames = [f"env_{index // per_world}/{name.lstrip('/')}" for index, name in enumerate(frames)]
        if any(not name for name in frames) or len(set(frames)) != len(frames):
            raise RuntimeError("Newton body topology requires unique nonempty frame names")
        self.core.set_topology(list(self.robot_joint_names), frames, body_is_robot)
