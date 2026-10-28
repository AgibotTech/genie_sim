#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""FUSED Isaac Sim node: PhysX physics AND tiled rendering in ONE process.

The split-process architecture (``isaacsim_tiled_render.py`` + a separate
engine node) pays for the split three times over: two Isaac Sim startups, two
GPU contexts, and a full serialise/deserialise of every body pose through ROS
on every step -- between two processes that are holding the very same USD
files. Worse, the render side has to DESTROY physics on its stage
(``_disable_physics_subtree``) so its own articulation does not fight the poses
being written in from outside.

This node inverts that last point: it keeps the physics alive on the multi-env
GridCloner stage and lets it drive the prims that are about to be rendered. No
pose ever leaves the process. See
``source/rlinf_geniesim/docs/ISAAC_FUSED_PHYSX_DESIGN.md``.

Wire protocol -- lockstep, exactly one request in flight:

    client (rlinf FusedIsaacBackend)          this node
    ------------------------------------      -------------------------------
    write actions / reset_mask into SHM
    set_command(CMD_STEP|CMD_RESET)
    sem.post_request()              ------>   sem.wait_request()
                                              apply actions
                                              step physics x K
                                              read joint/body/object state
                                              render tiled RGB -> tiled SHM
                                              publish_ok() / publish_error()
    sem.wait_response()             <------   sem.post_response()
    read state regions

THIS node creates BOTH shared-memory segments, because both are sized from
facts that only exist once the stage has been built here: the tiled image
segment from the camera tile count (as the render-only node has always done),
and the state segment from the articulation's DOF count plus the bodies found
by walking the stage. The client therefore attaches to them AFTER we signal
readiness -- see ``fused_shm.FusedStateShm``'s docstring for the full startup
order and for why the client can independently re-derive the same dimensions
from the schema file (which is what makes its header check a real cross-check).
The semaphore pair is the other way round: the client creates it, since it
needs no dimensions at all.

Usage (inside the geniesim_ros container, after sourcing ROS):
    python3 isaacsim_fused_sim.py --manifest /tmp/.../manifest.json \\
        --shm-name rlinf_tiled --state-shm rlinf_fused_state \\
        --req-sem /rlinf_fused_req --resp-sem /rlinf_fused_resp \\
        --schema-out /tmp/.../schema.json --ready-file /tmp/.../ready \\
        --num-envs 4 --physics-hz 120 --control-hz 30
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional


def _bootstrap_source_path() -> None:
    """Put ``source/`` on sys.path so ``rlinf_geniesim.*`` imports resolve.

    Same trick ``isaacsim_tiled_render.py`` uses -- this node is launched as a
    bare script from the colcon install tree, so the repo's python packages are
    not importable until we find them.
    """
    src = os.environ.get("SIM_REPO_ROOT")
    if src and os.path.isdir(os.path.join(src, "source")):
        sys.path.insert(0, os.path.join(src, "source"))
        return
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "source"
        if cand.is_dir():
            sys.path.insert(0, str(cand))
            return


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isaac Sim FUSED physics+render node")
    p.add_argument("--manifest", required=True, help="stage manifest.json path")
    p.add_argument("--shm-name", default="rlinf_tiled", help="TiledFrameShm segment name (images out)")
    p.add_argument("--state-shm", required=True, help="FusedStateShm segment name (actions in / state out)")
    p.add_argument("--req-sem", required=True, help="request semaphore name")
    p.add_argument("--resp-sem", required=True, help="response semaphore name")
    p.add_argument("--schema-out", default="", help="write the SceneSchema JSON here once the stage is up")
    p.add_argument("--ready-file", default="", help="touch this once initialised (client waits on it)")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--env-spacing", type=float, default=15.0)
    p.add_argument("--physics-hz", type=float, default=120.0)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--render-hz", type=float, default=30.0)
    p.add_argument("--headless", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--physics-params-file",
        default="",
        help="physics_params_rlinf.yaml -- per-joint-class PD gains. " "Empty = leave the USD DriveAPI gains alone.",
    )
    p.add_argument(
        "--reset-settle-time-s",
        type=float,
        default=1.0,
        help="settle the physics this long after a reset, matching " "GenieSimHandler's reset_settle_time_s default",
    )
    return p.parse_args(argv)


# Args + SimulationApp BEFORE any omni/isaacsim import -- Kit requires it.
_ARGS = parse_args([a for a in sys.argv[1:] if not a.startswith("__")])
_bootstrap_source_path()
# Hide our CLI from Kit's SimulationApp (it parses sys.argv itself).
sys.argv = [sys.argv[0]]

from genie_sim_engine_core.common.parent_lifetime import exit_with_parent  # noqa: E402

# Before Kit starts: a server killed during the slow startup must not leave this node behind.
exit_with_parent()

from isaacsim import SimulationApp  # noqa: E402
from genie_sim_engine_isaacsim.kit.launch_config import prepare_launch_config  # noqa: E402

# ``disable_viewport_updates``: see the long measurement note in
# isaacsim_tiled_render.py -- headless still renders the invisible viewport
# otherwise, measured -28% frame time with it on.
simulation_app = SimulationApp(
    prepare_launch_config(
        {
            "headless": bool(_ARGS.headless),
            "disable_viewport_updates": (
                bool(_ARGS.headless) and os.environ.get("RLINF_TILED_DISABLE_VIEWPORT", "1").strip() != "0"
            ),
        }
    )
)

# --- post-SimulationApp imports ---
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402

import _isaac_render_common as common  # noqa: E402
from rlinf_geniesim.comm.shm.sem import SemPair  # noqa: E402
from rlinf_geniesim.physics.fused_shm import (  # noqa: E402
    CMD_RESET,
    CMD_SHUTDOWN,
    CMD_STEP,
    FusedStateShm,
)
from rlinf_geniesim.render.tiled_shm import TiledFrameShm  # noqa: E402
from rlinf_geniesim.render.bridges._hotswap_protocol import (  # noqa: E402
    peek_randomize_generation,
    read_randomize_cmd_if_new,
)
from _render_randomize import (  # noqa: E402
    _apply_randomize_cmd,
    _read_camera_pose_baseline,
    _read_light_baseline,
)


def _np(x) -> np.ndarray:
    """Coerce whatever an Isaac batched view hands back into a host numpy array.

    These views return a torch tensor ON THE PHYSICS DEVICE when the
    ``SimulationContext`` was built with ``device="cuda:0"`` -- and
    ``np.asarray`` on a CUDA tensor does not silently copy, it raises
    ``TypeError: can't convert cuda:0 device type tensor to numpy``. Warp
    arrays show up on some code paths too. Funnelling every readback through
    one converter means a device or backend change cannot break half the
    state and leave the other half looking fine.
    """
    if hasattr(x, "detach"):  # torch.Tensor (possibly on GPU)
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):  # warp array / anything numpy-able
        return x.numpy()
    return np.asarray(x)


def _render_once(subframes: int = 1) -> None:
    """Force one RTX render + annotator populate (a bare app.update does not)."""
    # Isaac Sim's orchestrator advances the timeline by default, even when
    # called only to refresh RGB. Physics must advance only in sim.step(), or
    # state readback precedes an extra integration and no longer matches RGB.
    # Verified on the installed Isaac API: step exposes delta_time explicitly.
    rep.orchestrator.step(rt_subframes=max(1, int(subframes)), pause_timeline=False, delta_time=0.0)


# ---------------------------------------------------------------------------
# Physics views
# ---------------------------------------------------------------------------


class FusedPhysics:
    """Batched PhysX handles over the cloned multi-env stage.

    One ``Articulation`` view spanning ``/World/envs/env_*/<robot>`` gives
    batched joint control and joint state across every env in a single call --
    this is the whole reason multi-env PhysX is affordable, and it is the same
    mechanism Isaac Lab is built on.

    Per-LINK world poses need a second view: ``Articulation`` exposes joint
    state and root pose but has no batched link-transform getter (verified
    against the installed Isaac build -- ``get_body_*`` there is mass/inertia,
    not transforms). The obvious fallback, reading each link's xform off the
    USD stage, is both slow and can be STALE under GPU physics, where fabric is
    the source of truth and USD is only synced on demand. So links are read
    through a ``RigidPrim`` view instead, which pulls straight from the physics
    tensors.

    That view is constructed from an EXPLICIT ordered path list, never a regex.
    A regex leaves the ordering to Isaac's traversal, and the
    ``SceneSchema`` contract requires ``body_paths[i]`` to line up with
    ``get_body_poses()[:, i, :]`` forever. An explicit list makes that ordering
    ours and lets the schema simply quote it.
    """

    def __init__(self, stage, robot_prefix: str, num_envs: int, device: str, logger=print) -> None:
        self._stage = stage
        self._num_envs = int(num_envs)
        self._device = device
        self._log = logger
        self._robot_prefix = robot_prefix or "genie"

        self.articulation = None
        self.link_view = None
        self.object_view = None
        self.joint_names: List[str] = []
        self.body_rel_paths: List[str] = []  # env-relative, ordered
        self.object_rel_paths: List[str] = []  # env-relative, ordered
        self.default_joint_pos = None  # pose reset() returns to
        self.instance_poses: dict = {}  # authored per-instance object poses
        self.build_instance: Optional[int] = None
        self.table_materials: dict = {}
        self._active_instances = [0] * self._num_envs
        self.instance_body_poses: dict = {}
        self._body_pose_rows = []
        # Superset slots each instance uses (``instance_switch.name_alias``);
        # the others are parked with gravity off.
        self._slots_by_instance: dict = {}
        self._parked = None  # (num_envs, len(_body_pose_rows)) bool, as applied
        self._joint_prims = None  # lazy joint-name -> prim
        self.background_articulations = None
        # scene_cfg.randomization: what a seeded reset perturbs, plus the
        # per-env joint_pd multipliers (see genie_sim_engine_core.common.reset_randomization).
        self.randomization: dict = {}
        self.instance_layouts: dict = {}
        # (positions (N, 3), quats (N, 4)) the robot base returns to; a seeded
        # base offset is always applied relative to this, never accumulated.
        self.default_base_pose = None
        self._warned: set = set()

    # -- discovery ---------------------------------------------------

    def _env_path(self, i: int) -> str:
        return f"/World/envs/env_{i}"

    def discover(self) -> None:
        """Walk env_0 ONLY and record env-relative paths, then fan them out to
        every env by string substitution.

        Walking one env and replicating is not a shortcut -- it is the
        correctness argument. GridCloner produced every env as a deep copy of
        env_0, so any path that exists under env_0 exists under env_i at the
        same relative location. Discovering per-env independently would let a
        traversal-order difference in one env silently shift that env's
        columns relative to the schema.
        """
        from pxr import Usd
        from _fused_point_instancers import BODY_MARKER

        env0 = self._env_path(0)
        root = self._stage.GetPrimAtPath(env0)
        if not root or not root.IsValid():
            raise RuntimeError(
                f"fused: env_0 prim {env0} missing -- the " f"GridCloner stage was not built as expected"
            )

        robot_root = f"{env0}/{self._robot_prefix}"
        links: List[str] = []
        dup_links: List[str] = []
        objects: List[str] = []
        for prim in Usd.PrimRange(root):
            # Particle-scale rigid instances use a separate reset view. Adding
            # thousands of them to the task's object table changes its schema
            # and can exceed the teleop SHM capacity.
            if prim.GetCustomDataByKey(BODY_MARKER):
                continue
            # ``HasAPI`` alone is not the question: a prim can carry
            # RigidBodyAPI and still be authored ``rigidBodyEnabled = false``,
            # which is how rooms mark their decor as scenery. PhysX honours
            # that flag and builds no rigid body, so feeding such a path to a
            # RigidPrim view yields "did not match any rigid bodies" and then
            # a GPU-side assert. See common.rigid_body_is_enabled.
            if not common.rigid_body_is_enabled(prim):
                continue
            path = str(prim.GetPath())
            rel = path[len(env0) + 1 :]
            segments = rel.split("/")
            if path.startswith(robot_root + "/") or path == robot_root:
                # THE robot: the copy this node mounted at env_0/<prefix>.
                links.append(rel)
            elif self._robot_prefix in segments:
                # A DUPLICATE robot, not a task object. These are real and
                # routine, not a broken manifest: ``foreground_usda`` is often
                # the very same ``scene.usda`` used as the base stage, and that
                # file already contains its own robot -- so referencing it under
                # env_0/World brings a second (and, once cloned, a third) copy
                # of every robot link along with it.
                #
                # The render-only path never had to care (it strips physics from
                # the whole subtree), but here they would be SIMULATED: dozens
                # of extra articulated bodies spawned inside each other,
                # interpenetrating and fighting. Dropping them is what makes
                # fused physics match what the task actually describes.
                dup_links.append(rel)
            else:
                # Non-robot rigid bodies: the task's movable props. Static
                # scenery is excluded by the check above -- either it carries
                # no RigidBodyAPI at all, or it carries one that the asset
                # authored ``rigidBodyEnabled = false``. Both are what we
                # want: a static prop has no pose to report per step.
                objects.append(rel)
        self.body_rel_paths = sorted(links)
        self.object_rel_paths = sorted(objects)
        self._log(
            f"[fused] discovered {len(self.body_rel_paths)} robot link(s), "
            f"{len(self.object_rel_paths)} movable object(s) under env_0"
            + (
                f", dropped {len(dup_links)} duplicate robot link(s) " f"carried in by the foreground USD"
                if dup_links
                else ""
            )
        )
        # Sample the object paths: "0 objects" on a task that visibly has props
        # means they arrived without RigidBodyAPI, and the only way to tell that
        # apart from "discovery is broken" is to see what was actually found.
        if self.object_rel_paths:
            self._log(f"[fused] object sample: {self.object_rel_paths[:5]}")
        else:
            self._log(
                "[fused] NOTE: no movable objects found under env_0 -- "
                "if this task has props, they carry no RigidBodyAPI and "
                "nothing will fall or be grasped"
            )

    # -- views -------------------------------------------------------

    def initialize(self) -> None:
        """Create the batched views. MUST run after physics has been started
        (``SimulationContext.reset()``): the tensor APIs these views wrap do
        not exist until the PhysX scene is initialised, and constructing them
        earlier yields views whose ``count`` is 0."""
        from isaacsim.core.prims import Articulation, RigidPrim

        expr = f"/World/envs/env_*/{self._robot_prefix}"
        self.articulation = Articulation(expr)
        self.articulation.initialize()
        self.joint_names = list(self.articulation.dof_names)
        if self.articulation.count != self._num_envs:
            raise RuntimeError(
                f"fused: articulation view matched {self.articulation.count} "
                f"prim(s) for '{expr}' but the stage has {self._num_envs} "
                f"env(s) -- the robot prim is missing from some env, or the "
                f"robot_prefix is wrong"
            )
        self._log(f"[fused] articulation: {self.articulation.count} env(s) x " f"{len(self.joint_names)} dof")

        if self.body_rel_paths:
            paths = [f"{self._env_path(i)}/{rel}" for i in range(self._num_envs) for rel in self.body_rel_paths]
            self.link_view = RigidPrim(paths)
            self.link_view.initialize()
        if self.object_rel_paths:
            paths = [f"{self._env_path(i)}/{rel}" for i in range(self._num_envs) for rel in self.object_rel_paths]
            self.object_view = RigidPrim(paths)
            self.object_view.initialize()

        # Which array type do the setters want? Probe once rather than assume:
        # with device="cuda:0" these views run Isaac's TORCH backend, whose
        # setters call ``.to(...)`` on whatever they are handed -- a numpy array
        # raises ``AttributeError: 'numpy.ndarray' object has no attribute 'to'``.
        # Probing keeps this correct if the backend is ever switched to numpy
        # instead of encoding today's choice in every call site.
        self._uses_torch = hasattr(self.articulation.get_joint_positions(), "detach")
        self._log(f"[fused] view backend: " f"{'torch' if self._uses_torch else 'numpy'}")
        from _fused_background_reset import BackgroundArticulations

        self.background_articulations = BackgroundArticulations(self._stage, self._num_envs)
        self._log(f"[fused] background joints: {self.background_articulations.joint_paths}")

    def apply_initial_pose(self, mf: dict, env_origins) -> None:
        """Put the robot where the TASK says it starts: authored joint angles
        from ``manifest.init_joint_pos`` and base placement from
        ``manifest.robot_init_pose``.

        Skipping this does not fail loudly -- it produces a robot folded at its
        USD zero pose, standing at the world origin instead of backed off from
        the table. The head camera then stares at the tabletop from a few
        centimetres away and every frame comes back a near-featureless grey.

        Takes the RAW manifest dict, not the parsed ``common.Manifest``: that
        dataclass is the RENDER node's view and simply has no
        ``init_joint_pos`` / ``robot_init_pose`` fields, because a render-only
        node is told every pose over ROS and never needs to know where the
        robot starts. ``getattr`` on it returns None and the whole step
        silently does nothing.

        Units follow the existing engine exactly (``kit/stage.py``): a REVOLUTE
        joint's value is in DEGREES and must be converted for the tensor API,
        a PRISMATIC joint's is already in metres. Getting this wrong is silent
        too -- 42.3 radians is a perfectly legal number that wraps the joint
        several times round.

        The applied pose is also stashed as ``self.default_joint_pos`` so reset
        returns here rather than to the USD zero pose.
        """
        import math
        from pxr import UsdPhysics

        art = self.articulation
        n_env, n_dof = self._num_envs, len(self.joint_names)

        # --- joints
        init = (mf or {}).get("init_joint_pos") or {}
        pos = _np(art.get_joint_positions()).reshape(n_env, n_dof).copy()
        if init:
            name_to_idx = {n: i for i, n in enumerate(self.joint_names)}
            deg2rad = math.pi / 180.0
            applied, unknown = 0, []
            for jname, spec in init.items():
                idx = name_to_idx.get(jname)
                if idx is None:
                    idx = name_to_idx.get(f"{self._robot_prefix}_{jname}")
                if idx is None:
                    unknown.append(jname)
                    continue
                value = spec.get("value") if isinstance(spec, dict) else spec
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    unknown.append(jname)
                    continue
                jprim = self._joint_prim(self.joint_names[idx])
                is_prismatic = bool(jprim and jprim.IsA(UsdPhysics.PrismaticJoint))
                pos[:, idx] = value if is_prismatic else value * deg2rad
                applied += 1
            self._log(
                f"[fused] init_joint_pos: applied {applied}/{len(init)} "
                f"joint(s)" + (f", unknown: {unknown[:4]}" if unknown else "")
            )
            # A teleport must satisfy the same mimic relation as the solver.
            # Leaving a follower at the old angle makes the first physics tick
            # swing it through nearby props (scoop_popcorn's bucket). Save the
            # consistent pose below so episode resets cannot repeat that jump.
            from _fused_mimic_pose import synchronize_mimic_joint_positions

            pos = synchronize_mimic_joint_positions(
                self._stage.GetPrimAtPath(f"{self._env_path(0)}/{self._robot_prefix}"), self.joint_names, pos
            )
            art.set_joint_positions(self.to_backend(pos))
            art.set_joint_velocities(self.to_backend(np.zeros((n_env, n_dof), dtype=np.float32)))
            art.set_joint_position_targets(self.to_backend(pos))
        self.default_joint_pos = pos

        # --- base placement, per env. The authored position is relative to the
        # env, so the GridCloner origin has to be ADDED -- without it every env
        # stacks its robot at the same world point while the rest of each env
        # sits at its own origin.
        rip = (mf or {}).get("robot_init_pose") or {}
        bpos = rip.get("position") if isinstance(rip, dict) else None
        bquat = rip.get("quaternion") if isinstance(rip, dict) else None
        if bpos is not None:
            origins = (
                np.asarray(env_origins, dtype=np.float64).reshape(n_env, 3)
                if env_origins is not None
                else np.zeros((n_env, 3))
            )
            world = origins + np.asarray(bpos, dtype=np.float64).reshape(1, 3)
            quat = np.tile(
                np.asarray(bquat if bquat is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64).reshape(1, 4),
                (n_env, 1),
            )
            try:
                art.set_world_poses(self.to_backend(world.astype(np.float32)), self.to_backend(quat.astype(np.float32)))
                self.default_base_pose = (world.astype(np.float32), quat.astype(np.float32))
                self._log(f"[fused] robot_init_pose applied: base {bpos} " f"+ per-env origin")
            except Exception as exc:  # noqa: BLE001
                self._log(
                    f"[fused] WARN: robot base pose not applied ({exc!r}) " f"-- the robot will stand at the env origin"
                )

    def _joint_prim(self, joint_name: str):
        """Locate a joint's USD prim under env_0 so its type can be checked."""
        if self._joint_prims is None:
            from pxr import Usd, UsdPhysics

            self._joint_prims = {}
            root = self._stage.GetPrimAtPath(self._env_path(0))
            if root and root.IsValid():
                for prim in Usd.PrimRange(root):
                    if (
                        prim.IsA(UsdPhysics.Joint)
                        or prim.IsA(UsdPhysics.RevoluteJoint)
                        or prim.IsA(UsdPhysics.PrismaticJoint)
                    ):
                        self._joint_prims.setdefault(prim.GetName(), prim)
        return self._joint_prims.get(joint_name)

    def _is_mimic_follower(self, joint_name: str) -> bool:
        """Is this gripper joint a mimic FOLLOWER (vs. the master)?

        Detection copied from the benchmark PhysX path
        (``genie_sim_engine_isaacsim/kit/stage.py``'s ``_configure_drives``),
        including which schema names count, because the answer decides
        whether a joint gets driven or left to the constraint -- and getting
        it backwards silently locks the gripper.

        Followers carry a mimic schema; masters never do. The schema name
        varies by import pipeline, and benchmark enumerates the three cases:

          * Isaac Sim 6.2 after ``convert_joints_attributes`` ->
            ``PhysxMimicJointAPI:<axis>`` (an INSTANCE api, hence the
            prefix test rather than an equality test).
          * Isaac Sim 6.2 raw Newton output -> ``NewtonMimicAPI``.
          * Isaac Sim 4.x/5.x ``URDFParseAndImportFile`` -> neither. On that
            pipeline no mimic constraint exists at all, so every gripper
            joint is independently driven; falling through to the MASTER
            branch is the correct behaviour there (benchmark says so
            explicitly), which is what returning False does.

        Also accepts the plain-USD ``newton:mimicJoint`` RELATIONSHIP that
        ``_init_articulation`` keys on -- same file, second pass. It is
        checked too because it is visible under a vanilla pxr while the
        instance APIs may not be registered outside the kit runtime.

        Unknown joint (no prim found) -> False, i.e. treat as master. That
        matches the 4.x/5.x fall-through above and keeps a missing prim from
        silently zeroing a joint's gains.
        """

        prim = self._joint_prim(joint_name)
        if prim is None:
            return False
        try:
            applied = list(prim.GetAppliedSchemas())
        except Exception:  # noqa: BLE001
            applied = []
        if any(str(a).startswith("PhysxMimicJointAPI:") for a in applied):
            return True
        if "NewtonMimicAPI" in applied:
            return True
        # What this repo's assets actually carry: not an applied schema at
        # all, but a RELATIONSHIP named ``physxMimicJoint:<axis>:referenceJoint``
        # -- lowercase ``p``, axis in the middle. Neither of the two schema
        # names above matches it, so this relationship must be checked too or
        # followers get misclassified as masters.
        try:
            for rel in prim.GetRelationships():
                rn = rel.GetName()
                if rn.startswith("physxMimicJoint:") and rel.HasAuthoredTargets():
                    return True
                if rn == "newton:mimicJoint" and rel.HasAuthoredTargets():
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    def apply_drive_gains(self, params_file: str) -> None:
        """Apply the per-joint-class PD gains from ``physics_params*.yaml``.

        Without this the articulation runs on whatever the robot USD's
        DriveAPI happens to author, which is NOT what the rest of this repo
        means by "the robot's gains": every tuning conclusion, and the newton
        backend's runtime behaviour, come from this yaml (kp 32000-60000 on
        the arm vs whatever the asset shipped). Skipping it leaves the arm
        visibly lagging a commanded position instead of tracking it, even
        though the actions are wired correctly -- it is simply far too soft.

        Mirrors the per-class table in the newton path's mjwarp adapter
        (``_per_class_*``) so the two backends are driven by the same numbers
        from the same file, rather than this one quietly having its own feel.

        Gripper: follows the BENCHMARK PhysX path verbatim
        (``genie_sim_engine_isaacsim/kit/stage.py``'s ``_init_articulation``),
        because this backend IS PhysX and every gripper tuning conclusion in
        the repo was measured through that code:

          * MASTER -> ``drive_gripper.master_stiffness`` / ``master_damping``.
          * FOLLOWER -> kp=0, kd=0. The mimic constraint PhysX installs from
            ``PhysxMimicJointAPI`` is what propagates the master's motion; a
            non-zero PD on a follower FIGHTS that constraint, producing the
            low-frequency limit cycle that looks like a swinging gripper.
          * Master/follower is decided by the ABSENCE of a mimic schema, NOT
            by a name regex and NOT by ``HasAPI(DriveAPI)``: the URDF->USD
            importer authors a vestigial DriveAPI on every revolute joint,
            followers included, so a DriveAPI test hands master tuning to all
            five followers. A name regex is no better -- it encodes one
            asset's spelling, while the mimic schema is what the physics
            actually keys on.
          * Effort caps are PRESERVED from the articulation's own values
            (the URDF's ``<limit effort>``), with a 0-guard falling back to
            ``art_default.max_effort`` -- benchmark does exactly this rather
            than writing a number of its own.

        Driving all 16 gripper dof with master gains locks the closed
        four-bar linkage solid.
        """
        if not params_file:
            self._log("[fused] no physics params file given -- keeping the " "USD DriveAPI gains (arm will feel soft)")
            return
        try:
            from genie_sim_engine_core.common.params import load_physics_params
            from genie_sim_engine_newton.common.joint_classification import (
                JK_ARM,
                JK_ARM_MID,
                JK_ARM_SHOULDER,
                JK_ARM_WRIST,
                JK_BODY,
                JK_CHASSIS_DRIVE,
                JK_CHASSIS_STEER,
                JK_GRIPPER,
                JK_HEAD,
                classify_joint_by_name,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"[fused] WARN: cannot import gain tables ({exc!r}) -- " f"keeping USD DriveAPI gains")
            return
        try:
            p = load_physics_params(params_file)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[fused] WARN: cannot load {params_file} ({exc!r}) -- " f"keeping USD DriveAPI gains")
            return

        # Effort caps: read what the articulation already carries (the
        # URDF's own <limit effort>) so we can PRESERVE it per benchmark,
        # instead of substituting a number this file invented.
        try:
            prev_eff = _np(art.get_max_efforts()).reshape(self._num_envs, len(self.joint_names)).astype(np.float32)
        except Exception:  # noqa: BLE001 -- absent on some view versions
            prev_eff = None

        table = {
            JK_BODY: (p.art_body.kp, p.art_body.kd),
            JK_HEAD: (p.art_head.kp, p.art_head.kd),
            JK_ARM: (p.art_default.kp, p.art_default.kd),
            JK_ARM_SHOULDER: (p.art_arm_shoulder.kp, p.art_arm_shoulder.kd),
            JK_ARM_MID: (p.art_arm_mid.kp, p.art_arm_mid.kd),
            JK_ARM_WRIST: (p.art_arm_wrist.kp, p.art_arm_wrist.kd),
            JK_CHASSIS_DRIVE: (p.art_chassis_drive.kp, p.art_chassis_drive.kd),
            JK_CHASSIS_STEER: (p.art_chassis_steer.kp, p.art_chassis_steer.kd),
        }

        art = self.articulation
        n_env, n_dof = self._num_envs, len(self.joint_names)
        # Seeded from the CURRENT values so any dof we deliberately skip
        # (the gripper, per benchmark's pre-baked-USD rule) is written back
        # byte-identical rather than zeroed.
        kps = _np(art.get_gains()[0]).reshape(n_env, n_dof).astype(np.float32)
        kds = _np(art.get_gains()[1]).reshape(n_env, n_dof).astype(np.float32)
        before = (float(kps.min()), float(kps.max()))
        counts: dict = {}
        eff = prev_eff.copy() if prev_eff is not None else None
        for i, jn in enumerate(self.joint_names):
            kind = classify_joint_by_name(jn)
            if kind == JK_GRIPPER:
                # PRE-BAKED USD route (manifest ``robot_from_urdf`` false --
                # this repo's G2 asset): benchmark's rule is to leave the
                # gripper exactly as the USD authored it, because the asset
                # IS the tuning ("the gripper is already tuned in-USD ... so
                # we leave it alone", kit/stage.py's _configure_drives). The
                # yaml's own master_stiffness=100 / master_damping=20 are
                # described in-file as "benchmark/URDF-authored value" -- the
                # same numbers the asset carries, i.e. the yaml MIRRORS the
                # asset here rather than overriding it.
                #
                # Of the 16 gripper dof, only 2 (``outer_joint1``) carry a
                # DriveAPI and 2 (``inner_joint1``) carry the mimic
                # relationship; the remaining 12 are the closed four-bar's
                # coupler joints, authored with NO drive on purpose. Writing
                # any PD onto those 12 locks the linkage solid.
                #
                # Only the URDF route needs the master/follower split, and
                # ``_is_mimic_follower`` stays for that case (and for the log
                # line below, which is how we can SEE the asset's shape).
                counts["gripper_follower" if self._is_mimic_follower(jn) else "gripper_as_authored"] = (
                    counts.get("gripper_follower" if self._is_mimic_follower(jn) else "gripper_as_authored", 0) + 1
                )
                continue
            gains = table.get(kind)
            if gains is None:
                continue
            kps[:, i] = float(gains[0])
            kds[:, i] = float(gains[1])
            counts[kind] = counts.get(kind, 0) + 1
        art.set_gains(self.to_backend(kps), self.to_backend(kds))
        if eff is not None:
            try:
                art.set_max_efforts(self.to_backend(eff))
            except Exception as exc:  # noqa: BLE001
                self._log(f"[fused] WARN: set_max_efforts failed ({exc!r})")
        self._log(
            f"[fused] drive gains from {Path(params_file).name}: "
            f"kp {before[0]:g}..{before[1]:g} -> "
            f"{kps.min():g}..{kps.max():g} on "
            f"{sum(counts.values())}/{n_dof} dof {dict(sorted(counts.items()))}"
        )

    def load_instance_table(self, manifest_path: str) -> None:
        """Validate baked env-local body poses before physics starts.

        Locator poses alone cannot restore per-instance child transforms. Old
        instance-switch artifacts must be rebaked, never interpreted as bodies.
        """
        cfg = common.load_scene_config(manifest_path)
        self.randomization = dict(cfg.get("randomization") or {})
        sw = cfg.get("instance_switch") or {}
        self.instance_poses = sw.get("instance_poses") or {}
        self.instance_layouts = sw.get("instance_layout") or {}
        self.instance_body_poses = {}
        self._body_pose_rows = [
            (i, rel.removeprefix("World/Objects/"))
            for i, rel in enumerate(self.object_rel_paths)
            if rel.startswith("World/Objects/")
        ]
        bi = sw.get("build_instance")
        self.build_instance = int(bi) if bi is not None else None
        self.table_materials = sw.get("table_materials") or {}
        self._slots_by_instance = {str(k): set((v or {}).values()) for k, v in (sw.get("name_alias") or {}).items()}
        self._parked = np.zeros((self._num_envs, len(self._body_pose_rows)), dtype=bool)
        self._active_instances = [self.build_instance or 0] * self._num_envs
        if not sw:
            return
        if type(sw.get("body_pose_version")) is not int or sw["body_pose_version"] != 1:
            raise ValueError(
                "fused: instance body pose format is missing or unsupported; "
                "rebake the task with body_pose_version=1"
            )
        tables = sw.get("instance_body_poses")
        if not isinstance(tables, dict):
            raise ValueError("fused: missing instance_body_poses; rebake the task")
        ids = {str(i) for i in sw.get("group_instances", self.instance_poses)}
        if set(tables) != ids or (bi is not None and str(int(bi)) not in tables):
            raise ValueError("fused: instance body pose tables do not cover the instance group")
        expected = {key for _, key in self._body_pose_rows}
        for iid, table in tables.items():
            if not isinstance(table, dict) or set(table) != expected:
                raise ValueError(
                    f"fused: instance {iid} body pose paths do not match the stage; "
                    f"expected {sorted(expected)}, got {sorted(table) if isinstance(table, dict) else table}"
                )
            checked = {}
            for key, row in table.items():
                pose = np.asarray(row, dtype=np.float64)
                if (
                    pose.shape != (7,)
                    or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[3:]), 1.0, atol=1e-4, rtol=0)
                ):
                    raise ValueError(f"fused: invalid body pose for instance {iid}, {key}")
                checked[key] = pose.copy()
            self.instance_body_poses[iid] = checked
        self._log(
            f"[fused] instance body table v1: {len(tables)} instance(s), "
            f"{len(expected)} bodies, build={self.build_instance}"
        )

    @staticmethod
    def _short(rel_path: str) -> str:
        """Task schema's historical object name; reset uses exact body paths."""
        return rel_path.rsplit("/", 2)[-2] if "/" in rel_path else rel_path

    # -- superset parking ------------------------------------------------

    def _parking_changes(self, env_ids, instances):
        """Rows to give gravity back to, and to take it from, so each env's
        parked slots match ``instances`` (resolved ids, one per env).

        A slot is parked for an instance that does not name it in
        ``name_alias``. The superset parks those tens of metres away at table
        height; with gravity they fall to the floor and roll for the whole
        episode (wasted contacts, a polluted rest-state bake, false "the robot
        moved it" alarms). With gravity off, zero velocity and nothing within
        metres they stay put and soon sleep. Taking them out of the simulation
        instead (``disable_rigid_body_physics``) breaks the renderer's start-up
        on GPU: the ray-tracing pipeline fails to build.
        An instance missing from ``name_alias`` parks nothing, so a task
        without it keeps gravity on every slot as before.
        """
        enable, disable = [], []
        if self._parked is None or not self._slots_by_instance:
            return enable, disable
        n_obj = len(self.object_rel_paths)
        for env, instance in zip(env_ids, instances):
            used = self._slots_by_instance.get(str(instance))
            for col, (slot_index, body_key) in enumerate(self._body_pose_rows):
                parked = used is not None and body_key.split("/", 1)[0] not in used
                if parked != self._parked[env, col]:
                    (disable if parked else enable).append(env * n_obj + slot_index)
                    self._parked[env, col] = parked
        return enable, disable

    def _set_gravity(self, rows, on: bool) -> None:
        if not rows or self.object_view is None:
            return
        idx = self.to_backend(np.asarray(sorted(rows), dtype=np.int64), long=True)
        if on:
            self.object_view.enable_gravities(indices=idx)
        else:
            self.object_view.disable_gravities(indices=idx)
        # Wake the bodies either way. A parked body has long been asleep and
        # must wake to land on the table; and a gravity change on a body PhysX
        # thinks is asleep does not reach the GPU. Flag-type calls take host
        # indices (enable_gravities itself moves them to the CPU); device
        # indices make the backend refuse.
        physics_view = getattr(self.object_view, "_physics_view", None)
        if physics_view is not None and hasattr(physics_view, "wake_up"):
            physics_view.wake_up(idx.cpu() if hasattr(idx, "cpu") else idx)

    def apply_parking(self, env_ids: List[int]) -> None:
        """Turn gravity off for the current instances' unused superset slots
        (at startup, before the first step)."""
        enable, disable = self._parking_changes(env_ids, [self._active_instances[e] for e in env_ids])
        self._set_gravity(enable, True)
        self._set_gravity(disable, False)
        if disable:
            self._log(f"[fused] superset parking: gravity off for {len(disable)} parked bod(ies)")

    def apply_instance(self, env_ids: List[int], instances: List[int], env_origins) -> int:
        """Apply complete baked body frames only to the selected env/body rows.

        Slots the new instance uses get gravity back before their poses are
        written; slots it does not use are parked first and only then lose
        gravity, so none floats on the table."""
        if not self.instance_body_poses:
            return 0
        origins = (
            np.asarray(env_origins, dtype=np.float64).reshape(self._num_envs, 3)
            if env_origins is not None
            else np.zeros((self._num_envs, 3))
        )
        indices, positions, orientations = [], [], []
        resolved = []
        for env, instance in zip(env_ids, instances):
            if not 0 <= env < self._num_envs:
                raise ValueError(f"fused: invalid env {env}")
            resolved.append(self._active_instances[env] if instance is None or int(instance) < 0 else int(instance))
        enable, disable = self._parking_changes(env_ids, resolved)
        self._set_gravity(enable, True)
        for env, instance in zip(env_ids, resolved):
            key = str(instance)
            if key not in self.instance_body_poses:
                raise ValueError(f"fused: instance {instance} is not in the baked body table")
            for slot_index, body_key in self._body_pose_rows:
                pose = self.instance_body_poses[key][body_key]
                indices.append(env * len(self.object_rel_paths) + slot_index)
                positions.append(origins[env] + pose[:3])
                orientations.append(pose[3:])
        if not indices:
            return 0
        if self.object_view is None:
            raise RuntimeError("fused: body reset requested before object view initialization")
        idx = self.to_backend(np.asarray(indices, dtype=np.int64), long=True)
        self.object_view.set_world_poses(
            self.to_backend(np.asarray(positions, dtype=np.float32)),
            self.to_backend(np.asarray(orientations, dtype=np.float32)),
            indices=idx,
        )
        self.object_view.set_velocities(self.to_backend(np.zeros((len(indices), 6), dtype=np.float32)), indices=idx)
        self._set_gravity(disable, False)
        for env, instance in zip(env_ids, instances):
            if instance is not None and int(instance) >= 0:
                self._active_instances[env] = int(instance)
        return len(indices)

    def apply_reset_materials(self, env_ids, instances) -> None:
        from _instance_materials import apply_instance_materials

        selected = [int(i) if int(i) >= 0 else self._active_instances[e] for e, i in zip(env_ids, instances)]
        apply_instance_materials(self._stage, self.table_materials, env_ids, selected)
        for e, i in zip(env_ids, selected):
            self._active_instances[e] = i

    # -- seeded generalization ------------------------------------------

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            self._log(msg)

    def apply_joint_pd(self) -> None:
        """Scale arm, body and head PD gains per env from ``randomization.joint_pd``.

        The multipliers were drawn once at build time from the task-init seed
        (``task_loader.build_stage_manifest``) and are keyed by env index, so
        this only applies them -- once, at startup, on top of the gains
        ``apply_drive_gains`` just set. Grippers and the chassis keep their
        gains, as in Newton. A configuration that asks for scaling but cannot
        land on any env raises instead of running with the plain gains.
        """
        from genie_sim_engine_core.common.reset_randomization import joint_pd_eligible, scale_joint_pd

        jpd = self.randomization.get("joint_pd") or {}
        if not jpd:
            return
        from genie_sim_engine_newton.common.joint_classification import classify_joint_by_name

        art = self.articulation
        n_env, n_dof = self._num_envs, len(self.joint_names)
        kp_all, kd_all = art.get_gains()
        base_kp = _np(kp_all).reshape(n_env, n_dof).astype(np.float32)
        base_kd = _np(kd_all).reshape(n_env, n_dof).astype(np.float32)
        eligible = joint_pd_eligible(self.joint_names, classify_joint_by_name)
        kp, kd, requested, applied, touched = scale_joint_pd(jpd, range(n_env), base_kp, base_kd, eligible)
        if requested == 0:
            return
        if applied == 0:
            raise RuntimeError(
                f"[fused] joint_pd requested for {requested} env(s) but applied to 0 "
                f"({int(eligible.sum())} eligible dof among {n_dof})"
            )
        if applied < requested:
            self._log(f"[fused] WARN: joint_pd applied to {applied} of {requested} env(s)")
        if kp is not None and touched:
            art.set_gains(self.to_backend(kp), self.to_backend(kd))
            got = _np(art.get_gains()[0]).reshape(n_env, n_dof)
            if not np.allclose(got[touched], kp[touched], rtol=1e-4):
                raise RuntimeError("[fused] joint_pd gains did not land: readback differs from what was set")
        self._log(f"[fused] joint_pd: scaled {len(touched)} env(s) on " f"{int(eligible.sum())}/{n_dof} dof")

    def _instance_layout(self, env: int):
        """The env's active instance layout for the position constraints, or
        None when the task has none (no rest-state bake)."""
        return (self.instance_layouts or {}).get(str(self._active_instances[env]))

    def _object_reset_xy(self, env: int) -> list:
        """``[(object name, (x, y)), ...]`` env-local, sorted by name, from the
        env's active instance table. An object's position is its root body's
        (the shortest key below ``World/Objects/<name>``)."""
        from genie_sim_engine_core.common.reset_randomization import object_name

        table = self.instance_body_poses.get(str(self._active_instances[env])) or {}
        roots: dict = {}
        for key in sorted(table, key=lambda k: (k.count("/"), k)):
            name = object_name("World/Objects/" + key)
            if name is not None and name not in roots:
                roots[name] = (float(table[key][0]), float(table[key][1]))
        return sorted(roots.items())

    def randomize_reset(self, env_ids: List[int], seeds: List[int], env_origins):
        """Apply a reset's seeded generalization; return the arm joint offsets.

        For each env with ``seed >= 0`` this draws object xy, arm joint and
        robot base offsets from ``scene_cfg.randomization`` exactly as the
        Newton engine does (the shared ``reset_randomization.draw_reset``), with
        the instance's layout constraints when the rest-state bake recorded
        one (``instance_switch.instance_layout``). Object
        offsets move every body of the object on top of the instance pose that
        ``apply_instance`` just wrote; the base is placed at its default pose
        plus the offset (an unseeded reset puts it back at the default). Arm
        offsets are returned as ``{env: (num_dofs,) array}`` for ``_do_reset``
        to add to the joint reset target. Returns ``None`` when the
        configuration perturbs nothing, so such resets write nothing extra.

        Must run after ``apply_instance`` and before ``_do_reset``.
        """
        from genie_sim_engine_core.common.reset_randomization import draw_reset, is_active, object_name

        spec = self.randomization
        if not is_active(spec):
            return None
        n_dof = len(self.joint_names)
        arm_idx = sorted((i for i, n in enumerate(self.joint_names) if "_arm_" in n), key=lambda i: self.joint_names[i])
        bt = spec.get("init_base_xy_thresh")
        origins = (
            np.asarray(env_origins, dtype=np.float64).reshape(self._num_envs, 3)
            if env_origins is not None
            else np.zeros((self._num_envs, 3))
        )
        joint_offsets: dict = {}
        base_offsets: dict = {}
        indices, positions, orientations = [], [], []
        for env, seed in zip(env_ids, seeds):
            if seed is None or int(seed) < 0:
                if bt:
                    base_offsets[env] = (0.0, 0.0)
                continue
            objects = self._object_reset_xy(env)
            if not objects and float(spec.get("object_xy_radius", 0.03)) > 0 and self.object_rel_paths:
                self._warn_once(
                    "objects",
                    "[fused] WARN: object xy randomization needs the instance "
                    "body table; this task has none, so objects keep their poses",
                )
            draw = draw_reset(int(seed), spec, objects, len(arm_idx), has_base=True, layout=self._instance_layout(env))
            parts = []
            if draw.object_offsets:
                table = self.instance_body_poses[str(self._active_instances[env])]
                for slot, key in self._body_pose_rows:
                    off = draw.object_offsets.get(object_name("World/Objects/" + key))
                    if off is None:
                        continue
                    pose = table[key]
                    indices.append(env * len(self.object_rel_paths) + slot)
                    positions.append(origins[env] + pose[:3] + (off[0], off[1], 0.0))
                    orientations.append(pose[3:])
                parts.append(
                    f"{len(draw.object_offsets)} object(s): "
                    + ", ".join(f"{n}({d[0]:+.3f},{d[1]:+.3f})" for n, d in list(draw.object_offsets.items())[:6])
                )
            if draw.arm_offsets.any():
                offsets = np.zeros(n_dof, dtype=np.float32)
                offsets[arm_idx] = draw.arm_offsets
                joint_offsets[env] = offsets
                parts.append(f"arm offsets on {len(arm_idx)} joint(s) (thresh {spec.get('init_joint_thresh')})")
            if draw.base_offset is not None:
                base_offsets[env] = draw.base_offset
                parts.append(f"base ({draw.base_offset[0]:+.4f},{draw.base_offset[1]:+.4f})")
            if parts:
                self._log(f"[fused] reset-randomize env {env} seed {seed}: " + "; ".join(parts))
        if indices:
            idx = self.to_backend(np.asarray(indices, dtype=np.int64), long=True)
            self.object_view.set_world_poses(
                self.to_backend(np.asarray(positions, dtype=np.float32)),
                self.to_backend(np.asarray(orientations, dtype=np.float32)),
                indices=idx,
            )
            self.object_view.set_velocities(self.to_backend(np.zeros((len(indices), 6), dtype=np.float32)), indices=idx)
        if base_offsets:
            art = self.articulation
            if self.default_base_pose is None:
                pos, quat = art.get_world_poses()
                self.default_base_pose = (
                    _np(pos).reshape(self._num_envs, 3).astype(np.float32),
                    _np(quat).reshape(self._num_envs, 4).astype(np.float32),
                )
            envs = sorted(base_offsets)
            pos = np.array(self.default_base_pose[0][envs], dtype=np.float32)
            pos[:, :2] += np.asarray([base_offsets[e] for e in envs], dtype=np.float32)
            art.set_world_poses(
                self.to_backend(pos),
                self.to_backend(np.array(self.default_base_pose[1][envs], dtype=np.float32)),
                indices=self.to_backend(np.asarray(envs, dtype=np.int64), long=True),
            )
        return joint_offsets

    def to_backend(self, arr, *, long: bool = False):
        """numpy -> whatever the views' backend expects (see ``_uses_torch``)."""
        if not getattr(self, "_uses_torch", False):
            return arr
        import torch  # noqa: PLC0415 -- present in any Isaac runtime

        dev = getattr(self.articulation, "_device", None) or self._device
        t = torch.as_tensor(arr, device=dev)
        return t.to(torch.long) if long else t.to(torch.float32)

    # -- per-step I/O -------------------------------------------------

    def apply_actions(self, actions: np.ndarray) -> None:
        """``actions[:, j]`` is a POSITION TARGET for ``joint_names[j]``.

        Deliberately the same meaning the newton backends give it, so the same
        policy or the same recorded teleop tape produces the same motion on
        either backend. A different convention here would be near-impossible to
        debug from the outside: the robot would simply move differently.
        """
        self.articulation.set_joint_position_targets(self.to_backend(np.ascontiguousarray(actions, dtype=np.float32)))

    def read_joint_states(self, out: np.ndarray) -> None:
        pos = _np(self.articulation.get_joint_positions())
        vel = _np(self.articulation.get_joint_velocities())
        out[:, :, 0] = pos.reshape(self._num_envs, -1)
        out[:, :, 1] = vel.reshape(self._num_envs, -1)

    @staticmethod
    def _poses_into(view, num_envs: int, out: np.ndarray) -> None:
        """Scatter a flat ``(num_envs * B, ...)`` view read into ``(N, B, 7)``.

        The view was built env-major (all of env 0's paths, then env 1's), so a
        plain reshape recovers the two axes -- see ``initialize``.
        """
        pos, quat = view.get_world_poses()
        pos = _np(pos).reshape(num_envs, -1, 3)
        quat = _np(quat).reshape(num_envs, -1, 4)
        out[:, :, 0:3] = pos
        out[:, :, 3:7] = quat  # Isaac returns (w, x, y, z), same as our schema

    def read_body_poses(self, out: np.ndarray) -> None:
        """Fill ``(N, B, 7)`` with robot links FIRST, then movable objects.

        The body table the schema publishes is links + objects (see
        ``common.fused_body_tables`` for why objects have to be in it at all),
        so this array has to carry both in the same order. Slicing hands each
        view its own window: numpy basic slicing returns a VIEW, so both
        writes land in the caller's buffer.
        """
        n_links = len(self.body_rel_paths)
        if self.link_view is not None:
            self._poses_into(self.link_view, self._num_envs, out[:, :n_links])
        if self.object_view is not None:
            self._poses_into(self.object_view, self._num_envs, out[:, n_links:])

    def read_object_poses(self, out: np.ndarray) -> None:
        if self.object_view is not None:
            self._poses_into(self.object_view, self._num_envs, out)


def _log_effective_physics(stage) -> None:
    """Print the physics settings actually in force AFTER ``SimulationContext``
    has initialised.

    Not decoration. ``build_gridcloner_stage(physics=True)`` authors gravity and
    GPU dynamics onto the ``UsdPhysics.Scene`` before this process constructs a
    ``SimulationContext``, and that constructor defaults to ``set_defaults=True``
    -- which re-applies ITS OWN idea of the same settings. The two are expected
    to agree (Isaac derives gravity as ``-9.81 / metersPerUnit``, which is what
    the stage build computes too), but "expected to agree" is exactly the kind
    of assumption that silently stops being true across an Isaac upgrade. A
    gravity that quietly lost its unit conversion is 100x too weak on a
    centimetre stage: objects drift instead of falling, which reads as a broken
    grasp or a bad friction value and sends you tuning the wrong thing. One line
    of log makes that a five-second check instead.
    """
    from pxr import PhysxSchema, UsdGeom, UsdPhysics

    try:
        mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        scene_prim = None
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                scene_prim = prim
                break
        if scene_prim is None:
            print(
                "[fused] WARN: no UsdPhysics.Scene on the stage after "
                "SimulationContext init -- physics will not behave",
                flush=True,
            )
            return
        sc = UsdPhysics.Scene(scene_prim)
        mag = sc.GetGravityMagnitudeAttr().Get()
        direction = sc.GetGravityDirectionAttr().Get()
        px = PhysxSchema.PhysxSceneAPI(scene_prim)
        gpu = px.GetEnableGPUDynamicsAttr().Get()
        bp = px.GetBroadphaseTypeAttr().Get()
        in_ms2 = (float(mag) * mpu) if mag is not None else float("nan")
        print(
            f"[fused] effective physics @ {scene_prim.GetPath()}: "
            f"gravity={mag} stage-units/s^2 dir={direction} "
            f"(= {in_ms2:.2f} m/s^2 @ metersPerUnit={mpu:g}), "
            f"gpu_dynamics={gpu} broadphase={bp}",
            flush=True,
        )
        if mag is not None and not (9.0 <= in_ms2 <= 10.6):
            print(
                f"[fused] WARN: gravity resolves to {in_ms2:.3f} m/s^2, "
                f"which is not Earth gravity -- check the stage's "
                f"metersPerUnit against the authored magnitude",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 -- diagnostics must never stop the run
        print(f"[fused] effective-physics readback failed: {exc!r}", flush=True)


def _render_strategy_sweep(sensors, sim) -> None:
    """Try each plausible way of driving the RTX pass and report which one
    actually fills the annotator. Gated on ``RLINF_FUSED_RENDER_SWEEP=1``.

    Exists because "correctly-shaped, entirely zero" tells you the render
    product is wired up but nothing drew into it, and there are several
    independent candidates for whose job that is once a ``SimulationContext``
    is in the picture (it owns the render cadence; the replicator orchestrator
    owns it when there is no context, which is the render-only node's world).
    Guessing costs an Isaac boot per attempt; sweeping costs one.
    """
    import omni.replicator.core as _rep

    def _probe(label: str) -> None:
        for s_, _flat, _gh in sensors:
            d, _i = s_.get_data("rgb")
            a = _np(d) if d is not None else None
            print(
                f"[fused][sweep] {label}: "
                f"min={None if a is None else a.min()} "
                f"max={None if a is None else a.max()}",
                flush=True,
            )
            break

    strategies = [
        ("app.update x3", lambda: [simulation_app.update() for _ in range(3)]),
        ("sim.render()", lambda: sim.render()),
        ("sim.step(render=True)", lambda: sim.step(render=True)),
        ("orch.step(sub=1,pause=False)", lambda: _rep.orchestrator.step(rt_subframes=1, pause_timeline=False)),
        ("orch.step(sub=4,pause=True)", lambda: _rep.orchestrator.step(rt_subframes=4, pause_timeline=True)),
        ("orch.step() bare", lambda: _rep.orchestrator.step()),
    ]
    for label, fn in strategies:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[fused][sweep] {label}: RAISED {exc!r}", flush=True)
            continue
        _probe(label)


def build_schema(phys: FusedPhysics, static_prop_allowlist=()) -> dict:
    """The ``SceneSchema`` the rlinf side consumes.

    ``joint_names`` is quoted verbatim from ``Articulation.dof_names`` and the
    body ordering verbatim from the view's own path list -- neither is
    re-sorted here. That is what makes the ABC's alignment guarantee
    ("``body_paths[i]`` and ``get_body_poses()[:, i, :]`` refer to the same
    body") true by construction rather than by a convention someone has to
    remember.

    ``static_prop_allowlist`` -- the eval task's ``eval_static_props`` list
    (task_loader carries it as ``manifest["static_prop_aabbs"]``), i.e. WHICH
    background props to publish world boxes for, not the boxes themselves.
    Only point-named props are resolved (the market room alone has ~216) --
    see ``common.static_prop_world_aabbs`` for the resolution algorithm and
    why the box is world-frame, not body-local.
    """
    body_paths, body_kinds, body_link_names = common.fused_body_tables(phys.body_rel_paths, phys.object_rel_paths)
    _prop_names, _prop_aabbs = common.static_prop_world_aabbs(
        phys._stage, [phys._env_path(i) for i in range(phys._num_envs)], static_prop_allowlist
    )
    return {
        "body_paths": body_paths,
        "body_kinds": body_kinds,
        "body_link_names": body_link_names,
        # Static per-body box in each body's OWN frame, aligned with the table
        # above. Geometry does not change at runtime, so this ships once with
        # the schema instead of riding the per-step wire; the client composes
        # it with live poses (``_world_aabb_from_local``) exactly like the
        # newton path composes its ``/rlinf/body_aabb_local`` table.
        "body_local_aabbs": [
            common.local_aabb_center_half(phys._stage, f"{phys._env_path(0)}/{rel}") for rel in body_paths
        ],
        "joint_names": list(phys.joint_names),
        # Background DOFs are eval state, separate from robot action DOFs.
        "scene_joint_paths": list(phys.background_articulations.joint_paths),
        # NOT the leaf: every movable object's path ends in ".../entity", so a
        # leaf-name schema reports the SAME name for all of them ("entity" x6
        # on pick_block_color) and nothing downstream can tell them apart --
        # the success checkers look objects up BY NAME. Use the asset name one
        # level up, which is also the exact key the authored instance-pose
        # table is written with (see FusedPhysics._short).
        "object_names": [FusedPhysics._short(p) for p in phys.object_rel_paths],
        "cloth_meshes": [],
        # Static scene props (no rigid body -- see static_prop_world_aabbs):
        # ``static_props`` is the names that actually resolved (a subset of
        # the allowlist above, checker dialect e.g. "/World/storage_bin");
        # ``static_prop_world_aabbs`` is the paired WORLD-frame box per name,
        # keyed by that same string so the shim can look it up with exactly
        # what the checker asked for. Distinct key from the manifest's
        # ``static_prop_aabbs`` (that one is the input allowlist, a plain
        # list[str]) -- this is the computed output, a dict of boxes.
        "static_props": _prop_names,
        "static_prop_world_aabbs": _prop_aabbs,
    }


def main() -> int:
    manifest = common.Manifest.load(_ARGS.manifest)
    # The raw dict too: common.Manifest is the render node's view and drops
    # init_joint_pos / robot_init_pose, which this node genuinely needs.
    manifest_raw = json.loads(Path(_ARGS.manifest).read_text())
    print(f"[fused] scene: {manifest.scene_usda}", flush=True)
    n_envs = max(1, int(_ARGS.num_envs))

    # 1. Stage WITH physics alive. This is the one flag that separates this
    #    node from the render-only one: physics=True skips the
    #    _disable_physics_subtree call and clones with replicate_physics.
    stage, cam_paths, env_origins = common.build_gridcloner_stage(
        manifest, simulation_app, n_envs, float(_ARGS.env_spacing), physics=True
    )
    if not cam_paths:
        print("[fused] ERROR: no obs camera prims resolved", flush=True)
        return 2

    # Process the final per-env backgrounds, before PhysX parses the scene.
    from _fused_point_instancers import PointInstancerRigids

    instanced_rigids = PointInstancerRigids(stage)

    # Reject stale/malformed reset artifacts before starting PhysX.
    phys = FusedPhysics(stage, manifest.robot_prefix, n_envs, _ARGS.device)
    phys.discover()
    phys.load_instance_table(_ARGS.manifest)

    # Newton's ModelBuilder always adds its z=0 support plane. Fused must
    # supply the same support for foreground-only tasks (including sprite's
    # dynamic table_003); without it the entire table falls out of the scene.
    from isaacsim.core.api.objects import GroundPlane

    GroundPlane(prim_path="/World/GroundPlane", z_position=0.0, visible=False)

    # 2. Physics context, BEFORE the cameras. Created after the stage exists so
    #    it binds to the PhysicsScene the stage build authored.
    from isaacsim.core.api import SimulationContext

    physics_dt = 1.0 / max(float(_ARGS.physics_hz), 1e-3)
    sim = SimulationContext(
        physics_dt=physics_dt, rendering_dt=physics_dt, stage_units_in_meters=None, device=_ARGS.device
    )
    sim.reset()  # starts PhysX; the tensor views below need this first
    # Capture background mechanism state before Kit camera updates can advance
    # the timeline. An opened/fallen startup pose is not a reset target.
    phys.initialize()
    instanced_rigids.initialize()
    # Before the first step, or the parked superset slots fall.
    phys.apply_parking(list(range(n_envs)))

    # 3. Tiled camera sensors -- same grouping-by-height scheme the render-only
    #    node uses (a TiledCameraSensor needs a uniform tile size, and the
    #    benchmark's cameras differ: head 640x400, wrists 640x528).
    #
    # ORDER MATTERS, and not in the way the render-only node suggests. There,
    # sensors are built straight after the stage because there IS no
    # SimulationContext. Here, building them first produces correctly-SHAPED
    # but entirely ZERO tiles: ``SimulationContext.reset()`` restarts
    # the timeline underneath render products created before it, leaving
    # annotators attached to something that is no longer being drawn. Creating
    # them after ``sim.reset()`` binds them to the stage that is actually
    # running. An all-zero (or all-constant) image is the signature of this
    # class of bug and is easy to mistake for a camera-pose or lighting problem.
    obs = manifest.obs_cameras()
    sensors, n_cams, W, Hmax = common.build_tiled_sensor_group(stage, cam_paths, obs, n_envs)
    n_tiles = n_envs * n_cams
    # Let Kit process the newly-created render products before anything asks
    # them for data; the first orchestrator step otherwise races their setup.
    for _ in range(5):
        simulation_app.update()
    _log_effective_physics(stage)

    phys.apply_drive_gains(_ARGS.physics_params_file)
    phys.apply_joint_pd()
    # Place the robot as the task authored it BEFORE anything reads state or
    # renders -- see apply_initial_pose for what an un-posed robot looks like.
    phys.apply_initial_pose(manifest_raw, env_origins)
    phys.apply_reset_materials(list(range(n_envs)), [-1] * n_envs)

    # Render-side generalization (lights, material, camera pose): the same
    # apply code and cmd/ack files as the tiled node. Baselines are read once
    # per stage so every command perturbs around the authored values instead
    # of compounding on the previous reset's. The bridge writes a command
    # before a RESET; it is applied inside that reset, before its frame.
    _light_baseline: dict = _read_light_baseline(stage)
    _camera_pose_baseline: dict = _read_camera_pose_baseline(stage, cam_paths, obs, n_cams, n_envs)
    _manifest_dir = Path(_ARGS.manifest).resolve().parent
    last_randomize_generation = peek_randomize_generation(_manifest_dir)

    # Physics ticks per control step -- the lockstep ratio. Same definition the
    # newton path uses (physics_hz / control_hz), so a control step advances
    # the same amount of sim time on either backend.
    ticks = max(1, int(round(float(_ARGS.physics_hz) / max(float(_ARGS.control_hz), 1e-3))))
    print(
        f"[fused] lockstep: {ticks} physics tick(s) @ {_ARGS.physics_hz}Hz " f"per control step @ {_ARGS.control_hz}Hz",
        flush=True,
    )

    # Which background props to publish world AABBs for -- an explicit
    # per-scene allowlist task_loader carried into the manifest under
    # ``static_prop_aabbs`` (see resolve_benchmark_task/build_stage_manifest);
    # empty for every task whose checkers don't name a static prop.
    static_prop_allowlist = manifest_raw.get("static_prop_aabbs") or []
    schema = build_schema(phys, static_prop_allowlist)
    n_joints = len(schema["joint_names"])
    n_bodies = len(schema["body_paths"])
    n_objects = len(schema["object_names"])
    n_scene_joints = len(schema["scene_joint_paths"])
    if _ARGS.schema_out:
        Path(_ARGS.schema_out).write_text(json.dumps(schema, indent=2))
        print(f"[fused] schema -> {_ARGS.schema_out}", flush=True)

    # 4. SHM + semaphores. WE create the state segment: its size depends on the
    #    articulation's DOF count and on the bodies found by walking the stage,
    #    neither of which existed before the stage was built a few lines above.
    #    The client cannot size it in advance, so it attaches afterwards, using
    #    the schema file written just above to derive the same dimensions --
    #    see FusedStateShm's class docstring for the full startup order.
    state = FusedStateShm(
        _ARGS.state_shm,
        num_envs=n_envs,
        action_dim=n_joints,
        num_joints=n_joints,
        num_bodies=n_bodies,
        num_objects=n_objects,
        num_scene_joints=n_scene_joints,
        create=True,
    )
    shm = TiledFrameShm(_ARGS.shm_name, num_envs=n_tiles, height=Hmax, width=W, create=True)
    sems = SemPair(_ARGS.req_sem, _ARGS.resp_sem, create=False)
    print(
        f"[fused] tiled-SHM '{shm.name}' ({n_tiles},{Hmax},{W},3); "
        f"state-SHM '{state.name}' J={n_joints} B={n_bodies} O={n_objects}",
        flush=True,
    )

    frame_id = 0
    # Counts renders only so the diagnostic above fires on the first few and
    # then goes quiet -- an every-frame print would bury the run's real output.
    _render_diag = {"n": 0}
    #: Control steps taken since a frame was last actually rendered. 1 means
    #: "the previous step rendered", which is the every-step case and costs one
    #: subframe. On-demand rendering makes this climb while images are being
    #: declined, and the next real render has to make up all that motion at
    #: once -- exactly the case common.subframes_for_gap exists for.
    _steps_since_render = {"n": 1}
    # Performance breakdown accumulator. engine_ms is a black box on the
    # main-process side (physics+render are bundled into one CMD_STEP
    # round trip); here it is split into four segments, printing one
    # [fused-prof] line every 200 steps.
    import time as _time  # noqa: PLC0415

    _fprof = {"act": 0.0, "phys": 0.0, "readback": 0.0, "render": 0.0, "n": 0, "render_n": 0, "ticks": 0}

    def _readback_and_render(subframes: int, want_image: bool = True) -> None:
        """Fill every state region, then render. Order matters only in that
        both must complete before ``publish_ok`` -- the client reads them the
        instant the response semaphore is posted.

        ``want_image=False`` fills the state and returns without rendering, for
        a STEP whose caller said it will not read the frame. State readback is
        ~1 ms while the render is 42% (1 env) to 47% (4 envs) of the whole
        CMD_STEP on a 4090 with default cameras, so this is the difference
        between skipping a copy and skipping the work.

        No new frame is published in that case, deliberately: ``frame_id`` is
        the client's freshness watermark, and bumping it for a frame nobody
        rendered would make a stale buffer look new."""
        nonlocal frame_id
        _pt0 = _time.perf_counter()
        phys.read_joint_states(state.joint_states)
        phys.background_articulations.read_joint_positions(state.scene_joint_positions)
        phys.read_body_poses(state.body_poses)
        phys.read_object_poses(state.object_poses)
        state.sim_time[:] = sim.current_time
        _fprof["readback"] += _time.perf_counter() - _pt0
        if not want_image:
            return
        _pt1 = _time.perf_counter()

        # Drive the renderer through the SimulationContext as well as the
        # replicator orchestrator. The render-only node needs only the latter,
        # but it has no SimulationContext at all; once one exists it owns the
        # render pipeline's cadence, and orchestrator.step() alone produces
        # correctly-shaped all-ZERO tiles.
        try:
            sim.render()
        except Exception as exc:  # noqa: BLE001
            print(f"[fused] sim.render() failed: {exc!r}", flush=True)
        _render_once(subframes)
        _diag = _render_diag["n"] < 3
        for s, flat_idx, gh in sensors:
            data, _info = s.get_data("rgb")
            if _diag:
                try:
                    _a = _np(data) if data is not None else None
                    print(
                        f"[fused][render-diag {_render_diag['n']}] "
                        f"type={type(data).__name__} "
                        f"shape={None if _a is None else _a.shape} "
                        f"dtype={None if _a is None else _a.dtype} "
                        f"min={None if _a is None else _a.min()} "
                        f"max={None if _a is None else _a.max()} "
                        f"info_keys={list(_info)[:6] if _info else None}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"[fused][render-diag] readback failed: {exc!r}", flush=True)
            if data is None:
                continue
            arr = (data.numpy() if hasattr(data, "numpy") else np.asarray(data))[..., :3]
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)
            if int(arr.shape[0]) != len(flat_idx):
                continue
            for k, gi in enumerate(flat_idx):
                shm.rgb[gi, :gh] = arr[k]
        _render_diag["n"] += 1
        frame_id += 1
        # ``rendered_tick`` is the render-only node's frame-sync watermark
        # against an ASYNC pose producer. Here the render happened inside this
        # very call, after this step's physics -- the frame cannot be stale by
        # construction, so the frame counter itself is the watermark.
        shm.publish(frame_id, rendered_tick=frame_id)
        _fprof["render"] += _time.perf_counter() - _pt1
        _fprof["render_n"] += 1

    # First frame: settle + render once so the client's very first
    # observation is a real picture of the built stage rather than a black
    # buffer. Extra subframes because there is no temporal history to reuse yet
    # -- the same reason the render-only node bumps subframes after a reset,
    # which is why this asks for the same rule rather than repeating a number.
    for _ in range(2):
        sim.step(render=False)
    if os.environ.get("RLINF_FUSED_RENDER_SWEEP", "").strip() == "1":
        _render_strategy_sweep(sensors, sim)
    _readback_and_render(common.subframes_for_gap(None, after_reset=True))
    if _ARGS.ready_file:
        Path(_ARGS.ready_file).write_text(str(os.getpid()))
    print("[fused] ready", flush=True)

    # 5. Lockstep main loop.
    while simulation_app.is_running():
        try:
            sems.wait_request(timeout=1.0)
        except Exception:  # noqa: BLE001 -- timeout is the normal idle path
            # Pump the UI through render-only update. A bare app.update()
            # advances the playing timeline and violates lockstep while idle.
            sim.render()
            continue

        cmd, mask_valid = state.read_command()
        try:
            if cmd == CMD_SHUTDOWN:
                sems.post_response()
                break
            if cmd == CMD_STEP:
                _st0 = _time.perf_counter()
                phys.apply_actions(state.actions)
                _st1 = _time.perf_counter()
                for _ in range(ticks):
                    sim.step(render=False)
                _st2 = _time.perf_counter()
                _fprof["act"] += _st1 - _st0
                _fprof["phys"] += _st2 - _st1
                _fprof["n"] += 1
                _fprof["ticks"] = ticks
                # RESET below always renders: its caller is handed an
                # observation. Only STEP has a caller that can say it will not
                # look at the frame.
                #
                # Subframes scale with how many steps were skipped, not a
                # constant 1. RTX accumulates temporally, so the frame that
                # ends a run of declined images has no history covering that
                # motion and comes out soft -- the render-only node applies
                # the same scaling rule for on-demand rendering.
                _want = state.want_image()
                _readback_and_render(common.subframes_for_gap(_steps_since_render["n"]), want_image=_want)
                _steps_since_render["n"] = 1 if _want else _steps_since_render["n"] + 1
            elif cmd == CMD_RESET:
                env_ids = [i for i in range(n_envs) if state.reset_mask[i]] if mask_valid else list(range(n_envs))
                # Instance switch FIRST, then the joint reset: the switch
                # teleports objects, the joint reset puts the arm back, and
                # the settle afterwards lets both come to rest together. Doing
                # it the other way round settles the arm against the OLD
                # instance's objects and then yanks the objects out from under
                # it.
                wanted = [int(state.reset_instances[e]) for e in env_ids]
                n_w = phys.apply_instance(env_ids, wanted, env_origins)
                print(f"[fused] instance reset {dict(zip(env_ids, wanted))}" f" -> {n_w} object pose(s)", flush=True)
                phys.apply_reset_materials(env_ids, wanted)
                instanced_rigids.reset(env_ids, phys.to_backend)
                # Seed >= 0: the physics-side generalization on top of the
                # instance's authored state (object xy, robot base now; arm
                # joints through _do_reset). -1 is the verbatim snapshot.
                seeds = [int(state.reset_seeds[e]) for e in env_ids]
                joint_offsets = phys.randomize_reset(env_ids, seeds, env_origins)
                _do_reset(sim, phys, env_ids, joint_offsets)
                # Render-side generalization for this reset, if the bridge sent
                # one: after the instance materials (a randomized binding wins
                # on the same mesh, as on tiled) and before the reset frame.
                _rcmd = read_randomize_cmd_if_new(_manifest_dir, last_randomize_generation)
                if _rcmd is not None:
                    last_randomize_generation = _rcmd["generation"]
                    _apply_randomize_cmd(
                        stage,
                        _light_baseline,
                        _camera_pose_baseline,
                        _rcmd["render"],
                        _manifest_dir,
                        _rcmd["generation"],
                    )
                # after_reset, not the gap: a reset teleports the robot and
                # every object within one tick, so the gap would read as small
                # while nothing in the previous frame is still where it was.
                # (This also stops the 16 here from being a second, silently
                # drifting copy of RT_SUBFRAMES_RESET.)
                _readback_and_render(common.subframes_for_gap(None, after_reset=True))
                _steps_since_render["n"] = 1
            else:
                raise RuntimeError(f"unknown command {cmd}")
            state.publish_ok()
            # Rolling performance breakdown: prints one line to
            # stdout(sim.log) every 200 CMD_STEPs.
            # The sum of these four segments should come close to the
            # wait_ms the main process measures; the gap is IPC sync
            # overhead.
            if _fprof["n"] >= 200:
                _n = _fprof["n"]
                _rn = max(1, _fprof["render_n"])
                _total = _fprof["act"] + _fprof["phys"] + _fprof["readback"] + _fprof["render"]
                print(
                    "[fused-prof] {n} step(s): act={act:.2f} "
                    "phys={phys:.2f}(x{ticks}tick) readback={rb:.2f} "
                    "render={rd:.2f}ms/frame | rendered {rn}/{n} "
                    "({pct:.0f}%) | sum={total:.2f}ms/step".format(
                        n=_n,
                        act=_fprof["act"] / _n * 1000,
                        phys=_fprof["phys"] / _n * 1000,
                        ticks=_fprof["ticks"],
                        rb=_fprof["readback"] / _n * 1000,
                        rd=_fprof["render"] / _rn * 1000,
                        rn=_fprof["render_n"],
                        pct=_fprof["render_n"] / _n * 100,
                        total=_total / _n * 1000,
                    ),
                    flush=True,
                )
                for _k in _fprof:
                    _fprof[_k] = 0 if isinstance(_fprof[_k], int) else 0.0
        except Exception as exc:  # noqa: BLE001
            # Never die on a bad command: report it and stay in the loop. The
            # client re-raises this text, so a failure is diagnosable from the
            # trainer side without reading this node's log -- and the process
            # staying up means the next command still has somewhere to land.
            traceback.print_exc()
            state.publish_error(f"{type(exc).__name__}: {exc}")
        sems.post_response()

    print("[fused] shutting down", flush=True)
    try:
        # Both segments are OURS to unlink (we created them); the semaphores
        # are the client's, so we only detach from those. Unlinking a peer's
        # named object would pull it out from under a client that is still
        # running -- SemPair encodes the same creator-only rule.
        state.close(unlink=True)
        shm.close(unlink=True)
        sems.close(unlink=False)
    except Exception:  # noqa: BLE001
        pass
    simulation_app.close()
    return 0


def _do_reset(sim, phys: "FusedPhysics", env_ids: List[int], joint_offsets: Optional[dict] = None) -> None:
    """Reset the named envs to the articulation's default joint state.

    ``joint_offsets`` (``{env: (num_dofs,) array}``, from
    ``FusedPhysics.randomize_reset``) is added to that state for the envs it
    names -- the seeded arm joint generalization, applied to both the joint
    positions and the drive targets, as in Newton.

    Isaac's batched setters take an ``indices`` argument, so a subset reset is
    one call and never disturbs the envs left out of it -- which matters,
    because those are mid-episode and must not be perturbed by a neighbour
    resetting.

    Everything handed to those setters goes through ``phys.to_backend``: with
    a CUDA device the views run Isaac's torch backend and call ``.to(...)`` on
    both the values AND the indices, so a numpy array raises. This is
    invisible until a reset actually runs -- which, in a training loop, is
    after the expensive part has already started.
    """
    if not env_ids:
        return
    if phys.background_articulations is not None:
        phys.background_articulations.reset(env_ids, phys.to_backend)
    art = phys.articulation
    n_env, n_j = len(env_ids), len(phys.joint_names)
    idx = phys.to_backend(np.asarray(env_ids, dtype=np.int64), long=True)

    # Default joint positions. ``get_default_state()`` is the authored rest
    # pose; if this Isaac build does not expose it (or the stage never set
    # one), fall back to HOLDING the current positions rather than snapping to
    # zeros -- zeros is a valid pose for almost no robot and would slam the arm
    # into the table on the first reset.
    target = None
    if getattr(phys, "default_joint_pos", None) is not None:
        # The task's authored start pose, captured at startup. Preferred over
        # get_default_state(): the USD default is the zero pose, which is not
        # where this task begins.
        target = np.asarray(phys.default_joint_pos)[np.asarray(env_ids)]
    try:
        default = art.get_default_state() if target is None else None
        dpos = getattr(default, "joint_positions", None) if default is not None else None
        if dpos is not None:
            dpos = _np(dpos)
            target = dpos[np.asarray(env_ids)] if dpos.ndim == 2 else np.broadcast_to(dpos, (n_env, n_j))
    except Exception as exc:  # noqa: BLE001
        print(f"[fused] default-state unavailable ({exc!r}); holding current " f"joint positions on reset", flush=True)
    if target is None:
        cur = _np(art.get_joint_positions()).reshape(phys._num_envs, -1)
        target = cur[np.asarray(env_ids)]
    target = np.ascontiguousarray(target, dtype=np.float32)
    if joint_offsets:
        target = target.copy()
        for k, e in enumerate(env_ids):
            if e in joint_offsets:
                target[k] += joint_offsets[e]

    art.set_joint_positions(phys.to_backend(target), indices=idx)
    art.set_joint_velocities(phys.to_backend(np.zeros((n_env, n_j), dtype=np.float32)), indices=idx)
    # Re-assert the position targets too. Leaving the previous episode's
    # targets in place makes the controller immediately yank the freshly-reset
    # arm back toward where it just was, which reads as "the reset did not
    # take" rather than as a stale setpoint.
    art.set_joint_position_targets(phys.to_backend(target), indices=idx)

    # Settle, matching ``GenieSimHandler``'s own reset contract
    # (``reset_settle_time_s``, default 1.0s): the caller should get an
    # observation in which dropped objects have already come to rest, not one
    # mid-fall.
    settle = max(0.0, float(_ARGS.reset_settle_time_s))
    for _ in range(int(round(settle * float(_ARGS.physics_hz)))):
        sim.step(render=False)

    # Joint and root-pose writes do not move the links on GPU PhysX until a
    # kinematic update; a step does one, and so does ``sim.render()``, but the
    # state readback runs before the render. With no settle (the rest-state
    # bake) nothing else refreshes them, and the reset response carried the
    # previous episode's link poses. This advances no time.
    view = getattr(sim, "physics_sim_view", None) if sim is not None else None
    if view is not None:
        view.update_articulations_kinematic()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        try:
            simulation_app.close()
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
