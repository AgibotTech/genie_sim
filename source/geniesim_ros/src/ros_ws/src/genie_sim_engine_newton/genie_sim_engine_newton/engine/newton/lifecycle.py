# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Newton-standalone episode lifecycle mixin.

Provides ``_LifecycleMixin`` composed into ``_NewtonStandaloneBase`` (see
``engine_base.py``). Implements ``episode_reset(env_ids, seed, instances)``
-- the per-world reset contract, restoring selected worlds to the build-time
init state without rebuilding the model.

The ROS serving layer lives in ``genie_sim_engine_core.common.lifecycle_ros``
(``RosResetCoordinator``), which handles the ``/reset_envs`` subscriber,
thread-safe queue, reset_epoch credential, and activity stamping. This mixin
is the Newton-specific implementation that actually rewrites state.

How it works
------------

At build (end of ``_phase_finalize_init_state``) we snapshot:
  * ``model.joint_q``            — the FULL per-world init configuration
    (robot joints AT init pose + free-body objects at their spawn pose).
    ``model.joint_q`` is the template and is NOT mutated by stepping, so
    it stays a stable init reference for the engine's lifetime.
  * ``control.joint_target_q`` — the init PD targets (copied, so later
    ``apply_commands`` writes don't disturb the snapshot).
  * ``model.particle_q`` — initial particle positions for single-world scenes.
    The first finite settled snapshot updates particle and joint baselines
    together. Particle reset restores both existing buffers in place and
    clears particle velocities and forces before committing the reset epoch.

``episode_reset(env_ids)`` rewrites, for each selected world's slice (world-major
from ``replicate``):
  * ``state_0/state_1.joint_q`` ← init joint_q slice
  * ``state_0/state_1.joint_qd`` ← 0   (zero velocity)
  * ``control.joint_target_q`` ← init target slice (else PD yanks the
    just-reset joints back toward the last command)
then re-runs ``eval_fk`` so body poses refresh for the read between reset
and the next step. Task scenes apply the selected/current instance's baked
body and robot poses before seeded randomization and ``eval_fk``.

How the rewritten state reaches physics on the mjwarp path: the adapter
builds ``SolverMuJoCo(update_data_interval=0)`` (the historical
interval=1 per-substep re-sync pumped energy into resting contacts —
see the adapter's ``build_solver``), so after the state writes this
mixin calls ``adapter.sync_state_into_solver(state_0)`` for a one-shot
push of joint_q/joint_qd into ``mjw_data.qpos/qvel``.  ``.assign()``
mutates the buffers in place so the captured CUDA graph's pointers stay
valid.

Trigger path: ``RosResetCoordinator`` (engine_core) owns the always-on
rclpy subscriber on ``/reset_envs`` (``std_msgs/Int32MultiArray`` of world
indices; empty = all worlds) and the thread-safe pending list; its
``drain_resets`` -- called from ``apply_commands``, on the loop thread
between frames -- calls ``episode_reset`` below.  This mixin used to carry
a second copy of that queue; it was dead from the day the coordinator
landed and is gone.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def _body_name_matches(label: str, name: str) -> bool:
    """Does the FREE-joint body ``label`` answer to the requested ``name``?

    Three accepted forms, because three callers already speak three dialects:

    * the full Newton body label, as ``/tf_scene`` publishes it
    * the leaf-PARENT segment -- ``"mug"`` for ``"/World/Objects/mug/base_link"``
      -- the historical object shorthand, so a teleport can name a body the
      same way the task files do
    * any trailing path segment, so ``"base_link"`` or ``"mug/base_link"``
      resolve without the caller knowing the scene root

    Deliberately NOT a substring test: ``"cup"`` must not match ``"cupboard"``.
    """
    if label == name:
        return True
    short = label.rsplit("/", 2)[-2] if "/" in label else label
    return short == name or label.endswith("/" + name)


def _finite(arr) -> bool:
    """Is every value in ``arr`` finite? Returns True when it cannot tell.

    Not blocking on "cannot tell" is deliberate: within this module the
    function is only used in GUARD positions, and a false positive turns a
    normal reset into a "fall back to the build-time snapshot", which costs
    more than a false negative. Anything that slips through is caught
    downstream (sim_handler's outbound guard, vec_env's obs guard).
    """
    try:
        return bool(np.isfinite(np.asarray(arr)).all())
    except Exception:  # noqa: BLE001 — a guard must not create new failures
        return True


# ---------------------------------------------------------------------------
# Joint PD gain randomization: pure helpers, no ``self`` / Newton / warp
# dependency, so they are unit-testable with plain fakes.
# ``_LifecycleMixin`` methods below are thin glue around these two --
# reading scene_cfg / model arrays, logging, and raising -- the actual
# numeric logic lives here where a test can drive it directly.
# ---------------------------------------------------------------------------


def _compute_joint_pd_baseline(ke_arr, kd_arr, jindex):
    """Snapshot ``joint_target_ke``/``kd`` + the arm/body/head eligibility
    mask, once, at build time (see ``_snapshot_reset_state``'s own
    docstring for WHY it must be exactly here and WHY the mask excludes
    gripper/chassis). Returns ``None`` when there is nothing to cache
    (missing arrays / empty jindex); otherwise ``(ke_snapshot, kd_snapshot,
    eligible_mask)``, all ``np.ndarray``.
    """
    if ke_arr is None or kd_arr is None or jindex is None or not len(jindex):
        return None
    from genie_sim_engine_newton.common.joint_classification import (  # noqa: PLC0415
        JK_ARM,
        JK_ARM_MID,
        JK_ARM_SHOULDER,
        JK_ARM_WRIST,
        JK_BODY,
        JK_HEAD,
        classify_joint_by_name,
    )

    eligible_kinds = {JK_BODY, JK_ARM, JK_ARM_SHOULDER, JK_ARM_MID, JK_ARM_WRIST, JK_HEAD}
    ke_snapshot = ke_arr.numpy().copy()
    kd_snapshot = kd_arr.numpy().copy()
    mask = np.zeros(len(ke_snapshot), dtype=bool)
    for slc in jindex.slices():
        if not slc.is_actuated:
            continue
        if classify_joint_by_name(slc.name) not in eligible_kinds:
            continue
        for k in range(slc.dof_count):
            idx = slc.qd_start + k
            if idx < len(mask):
                mask[idx] = True
    return ke_snapshot, kd_snapshot, mask


def _scale_joint_pd(jpd_map, env_ids, baseline_ke, baseline_kd, eligible, qd_per):
    """Pure numeric core of ``_apply_joint_pd_randomization`` -- no I/O, no
    ``self``, no raising. ``jpd_map`` is ``scene_cfg["randomization"]
    ["joint_pd"]`` (``{"<world>": {"kp_scale": .., "kd_scale": ..}}``,
    already resolved by ``Sampler.sample()`` at task-build time -- this
    function draws nothing).

    Returns ``(ke_or_None, kd_or_None, requested, applied, touched)``:
    ``ke``/``kd`` are ``None`` when the baseline/mask/``qd_per`` aren't
    usable at all (caller decides what that means -- see
    ``_apply_joint_pd_randomization``'s raise/warn policy); otherwise they
    are full-length arrays with every requested world's slice rescaled
    (RELATIVE TO ``baseline_ke``/``baseline_kd``, never to a running value
    -- this is what makes repeated calls not compound). ``requested`` counts
    worlds (in ``env_ids``) that have a ``joint_pd`` entry at all;
    ``applied`` counts worlds that got at least one DOF rescaled OR whose
    entry was exactly identity (``kp_scale == kd_scale == 1.0`` is honoured
    by construction, without a write); ``touched`` lists the worlds whose
    ke/kd were actually rewritten (excludes identity worlds, since nothing
    changed for them).
    """
    requested_entries = []
    for w in env_ids:
        entry = jpd_map.get(str(w))
        if entry:
            requested_entries.append((w, entry))
    if not requested_entries:
        return None, None, 0, 0, []
    requested = len(requested_entries)
    applied = 0
    touched: list = []
    ke = kd = None
    have_baseline = baseline_ke is not None and baseline_kd is not None and eligible is not None and bool(qd_per)
    for w, entry in requested_entries:
        kp_scale = float(entry.get("kp_scale", 1.0))
        kd_scale = float(entry.get("kd_scale", 1.0))
        if kp_scale == 1.0 and kd_scale == 1.0:
            # Identity IS the requested transform, honoured by construction
            # (baseline * 1.0 == baseline byte-for-byte) regardless of
            # whether a baseline/mask happens to be cached -- skipping the
            # write is correct even when the engine has no way to do a REAL
            # scale, so this counts as applied, not missing, unconditionally.
            applied += 1
            continue
        if not have_baseline:
            continue  # a real scale was requested but there is nothing to
            # scale from -- left un-applied; the caller's
            # requested/applied comparison surfaces this loudly.
        if ke is None:
            # Lazily copy the baseline on the FIRST world that actually
            # needs a real (non-identity) scale -- an all-identity config
            # (or an all-unusable one) never touches these arrays at all.
            ke = baseline_ke.copy()
            kd = baseline_kd.copy()
        lo, hi = w * qd_per, (w + 1) * qd_per
        if hi > len(ke) or hi > len(baseline_ke) or hi > len(eligible):
            continue
        world_mask = eligible[lo:hi]
        if not world_mask.any():
            continue
        ke[lo:hi] = np.where(world_mask, baseline_ke[lo:hi] * kp_scale, ke[lo:hi])
        kd[lo:hi] = np.where(world_mask, baseline_kd[lo:hi] * kd_scale, kd[lo:hi])
        touched.append(w)
        applied += 1
    return ke, kd, requested, applied, touched


class _LifecycleMixin:
    # ------------------------------------------------------------------
    # Build-time snapshot (called from _phase_finalize_init_state)
    # ------------------------------------------------------------------
    def _snapshot_reset_state(self) -> None:
        self._reset_init_joint_q = None
        self._reset_init_target = None
        self._reset_init_particle_q = None
        # reset_epoch causal credential: incremented by ``episode_reset()`` below only
        # after a whole reset batch has been fully applied. The main loop writes
        # the current value into the header.frame_id of every /joint_states, so
        # once a caller sees the epoch pass its pre-reset baseline, the joint
        # state that comes with it is guaranteed to be post-reset.
        # Monotonic, process-local, starting at 0.
        self._reset_epoch: int = 0
        try:
            if self._model is not None and self._model.joint_q is not None:
                self._reset_init_joint_q = self._model.joint_q.numpy().copy()
            particles = getattr(self._model, "particle_q", None)
            if particles is not None:
                self._reset_init_particle_q = particles.numpy().copy()
            ctl = self._control
            if ctl is not None and getattr(ctl, "joint_target_q", None) is not None:
                self._reset_init_target = ctl.joint_target_q.numpy().copy()
            # This build-time copy is kept SEPARATELY and never overwritten.
            # It is the only reference in the whole flow that is clean by
            # definition: the model has just been built and not a single physics
            # step has run, so it cannot contain NaN.
            # After resnapshot_settled, _reset_init_joint_q is replaced by the
            # "settled" state, which is taken from live physics and could in
            # principle already have diverged; when that happens, reset() falls
            # back on this copy (see its own guard).
            self._reset_build_joint_q = None if self._reset_init_joint_q is None else self._reset_init_joint_q.copy()
            self._logger.info(
                "[newton-standalone] reset snapshot: joint_q size="
                f"{0 if self._reset_init_joint_q is None else len(self._reset_init_joint_q)}, "
                f"target size={0 if self._reset_init_target is None else len(self._reset_init_target)}, "
                f"num_envs={int(getattr(self, '_num_envs', 1) or 1)}"
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] _snapshot_reset_state failed: {exc}")

        # ------------------------------------------------------------------
        # Joint PD gain randomization: the ONE authorized engine-side
        # change for that dimension (see rlinf_geniesim/randomize/spec.py
        # + sampler.py module docstrings). Cache a read-only snapshot of
        # joint_target_ke/kd right here: prepare_model has already
        # authored the per-class gains, and no reset has run yet -- this
        # is the earliest and only point in the whole lifecycle where
        # ke/kd are guaranteed to be the authored-but-never-scaled
        # baseline. episode_reset always scales THIS snapshot, never the
        # live array: scaling the live value instead would let repeated
        # resets compound the same multiplier again and again (the same
        # failure mode camera_pose guards against with its
        # "..._repeated_resets_do_not_compound" test).
        #
        # Also cache which DOFs joint_pd is even allowed to touch: arm / body
        # / head only. Gripper gains are excluded on purpose -- they are the
        # difference between holding and dropping a grasp, and this repo has
        # a dedicated static-grasp regression rig tuning them; chassis gains
        # are excluded because the 4-wheel-steer base has its own tuning.
        # Folding either into "generalization" would turn it into "randomly
        # sabotage the grasp/drive", and the failure would look like a worse
        # return, never an error -- exactly the failure shape this repo is
        # least tolerant of. Classification is name-only (no scene-
        # ownership / mimic / chassis drive-vs-steer refinement --
        # prepare_model needs those to build actuators correctly, this does
        # not): the only way name-only classification can go wrong here is
        # mis-INCLUDING a non-robot joint, and every such joint already has
        # ke=kd=0 from prepare_model (scene/passive joints never get a
        # non-zero PD gain), so 0 * scale is still 0 -- harmless.
        self._pd_ke_build_baseline = None
        self._pd_kd_build_baseline = None
        self._joint_pd_eligible_mask = None
        try:
            ke_arr = getattr(self._model, "joint_target_ke", None) if self._model is not None else None
            kd_arr = getattr(self._model, "joint_target_kd", None) if self._model is not None else None
            jindex = getattr(self, "_jindex", None)
            baseline = _compute_joint_pd_baseline(ke_arr, kd_arr, jindex)
            if baseline is not None:
                ke_snapshot, kd_snapshot, mask = baseline
                self._pd_ke_build_baseline = ke_snapshot
                self._pd_kd_build_baseline = kd_snapshot
                self._joint_pd_eligible_mask = mask
                self._logger.info(
                    "[newton-standalone] joint_pd baseline cached: "
                    f"{int(mask.sum())}/{len(mask)} dof(s) eligible "
                    "(arm/body/head only -- gripper + chassis excluded)"
                )
        except Exception as exc:  # noqa: BLE001 -- caching must never block the build
            self._logger.warn(f"[newton-standalone] joint_pd baseline caching failed: {exc}")

        # Reject old locator-only artifacts before serving any reset. Body
        # poses are baked per instance from composed USD, never reconstructed
        # from a build-instance delta or a settled/live physics snapshot.
        self._validate_instance_body_poses()
        self._active_instances = {}

    # ------------------------------------------------------------------
    # Composed USD body poses -> Newton free-joint coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def _qmul(a, b):
        """Quaternion product; in and out are (w, x, y, z). ``a * b`` means
        rotate by b first, then by a."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )

    @staticmethod
    def _instance_pose_error(detail):
        return ValueError(
            f"[reset-instance] invalid instance body poses: {detail}; "
            "re-bake the task scene with body_pose_version=1"
        )

    @staticmethod
    def _instance_body_key(label):
        """Exact USD body path below Objects, including every asset layer."""
        prefix = "/World/Objects/"
        return label.partition(prefix)[2] if prefix in label else None

    def _validate_instance_body_poses(self):
        """Validate the complete baked table, including non-free body rows.

        Extra fixed/kinematic/articulated bodies are valid: only FREE object
        bodies have independent coordinates for this reset path to write.
        Every such free body must nevertheless appear in every instance.
        """
        sw = (getattr(self, "_scene_cfg", None) or {}).get("instance_switch") or {}
        if not sw:
            return None
        tables = sw.get("instance_body_poses")
        version = sw.get("body_pose_version")
        if type(version) is not int or version != 1 or not isinstance(tables, dict):
            raise self._instance_pose_error(
                "expected body_pose_version=1 and instance_body_poses; "
                "locator-only instance_poses are not sufficient"
            )
        expected = {str(i) for i in sw.get("group_instances", [])}
        expected.update(str(i) for i in (sw.get("instance_poses") or {}))
        if sw.get("build_instance") is not None:
            expected.add(str(sw["build_instance"]))
        missing = expected - tables.keys()
        if missing:
            raise self._instance_pose_error(f"missing instance rows {sorted(missing)}")
        required = set()
        for w in range(int(getattr(self, "_num_envs", 1) or 1)):
            for slc, label in self._world_free_joints(w):
                key = self._instance_body_key(label)
                if key is not None:
                    required.add(key)
                    self._instance_joint_frames(slc, key)
        for inst, table in tables.items():
            if not isinstance(inst, str) or not isinstance(table, dict):
                raise self._instance_pose_error("instance keys must be strings and rows mappings")
            missing = required - table.keys()
            if missing:
                raise self._instance_pose_error(f"instance {inst} missing FREE body poses {sorted(missing)}")
            for key, value in table.items():
                if not isinstance(key, str) or not key or any(part in ("", ".", "..") for part in key.split("/")):
                    raise self._instance_pose_error(f"instance {inst} invalid body path {key!r}")
                self._checked_instance_pose(value, f"instance {inst} body {key}")
        return tables

    def _checked_instance_pose(self, value, label):
        try:
            pose = np.asarray(value, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise self._instance_pose_error(f"{label}: expected numeric xyz+wxyz") from exc
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise self._instance_pose_error(f"{label}: expected 7 finite xyz+wxyz values")
        norm = np.linalg.norm(pose[3:])
        if not np.isclose(norm, 1.0, rtol=0.0, atol=1e-4):
            raise self._instance_pose_error(f"{label}: quaternion must have unit length (got {norm})")
        pose = pose.copy()
        pose[3:] /= norm
        return pose

    def _instance_joint_frames(self, slc, key):
        """USD-imported free bodies use world parent and identity anchors.

        Preserve the general world-parent FK contract even for nonidentity
        anchors: body = X_p * joint_q * inverse(X_c). A parented FREE joint
        is not emitted by Newton's USD floating-body importer; reject it
        instead of treating the parent's moving frame as world space.
        """
        j = slc.joint_index
        if getattr(self, "_instance_frame_model", None) is not self._model:
            arrays = [getattr(self._model, name, None) for name in ("joint_parent", "joint_X_p", "joint_X_c")]
            if any(array is None for array in arrays):
                raise self._instance_pose_error(f"body {key}: missing Newton joint frame metadata")
            # Object FREE-joint anchors are model constants. Copy once, not
            # one device->host transfer per body in each reset. This cache
            # contains no live joint_q/body_q and is invalidated on rebuild.
            self._instance_frame_arrays = tuple(array.numpy().copy() for array in arrays)
            self._instance_frame_model = self._model
        parent, xp, xc = self._instance_frame_arrays
        if int(parent[j]) != -1:
            raise self._instance_pose_error(f"body {key}: FREE joint must have world parent")
        frames = []
        for name, array in (("parent", xp), ("child", xc)):
            # Warp transforms, like Newton free-joint q, store xyzw.
            frame = np.asarray(array[j], dtype=np.float64)
            frames.append(self._checked_instance_pose(frame[[0, 1, 2, 6, 3, 4, 5]], f"body {key} {name} joint frame"))
        return frames

    @classmethod
    def _compose_instance_pose(cls, a, b):
        # p(a*b) = p(a) + R(a)*p(b), quaternion convention wxyz.
        q = np.asarray(a[3:])
        v = np.asarray(b[:3])
        rotated = v + 2 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)
        return np.concatenate((a[:3] + rotated, cls._qmul(q, b[3:])))

    def _body_pose_to_free_joint(self, pose, slc, key):
        xp, xc = self._instance_joint_frames(slc, key)
        inverse_rotation = np.concatenate(([xp[3]], -xp[4:]))
        # Construct inverse X_p, then q = inverse(X_p) * body * X_c.
        inverse_parent = self._compose_instance_pose(
            np.concatenate((np.zeros(3), inverse_rotation)), np.concatenate((-xp[:3], [1, 0, 0, 0]))
        )
        joint_pose = self._compose_instance_pose(self._compose_instance_pose(inverse_parent, pose), xc)
        return joint_pose[[0, 1, 2, 4, 5, 6, 3]]

    def resnapshot_settled(self) -> None:
        """Re-capture the reset reference from the CURRENT state (call once
        when the first client command arrives — by then the scene has settled:
        objects authored a few cm above their supports have landed and damped
        out). Pure engine scenes restart from this settled configuration;
        task resets override object bodies with the baked instance table."""
        try:
            s0 = self._state_0
            if s0 is None or getattr(s0, "joint_q", None) is None:
                return
            jq_now = s0.joint_q.numpy()
            # This is the ONLY place in the whole flow where live physics
            # state overwrites the reset baseline — and therefore the only place
            # that can turn one numerical divergence into permanent breakage:
            # once NaN lands in the baseline, every later reset faithfully
            # "restores" to NaN, unrecoverable short of restarting the server.
            # The settling phase runs physics (objects falling, contact solving),
            # so a single blow-up would reach here; it has to be blocked.
            if not _finite(jq_now):
                self._logger.warn(
                    "[newton-standalone] settled state has non-finite values; "
                    "REFUSING to overwrite the reset baseline with it — doing so "
                    "would make every later reset restore to NaN. Keeping the "
                    "build-time snapshot as the baseline; this episode's physics "
                    "is no longer trustworthy, restarting the task is advised."
                )
                return
            particles = getattr(s0, "particle_q", None)
            points = None
            if particles is not None:
                points = particles.numpy()
                if not _finite(points):
                    raise ValueError("settled particles are non-finite; keeping the build-time particle baseline")
            self._reset_init_joint_q = jq_now.copy()
            if points is not None:
                self._reset_init_particle_q = points.copy()
            ctl = self._control
            if ctl is not None and getattr(ctl, "joint_target_q", None) is not None:
                self._reset_init_target = ctl.joint_target_q.numpy().copy()
            self._logger.info(
                "[newton-standalone] reset reference re-snapshotted from the " "settled state (first client command)"
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] resnapshot_settled failed: {exc}")

    # ------------------------------------------------------------------
    # Shared: free-joint enumeration for world w (objects / robot root)
    # ------------------------------------------------------------------
    def _world_free_joints(self, w: int):
        """Yield ``(slice, child_body_label)`` for world ``w``'s FREE joints.
        World membership via the child-body INDEX (bodies are world-major);
        labels carry no env_ prefix (the ROS publisher adds it)."""
        jindex = getattr(self, "_jindex", None)
        if jindex is None:
            return
        child_arr = getattr(self._model, "joint_child", None)
        child = child_arr.numpy() if child_arr is not None else None
        body_paths = getattr(self, "_body_paths", None) or list(getattr(self._model, "body_label", []) or [])
        ne = int(getattr(self, "_num_envs", 1) or 1)
        nb_per = len(body_paths) // ne if ne else len(body_paths)
        for slc in jindex.slices():
            if slc.joint_type != 4 or child is None:  # 4 = JointType.FREE
                continue
            if slc.joint_index >= len(child):
                continue
            bi = int(child[slc.joint_index])
            if bi >= len(body_paths):
                continue
            if ((bi // nb_per) if nb_per else 0) != w:
                continue
            yield slc, str(body_paths[bi])

    # ------------------------------------------------------------------
    # In-group instance switching (L1b generalization)
    # ------------------------------------------------------------------
    def _apply_instance_poses(self, w: int, inst_id: int, jq, tgt=None, base_mocap_out=None, _body_tables=None) -> None:
        """Apply composed body poses in env-local meters, then robot state.

        ``instance_body_poses`` is already the complete per-instance USD
        body frame. Old locator poses/name aliases are metadata only; adding
        any build-instance asset delta here would apply that transform twice.
        ``tgt`` and ``base_mocap_out`` retain the per-instance robot channels.
        """
        sw = (getattr(self, "_scene_cfg", None) or {}).get("instance_switch") or {}
        tables = _body_tables if _body_tables is not None else self._validate_instance_body_poses()
        if tables is None:
            return
        table = tables.get(str(int(inst_id)))
        if table is None:
            raise self._instance_pose_error(f"missing instance {inst_id}")
        writes = []
        for slc, bname in self._world_free_joints(w):
            key = self._instance_body_key(bname)
            if key is None:
                continue
            pose = self._checked_instance_pose(table[key], f"instance {inst_id} body {key}")
            writes.append((slc, self._body_pose_to_free_joint(pose, slc, key)))
        for slc, pose in writes:
            jq[slc.q_start : slc.q_start + 7] = pose
        if writes:
            self._logger.info(
                f"[reset-instance] world {w} -> instance {inst_id}: " f"{len(writes)} body pose(s) applied"
            )

        # ---- Per-instance ROBOT initial state (joints + base xy) ---------
        # Joints: same write pattern as _randomize_world_objects's arm
        # section below -- jq AND the PD target must be written TOGETHER.
        # Writing jq alone lets the live PD pull the joint back toward the
        # old command within ~0.2-0.5s.
        jtab = (sw.get("robot_joints") or {}).get(str(int(inst_id)))
        jindex = getattr(self, "_jindex", None)
        if jtab and jindex is not None:
            child_arr = getattr(self._model, "joint_child", None)
            child = child_arr.numpy() if child_arr is not None else None
            body_paths = getattr(self, "_body_paths", None) or list(getattr(self._model, "body_label", []) or [])
            ne = int(getattr(self, "_num_envs", 1) or 1)
            nb_per = len(body_paths) // ne if ne else len(body_paths)
            n_j = 0
            for slc in jindex.slices():
                if not slc.is_actuated or slc.name not in jtab:
                    continue
                if child is None or slc.joint_index >= len(child):
                    continue
                if ((int(child[slc.joint_index]) // nb_per) if nb_per else 0) != w:
                    continue
                val = float(jtab[slc.name])
                jq[slc.q_start] = val
                if tgt is not None and slc.qd_start < len(tgt):
                    tgt[slc.qd_start] = val
                n_j += 1
            if n_j:
                self._logger.info(
                    f"[reset-instance] world {w} -> instance {inst_id}: "
                    f"{n_j} robot joint(s) set to the instance's authored pose"
                )
        elif jtab:
            self._logger.warn(
                "[reset-instance] no joint index (_jindex) -- robot joints "
                "skipped; this instance will run on the build instance's "
                "joint angles"
            )

        # Base: goes through the mocap channel (this robot's base is a
        # world-fixed root, emitted by Newton's MJC converter as a native
        # mocap body with no free joint to write). The table stores the
        # offset RELATIVE to the build instance, because
        # apply_base_mocap_offsets adds its offset on top of the build-time
        # snapshot. xy only -- that function never writes z/orientation, and
        # the task loader has already validated that every row in this batch
        # shares the same z/orientation; a mismatch is an error there, not a
        # silent drop here.
        btab = (sw.get("robot_base_xy") or {}).get(str(int(inst_id)))
        if btab is not None and base_mocap_out is not None:
            base_mocap_out[w] = (float(btab[0]), float(btab[1]))
            self._logger.info(
                f"[reset-instance] world {w} -> instance {inst_id}: "
                f"base xy offset ({btab[0]*1000:+.1f}, {btab[1]*1000:+.1f}) mm"
            )

    # ------------------------------------------------------------------
    # Seeded object randomization (deterministic; GRPO-group friendly)
    # ------------------------------------------------------------------
    def _randomize_world_objects(
        self, w: int, seed: int, jq, q_per: int, tgt=None, base_mocap_out=None, instance=None
    ) -> None:
        """Seeded per-env initial-state randomization for world ``w`` on the
        just-restored buffers. Deterministic in ``seed`` alone. Three parts,
        drawn from ONE rng stream in a fixed order (objects -> arm -> base) so
        the same seed reproduces the whole initial world state:

        1. OBJECT xy: uniform +-object_xy_radius with min-gap rejection
           sampling (never pushes objects into each other); support prims
           excluded via ``exclude`` substrings.
        2. ARM joints (benchmark ``init_joint`` generalization): offsets
           ~ U(-init_joint_thresh, +init_joint_thresh) [rad] on every actuated
           ``_arm_`` joint — applied to joint_q AND the PD target.
        3. ROBOT BASE xy (benchmark ``init_base``): offset ~ U(-x,+x)/U(-y,+y)
           metres (init_base_xy_thresh=[x, y]) applied on whichever base the
           build has — the root FREE joint (offset lands in ``jq``) OR, when
           there is no free root but the adapter exposes a native mocap base
           (``pin_base_to_world:false`` builds), recorded into
           ``base_mocap_out[w]`` for the caller to apply post-eval_fk via
           ``adapter.apply_base_mocap_offsets``. The rng draw is IDENTICAL
           in both cases (exactly two uniforms, same stream position), so a
           given seed yields the same base offset regardless of build type.

        Spec: ``scene_cfg.randomization`` — the benchmark eval task's
        ``generalization`` block feeds init_joint/init_base via the task
        loader; explicit yaml keys win.

        Task 3 (2026-07-28, RLinf interface batch 1 — random seed contract):
        dropped
        the separate ``randomization.enabled`` door in front of the OBJECT
        step. This method is only ever called for an env that already has a
        real (>=0) seed — see ``episode_reset()``'s own ``if seeds and w in seeds``
        call site, fed from the coordinator's own ``if int(v) >= 0`` filter —
        so "a seed was given" IS the request to randomize; a second boolean
        gate on top of that let ``enabled: false`` (the training template's
        own default — ``task_init_pick_block_color.yaml``) silently swallow
        a real per-reset seed, which is exactly backwards from the contract
        this task establishes (``seed >= 0`` => generalize, ``seed == -1``
        => verbatim, nothing in between). Object-xy randomization is now
        gated on ``object_xy_radius > 0`` alone — a real magnitude knob, not
        an on/off door — same as arm/base already were on ``jt > 0`` / ``bt``.

        The draw itself lives in ``genie_sim_engine_core.common.reset_randomization``
        (shared with the fused PhysX node), unchanged; this method collects the
        candidates in the order it always has and applies the offsets. When the
        world's instance has a layout (``instance_switch.instance_layout``,
        recorded by the rest-state bake), the object offsets are constrained
        there: no new overlaps, carried or touching objects move together,
        objects on a surface stay on it.
        """
        from genie_sim_engine_core.common.reset_randomization import draw_reset, is_active, object_name  # noqa: PLC0415

        spec = (getattr(self, "_scene_cfg", None) or {}).get("randomization") or {}
        jt = float(spec.get("init_joint_thresh", 0.0) or 0.0)
        bt = spec.get("init_base_xy_thresh")
        radius = float(spec.get("object_xy_radius", 0.03))
        if not is_active(spec):
            return
        jindex = getattr(self, "_jindex", None)
        if jindex is None or not q_per:
            return

        # Candidates in free-joint order (the draw's rng order depends on it).
        objects = []  # (name, slice)
        robot_root_slc = None
        for slc, bname in self._world_free_joints(w):
            if "Objects" in bname:
                # The task object, as the layout names it; an articulated
                # asset's free body sits deeper (``<object>/entity/body``).
                name = object_name(bname) or (bname.rsplit("/", 2)[-2] if "/" in bname else bname)
                objects.append((name, slc))
            elif robot_root_slc is None:
                robot_root_slc = slc  # the only non-object free joint
        counts: dict = {}
        for name, _ in objects:
            counts[name] = counts.get(name, 0) + 1
        objects = [(name if counts[name] == 1 else f"{name}#{k}", slc) for k, (name, slc) in enumerate(objects)]

        arm = []
        if jt > 0:
            child_arr = getattr(self._model, "joint_child", None)
            child = child_arr.numpy() if child_arr is not None else None
            body_paths = getattr(self, "_body_paths", None) or list(getattr(self._model, "body_label", []) or [])
            ne = int(getattr(self, "_num_envs", 1) or 1)
            nb_per = len(body_paths) // ne if ne else len(body_paths)
            for slc in jindex.slices():
                if "_arm_" not in slc.name or not slc.is_actuated:
                    continue
                if child is None or slc.joint_index >= len(child):
                    continue
                if ((int(child[slc.joint_index]) // nb_per) if nb_per else 0) != w:
                    continue
                arm.append(slc)

        # The base has three representations:
        #   * FREE-joint root -- the offset is written straight into jq;
        #   * MOCAP body root -- the tabletop configuration with
        #     pin_base_to_world:false; the draw is recorded into
        #     base_mocap_out[w] and applied by the adapter after reset();
        #   * neither -- unsupported, warned about once.
        _has_mocap_base = (
            base_mocap_out is not None
            and robot_root_slc is None
            and getattr(self, "_adapter", None) is not None
            and getattr(self._adapter, "has_base_mocap", None) is not None
            and self._adapter.has_base_mocap()
        )
        layout = None
        if instance is not None:
            sw = (getattr(self, "_scene_cfg", None) or {}).get("instance_switch") or {}
            layout = (sw.get("instance_layout") or {}).get(str(int(instance)))

        draw = draw_reset(
            int(seed),
            spec,
            [(name, (float(jq[slc.q_start]), float(jq[slc.q_start + 1]))) for name, slc in objects],
            len(arm),
            has_base=bool(robot_root_slc is not None or _has_mocap_base),
            layout=layout,
        )

        moved = []
        for name, slc in objects:
            off = draw.object_offsets.get(name)
            if off is None:
                continue
            jq[slc.q_start] += off[0]
            jq[slc.q_start + 1] += off[1]
            moved.append(f"{name}(+{off[0]:.3f},{off[1]:.3f})")
        for k, slc in enumerate(arm):
            off = float(draw.arm_offsets[k])
            jq[slc.q_start] += off
            if tgt is not None and slc.qd_start < len(tgt):
                tgt[slc.qd_start] += off
        base_d = draw.base_offset
        base_via = ""
        if base_d is not None and robot_root_slc is not None:
            jq[robot_root_slc.q_start] += base_d[0]
            jq[robot_root_slc.q_start + 1] += base_d[1]
            base_via = "free"
        elif base_d is not None:
            base_mocap_out[w] = base_d
            base_via = "mocap"
        elif bt and not getattr(self, "_base_warn_done", False):
            self._base_warn_done = True
            self._logger.warn(
                "[reset-randomize] init_base requested but this robot build "
                "has neither a free-joint root nor a native mocap base "
                "(root_joint state unexpected) — base perturbation "
                "unsupported, skipping"
            )

        parts = []
        if moved:
            parts.append(f"{len(moved)} object(s): " + ", ".join(moved[:6]))
        if draw.constrained:
            parts.append("constrained " + "; ".join(draw.constrained[:6]))
        if arm:
            parts.append(f"arm offsets on {len(arm)} joint(s) (thresh {jt})")
        if base_d:
            parts.append(f"base{'/' + base_via if base_via else ''} " f"(+{base_d[0]:.4f},+{base_d[1]:.4f})")
        if parts:
            self._logger.info(f"[reset-randomize] world {w} seed {seed}: " + "; ".join(parts))
        elif radius > 0 and not getattr(self, "_rand_warn_done", False):
            self._rand_warn_done = True
            free_names = [s2.name for s2 in jindex.slices() if s2.joint_type == 4][:12]
            self._logger.warn(
                "[reset-randomize] object_xy_radius > 0 but nothing to "
                f"randomize — free joints present: {free_names}"
            )

    # ------------------------------------------------------------------
    # Joint PD gain randomization
    # ------------------------------------------------------------------
    def _apply_joint_pd_randomization(self, env_ids: Sequence[int], qd_per: int) -> None:
        """Scale ``joint_target_ke``/``joint_target_kd`` per world from
        ``scene_cfg.randomization.joint_pd`` -- the ONE authorized engine
        change for the generalization layer's joint_pd dimension (see
        ``rlinf_geniesim/randomize/spec.py`` and ``sampler.py``).

        ``joint_pd`` here is NOT a range for this method to draw from -- the
        concrete per-world multipliers are already drawn, once, at
        task-build time by ``Sampler.sample()`` (its own independent rng
        stream, decoupled from this engine's object/arm/base randomization
        stream in ``_randomize_world_objects``) and baked into
        ``scene_cfg["randomization"]["joint_pd"]`` before this engine process
        even starts, keyed by world index AS A STRING (``{"0":
        {"kp_scale": 1.03, "kd_scale": 0.97}, ...}`` -- the same
        string-keyed-by-id convention ``instance_switch.robot_base_xy``
        already uses). This method only APPLIES that already-resolved
        table: it draws nothing and opens no new IPC ("go through the
        existing scene_cfg.randomization channel, don't build a new IPC").

        Always scales ``self._pd_ke_build_baseline``/``_pd_kd_build_baseline``
        (see ``_snapshot_reset_state``), never the live array -- rescaling
        the live value on every reset would compound the same multiplier
        again and again, the same failure mode camera_pose guards against.
        Only DOFs in ``self._joint_pd_eligible_mask`` (arm/body/head) are
        touched; gripper and chassis gains are never written here.

        Readback discipline: count how many of the worlds in
        THIS ``env_ids`` call actually have a ``joint_pd`` entry
        (``requested``) versus how many actually got at least one DOF
        rescaled (``applied``). ``requested > 0`` and ``applied == 0`` raises
        -- the config asked for this dimension and the engine had no way to
        honour it (missing baseline / empty eligible mask / mis-shaped
        ``qd_per``), which is a build-time configuration bug that must not be
        swallowed. ``0 < applied < requested`` warns loudly but does not
        raise, so the worlds that DID succeed keep their write. ``requested
        == 0`` is a silent no-op -- ``joint_target_ke``/``kd`` are never even
        read, which is the first half of the "not configured -> byte-
        identical" guarantee (the second half is the per-world identity
        shortcut below, for "configured but scale == (1.0, 1.0)").
        """
        jpd_map = ((getattr(self, "_scene_cfg", None) or {}).get("randomization") or {}).get("joint_pd") or {}
        if not jpd_map:
            return
        baseline_ke = getattr(self, "_pd_ke_build_baseline", None)
        baseline_kd = getattr(self, "_pd_kd_build_baseline", None)
        eligible = getattr(self, "_joint_pd_eligible_mask", None)
        ke, kd, requested, applied, touched = _scale_joint_pd(
            jpd_map, env_ids, baseline_ke, baseline_kd, eligible, qd_per
        )
        if requested == 0:
            return
        if applied == 0:
            raise RuntimeError(
                f"[joint_pd] requested for {requested} world(s) but applied "
                f"to 0 -- baseline_cached={baseline_ke is not None}, "
                f"eligible_mask_cached={eligible is not None}, qd_per={qd_per} "
                "-- see _snapshot_reset_state's joint_pd baseline caching"
            )
        if applied < requested:
            self._logger.warn(
                f"[joint_pd] partial application: requested {requested} "
                f"world(s), applied to only {applied} (touched: {touched})"
            )
        if ke is not None and kd is not None and touched:
            ke_arr = getattr(self._model, "joint_target_ke", None) if self._model is not None else None
            kd_arr = getattr(self._model, "joint_target_kd", None) if self._model is not None else None
            if ke_arr is None or kd_arr is None:
                return
            ke_arr.assign(ke)
            kd_arr.assign(kd)
            # Independent observation: assign()
            # not raising is not proof the value landed -- read it back and
            # compare against what was intended, the same way camera_pose
            # Get()s after Set() and material independently re-resolves the
            # bound material rather than trusting the call that issued the
            # write.
            readback_ke = ke_arr.numpy()
            readback_kd = kd_arr.numpy()
            for w in touched:
                lo, hi = w * qd_per, (w + 1) * qd_per
                if not np.allclose(readback_ke[lo:hi], ke[lo:hi]) or not np.allclose(readback_kd[lo:hi], kd[lo:hi]):
                    self._logger.warn(
                        f"[joint_pd] world {w}: assign() did not raise, but "
                        "the independent readback disagrees with the "
                        "intended value"
                    )

    # ------------------------------------------------------------------
    # Shared state-write commit path (episode_reset AND teleport)
    # ------------------------------------------------------------------
    def _commit_state_write(self, env_ids, jq, qd, tgt=None, base_mocap_off=None, label="reset") -> None:
        """Land an out-of-band ``joint_q``/``joint_qd`` write on live physics.

        Every caller that rewrites state between frames owes the SAME seven
        steps, and each one below carries the rationale it was learned from:

        1. assign into ``state_0`` AND ``state_1`` -- the ping-pong pair
        2. assign the PD targets, or a live drive yanks the joints back
        3. apply per-world base mocap offsets before FK
        4. ``eval_fk``, so ``body_q`` matches the new ``joint_q``
        5. push ``qpos``/``qvel`` and ``ctrl`` into the solver's own data
        6. drop the solver's cross-step carry-over for the written worlds
        7. flush the async host mirrors the publishers read

        Step 7 is why this is a shared helper and not a copy: a state write
        applied while nothing is stepping publishes the STALE pre-write pose,
        because only ``step()`` refreshes those mirrors.  ``teleport`` applies
        while paused by contract, so it would hit that on every call.

        ``jq``/``qd``/``tgt`` are host arrays already carrying the desired
        values -- this method does not compute them, it commits them.  ``None``
        for any of the three skips that assignment.  ``label`` only names the
        caller in the diagnostics.
        """
        import newton  # noqa: PLC0415

        s0 = self._state_0
        s1 = self._state_1
        ctl = self._control
        _ad = getattr(self, "_adapter", None)

        for st in (s0, s1):
            if st is None:
                continue
            if jq is not None and getattr(st, "joint_q", None) is not None:
                st.joint_q.assign(jq)
            if qd is not None and getattr(st, "joint_qd", None) is not None:
                st.joint_qd.assign(qd)
        if tgt is not None and ctl is not None and getattr(ctl, "joint_target_q", None) is not None:
            ctl.joint_target_q.assign(tgt)

        # MOCAP-body robot base (pin_base_to_world:false builds): apply
        # the per-world base-xy offset BEFORE eval_fk. The adapter writes
        # model.joint_X_p (so eval_fk below — and every later step's
        # eval_articulation_fk — derives the offset base body_q, making
        # the published body_poses reflect it in steady state) AND
        # mjw_data.mocap_pos (MuJoCo's per-step hold, so the base stays
        # put under free-run). No-op on free-joint / weld builds. Runs
        # between frames, so the in-place .assign() writes are graph-safe.
        if base_mocap_off and _ad is not None and getattr(_ad, "apply_base_mocap_offsets", None) is not None:
            _ad.apply_base_mocap_offsets(base_mocap_off, self._logger)

        # Refresh body poses for the rewritten joint_q so the read
        # between this reset and the next physics step is consistent.
        if getattr(s0, "joint_q", None) is not None:
            newton.eval_fk(self._model, s0.joint_q, s0.joint_qd, s0)
            # A clean joint_q does not imply a clean body_q: body_q is
            # derived by eval_fk from joint_q AND the MODEL, so corruption
            # on the model side (joint_X_p / inertia / mocap) becomes
            # visible here for the first time. Reporting it separates
            # "joints are broken" from "the model is broken" — the two lead
            # investigations in completely different directions, while
            # downstream only ever sees a sea of NaN.
            _bq = getattr(s0, "body_q", None)
            if _bq is not None and not _finite(_bq.numpy() if hasattr(_bq, "numpy") else _bq):
                self._logger.warn(
                    f"[newton-standalone] {label}: joint_q was written with "
                    "finite values, but the body_q derived by eval_fk is "
                    "still non-finite — the corruption is on the MODEL side "
                    "(joint_X_p / instance-pose table / mocap offset), not "
                    "in the joint state. No state write fixes this class; the "
                    "task has to be rebuilt."
                )

        # Clear lagged rigid/particle feedback and sub-solver transient state.
        # flags=0 preserves the authored coordinates restored above; the
        # following sync re-applies PD targets after Newton clears ctrl.
        coupled = getattr(self, "_coupled_solver", None)
        if coupled is not None:
            reset_options = {"flags": 0}
            worlds = int(getattr(self, "_num_envs", 1) or 1)
            if len(set(env_ids)) < worlds:
                import warp as wp

                reset_options["world_mask"] = wp.array(
                    [w in env_ids for w in range(worlds)] + [False], dtype=wp.bool, device=self._model.device
                )
            coupled.reset(s0, **reset_options)
            self._logger.info(
                "[newton-standalone] coupled solver transient state reset; authored coordinates preserved"
            )

        # Push the restored state into the live mjw_data. The adapter runs
        # with ``update_data_interval=0`` and does no per-substep sync back,
        # so without this explicit push the PD targets would restore the
        # robot while free bodies (objects, the kinematic tabletop's pin
        # reference) stayed at their pre-reset physics state.
        # A reset runs between frames with no CUDA graph in flight, so
        # launching kernels directly is safe.
        sync = getattr(self._adapter, "sync_state_into_solver", None) if hasattr(self, "_adapter") else None
        if sync is not None:
            sync(s0, self._logger)
        # ...and the control side needs pushing too. The step above only
        # sends qpos/qvel into the solver, whereas what the actuators
        # actually read is ``mjw_data.ctrl``; ``_apply_mjc_control`` runs
        # only inside solver.step, so without this push ctrl still holds the
        # PRE-RESET command.
        # Measured 2026-08-05: after resetting a diverged run, all five
        # state probes read 0.000000 (position / velocity / PD target /
        # warmstart / act all clean), yet ctrl was 1.1-1.8 rad off the
        # baseline; the next frame qfrc_actuator spiked to 892-22429 (130-350
        # when healthy), the arm was driven into its limits and velocity
        # climbed from 0 to 2388 rad/s — the reset restored position only,
        # not control.
        _syncc = getattr(self._adapter, "sync_control_into_solver", None) if hasattr(self, "_adapter") else None
        if _syncc is not None:
            _syncc(s0, ctl, self._logger)
        # ...and drop the solver's cross-step carry-over for the same
        # worlds. The push above restores qpos/qvel; the carry-over
        # (warm start / actuator activation) still describes the
        # PRE-teleport configuration and is stale by definition. It is
        # also what made an integrator blow-up unrecoverable: reset put
        # joint_q back to init while qacc_warmstart kept its NaNs and
        # re-poisoned the next constraint solve. See the adapter method's
        # own docstring.
        _cc = getattr(self._adapter, "clear_solver_carryover", None) if hasattr(self, "_adapter") else None
        if _cc is not None:
            _cc(env_ids, self._logger)

        # FLUSH THE ASYNC HOST MIRRORS. The /joint_states + /tf publishers
        # read the ping-pong device->host mirrors, which only step()
        # refreshes — a reset applied while the engine is idle-paused
        # would otherwise publish (and hand the RL client) the STALE
        # pre-reset state -- and a teleport, which by contract applies WHILE
        # PAUSED, lands in that window every time. Refresh twice + synchronize
        # so the publish that follows this reset carries the restored
        # (and possibly seed-randomized) configuration.
        try:
            import warp as _wp  # noqa: PLC0415

            if getattr(self, "_pq_mirror", None) is not None:
                self._update_record_snapshot_mirror()
            for _mir_name, _src in (
                ("_jq_mirror", getattr(s0, "joint_q", None)),
                ("_jqd_mirror", getattr(s0, "joint_qd", None)),
                ("_bq_mirror", getattr(s0, "body_q", None)),
                ("_pq_mirror", getattr(s0, "particle_q", None)),
            ):
                _mir = getattr(self, _mir_name, None)
                if _mir is None or _src is None:
                    continue
                _stream = _wp.get_stream(_src.device)
                _mir.refresh(_src, _stream)
                _mir.refresh(_src, _stream)
            _wp.synchronize_device()
        except Exception as exc3:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] {label} mirror flush failed: {exc3}")

    # ------------------------------------------------------------------
    # Out-of-band pose writes (SimulationControl.teleport)
    # ------------------------------------------------------------------
    def _teleport_targets(self, req, ne: int):
        """Resolve one request to ``(worlds, body_name)``.

        ``env_<w>/<name>`` addresses a single world; a bare name at
        ``num_envs > 1`` addresses EVERY world, which is what a batched RL
        caller means when it teleports "the mug".  An explicit
        ``TeleportRequest.env_ids`` narrows either form.
        """
        name = str(req.body)
        envs = None
        if name.startswith("env_"):
            head, sep, rest = name.partition("/")
            if sep:
                try:
                    envs, name = [int(head[4:])], rest
                except ValueError:
                    pass  # a body genuinely named "env_something" -- leave it
        want = getattr(req, "env_ids", None)
        if want:
            want = {int(w) for w in want}
            envs = sorted(want) if envs is None else [w for w in envs if w in want]
        if envs is None:
            envs = list(range(ne))
        return [w for w in envs if 0 <= w < ne], name

    def teleport(self, requests) -> int:
        """Write explicit body poses out of band with the physics.

        FREE joints only, and that is a rejection rather than an approximation:
        a body welded to the world or hung off a revolute joint has no 6-DoF
        state to write, so "teleporting" it could only mean solving IK for
        whatever chain holds it.  The reference draws the same line
        (``ApplyTeleportTransform``), and an unmatched body is WARNED about,
        never raised -- one bad name in a batch must not lose the rest.

        Returns the number of REQUESTS applied (a request that fans out over
        several worlds counts once), so the caller can log the shortfall
        instead of assuming a batch landed.

        Two contracts inherited from the reference and enforced here:

        * velocity is zeroed with the pose (``zero_velocity``, default True).
          A body that keeps its pre-teleport velocity leaves the requested
          pose on the very next step.
        * the write is legal while PAUSED, and that is the main use.  The
          publish path reads async host mirrors that only ``step()``
          refreshes, so the commit runs through ``_commit_state_write``, whose
          step 7 flushes them -- otherwise a paused teleport would publish the
          pose it just replaced.

        ``reset_epoch`` is deliberately NOT bumped.  A teleport is not an
        episode boundary; the epoch is the credential an RL wrapper waits on
        to know its reset landed, and issuing one here would tell it an
        episode restarted when none did.
        """
        reqs = list(requests or ())
        if not reqs:
            return 0
        s0 = self._state_0
        if s0 is None or getattr(s0, "joint_q", None) is None:
            self._logger.warn("[teleport] no live state yet; %d request(s) ignored" % len(reqs))
            return 0
        if getattr(self, "_jindex", None) is None:
            self._logger.warn("[teleport] no joint index; cannot resolve free joints")
            return 0

        ne = int(getattr(self, "_num_envs", 1) or 1)
        jq = s0.joint_q.numpy().copy()
        _qdw = getattr(s0, "joint_qd", None)
        qd = _qdw.numpy().copy() if _qdw is not None else None

        applied = 0
        touched: set = set()
        missed: list = []
        for req in reqs:
            envs, name = self._teleport_targets(req, ne)
            hits = 0
            for w in envs:
                for slc, bname in self._world_free_joints(w):
                    if not _body_name_matches(bname, name):
                        continue
                    q0 = slc.q_start
                    pos = tuple(float(v) for v in req.position)
                    jq[q0] = pos[0]
                    jq[q0 + 1] = pos[1]
                    jq[q0 + 2] = pos[2]
                    # Canonical Python orientation is wxyz (ROS gives xyzw and
                    # the C++ callback already reordered it); Newton's
                    # free-joint q slots are xyzw, Warp-native. Same single
                    # reorder as _apply_instance_poses -- verified in Spike S1.
                    qo = tuple(float(v) for v in req.orientation_wxyz)
                    jq[q0 + 3] = qo[1]  # qx
                    jq[q0 + 4] = qo[2]  # qy
                    jq[q0 + 5] = qo[3]  # qz
                    jq[q0 + 6] = qo[0]  # qw
                    if getattr(req, "zero_velocity", True) and qd is not None:
                        # dof_count, not a hard 6: the slice knows its own
                        # width, and a widthless joint's q_count (7) is not it.
                        qd[slc.qd_start : slc.qd_start + slc.dof_count] = 0.0
                    hits += 1
                    touched.add(w)
            if hits:
                applied += 1
                self._logger.info(
                    f"[teleport] {name} -> pos={tuple(round(float(v), 4) for v in req.position)} "
                    f"quat_wxyz={tuple(round(float(v), 4) for v in req.orientation_wxyz)} "
                    f"in world(s) {sorted(set(envs))} ({hits} free joint(s))"
                )
            else:
                missed.append(name)

        if missed:
            # Name the candidates: the overwhelmingly common failure is a body
            # that IS in the scene but is welded or hinged, and a bare "not
            # found" sends the reader looking for a typo instead.
            free_names = sorted({b for w in range(ne) for _s, b in self._world_free_joints(w)})
            self._logger.warn(f"[teleport] no FREE joint for {missed} -- ignored. Teleportable bodies: {free_names}")
        if not applied:
            return 0
        # tgt is NOT touched: a teleport moves a free body, which has no PD
        # target to restore, and rewriting the robot's targets here would
        # silently undo whatever command is in flight.
        self._commit_state_write(sorted(touched), jq, qd, tgt=None, label="teleport")
        return applied

    # ------------------------------------------------------------------
    # The state write
    # ------------------------------------------------------------------
    def episode_reset(
        self, env_ids: Optional[Sequence[int]] = None, seeds: Optional[dict] = None, instances: Optional[dict] = None
    ) -> None:
        """``seeds``: {env_id: seed} — after restoring a world's snapshot,
        apply DETERMINISTIC object-pose randomization seeded per env (see
        _randomize_world_objects — a present, non-negative seed is the
        WHOLE trigger now; task 3, 2026-07-28, dropped the separate
        scene_cfg.randomization.enabled door that used to ALSO have to be
        on). Same seed ⇒ identical initial state (GRPO group sampling
        relies on this); no seed entry means no random perturbation. Task
        scenes restore the selected/current instance's baked poses first."""
        ne = int(getattr(self, "_num_envs", 1) or 1)
        if env_ids is None:
            env_ids = list(range(ne))
        env_ids = [int(w) for w in env_ids if 0 <= int(w) < ne]
        if not env_ids:
            return
        if getattr(self, "_reset_init_joint_q", None) is None or self._state_0 is None:
            self._logger.warn("[newton-standalone] reset: no init snapshot / state; skipping")
            return
        # Configuration errors must fail outside the runtime recovery catch,
        # before resnapshotting or writing any selected world's state.
        body_tables = self._validate_instance_body_poses()
        reset_instances = {}
        if body_tables is not None:
            sw = self._scene_cfg["instance_switch"]
            active = getattr(self, "_active_instances", None) or {}
            for w in env_ids:
                requested = int((instances or {}).get(w, -1))
                # The ROS coordinator drops -1; direct callers may retain
                # it. Both spell "reset the current instance", not "restore
                # the build instance's snapshot after an instance switch".
                inst = requested if requested >= 0 else active.get(w, sw.get("build_instance"))
                if inst is not None:
                    inst = int(inst)
                    if str(inst) not in body_tables:
                        raise self._instance_pose_error(f"missing instance {inst}")
                    reset_instances[w] = inst

        # First client reset captures a finite settled baseline for pure
        # engine scenes and channels absent from task tables. The baked
        # instance body poses below always override this live snapshot.
        if not getattr(self, "_reset_resnapshotted", False):
            self._reset_resnapshotted = True
            self.resnapshot_settled()
        try:
            s0 = self._state_0
            jq = s0.joint_q.numpy().copy() if getattr(s0, "joint_q", None) is not None else None
            qd = s0.joint_qd.numpy().copy() if getattr(s0, "joint_qd", None) is not None else None
            ctl = self._control
            tgt = (
                ctl.joint_target_q.numpy().copy()
                if ctl is not None and getattr(ctl, "joint_target_q", None) is not None
                else None
            )
            init_q = self._reset_init_joint_q
            init_tgt = self._reset_init_target

            q_per = (len(jq) // ne) if (jq is not None and len(jq) % ne == 0) else 0
            qd_per = (len(qd) // ne) if (qd is not None and len(qd) % ne == 0) else 0
            tgt_per = (len(tgt) // ne) if (tgt is not None and len(tgt) % ne == 0) else 0

            # Per-world base-xy offsets for a MOCAP-body robot base
            # (pin_base_to_world:false builds — no root free joint, so the
            # offset can't ride in joint_q). Default (0,0) per reset world =
            # restore that base to the build-time init pose, so a prior
            # reset's offset never accumulates; seeded worlds get their drawn
            # offset written by _randomize_world_objects below. Applied by the
            # adapter AFTER eval_fk (mjw mocap_pos + model.joint_X_p).
            _ad = getattr(self, "_adapter", None)
            _cfg_now = getattr(self, "_scene_cfg", None) or {}
            _mocap_base_active = (
                (
                    bool((_cfg_now.get("randomization") or {}).get("init_base_xy_thresh"))
                    # Per-instance base poses (instance_switch.robot_base_xy) ride
                    # the SAME mocap channel below. Without this clause the gate
                    # never opens from that config alone, _apply_instance_poses
                    # always receives base_mocap_out=None, and a base offset it
                    # writes is silently never applied.
                    or bool((_cfg_now.get("instance_switch") or {}).get("robot_base_xy"))
                )
                and _ad is not None
                and getattr(_ad, "has_base_mocap", None) is not None
                and _ad.has_base_mocap()
            )
            base_mocap_off: dict = {}

            pre_dev = {}  # world -> max|joint_q - init| BEFORE the write (proof of teleport)
            # world -> max|joint_qd| BEFORE the reset. Divergence signs itself
            # by velocity exploding first and position turning NaN afterwards,
            # so this number is direct evidence of whether physics was already
            # diverging at the end of the episode — NaN in obs only shows that
            # it HAS blown up, never when it started.
            # (The first open item in RLINF_GENIESIM_OBS_NAN_REPORT.html section
            # 3.3 is exactly this.)
            pre_vel = {}
            for w in env_ids:
                # Restore the PD targets to their initial values FIRST, then
                # run the instance-pose / object randomisation below — the other
                # order lets this restore clobber the randomisation's companion
                # writes to ``tgt``, leaving joint_q offset while the PD targets
                # sit at their initial values, and a live PD drags the arm back
                # within a fraction of a second.
                if tgt_per and init_tgt is not None and len(init_tgt) == len(tgt):
                    tgt[w * tgt_per : (w + 1) * tgt_per] = init_tgt[w * tgt_per : (w + 1) * tgt_per]
                if _mocap_base_active:
                    base_mocap_off[w] = (0.0, 0.0)  # restore-to-init; seeded worlds override below
                if q_per and init_q is not None and len(init_q) == len(jq):
                    cur = jq[w * q_per : (w + 1) * q_per]
                    ini = init_q[w * q_per : (w + 1) * q_per]
                    pre_dev[w] = float(max(abs(cur - ini))) if len(cur) else 0.0
                    jq[w * q_per : (w + 1) * q_per] = ini
                    if w in reset_instances:
                        # Ordering matters: this call runs BEFORE
                        # _randomize_world_objects below, which -- if the two
                        # channels were ever both configured -- would overwrite
                        # base_mocap_off[w] with its own (seeded) draw.
                        # task_loader.check_base_xy_channel_conflict raises at
                        # build time when instance_switch.robot_base_xy and
                        # randomization.init_base_xy_thresh are both non-empty,
                        # so the two writers never actually collide in
                        # practice -- this ordering is what makes that mutex
                        # the correct fallback should the gate ever be
                        # loosened.
                        self._apply_instance_poses(
                            w,
                            reset_instances[w],
                            jq,
                            tgt=tgt,
                            base_mocap_out=(base_mocap_off if _mocap_base_active else None),
                            _body_tables=body_tables,
                        )
                    if seeds and w in seeds:
                        self._randomize_world_objects(
                            w,
                            int(seeds[w]),
                            jq,
                            q_per,
                            tgt=tgt,
                            base_mocap_out=base_mocap_off if _mocap_base_active else None,
                            instance=reset_instances.get(w),
                        )
                if qd_per:
                    seg = qd[w * qd_per : (w + 1) * qd_per]
                    try:
                        pre_vel[w] = float(np.max(np.abs(seg))) if len(seg) else 0.0
                    except Exception:  # noqa: BLE001 — telemetry only; must
                        # never block the reset
                        pre_vel[w] = float("nan")
                    qd[w * qd_per : (w + 1) * qd_per] = 0.0

            # Still non-finite after the restore — the last line of defence. By
            # now jq has been through three writes: the init_q overwrite,
            # _apply_instance_poses and _randomize_world_objects. If the result
            # is not clean, fall back to the BUILD-TIME snapshot: by definition
            # it never ran physics and is the only reference in the whole flow
            # guaranteed to be finite. Without this, a corrupted baseline turns
            # reset into "faithfully restoring to NaN" while the caller sees a
            # successful reset.
            build_q = getattr(self, "_reset_build_joint_q", None)
            for w in env_ids:
                if not q_per or jq is None:
                    continue
                seg = jq[w * q_per : (w + 1) * q_per]
                if _finite(seg):
                    continue
                if build_q is not None and len(build_q) == len(jq):
                    jq[w * q_per : (w + 1) * q_per] = build_q[w * q_per : (w + 1) * q_per]
                    self._logger.warn(
                        f"[newton-standalone] reset: world {w} restored a "
                        f"joint_q that is still non-finite — fell back to the "
                        f"BUILD-TIME snapshot. The baseline or the instance-pose "
                        f"table may be corrupted; this episode starts from the "
                        f"build configuration rather than the settled one, so "
                        f"objects may fall again."
                    )
                else:
                    self._logger.warn(
                        f"[newton-standalone] reset: world {w} restored a "
                        f"joint_q that is still non-finite, and there is NO "
                        f"build-time snapshot to fall back to — this world's "
                        f"state is unusable and downstream will see NaN "
                        f"observations."
                    )

            # Scene-plugin builds with particles are single-world (native
            # replication rejects these builds). Restore both ping-pong
            # buffers in place before the shared commit flushes render mirrors;
            # replacing arrays would invalidate captured CUDA graph pointers.
            points = getattr(self, "_reset_init_particle_q", None)
            if points is not None and len(points):
                if ne != 1:
                    raise NotImplementedError("particle reset requires a single-world particle build")
                for state in (self._state_0, self._state_1):
                    if state is None:
                        continue
                    state.particle_q.assign(points)
                    state.particle_qd.zero_()
                    if getattr(state, "particle_f", None) is not None:
                        state.particle_f.zero_()

            self._commit_state_write(
                env_ids, jq, qd, tgt=tgt, base_mocap_off=base_mocap_off if _mocap_base_active else None, label="reset"
            )

            # Joint PD gain randomization: re-apply this world's
            # PD-gain multiplier from the build-time baseline. Independent of
            # the jq/qd/tgt state write above -- ke/kd are a completely
            # separate pair of dof-sized arrays -- so it runs as its own
            # step rather than being woven into the loop above.
            self._apply_joint_pd_randomization(env_ids, qd_per)

            # [diag-reset-z] one-shot: trace a single OBJECT free joint through
            # the restore stages to find where its height goes wrong.
            if not getattr(self, "_reset_zdiag_done", False):
                self._reset_zdiag_done = True
                try:
                    jindex2 = getattr(self, "_jindex", None)
                    child_arr = getattr(self._model, "joint_child", None)
                    bp = list(getattr(self._model, "body_label", []) or [])
                    ch = child_arr.numpy() if child_arr is not None else None
                    for slc in jindex2.slices() if jindex2 else []:
                        if slc.joint_type != 4 or ch is None:
                            continue
                        bi = int(ch[slc.joint_index])
                        if bi >= len(bp) or "Objects" not in str(bp[bi]) or "table" in str(bp[bi]):
                            continue
                        jq_now = s0.joint_q.numpy()
                        bq = s0.body_q.numpy()
                        self._logger.info(
                            f"[diag-reset-z] {bp[bi]}: template joint_q z="
                            f"{float(init_q[slc.q_start + 2]):.4f}, restored joint_q z="
                            f"{float(jq_now[slc.q_start + 2]):.4f}, body_q after eval_fk z="
                            f"{float(bq[bi, 2]):.4f}"
                        )
                        break
                except Exception as exc2:  # noqa: BLE001
                    self._logger.warn(f"[diag-reset-z] failed: {exc2}")

            dev_str = ", ".join(f"w{w}:{pre_dev.get(w, 0.0):.3f}" for w in env_ids)
            vel_str = ", ".join(f"w{w}:{pre_vel.get(w, 0.0):.3f}" for w in env_ids)
            self._logger.info(
                f"[newton-standalone] reset applied to world(s) {env_ids} "
                f"(q_per={q_per} qd_per={qd_per} tgt_per={tgt_per}); "
                f"pre-reset max|joint_q-init| per world: {dev_str} → all set to init"
                f"; pre-reset max|joint_qd| per world: {vel_str}"
            )

            # reset_epoch causal credential: incremented only after every
            # write above has landed. The main loop's next /joint_states carries
            # this value, and this method runs sequentially on the same loop
            # thread as the publish, so that publish cannot pair an old epoch
            # with new state.
            # Inside the try (not finally): a failed reset leaves the state
            # unchanged, so the epoch must not advance — that would issue a
            # credential for stale state.
            if reset_instances:
                active = dict(getattr(self, "_active_instances", None) or {})
                active.update(reset_instances)
                self._active_instances = active
            self._reset_epoch += 1
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[newton-standalone] reset failed: {exc}")

    # ------------------------------------------------------------------
    # rclpy trigger listener (always-on, separate participant)
    # ------------------------------------------------------------------
