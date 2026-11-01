# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""FEM cloth pin controller for the cloth-fold task.

The inference model only commands the gripper open/close; it never tells the sim
to grab the deformable cloth. Real gripper-vs-FEM contact is unreliable, so this
controller makes the cloth follow the gripper directly via a velocity servo on
the FEM simulation nodes (the approach proven in the standalone demo
cloth_fold_fem_tshirt001_gripper_standalone.py, run with --pin_pos_follow 0):

  - When a gripper closes, cloth nodes that actually fall inside that gripper's
    jaw_pocket box (a measured region between the fingers, see robot_cfg
    "gripper.jaw_pocket" / DEFAULT_JAW_POCKET below) are "pinned": each physics
    step their nodal velocity is set to a feed-forward (anchor velocity) plus a
    P term pulling them toward (anchor world pos + the offset frozen at grasp
    time). The rest of the cloth follows via FEM elasticity and folds naturally.
  - When the gripper opens, the pinned nodes' velocities are zeroed and released.

It also disables the gripper collision meshes once at task start: with the pin in
charge of grasping, gripper colliders fighting the cloth/table only hurts.

Runs on the physics thread (APICore.cloth_pin_step), so it reads link poses via
get_world_pose directly (NOT api_core.get_link_world_pose, which re-enters
run_on_physics_loop and would deadlock).

FEM access uses a private create_simulation_view("torch") (NOT the shared
SimulationManager view, which reports 0 nodes for this asset) +
create_surface_deformable_body_view + get/set_simulation_nodal_velocities.
"""

from pxr import Usd, UsdPhysics
import threading
import time

from geniesim_benchmark.plugins.logger import Logger

logger = Logger()

# Gripper joint command is passthrough: 0.0 fully open, -0.91 fully closed.
GRIPPER_CLOSE_THRESHOLD = -0.75
# Debug sphere radius only (visualization). Grasp gating itself uses the
# jaw_pocket box below, not a sphere.
PIN_RADIUS = 0.025
PIN_MIN_SUPPORT_NODES = 60      # if contact is real, pull nearest nodes up to this
# Minimum jaw_pocket inside_count required to trust a grasp as real contact
# (not just 1-2 stray nodes grazing the box edge) before the pin engages.
PIN_MIN_INSIDE_COUNT = 5
# Fallback jaw-pocket geometry (gripper-local box, parent link + center offset +
# half extents) used only if robot_cfg has no "gripper.jaw_pocket" entry. The real
# values are measured per-robot by manually closing the gripper in Isaac Sim and
# drawing a box between the fingers (see robot_cfg/G2_90d.json for the G2_90d
# values these mirror). Node selection tests whether cloth nodes actually fall
# inside this box (the physical space between the closed jaws) — replacing the
# old hand-tuned ellipsoid/offset that had no basis in the real gripper geometry.
DEFAULT_JAW_POCKET = {
    "left": {
        "parent_link": "gripper_l_base_link",
        "center": (0.0, 0.0, 0.2025),
        "half_extents": (0.014, 0.016, 0.04),
        "quat_xyzw": (0.0, 0.0, 0.0, 1.0),
    },
    "right": {
        "parent_link": "gripper_r_base_link",
        "center": (0.0, 0.0, 0.2025),
        "half_extents": (0.014, 0.016, 0.04),
        "quat_xyzw": (0.0, 0.0, 0.0, 1.0),
    },
}
# Velocity servo gains — aligned with the standalone demo, which runs stably at
# 360Hz (set physics_step: 360 in the task yaml). At 120Hz these lagged/jittered.
# Pure velocity servo: FEM stays in charge of positions, so it doesn't jitter the
# cloth the way directly writing node positions does. It leaves a small (~2cm)
# steady-state lag while lifting; that's accepted (position-write fixes all jitter).
PIN_GAIN = 3.0   # P term on (target - current)/dt
PIN_VMAX = 3.0   # m/s clamp (FEM contact solver blows up above this)
PIN_FF = 1.0     # feed-forward fraction of the anchor's own velocity
# Master switch. When False the controller still disables gripper colliders and
# detects/logs grasps, but does NOT touch the cloth — useful to confirm that with
# colliders off the cloth is NOT lifted by friction (so any lift is the pin's).
PIN_ENABLED = True
# The cloth-pin debug sphere (a translucent ball of radius PIN_RADIUS drawn at the
# grasp point to visualize which region is pinned) is toggled per-run via the
# config flag app.show_pin_sphere (read as self.api.show_pin_sphere), not here.


class ClothPinController:
    def __init__(self, api_core):
        self.api = api_core
        self._reset_runtime()

    def _reset_runtime(self):
        self._sim_view = None
        self._cloth_view = None
        self._body_idx = None
        self._torch = None
        self._device = "cuda:0"
        self._view_failed = False
        self._colliders_disabled = False
        self._cloth_path = None
        self._rest_positions = None    # captured once per instance, right after the FEM view first cooks
        self._art_view = None          # physics-tensor ArticulationView (real-time link poses)
        self._link_idx = {}            # arm -> link index into get_link_transforms
        self._last_jaw_pocket = {"left": None, "right": None}  # for the video overlay
        # Policy/recorder readers run on a different thread. Protect the
        # two-arm snapshot so a read cannot mix arms from adjacent physics ticks.
        self._pin_lag_lock = threading.Lock()
        self._pin_lag = {
            "left": {"active": False, "lag_m": None},
            "right": {"active": False, "lag_m": None},
        }
        self._release_lag = {
            "left": {"lag_m": None, "expires_at": None},
            "right": {"lag_m": None, "expires_at": None},
        }
        self._pin_lag_tick = 0
        self._pin_lag_invalid_warn_n = 0
        # 12 physics ticks at the cloth tasks' 360 Hz = the policy loop's
        # nominal 30 Hz. Converting a CUDA scalar to float synchronizes the
        # device, so do that only as often as the consumer can use a new value.
        self._pin_lag_update_every = 12
        # per-arm: closed flag, pinned node indices, frozen offset, prev anchor pos
        self._arm_state = {
            "left": {"closed": False, "idx": None, "offset": None, "anchor_prev": None},
            "right": {"closed": False, "idx": None, "offset": None, "anchor_prev": None},
        }

    def invalidate(self):
        """Drop the cached view + collider state (call on Workspace/robot reload)."""
        self._reset_runtime()

    # ------------------------------------------------------------------ view

    def _find_deformable_prim(self):
        if self._cloth_path is not None:
            return self._cloth_path
        stage = self.api._stage
        if stage is None:
            return None
        ws = stage.GetPrimAtPath("/Workspace")
        if not ws or not ws.IsValid():
            return None
        # Match the BODY schema (PhysxSurfaceDeformableBodyAPI); the cloth material
        # prim also carries a "...SurfaceDeformableMaterialAPI" and must not match.
        for prim in Usd.PrimRange(ws):
            if any("SurfaceDeformableBody" in s for s in prim.GetAppliedSchemas()):
                self._cloth_path = str(prim.GetPath())
                return self._cloth_path
        return None

    def _ensure_view(self):
        if self._cloth_view is not None:
            return True
        cloth_path = self._find_deformable_prim()
        if cloth_path is None:
            return False
        try:
            import torch
            from omni.physics.tensors import create_simulation_view

            # A private "torch" view (like the standalone). The shared
            # SimulationManager warp view reports 0 sim nodes for this asset.
            # Build the simulation view once; the body view is (re)created each
            # retry until the FEM finishes cooking (max nodes > 0).
            if self._sim_view is None:
                self._sim_view = create_simulation_view("torch")
                self._sim_view.set_subspace_roots("/")
            cloth_view = self._sim_view.create_surface_deformable_body_view(cloth_path)
            if cloth_view is None or cloth_view.count == 0:
                return False  # not on stage yet — retry
            n = cloth_view.max_simulation_nodes_per_body
            if n == 0:
                # FEM not cooked yet (cook is async; cloth_pin_step runs from the
                # very first physics step). Do NOT latch — retry next step until
                # the sim mesh nodes appear.
                return False
            self._torch = torch
            self._cloth_view = cloth_view
            self._body_idx = torch.tensor([0], dtype=torch.int32, device=self._device)
            self._rest_positions = cloth_view.get_simulation_nodal_positions().clone()
            logger.info(f"[ClothPin] surface deformable view ready: {cloth_path} nodes={n}")
            return True
        except Exception as e:
            logger.error(f"[ClothPin] failed to build deformable view: {e!r}")
            self._sim_view = None
            self._cloth_view = None
            return False

    # ------------------------------------------------ gripper collider toggle

    def _disable_gripper_colliders(self):
        if self._colliders_disabled:
            return
        stage = self.api._stage
        robot_root = getattr(self.api, "robot_prim_path", "/genie")
        n = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
            path = str(prim.GetPath())
            if "gripper" not in path:
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            attr = prim.GetAttribute("physics:collisionEnabled")
            if attr and attr.Get():
                attr.Set(False)
                n += 1
        self._colliders_disabled = True
        logger.info(f"[ClothPin] disabled {n} gripper collision mesh(es) under {robot_root}")

    # ----------------------------------------------------------- gripper read

    def _gripper_closed(self, arm):
        finger_names = getattr(self.api, "finger_names", None)
        if not finger_names or arm not in finger_names or not finger_names[arm]:
            return False
        outer_joint = finger_names[arm][0]  # idx32/idx72_gripper_*_outer_joint1
        try:
            val = self.api._get_joint_state_dict().get(outer_joint)
        except Exception:
            return False  # joint state not ready yet (early startup)
        if val is None:
            return False
        return float(val) < GRIPPER_CLOSE_THRESHOLD

    def _jaw_pocket_cfg(self, arm):
        jp = getattr(self.api, "jaw_pocket", None)
        cfg = jp.get(arm) if isinstance(jp, dict) else None
        return cfg if cfg else DEFAULT_JAW_POCKET[arm]

    def _ensure_anchor_links(self):
        """Resolve each gripper's jaw_pocket parent-link index into the
        physics-tensor link transform array (once). Returns True if both
        arms resolved."""
        if self._art_view is not None and len(self._link_idx) == 2:
            return True
        art = self.api._get_articulation()
        if art is None:
            return False
        core = getattr(art, "_articulation_view", None)       # isaacsim core Articulation
        pv = getattr(core, "_physics_view", None) if core is not None else None  # tensors ArticulationView
        if core is None or pv is None:
            return False  # physics view not initialized yet — retry next step
        try:
            names = list(core.body_names)
        except Exception:
            return False
        link_idx = {}
        for arm in ("left", "right"):
            name = self._jaw_pocket_cfg(arm).get("parent_link") or f"gripper_{arm[0]}_base_link"
            if name in names:
                link_idx[arm] = names.index(name)
        if len(link_idx) != 2:
            logger.warning(f"[ClothPin] could not resolve jaw_pocket parent links; body_names sample={names[:6]}")
            return False
        self._art_view = pv
        self._link_idx = link_idx
        logger.info(f"[ClothPin] anchor link indices {link_idx} (jaw_pocket parent links, real-time transforms)")
        return True

    def _anchor_pose(self, arm):
        """Gripper grasp-point world pose (pos, quat xyzw), read from the
        physics-tensor articulation view (real-time). get_world_pose() reads
        USD/Fabric which is STALE for physics-driven links under the GPU pipeline.

        The anchor is the jaw_pocket box center: the parent link's real-time pose,
        offset/rotated by the measured box center + orientation (see
        DEFAULT_JAW_POCKET / robot_cfg "gripper.jaw_pocket")."""
        if not self._ensure_anchor_links():
            return None, None
        T = self._art_view.get_link_transforms()  # (count, max_links, 7): xyz + quat(xyzw)
        row = T[0, self._link_idx[arm]].clone()   # clone: buffer is updated in place
        pos = row[:3]
        quat = row[3:7]
        cfg = self._jaw_pocket_cfg(arm)
        center = cfg.get("center", (0.0, 0.0, 0.0))
        jaw_quat = cfg.get("quat_xyzw", (0.0, 0.0, 0.0, 1.0))
        pos = pos + self._quat_rotate(quat, center)
        quat = self._quat_mul(quat, jaw_quat)
        return pos, quat

    def _anchor_pos(self, arm):
        pos, _ = self._anchor_pose(arm)
        return pos

    def _quat_rotate(self, quat, v):
        """Rotate vector v (tuple/len-3) by quaternion quat (qx,qy,qz,qw)."""
        torch = self._torch
        q = quat
        u = torch.tensor(v, dtype=torch.float32, device=self._device)
        qv = q[:3]
        w = q[3]
        # v' = v + 2*qv x (qv x v + w*v)
        t = 2.0 * torch.linalg.cross(qv, u)
        return u + w * t + torch.linalg.cross(qv, t)

    def _quat_rotate_inv_vecs(self, quat, V):
        """Rotate Nx3 vectors by quat^-1. Used to express cloth nodes in the
        gripper's local frame for grasp-region selection."""
        torch = self._torch
        qv = -quat[:3]
        w = quat[3]
        qvN = qv.unsqueeze(0).expand_as(V)
        t = 2.0 * torch.linalg.cross(qvN, V, dim=1)
        return V + w * t + torch.linalg.cross(qvN, t, dim=1)

    def _quat_mul(self, q1, q2):
        """Compose q1 (xyzw) then q2 (xyzw), both as len-4 tensors/tuples -> xyzw tensor."""
        torch = self._torch
        q2 = torch.tensor(q2, dtype=torch.float32, device=self._device) if not torch.is_tensor(q2) else q2
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([x, y, z, w])

    # --------------------------------------------------------- attach/release

    def _cloth_positions(self):
        return self._cloth_view.get_simulation_nodal_positions()  # (1, N, 3) cuda

    def _attach(self, arm):
        anchor, quat = self._anchor_pose(arm)
        if anchor is None:
            return
        torch = self._torch
        P = self._cloth_positions()
        rel = P[0] - anchor.unsqueeze(0)
        d3 = torch.sqrt((rel ** 2).sum(dim=1))
        local = self._quat_rotate_inv_vecs(quat, rel)

        # Contact gate: a grasp is only allowed if at least one simulation node
        # actually falls inside the jaw_pocket box (the measured physical space
        # between the closed jaws), not just "close to the gripper". This is the
        # inside_count check — it prevents the old failure mode where the gripper
        # closes on empty air (nothing between the fingers) but still pins nodes
        # that merely happen to be nearby.
        cfg = self._jaw_pocket_cfg(arm)
        ext = torch.tensor(cfg.get("half_extents", (0.035, 0.035, 0.018)), dtype=torch.float32, device=self._device)
        inside = (local.abs() < ext.unsqueeze(0)).all(dim=1)
        contact = torch.where(inside)[0]
        if contact.numel() < PIN_MIN_INSIDE_COUNT:
            st = self._arm_state[arm]
            st["_diag_n"] = st.get("_diag_n", 0) + 1
            if st["_diag_n"] % 120 == 1:
                logger.info(
                    f"[ClothPin] {arm} closed but jaw_pocket inside_count={contact.numel()} "
                    f"< {PIN_MIN_INSIDE_COUNT} (nearest={float(d3.min()):.3f}m); not pinned"
                )
            return

        # Once real contact is established, gather a slightly larger local support
        # patch so the soft velocity servo has enough nodes to pull the cloth,
        # without letting the support region itself decide whether an air-grasp
        # succeeds (that's already settled by the box test above). Centered on
        # the contact centroid (not the anchor) so the expansion follows
        # wherever the actual grasped patch is inside the jaw_pocket box rather
        # than uniformly around the box's geometric center; no distance cap, so
        # this always reaches PIN_MIN_SUPPORT_NODES (short only if the mesh has
        # fewer nodes total) — this can pull in a nearby second cloth layer if
        # the fold brings it close to the contact patch — accepted for now.
        idx = contact
        cen = P[0, contact].mean(0)
        d_cen = torch.sqrt(((P[0] - cen.unsqueeze(0)) ** 2).sum(dim=1))
        if idx.numel() < PIN_MIN_SUPPORT_NODES:
            k = min(PIN_MIN_SUPPORT_NODES, int(d_cen.numel()))
            nearest = torch.topk(d_cen, k, largest=False).indices
            idx = torch.unique(torch.cat((idx, nearest)))

        st = self._arm_state[arm]
        st["idx"] = idx
        st["offset"] = (P[0, idx] - anchor.unsqueeze(0)).clone()
        st["anchor_prev"] = anchor
        st["anchor0"] = anchor.clone()              # anchor at attach (for diag)
        st["cen0"] = P[0, idx].mean(0).clone()      # pinned centroid at attach
        with self._pin_lag_lock:
            self._pin_lag[arm] = {"active": True, "lag_m": 0.0}
        logger.info(
            f"[ClothPin] {arm} attached {idx.numel()} cloth node(s) "
            f"inside_count={contact.numel()} nearest={float(d3.min()):.3f}m"
        )
        self._update_debug_sphere(arm, anchor)

    def _inside_count(self, arm, P):
        """Current inside-jaw_pocket cloth node count for `arm`, live (not just
        at the moment of attach) — used by the diagnostic log below."""
        anchor, quat = self._anchor_pose(arm)
        if anchor is None:
            return None
        torch = self._torch
        rel = P[0] - anchor.unsqueeze(0)
        local = self._quat_rotate_inv_vecs(quat, rel)
        cfg = self._jaw_pocket_cfg(arm)
        ext = torch.tensor(cfg.get("half_extents", (0.035, 0.035, 0.018)), dtype=torch.float32, device=self._device)
        return int((local.abs() < ext.unsqueeze(0)).all(dim=1).sum())

    def _log_live_jaw_pocket_counts(self):
        """Refresh both grippers' inside_count every ~0.5s regardless of
        open/closed/pinned state, caching the result on `self._last_jaw_pocket`
        for the video overlay (LocalRecorder). Only runs while that cache is
        actually consumable (recording, or app.log_jaw_pocket_counts for the
        text-log debug aid) so idle benchmark runs pay no extra GPU sync."""
        recording = bool(getattr(self.api, "local_recorder", None) and self.api.local_recorder.is_recording)
        want_log = getattr(self.api, "log_jaw_pocket_counts", False)
        if not (want_log or recording):
            return
        self._live_diag_n = getattr(self, "_live_diag_n", 0) + 1
        if self._live_diag_n % 180 != 1:
            return
        P = self._cloth_positions()
        left = self._inside_count("left", P)
        right = self._inside_count("right", P)
        self._last_jaw_pocket = {"left": left, "right": right}
        if want_log:
            logger.info(f"[ClothPin] live jaw_pocket counts: left={left} right={right}")

    def get_jaw_pocket_counts(self):
        """Latest cached {"left": N, "right": N} inside_count, refreshed by
        `_log_live_jaw_pocket_counts` (~2x/s while recording). Both values are
        None for rigid tasks (no deformable) or before the first refresh."""
        return self._last_jaw_pocket

    def get_pin_lag_snapshot(self):
        """Return a Python-only copy of each arm's cached pin-position lag.

        The physics thread is the sole writer. Readers never touch the CUDA FEM
        tensors, and cannot mutate the controller's internal cache.
        """
        with self._pin_lag_lock:
            return {
                arm: {"active": bool(state["active"]), "lag_m": state["lag_m"]}
                for arm, state in self._pin_lag.items()
            }

    def get_release_lag_snapshot(self):
        """Return per-arm release lag while its two-second display TTL is live."""
        now = time.monotonic()
        with self._pin_lag_lock:
            return {
                arm: {
                    "lag_m": state["lag_m"]
                    if state["expires_at"] is not None and now < state["expires_at"]
                    else None
                }
                for arm, state in self._release_lag.items()
            }

    def _compute_arm_pin_lag(self, arm, positions=None):
        st = self._arm_state[arm]
        idx = st["idx"]
        if idx is None or not idx.numel() or st["offset"] is None:
            return None
        anchor = self._anchor_pos(arm)
        if anchor is None:
            return None
        pos = self._cloth_positions() if positions is None else positions
        target = anchor.unsqueeze(0) + st["offset"]
        lag = (target.mean(dim=0) - pos[0, idx].mean(dim=0)).norm()
        return float(lag.item()) if bool(self._torch.isfinite(lag).item()) else None

    def _release(self, arm):
        st = self._arm_state[arm]
        idx = st["idx"]
        release_lag_m = self._compute_arm_pin_lag(arm)
        if idx is not None and idx.numel():
            vel = self._cloth_view.get_simulation_nodal_velocities().clone()
            vel[0, idx] = 0.0  # zero so released nodes don't fling
            self._cloth_view.set_simulation_nodal_velocities(vel, self._body_idx)
            lag_text = "-" if release_lag_m is None else f"{release_lag_m:.6f}"
            logger.info(f"[ClothPin] {arm} released lag_m={lag_text}")
        st["idx"] = None
        st["offset"] = None
        st["anchor_prev"] = None
        with self._pin_lag_lock:
            self._pin_lag[arm] = {"active": False, "lag_m": None}
            self._release_lag[arm] = {
                "lag_m": release_lag_m,
                "expires_at": time.monotonic() + 2.0 if release_lag_m is not None else None,
            }
        self._hide_debug_sphere(arm)

    def restore_rest_state(self):
        """Snap the cloth FEM nodes back to the pose captured when the view
        first cooked for this scene instance (see _ensure_view). env.reset()
        already re-teleports the robot/rigidbody prims via reset_one_frame,
        but that path never touches deformables (they're neither an
        articulation nor rigidBodyEnabled) — this is the cloth-side
        counterpart, called from api_core._reset_env."""
        if self._cloth_view is None or self._rest_positions is None:
            return
        self._cloth_view.set_simulation_nodal_positions(self._rest_positions.clone(), self._body_idx)
        zeros = self._torch.zeros_like(self._rest_positions)
        self._cloth_view.set_simulation_nodal_velocities(zeros, self._body_idx)
        for arm in ("left", "right"):
            self._release(arm)
        logger.info("[ClothPin] restored cloth to rest state")

    # ----------------------------------------------------- debug visualization

    def _debug_sphere_path(self, arm):
        return f"/World/ClothPinDebug/pin_{arm}"

    def _update_debug_sphere(self, arm, anchor):
        """Show a translucent sphere (radius=PIN_RADIUS) at the pinned region."""
        if not self.api.show_pin_sphere:
            return
        try:
            from pxr import UsdGeom, Gf, Sdf, Vt
            stage = self.api._stage
            path = self._debug_sphere_path(arm)
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                sph = UsdGeom.Sphere.Define(stage, path)
                sph.CreateRadiusAttr(PIN_RADIUS)
                sph.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(1.0, 0.2, 0.2)]))
                sph.CreateDisplayOpacityAttr(Vt.FloatArray([0.35]))
                UsdGeom.Imageable(sph).MakeVisible()
                prim = sph.GetPrim()
            xf = UsdGeom.Xformable(prim)
            a = anchor.detach().cpu().numpy()
            # reset xform ops to a single translate each update
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(float(a[0]), float(a[1]), float(a[2])))
            UsdGeom.Imageable(prim).MakeVisible()
        except Exception as e:
            logger.warning(f"[ClothPin] debug sphere update failed: {e!r}")

    def _hide_debug_sphere(self, arm):
        if not self.api.show_pin_sphere:
            return
        try:
            from pxr import UsdGeom
            prim = self.api._stage.GetPrimAtPath(self._debug_sphere_path(arm))
            if prim and prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
        except Exception:
            pass

    def _pin_servo(self, dt):
        torch = self._torch
        self._pin_lag_tick += 1
        update_pin_lag = self._pin_lag_tick % self._pin_lag_update_every == 0
        active = [a for a in ("left", "right") if self._arm_state[a]["idx"] is not None]
        if not active:
            return
        pos = self._cloth_positions()
        vel = self._cloth_view.get_simulation_nodal_velocities().clone()
        for arm in active:
            st = self._arm_state[arm]
            idx = st["idx"]
            anchor = self._anchor_pos(arm)
            if anchor is None:
                continue
            # Move the debug sphere with the gripper (throttled — USD edits aren't
            # free at 360Hz).
            st["_sph_n"] = st.get("_sph_n", 0) + 1
            if self.api.show_pin_sphere and st["_sph_n"] % 12 == 0:
                self._update_debug_sphere(arm, anchor)
            tgt = anchor.unsqueeze(0) + st["offset"]   # (M,3) where each pinned node should be
            cur = pos[0, idx]
            if update_pin_lag:
                lag_m = self._compute_arm_pin_lag(arm, positions=pos)
                lag_is_finite = lag_m is not None
                with self._pin_lag_lock:
                    self._pin_lag[arm] = {"active": True, "lag_m": lag_m}
                if not lag_is_finite:
                    # Publish an invalid sample so the policy takes its configured
                    # fixed-substep fallback instead of trusting stale/NaN feedback.
                    self._pin_lag_invalid_warn_n += 1
                    if self._pin_lag_invalid_warn_n % 120 == 1:
                        logger.warning(f"[ClothPin] {arm} pin lag is non-finite; adaptive feedback unavailable")
            # Feed-forward velocity: the physical anchor only updates at render-frame
            # boundaries, so the per-step delta is 0 between updates. When it does
            # change, estimate the true velocity over the elapsed steps and hold it.
            anchor_prev = st["anchor_prev"]
            moved = float((anchor - anchor_prev).norm())
            st["_since"] = st.get("_since", 0) + 1
            if moved > 1e-6:
                ff_vel = (anchor - anchor_prev) / (st["_since"] * dt)
                st["ff_vel"] = torch.clamp(ff_vel * PIN_FF, -PIN_VMAX, PIN_VMAX)
                st["anchor_prev"] = anchor
                st["_since"] = 0
            v_ff = st.get("ff_vel")
            if v_ff is None:
                v_ff = torch.zeros(3, device=self._device)

            # Velocity servo: feed-forward (anchor velocity) + P term toward target.
            vel[0, idx] = v_ff.unsqueeze(0) + torch.clamp((tgt - cur) / dt * PIN_GAIN, -PIN_VMAX, PIN_VMAX)
        self._cloth_view.set_simulation_nodal_velocities(vel, self._body_idx)

    # ----------------------------------------------------------------- step

    def step(self, dt):
        # Cloth task? (a surface deformable under /Workspace). Rigid tasks no-op.
        if self._find_deformable_prim() is None:
            return
        # Disable gripper colliders before any grasp (needs only the robot prim),
        # unless the task opts into real gripper-vs-cloth collision.
        if not getattr(self.api, "enable_gripper_collision", False):
            self._disable_gripper_colliders()

        gripper_closed = {a: self._gripper_closed(a) for a in ("left", "right")}

        if not self._ensure_view():
            for arm in ("left", "right"):
                self._arm_state[arm]["closed"] = gripper_closed[arm]
            return

        self._log_live_jaw_pocket_counts()

        for arm in ("left", "right"):
            st = self._arm_state[arm]
            closed = gripper_closed[arm]
            if closed:
                if PIN_ENABLED and st["idx"] is None:
                    # Attempt attach whenever the gripper is closed but nothing is
                    # pinned yet — handles the gripper closing in the air first and
                    # only later reaching the cloth (no fresh open->close edge).
                    self._attach(arm)
                elif not PIN_ENABLED and not st["closed"]:
                    logger.info(f"[ClothPin] {arm} closed (pin disabled)")
            elif not closed and st["closed"]:
                if PIN_ENABLED:
                    self._release(arm)
            st["closed"] = closed

        if PIN_ENABLED:
            self._pin_servo(dt)

    # ----------------------------------------------- shared read (for checker)

    def _get_cloth_nodal_positions(self, indices=None):
        if not self._ensure_view():
            return None
        pos = self._cloth_positions()[0]
        if indices is not None:
            pos = pos[indices]
        return pos
