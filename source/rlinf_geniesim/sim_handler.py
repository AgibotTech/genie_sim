# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""GenieSimHandler: transport-agnostic sim core wrapping a physics engine and
a render bridge. Writes observations into a transport-provided ResultBuffer."""
from __future__ import annotations
from typing import Optional, Sequence
import logging
import sys
import numpy as np
from rlinf_geniesim.perf import PerfLogger

from rlinf_geniesim.comm.base import (
    SimHandler,
    SimDims,
    ResultBuffer,
)
from rlinf_geniesim.render.bridges.base import VecRenderBackend
from rlinf_geniesim.sim_clock import SimClock


_LOGGER = logging.getLogger(__name__)


def _copy_into(dst: np.ndarray, src: np.ndarray) -> None:
    """Copy src into dst, trimming/padding the body/object axis if needed."""
    if dst.shape == src.shape:
        dst[:] = src
        return
    n = min(dst.shape[1], src.shape[1]) if dst.ndim >= 2 else 0
    if dst.ndim >= 2 and n:
        dst[:, :n] = src[:, :n]


class GenieSimHandler(SimHandler):
    def __init__(
        self,
        engine,
        bridge,
        dims: SimDims,
        eval_cfg: Optional[dict] = None,
        reset_settle_time_s: float = 1.0,
        policy_step_dt: Optional[float] = None,
        eval_interval: int = 1,
    ) -> None:
        self._engine = engine
        self._bridge = bridge
        self._dims = dims
        # Post-reset physics settle window.
        #
        # Objects spawn slightly above the table and fall for about a second;
        # the official benchmark's reset free-runs physics over that same
        # window. Without it, a lockstep reset returns an observation of
        # objects still in mid-air.
        #
        # The window runs inside ``reset()``, so a reset env is ready to act
        # the moment the call returns. With the rest-state bake in place the
        # objects start at rest and the window is zero (see
        # ``rlinf_geniesim/reset_bake.py``).
        #
        # ``policy_step_dt`` — the simulated duration of one engine.step() —
        # is computed by the caller from the lockstep control_hz; there is no
        # general way to derive it here. None, or
        # ``reset_settle_time_s <= 0``, disables settling.
        self._reset_settle_time_s = float(reset_settle_time_s)
        self._policy_step_dt = float(policy_step_dt) if policy_step_dt else None
        if self._policy_step_dt and self._policy_step_dt > 0 and self._reset_settle_time_s > 0:
            import math as _math

            self._settle_steps = int(_math.ceil(self._reset_settle_time_s / self._policy_step_dt))
        else:
            self._settle_steps = 0
        # Global sim clock: one authoritative tick threaded poses -> render node
        # -> rendered frame -> obs so a step's rgb/state/reward share a tick.
        # Decision: the tick NEVER resets (reset() must not touch it).
        self._clock = SimClock()
        # Build per-environment benchmark checkers after object metadata is available.
        self._eval = dict(eval_cfg or {})
        self._eval_on = bool(self._eval.get("enabled"))
        self._runners: Optional[list] = None
        self._eval_failed = False
        self._ep_step = np.zeros(int(dims.num_envs), dtype=np.int64)
        # Scoring interval in steps. 1 scores every step (default); 30
        # matches the benchmark.
        #
        # The checker must genuinely not tick, rather than tick and withhold:
        # it latches, so one tick permanently records a momentary condition.
        # Ticking every step gives 30x the chances to catch a transient state
        # and biases success upward.
        self._eval_interval = max(1, int(eval_interval or 1))
        # Per-env cache holding the previous values between scoring steps;
        # cleared on reset.
        self._eval_hold_reward = np.zeros(int(dims.num_envs), dtype=np.float32)
        self._eval_hold_succ = np.zeros(int(dims.num_envs), dtype=bool)
        self._eval_hold_done = np.zeros(int(dims.num_envs), dtype=bool)
        self._eval_hold_stage: list = [[] for _ in range(int(dims.num_envs))]
        # The action each env last executed. While another env settles inside
        # its reset, the envs that are not resetting keep executing it; None
        # until the first step.
        self._last_actions: Optional[np.ndarray] = None
        # Per-env ACTIVE task instance (in-topology-group switching at reset).
        base_inst = int(self._eval.get("instance_id", 0))
        self._env_instance = [base_inst] * int(dims.num_envs)
        self._env_has_reset = np.zeros(int(dims.num_envs), dtype=bool)
        # Per-env ACTIVE instruction id (instructions.json 1-based). It picks
        # the success-checker tree problem{id} — the client picks the SAME id
        # for the language string, so goal text and reward stay welded.
        self._env_instruction = [1] * int(dims.num_envs)
        # Superset instance ids, merged into one scene after asset
        # de-duplication. sim_server defaults to every instance of the subtask
        # when the key is absent, so a non-empty value is the normal case;
        # only an explicit empty list opts out.
        self._superset = list(self._eval.get("superset_instances") or [])
        # Per-instance checker alias table, mapping the prim names the
        # checker uses to the real prim names in the model. Without it the
        # checker finds nothing and silently scores zero. Empty only when
        # sim_server could not resolve them either.
        self._name_alias = dict(self._eval.get("name_alias") or {})
        # Per-env one-shot dedup for the non-member fallback warning below —
        # avoid spamming stderr every reset while a live tool keeps retrying
        # the same bad instance id.
        # Same, for the non-superset in-group-switch guard below (an env
        # asked for an instance outside the current topology group).
        import time as _time

        self._perf_logger = PerfLogger()
        self._step_prof = {"engine": 0.0, "read": 0.0, "eval": 0.0, "drive": 0.0, "n": 0, "t_log": _time.monotonic()}

    # ------------------------------------------------------------------
    # Benchmark evaluation helpers.

    def _schema_has_objects(self) -> bool:
        """True once the engine publishes object bodies (World/Objects/.../entity)
        — the checker shim needs them to resolve obj ids."""
        try:
            schema = self._engine.get_scene_schema()
            names = (
                schema.get("body_link_names") if isinstance(schema, dict) else getattr(schema, "body_link_names", [])
            )
            return any("/Objects/" in str(n) for n in (names or []))
        except Exception:  # noqa: BLE001
            return False

    def _build_runner(self, env_id: int, instance_id: Optional[int] = None, instruction_id: int = 1):
        from rlinf_geniesim.eval.checker_runner import CheckerRunner

        iid = int(self._eval.get("instance_id", 0) if instance_id is None else instance_id)
        return CheckerRunner(
            self._engine,
            env_id=env_id,
            task_name=self._eval["task_name"],
            sub_task_name=self._eval["sub_task_name"],
            instance_id=iid,
            problem_key=f"problem{int(instruction_id)}",
            robot_cfg=self._eval.get("robot_cfg", "G2_omnipicker"),
            alias=self._name_alias.get(str(iid)),
        )

    def _runner_for(self, env_id: int, instance_id: int, instruction_id: int = 1):
        """Build a fresh checker for the episode.

        The tree accumulates progress, so reusing it would retain prior success.
        """
        return self._build_runner(env_id, instance_id, instruction_id)

    def _ensure_runners(self) -> None:
        """Lazily build one checker per env once the scene schema is ready.
        On any failure, disable eval (keep default rewards) rather than crash."""
        if (not self._eval_on) or self._eval_failed or self._runners is not None:
            return
        if not self._schema_has_objects():
            return
        try:
            # instance-aware: each env's runner is built from whatever
            # instance is ALREADY assigned to it in self._env_instance (the
            # base instance for envs that haven't reset yet, or the
            # round-robin/switched instance for envs that have) -- NOT
            # unconditionally the base instance, so the checker's problems.json
            # always matches the physically-active instance in that env.
            self._runners = [
                self._build_runner(i, self._env_instance[i], self._env_instruction[i])
                for i in range(int(self._dims.num_envs))
            ]
            self._ep_step[:] = 0
            self._eval_hold_reward[:] = 0.0
            self._eval_hold_succ[:] = False
            self._eval_hold_done[:] = False
            self._eval_hold_stage = [[] for _ in range(int(self._dims.num_envs))]
            print(
                f"[sim_handler] eval checkers built for "
                f"{self._dims.num_envs} env(s) "
                f"(task={self._eval.get('sub_task_name')})",
                flush=True,
            )
            # One-shot sanity: show each env's checker resolves REAL per-env
            # gripper + object poses (non-zero => shim mapped the engine labels).
            for i, r in enumerate(self._runners):
                # Query by the checker's naming, then map through the alias table to
                # the prim names that exist in the model; the two conventions differ.
                m = getattr(r.shim, "_obj2idx", {})
                pfx = getattr(r.shim, "_obj_prefix", "")
                alias = self._name_alias.get(str(self._env_instance[i])) or {}
                kin = getattr(r.shim, "_kinematic_alias_keys", None) or set()
                oid = None
                if alias:
                    # Prefer a non-kinematic key: a kinematic object is
                    # STRUCTURALLY unresolvable (the engine demotes it to a
                    # static collider before assigning a body index — see
                    # ApiCoreShim's kinematic_alias_keys doc), so picking one
                    # first would print a false "unresolved" pose on an
                    # otherwise fully-healthy alias table.
                    oid = next((k for k in alias if k not in kin), None) or next(iter(alias), None)
                elif pfx:
                    # No alias table for this instance (name_alias resolution
                    # failed upstream, or a bare dev setup with no benchmark
                    # data) — fall back to the prefix-scan strategy in case
                    # prims really are ``i{id}_``-prefixed (kept for
                    # completeness; asset-dedup superset stages never hit
                    # this branch since their alias table is always filled).
                    oid = next((k[len(pfx) :] for k in m if k.startswith(pfx)), None)
                weak = oid is None
                if weak:
                    # LAST resort: an alias-blind key from the raw registry.
                    # This does NOT verify the checker's name mapping at all
                    # — it only proves the engine published SOME body, which
                    # would print a healthy-LOOKING non-zero pose even when
                    # the alias table (or prefix scheme) totally failed to
                    # resolve — the exact false-negative this probe exists to
                    # catch. Flag it loudly instead of silently degrading.
                    oid = next(iter(m), None)
                gpos, _ = r.shim.get_obj_world_pose("/genie/gripper_r_center_link")
                opos = r.shim.get_obj_world_pose(f"/Workspace/Objects/{oid}")[0] if oid else None
                print(
                    f"[sim_handler]   env{i}: gripper_r={tuple(round(v,3) for v in gpos)} "
                    f"obj({oid}{f' @{pfx}' if pfx else ''})="
                    f"{tuple(round(v,3) for v in opos) if opos else None}"
                    + (
                        " — CANARY WEAK: no alias table / prefix match for "
                        "this instance, this only proves SOME body exists, "
                        "NOT that checker name-mapping resolved"
                        if weak
                        else ""
                    ),
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            import traceback

            print(
                f"[sim_handler] eval disabled — checker build failed: {e!r}", file=__import__("sys").stderr, flush=True
            )
            traceback.print_exc()
            self._eval_failed = True
            self._runners = None

    def _write_eval(self, out: ResultBuffer) -> None:
        """Tick each env's checker and write reward/termination/success.
        Falls back to defaults when eval is off/not yet built.

        Only policy steps get here: settle substeps inside ``reset()`` never
        call it, so a checker never sees the transient poses of a settle."""
        self._ensure_runners()
        N = int(self._dims.num_envs)
        if self._runners is None:
            out.rewards[:] = 0.0
            out.terminations[:] = False
            out.truncations[:] = False
            if out.success is not None:
                out.success[:] = False
            return
        self._ep_step += 1
        succ = np.zeros(N, dtype=bool)
        done = np.zeros(N, dtype=bool)
        # Staged reward: completed_steps()/total_steps per env when the
        # runner exposes a real total_steps, falling back to the sparse 0/1
        # success below when it does not.
        reward = np.zeros(N, dtype=np.float32)
        # One stage-score list per env, appended alongside the tick loop
        # rather than in a second pass, so a failed tick's entry is
        # unambiguously `[]` rather than whatever a half-broken runner would
        # snapshot afterwards.
        stage_scores: list = []
        for i, r in enumerate(self._runners):
            # Match benchmark sampling: exclude step zero and hold scores
            # between intervals, including an episode's final partial interval.
            if self._eval_interval > 1 and not (self._ep_step[i] != 0 and self._ep_step[i] % self._eval_interval == 0):
                reward[i] = self._eval_hold_reward[i]
                succ[i] = self._eval_hold_succ[i]
                done[i] = self._eval_hold_done[i]
                stage_scores.append(list(self._eval_hold_stage[i]))
                continue
            ss: list = []
            ts = None
            try:
                d, s = r.step(int(self._ep_step[i]))
                try:
                    ss = r.stage_scores()
                except Exception:  # noqa: BLE001 - snapshot is best-effort
                    ss = []
                ts = getattr(r, "total_steps", None)
            except Exception as e:  # noqa: BLE001
                print(f"[sim_handler] checker env {i} tick error: {e!r}", file=__import__("sys").stderr, flush=True)
                d, s = False, False
            done[i] = d
            succ[i] = s
            stage_scores.append(ss)
            if ts:
                try:
                    reward[i] = min(1.0, int(r.completed_steps()) / ts)
                except Exception:  # noqa: BLE001 - degrade to sparse, not crash
                    reward[i] = float(s)
            else:
                reward[i] = float(s)  # sparse success reward (0/1)
            # Scoring step: cache these values for the interval that follows.
            self._eval_hold_reward[i] = reward[i]
            self._eval_hold_succ[i] = succ[i]
            self._eval_hold_done[i] = done[i]
            self._eval_hold_stage[i] = list(ss)
        out.rewards[:] = reward
        out.terminations[:] = done  # episode ended (success OR timeout)
        out.truncations[:] = done & ~succ  # ended without success = timeout
        if out.success is not None:
            out.success[:] = succ
        # Plain python-list attribute (not a ResultBuffer field): the shm
        # transport copies only the fixed ndarray fields and ignores this;
        # the zmq wire ships it as an optional header side-field (see _wire.py).
        out.stage_scores = stage_scores

    def handshake(self) -> tuple[SimDims, dict]:
        return self._dims, self._engine.get_scene_schema()

    def on_serve_start(self, frame_region: Optional[object] = None) -> None:
        # Wire the render bridge to the transport's output: point its image
        # output at the Frame SHM the ResultBuffer views (so out.images
        # reflects what it renders), and bind the engine's scene schema.
        # The bridge requires BOTH before the first update_poses/render()
        # (it raises otherwise), so this hook is what makes the new pluggable
        # path actually drive a real renderer.
        if frame_region is not None:
            self._bridge.attach_shm(frame_region, layout=None)
        schema = self._engine.get_scene_schema()
        self._bridge.bind_scene_schema(schema)
        # Explicit PHYSICS-READY marker (the engine must already answer schema/
        # state queries to get here — handshake ran before this hook).
        names = (schema.get("joint_names") if isinstance(schema, dict) else getattr(schema, "joint_names", None)) or []
        bodies = (
            schema.get("body_link_names") if isinstance(schema, dict) else getattr(schema, "body_link_names", None)
        ) or []
        print(
            f"[sim_handler] PHYSICS READY: {self._dims.num_envs} env(s), "
            f"{len(names)} joint(s), {len(bodies)} bodies",
            flush=True,
        )
        self._warn_if_bodies_truncated(bodies)

    def _warn_if_bodies_truncated(self, bodies: Sequence[str]) -> None:
        """Warn once when the transport buffer cannot hold the schema's bodies.

        Per-step copies truncate to fit. Object bodies follow robot links, so an
        undersized buffer can omit the poses needed by evaluation.
        """
        n_rows = int(self._dims.num_bodies)
        over = [nm for i, nm in enumerate(bodies) if i >= n_rows and "/Objects/" in nm]
        if not over:
            return
        print(
            f"[sim_handler] WARNING: body_poses has only {n_rows} rows "
            f"while the schema has {len(bodies)} bodies — {len(over)} "
            f"objects fall outside and their poses are always zero on the "
            f"wire (_copy_into truncates silently): {over[:5]}"
            f"{' …' if len(over) > 5 else ''}\n"
            f"    Affected: teleop tape object trajectories, flyaway "
            f"criteria, and the evaluation's AABB checks.\n"
            f"    The shm layout is fixed when the task is built, so this "
            f"needs a restart: --num-bodies="
            f"{len(bodies)}",
            file=sys.stderr,
            flush=True,
        )

    def warmup_physics(self, ticks: int = 2) -> None:
        """Build lazy checkers and run warmup physics steps before serving.

        Hold current joint positions for at least one step, or ``ticks`` if larger.
        Initialization failures are logged and do not prevent serving.
        """
        import time as _time  # noqa: PLC0415

        t0 = _time.monotonic()
        try:
            self._ensure_runners()
            state = np.asarray(self._engine.get_state())
            if state.ndim == 2 and state.shape[1] >= 2:
                hold = np.ascontiguousarray(state[:, 0::2]).astype(np.float32)
                for _ in range(max(1, int(ticks))):
                    self._engine.step(hold)
        except Exception as exc:  # noqa: BLE001
            print(
                f"[sim_handler] WARN: warm-up failed "
                f"({type(exc).__name__}: {exc}); serving anyway, the first "
                f"client step will be slower",
                flush=True,
            )
            return
        print(
            f"[sim_handler] PHYSICS WARM: {(_time.monotonic()-t0)*1000:.0f}ms "
            f"({self._dims.num_envs} env(s); the first step's checker "
            f"construction and engine initialisation were paid before "
            f"serving)",
            flush=True,
        )

    def warmup_render(self, timeout_s: float = 300.0) -> None:
        """Publish current poses and wait for an initial frame before serving.

        Retry dropped frames until the timeout. A timeout is logged and serving
        continues; physics-only backends skip this warmup.
        """
        import time as _time  # noqa: PLC0415

        if getattr(self._bridge, "BACKEND_NAME", "") == "none":
            print("[sim_handler] RENDER: none (physics-only) — no warmup needed", flush=True)
            return
        t0 = _time.monotonic()
        while True:
            tick = self._clock.advance()
            body_names = None
            _get_names = getattr(self._engine, "get_body_link_names", None)
            if _get_names is not None:
                body_names = _get_names()
            self._bridge.update_poses(
                body_poses=self._engine.get_body_poses(),
                sim_time=self._engine.get_sim_time(),
                body_names=body_names,
                tick=tick,
            )
            res = self._bridge.render()
            dt = _time.monotonic() - t0
            if not getattr(res, "dropped", False):
                print(f"[sim_handler] RENDER READY: first frame rendered " f"({dt:.1f}s after physics)", flush=True)
                return
            if dt > timeout_s:
                print(
                    f"[sim_handler] WARN: render warmup still dropping frames "
                    f"after {dt:.0f}s — serving anyway (renderer may catch up "
                    f"on the first client step)",
                    flush=True,
                )
                return

    def _settle_hold_for_others(self, measured: np.ndarray) -> np.ndarray:
        """The action rows a settle feeds the envs that are NOT resetting:
        the action each one last executed.

        Holding them at their measured joint positions instead rewrites their
        command for the whole settle window -- a position-controlled gripper
        closed on an object gets a target equal to where it already is, its
        squeeze drops to zero, and the object can slip out while the env is
        not even being reset. Before any step there is no last action and the
        measured pose is the only sensible hold. ``_settle`` replaces the
        resetting rows with their own post-reset pose either way.
        """
        last = self._last_actions
        if last is None or last.shape != measured.shape:
            return measured
        return np.array(last, dtype=np.float32, copy=True)

    def _settle(self, ids, hold: np.ndarray) -> None:
        """Settle reset environments at their measured poses; others hold their last action.

        Settling advances physics without checker ticks, episode steps, rendering,
        or clock updates. The reset response renders once after this window."""
        if self._settle_steps <= 0:
            return
        actions = np.array(self._settle_hold_for_others(hold), dtype=np.float32, copy=True)
        actions[ids] = hold[ids]
        for _ in range(self._settle_steps):
            # Settling frames are never consumed; render only the final response.
            self._engine_step(actions, need_images=False)

    def _engine_step(self, actions: np.ndarray, need_images: bool) -> None:
        """``engine.step`` with the image request attached where supported.

        ``VecPhysicsBackend.step(actions)`` is the ABC signature and the
        newton/upstream backends still have exactly that -- their render is a
        separate free-running process, so there is nothing for them to skip.
        Passing the keyword unconditionally would break all of them with a
        TypeError, so this asks first. Checked against the FUNCTION rather than
        with try/except: a TypeError raised from deep inside a real step() is
        not a signature mismatch, and swallowing it would step twice."""
        fn = getattr(self._engine, "step", None)
        if getattr(self, "_engine_takes_need_images", None) is None:
            try:
                import inspect  # noqa: PLC0415

                self._engine_takes_need_images = "need_images" in inspect.signature(fn).parameters
            except (TypeError, ValueError):  # builtins / C callables
                self._engine_takes_need_images = False
        if self._engine_takes_need_images:
            fn(actions, need_images=need_images)
        else:
            fn(actions)

    def step(self, actions: np.ndarray, out: ResultBuffer, need_images: bool = True, pre_copy_callback=None) -> None:
        import time as _time

        _t0 = _time.monotonic()
        _rsnap = dict(getattr(self._bridge, "_render_prof", {}) or {})
        _render_result = None
        self._engine_step(actions, need_images)
        self._last_actions = np.array(actions, dtype=np.float32, copy=True)
        _t1 = _time.monotonic()
        # Advance the global clock exactly once per step. This tick is the
        # barrier TARGET threaded to the render node via update_poses(tick=...)
        # and stamped onto obs.
        tick = self._clock.advance()
        # Publish poses before state reads and evaluation so rendering can overlap host work.
        body_poses = self._engine.get_body_poses()
        sim_time = self._engine.get_sim_time()  # fetch once, reuse
        if out.global_tick is not None:
            out.global_tick[...] = tick
        body_names = None
        _get_names = getattr(self._engine, "get_body_link_names", None)
        if _get_names is not None:
            body_names = _get_names()
        self._bridge.update_poses(
            body_poses=body_poses, sim_time=sim_time, body_names=body_names, tick=tick, will_collect=need_images
        )
        # --- overlapped with the node's render from here ---
        out.states[:] = self._engine.get_state()
        _copy_into(out.joint_states, self._engine.get_joint_states())
        _copy_into(out.body_poses, body_poses)
        _copy_into(out.object_poses, self._engine.get_object_poses())
        out.sim_time[:] = sim_time
        _t2 = _time.monotonic()
        # Evaluate rewards, termination and success for each environment.
        self._write_eval(out)
        _t3 = _time.monotonic()
        # With need_images false, only the render barrier and frame fetch are
        # skipped; pose streaming and the engine and checker reads above ran
        # unconditionally, exactly as on a rendering step.
        #
        # This does not affect the next need_images=True step: the render node
        # renders every tick anyway, and the barrier waits for a frame newer
        # than the last consumed one, so skipping merely leaves the high-water
        # mark in place.
        #
        # ``pre_copy_callback``, supplied by the transport, runs between the
        # barrier and the fetch, when every non-image field is final and the
        # transport can pack while the GPU frame is still in flight. A bridge
        # that only implements a whole render() has no such gap and the
        # callback runs straight after. Either way it runs exactly once.
        if need_images:
            # getattr, not a bare attribute read: the tests' duck-typed fake
            # bridges don't subclass VecRenderBackend and have neither
            # method — they must land on the render() path, not AttributeError.
            if getattr(type(self._bridge), "render_barrier", None) not in (None, VecRenderBackend.render_barrier):
                self._bridge.render_barrier()
                if pre_copy_callback is not None:
                    pre_copy_callback()
                _render_result = self._bridge.render_copy()
            else:
                _render_result = self._bridge.render()
                if pre_copy_callback is not None:
                    pre_copy_callback()
        elif pre_copy_callback is not None:
            pre_copy_callback()
        _t4 = _time.monotonic()
        self._log_timing("step", _t4 - _t0, _t1 - _t0, _t2 - _t1, _t3 - _t2, _t4 - _t3, _rsnap, _render_result)
        # Report the host-side timing breakdown at DEBUG every five seconds.
        p = self._step_prof
        p["engine"] += _t1 - _t0
        p["read"] += _t2 - _t1
        p["eval"] += _t3 - _t2
        p["drive"] += _t4 - _t3
        p["n"] += 1
        if _t4 - p["t_log"] >= 5.0 and p["n"]:
            n = p["n"]
            extra = ""
            bp = getattr(self._bridge, "_render_prof", None)
            if bp and bp.get("n"):
                bn = bp["n"]
                extra = (
                    f" | render: pub={bp['pub']/bn*1000:.1f} "
                    f"barrier={bp['barrier']/bn*1000:.1f} "
                    f"copy={bp['copy']/bn*1000:.1f}ms"
                )
                for k in ("pub", "barrier", "copy"):
                    bp[k] = 0.0
                bp["n"] = 0
            _LOGGER.debug(
                f"[handler-prof] {n} step(s): engine={p['engine']/n*1000:.1f} "
                f"read={p['read']/n*1000:.1f} eval={p['eval']/n*1000:.1f} "
                f"drive+render={p['drive']/n*1000:.1f}ms{extra}"
            )
            for k in ("engine", "read", "eval", "drive"):
                p[k] = 0.0
            p["n"] = 0
            p["t_log"] = _t4

    def _log_timing(self, event, total, engine, read, evaluate, drive, snapshot=None, render_result=None):
        # Fused engine.step includes physics AND rendering. These CPU spans
        # overlap node execution and must not be summed as GPU timings.
        fused = self._engine is self._bridge
        prof = getattr(self._bridge, "_render_prof", {}) or {}
        delta = lambda k: (prof.get(k, 0) - (snapshot or {}).get(k, 0)) * 1000
        tick = getattr(self._bridge, "_current_tick", None)
        shm = getattr(self._bridge, "_tiled_shm", None)
        rendered = getattr(shm, "rendered_tick", None)
        self._perf_logger.log(
            event=event,
            engine_scope="physics+render" if fused else "physics",
            total_ms=total * 1000,
            engine_ms=engine * 1000,
            send_ms=getattr(self._engine, "_last_send_ms", 0),
            wait_ms=getattr(self._engine, "_last_wait_ms", 0),
            read_ms=read * 1000,
            eval_ms=evaluate * 1000,
            drive_ms=drive * 1000,
            render_pub_ms=delta("pub") if snapshot is not None else 0,
            render_barrier_ms=delta("barrier") if snapshot is not None else 0,
            render_copy_ms=delta("copy") if snapshot is not None else 0,
            dropped=int(bool((render_result or {}).get("dropped", False))),
            frame_age_ticks=(max(0, int(tick) - int(rendered)) if tick is not None and rendered is not None else ""),
            stall_total=getattr(self._bridge, "_barrier_stall_total", 0),
        )

    def _drive_and_render(self, out: ResultBuffer, tick: int, body_poses: np.ndarray, sim_time: np.ndarray) -> None:
        """Stamp obs with ``tick``, push the latest poses to the renderer, and
        render one frame into Frame SHM. ``body_names`` = the engine's live
        /tf_render frame names (absolute USD paths) so each pose maps to the
        right prim/anchor; falls back to schema. Shared by ``step`` and ``reset``
        so reset re-renders a fresh frame (not a stale pre-reset image)."""
        if out.global_tick is not None:
            out.global_tick[...] = tick
        body_names = None
        _get_names = getattr(self._engine, "get_body_link_names", None)
        if _get_names is not None:
            body_names = _get_names()
        self._bridge.update_poses(body_poses=body_poses, sim_time=sim_time, body_names=body_names, tick=tick)
        self._bridge.render()

    def _fill_reset_response(self, out: ResultBuffer) -> None:
        """reset()'s single "respond now" tail — fill ``out`` with the
        engine's CURRENT snapshot, render exactly ONE fresh frame, zero
        reward/termination/truncation/success. Used EXACTLY ONCE per
        reset() call, after the settle (if any) has run, so the engine's
        current state IS the settled one and the whole reset+settle costs
        exactly ONE render/clock-advance."""
        out.states[:] = self._engine.get_state()
        # Refresh all state arrays before deriving post-reset observations and hold targets.
        _copy_into(out.joint_states, self._engine.get_joint_states())
        _copy_into(out.body_poses, self._engine.get_body_poses())
        _copy_into(out.object_poses, self._engine.get_object_poses())
        out.sim_time[:] = self._engine.get_sim_time()
        # Advance the monotonic clock once and render a fresh frame for the settled state.
        reset_tick = self._clock.advance()
        self._drive_and_render(out, reset_tick, self._engine.get_body_poses(), self._engine.get_sim_time())
        # reset() already rebuilt the checkers and cleared ep_step per
        # instance. Rebuilding unconditionally here would overwrite them with
        # the base instance's runner and score non-base envs against the wrong
        # problems.json.
        out.rewards[:] = 0.0
        out.terminations[:] = False
        out.truncations[:] = False
        if out.success is not None:
            out.success[:] = False

    def reset(self, mask: np.ndarray, out: ResultBuffer, seeds=None, instances=None, instructions=None) -> None:
        """Reset the masked envs; every one of them is ready to act when this
        returns.

        Any settle (``reset_settle_time_s``) runs inside this call, holding
        the reset envs at their post-reset pose while the others keep their
        last action (see ``_settle``); with the rest-state bake in place the
        objects start at rest and there is no settle at all. The fused
        engine advances no other env's physics during a reset; Newton's
        keeps pace with the wall clock, so the others move on only by the
        ~0.03 s the reset takes."""
        import time as _time

        _reset_start = _time.monotonic()
        env_ids = [i for i, m in enumerate(np.asarray(mask).ravel()) if bool(m)]
        if not env_ids:
            self._fill_reset_response(out)
            return
        ids = env_ids

        # seeds / instances: (N,) int64 aligned with envs; forward only the
        # reset envs' entries (engine treats -1 / missing as no-op).
        def _pick(arr):
            if arr is None:
                return None
            a = np.asarray(arr).ravel()
            return [int(a[i]) for i in ids]

        seed_list = _pick(seeds)
        inst_list = _pick(instances)
        instr_list = _pick(instructions)
        # Superset resolution; an explicit instance wins. Without a superset
        # inst_list passes through unchanged. With one:
        #   - absent / negative  -> retain the current instance after its first reset
        #   - first reset        -> round-robin over the superset
        #   - a superset member  -> use the caller's value
        #   - a real non-member   -> ValueError. Substituting a round-robin
        #     instance instead would be a silent wrong answer: the env runs
        #     another instance's scene while the caller's bookkeeping still
        #     says otherwise, and the merged scene has no objects for the
        #     requested instance anyway.
        if self._superset:
            defaults = [
                self._env_instance[e] if self._env_has_reset[e] else self._superset[e % len(self._superset)]
                for e in ids
            ]
            if inst_list is None:
                inst_list = defaults
            else:
                resolved = []
                bad = [
                    (int(e), int(inst_list[k]))
                    for k, e in enumerate(ids)
                    if inst_list[k] >= 0 and inst_list[k] not in self._superset
                ]
                if bad:
                    raise ValueError(
                        "reset requested instance(s) outside the loaded "
                        "superset: "
                        + ", ".join(f"env {e} -> instance {v}" for e, v in bad)
                        + f"; loaded superset is {self._superset}. The built "
                        "stage does not carry those instances' objects, so "
                        "the switch cannot happen — reload the server onto a "
                        "task-init whose superset covers them (or pick a "
                        "different instance). Refusing rather than silently "
                        "running a different instance."
                    )
                for k, e in enumerate(ids):
                    v = inst_list[k]
                    resolved.append(v if v >= 0 else defaults[k])
                inst_list = resolved
        elif self._name_alias and inst_list is not None:
            # Non-superset within-group switching, same rule: asking for an
            # instance outside this topology group finds nothing on either
            # side, since neither the engine's instance poses nor the checker
            # has an alias for it. Warning and proceeding would hand the caller
            # an env that never switched while its label says it did.
            bad = [(int(e), int(v)) for e, v in zip(ids, inst_list) if v >= 0 and str(int(v)) not in self._name_alias]
            if bad:
                raise ValueError(
                    "reset requested instance(s) outside the loaded topology "
                    "group: " + ", ".join(f"env {e} -> instance {v}" for e, v in bad) + f"; loaded group is "
                    f"{sorted(int(k) for k in self._name_alias)}. Object "
                    "poses would not switch and the success checker has no "
                    "name alias for those instances — reload the server onto "
                    "the target instance instead of resetting into it."
                )
        self._engine.reset(env_ids=env_ids, seeds=seed_list, instances=inst_list)
        # The pose the reset envs hold through the settle, read before
        # anything else can move them.
        hold_now = None
        if self._settle_steps > 0 or self._last_actions is not None:
            hold_now = np.ascontiguousarray(self._engine.get_joint_states()[:, :, 0], dtype=np.float32)
        # A reset env's "last action" is now its start pose: a neighbour's
        # settle before the next step must hold it there, not drive it back
        # to the previous episode's last command.
        if self._last_actions is not None and hold_now is not None and self._last_actions.shape == hold_now.shape:
            self._last_actions[ids] = hold_now[ids]
        # Per-episode bookkeeping: a reset env starts a new checker episode.
        # The decision tree is per instance, so switching instance swaps in
        # that instance's runner, built lazily and cached by (env, instance).
        # Independent of settling: building a runner only parses json.
        for k, e in enumerate(ids):
            self._env_has_reset[e] = True
            if inst_list is not None and inst_list[k] >= 0:
                self._env_instance[e] = int(inst_list[k])
            if instr_list is not None and instr_list[k] >= 1:
                self._env_instruction[e] = int(instr_list[k])
            self._ep_step[e] = 0
            # The hold cache must be cleared too, or the first interval-1
            # steps of the new episode re-send the previous episode's reward
            # and success, silently reporting success at episode start.
            self._eval_hold_reward[e] = 0.0
            self._eval_hold_succ[e] = False
            self._eval_hold_done[e] = False
            self._eval_hold_stage[e] = []
            if self._runners is not None:
                try:
                    self._runners[e] = self._runner_for(e, self._env_instance[e], self._env_instruction[e])
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[sim_handler] checker swap env {e} -> instance "
                        f"{self._env_instance[e]} problem"
                        f"{self._env_instruction[e]} failed: {exc!r}; "
                        f"falling back to problem1",
                        file=__import__("sys").stderr,
                        flush=True,
                    )
                    try:
                        self._env_instruction[e] = 1
                        self._runners[e] = self._runner_for(e, self._env_instance[e], 1)
                    except Exception as exc2:  # noqa: BLE001
                        print(
                            f"[sim_handler] fallback checker env {e} also " f"failed: {exc2!r}",
                            file=__import__("sys").stderr,
                            flush=True,
                        )
        table_materials = self._eval.get("table_materials") or {}
        material_payload = {}
        mesh_rel = table_materials.get("mesh_rel")
        if mesh_rel:
            by_instance = table_materials.get("by_instance") or {}
            for e in ids:
                material_payload[str(int(e))] = {mesh_rel: by_instance[str(self._env_instance[e])]}
        if material_payload:
            self._bridge.on_reset(ids, seeds=seed_list, material_bindings=material_payload)
        else:
            # seed_list is aligned with env_ids by index (see _pick above) —
            # the render-side generalization channel (lights, etc.) needs it
            # to pick which env's seed sources this reset's shared
            # grid-wide values. ``None`` when the caller passed no ``seeds``
            # at all; ``IsaacTiledBackend.on_reset`` treats that the same as
            # "nothing to generalize this reset" rather than raising.
            self._bridge.on_reset(ids, seeds=seed_list)

        if hold_now is not None and self._settle_steps > 0:
            self._settle(ids, hold_now)
        self._recover_non_finite_reset(ids, env_ids, seed_list, inst_list)

        self._fill_reset_response(out)
        self._log_timing("reset", _time.monotonic() - _reset_start, 0, 0, 0, 0)

    def _recover_non_finite_reset(self, ids, env_ids, seed_list, inst_list) -> None:
        """Before handing a reset back, confirm the state is finite; reset
        once more if it is not.

        A reset is an idempotent teleport, so retrying costs nothing
        semantically, and a diverged solver is exactly the kind of fault that
        clearing cross-step carryover recovers from.

        If the retry also fails, report but do not alter: send the NaN through
        so the client's observation guard reports it with full context.
        Substituting zeros would be worse — plausible-looking bad data.
        """
        js = self._engine.get_joint_states()
        if np.isfinite(js).all():
            return
        import sys as _sys

        n_bad = int((~np.isfinite(js)).sum())
        print(
            f"[sim_handler] state after reset holds {n_bad} non-finite "
            f"values (envs {ids}); resetting once more, since a diverged "
            f"solver usually recovers once carryover is cleared",
            file=_sys.stderr,
            flush=True,
        )
        try:
            self._engine.reset(env_ids=env_ids, seeds=seed_list, instances=inst_list)
            if self._settle_steps > 0:
                self._settle(ids, np.ascontiguousarray(self._engine.get_joint_states()[:, :, 0], dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 — a failed retry must not
            # swallow another reset
            print(f"[sim_handler] retry reset failed: {exc!r}", file=_sys.stderr, flush=True)
            return
        js2 = self._engine.get_joint_states()
        if np.isfinite(js2).all():
            print("[sim_handler] retry reset recovered; state is finite", file=_sys.stderr, flush=True)
        else:
            print(
                f"[sim_handler] retry reset still leaves "
                f"{int((~np.isfinite(js2)).sum())} non-finite values; "
                f"sending as-is for the client's observation guard to "
                f"report. This task's physics is no longer trustworthy — "
                f"rebuild the task rather than resetting again.",
                file=_sys.stderr,
                flush=True,
            )

    def renderer_alive(self) -> bool:
        """False once the render subprocess (e.g. Isaac Sim GUI) has exited, so
        the server's watchdog can tear the whole sim down with it. True for
        renderers with no watchable subprocess."""
        chk = getattr(self._bridge, "render_proc_alive", None)
        try:
            return bool(chk()) if chk else True
        except Exception:  # noqa: BLE001
            return True

    def close(self, keep_bridge: bool = False) -> None:
        """Tear down the engine (and, by default, the render bridge too).

        ``keep_bridge=True`` closes ONLY the engine — the render subprocess
        stays alive and the bridge object remains usable — for the
        render hot-swap reload path: the caller
        pulls the bridge out of the old handler, cold-rebuilds a new engine,
        then calls ``bridge.reload_stage(...)`` to re-point the SAME render
        node at the new stage in place. The default (``keep_bridge=False``)
        path is byte-for-byte the pre-existing behavior.
        """
        self._perf_logger.close()
        targets = (self._engine,) if keep_bridge else (self._bridge, self._engine)
        for obj in targets:
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["GenieSimHandler"]
