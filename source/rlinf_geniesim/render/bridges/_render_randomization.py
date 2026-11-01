# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""Shared render randomization preparation and readback validation.

The bridge scans scene material candidates, resolves camera paths and records
what the render subprocess applied for each reset."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from ._hotswap_protocol import poll_randomize_ack, write_randomize_cmd
from ...randomize import RandomizationPlan, RandomizationSpec, Sampler
from ...randomize.material import scan_material_candidates


class RenderReloadError(RuntimeError):
    """A render reload or randomization request could not be confirmed.

    Callers must not continue with an unverified observation; they own recovery or
    cold rebuild after this exception."""


class RenderRandomizationMixin:
    """See the module docstring."""

    def _scan_material_candidates_once(self, material_spec: dict) -> dict:
        """Cache material candidates from the loaded foreground and background layers.

        Candidate paths use the scopes under which those layers are mounted in the
        assembled stage. The cached mapping is invalidated when the manifest changes."""
        if self._material_candidates is not None:
            return self._material_candidates
        try:
            manifest = json.loads(Path(self._stage_manifest).read_text())
        except Exception as e:  # noqa: BLE001 — config/env problem, not an
            # asset limitation: never cache this as "no candidates" — see
            # this function's docstring above.
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: material randomization is "
                f"enabled but the stage manifest {self._stage_manifest!r} "
                f"could not be read/parsed ({e!r}) — this is a config/"
                f"environment problem, not an asset limitation, and must "
                f"not be silently treated as 'no material candidates this "
                f"scene'"
            ) from e

        base_dir = (
            Path(manifest["base_path"]) if manifest.get("base_path") else Path(self._stage_manifest).resolve().parent
        )

        def _resolve(p: str) -> str:
            if not p:
                return ""
            pp = Path(p)
            return str(pp if pp.is_absolute() else (base_dir / pp).resolve())

        mode = material_spec.get("mode", "ml_prefix")
        keywords = material_spec.get("keywords") or []
        candidates: dict = {}
        attempted = 0
        failures: list = []
        for key, scope in (("foreground_usda", "World"), ("background_usda", "Background")):
            configured = manifest.get(key, "")
            if not configured:
                continue  # this task simply never configured this source — not a failure, just absent
            usd_path = _resolve(configured)
            attempted += 1
            if not Path(usd_path).is_file():
                failures.append(f"{key}={usd_path!r} (file not found)")
                continue
            try:
                found = scan_material_candidates(usd_path, mesh_scope=scope, mode=mode, keywords=keywords)
            except Exception as e:  # noqa: BLE001 — recorded, judged below;
                # one bad source shouldn't by itself sink the other source.
                failures.append(f"{key}={usd_path!r} ({e!r})")
                continue
            candidates.update(found)

        if attempted and len(failures) == attempted:
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: material randomization is "
                f"enabled but every configured source failed to scan: "
                f"{'; '.join(failures)} — this is a config/environment "
                f"problem, not an asset limitation (an asset with no ml_ "
                f"variants scans successfully and just returns no "
                f"candidates)"
            )
        if failures:
            print(
                f"[{type(self).__name__}] WARN: material scan — "
                f"{len(failures)}/{attempted} configured source(s) failed "
                f"to scan ({'; '.join(failures)}); continuing with the "
                f"source(s) that scanned successfully",
                flush=True,
            )

        if not candidates:
            print(
                f"[{type(self).__name__}] material randomization enabled "
                f"but no multi-material mesh found in this scene (asset "
                f"has no ml_ variants — an asset limitation, not a config "
                f"error, see docs/GENERALIZATION_DESIGN.md"
                f"); staying a quiet no-op",
                flush=True,
            )
        self._material_candidates = candidates
        return candidates

    def _camera_pose_candidates(self) -> list:
        """Read camera identifiers from the current manifest.

        Use manifest camera paths so sampled perturbations address the same cameras the
        render node loaded."""
        try:
            manifest = json.loads(Path(self._stage_manifest).read_text())
        except Exception as e:  # noqa: BLE001 — config/env problem, see
            # docstring above and _scan_material_candidates_once's parallel
            # branch: never silently treat this as "no camera candidates".
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: camera_pose randomization "
                f"is enabled but the stage manifest "
                f"{self._stage_manifest!r} could not be read/parsed ({e!r}) "
                f"— this is a config/environment problem, not 'no cameras "
                f"in this scene', and must not be silently treated as 'no "
                f"camera pose candidates this reset'"
            ) from e
        cams = manifest.get("cameras") or []
        return [c["path"] for c in cams if not c.get("is_free_cam") and c.get("path")]

    def _send_render_randomization(
        self, env_ids: Sequence[int], seeds: Optional[Sequence[int]] = None, material_bindings: Optional[dict] = None
    ) -> Optional[dict]:
        """First half of a reset's render-side generalization: sample this
        reset's values and write ``randomize_cmd.json``. Returns what
        ``_await_render_randomization`` needs to collect and check the ack, or
        ``None`` when there is nothing to send. See ``IsaacTiledBackend.on_reset``
        for the full contract; the fused backend calls the two halves around
        its own reset command so the values land before the reset frame."""
        lights_spec = (self._render_randomization or {}).get("lights") or {}
        material_spec = (self._render_randomization or {}).get("material") or {}
        camera_pose_spec = (self._render_randomization or {}).get("camera_pose") or {}
        lights_on = bool(lights_spec.get("enable", False))
        # Lights are one shared value for the whole grid: changing them on
        # a partial reset would also change them for other envs whose
        # episode is still running, so lights only change on a reset that
        # covers every env.
        if lights_on and len({int(e) for e in env_ids}) < self._num_envs:
            lights_on = False
        material_on = bool(material_spec.get("enable", False))
        camera_pose_on = bool(camera_pose_spec.get("enable", False))
        authored = {str(e): dict(b) for e, b in (material_bindings or {}).items() if b}
        if not lights_on and not material_on and not camera_pose_on and not authored:
            return None
        if seeds is None and not authored:
            return None
        seeds_map = {int(e): int(s) for e, s in zip(env_ids, seeds if seeds is not None else [-1] * len(env_ids))}
        if not any(s >= 0 for s in seeds_map.values()) and not authored:
            return None  # all -1 (the verbatim-replay contract) -- produces no command

        material_candidates = self._scan_material_candidates_once(material_spec) if material_on else {}
        camera_pose_candidates = self._camera_pose_candidates() if camera_pose_on else []
        render_spec: dict = {}
        if lights_on:
            render_spec["lights"] = lights_spec
        if material_on:
            render_spec["material"] = material_spec
        if camera_pose_on:
            render_spec["camera_pose"] = camera_pose_spec
        spec = RandomizationSpec(physics={}, render=render_spec)
        plans = Sampler(spec).sample(
            seeds_map, material_candidates=material_candidates, camera_pose_candidates=camera_pose_candidates
        )

        render_payload: dict = {}
        for plan in plans.values():
            if plan.render.get("lights"):
                render_payload["lights"] = plan.render["lights"]
                break  # one shared value for the whole grid -- any one plan has the whole thing
        # Establish the selected instance's authored baseline on EVERY reset,
        # then let enabled randomization override the meshes it selects.
        material_payload = {e: dict(b) for e, b in authored.items()}
        for env_id, plan in plans.items():
            if plan.render.get("material"):
                material_payload.setdefault(str(env_id), {}).update(plan.render["material"])
        for env_id, bindings in material_payload.items():
            e = int(env_id)
            if e not in plans:
                plans[e] = RandomizationPlan(seed=seeds_map.get(e, -1), env_id=e)
            plans[e].render["material"] = dict(bindings)
        if material_payload:
            render_payload["material"] = material_payload
        camera_pose_payload = {
            str(env_id): plan.render["camera_pose"] for env_id, plan in plans.items() if plan.render.get("camera_pose")
        }
        if camera_pose_payload:
            render_payload["camera_pose"] = camera_pose_payload
        if not render_payload:
            return None  # none of the three dimensions drew anything to change -> nothing to do

        control_dir = self._reload_control_dir
        if not control_dir:
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: no cmd/ack control "
                f"directory cached — start() must launch the render node "
                f"before on_reset() can send a randomize command"
            )
        if not self.render_proc_alive():
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: render subprocess is not " f"alive — cannot send randomize command"
            )

        self._randomize_generation += 1
        generation = self._randomize_generation
        write_randomize_cmd(control_dir, generation, render_payload)
        return {
            "generation": generation,
            "control_dir": control_dir,
            "authored": authored,
            "render_payload": render_payload,
            "plans": plans,
            "material_payload": material_payload,
            "camera_pose_payload": camera_pose_payload,
        }

    def _await_render_randomization(self, pending: dict) -> None:
        """Second half: wait for ``randomize_ack.json`` and check every
        dimension's readback (see ``IsaacTiledBackend.on_reset``)."""
        generation = pending["generation"]
        control_dir = pending["control_dir"]
        authored = pending["authored"]
        render_payload = pending["render_payload"]
        plans = pending["plans"]
        material_payload = pending["material_payload"]
        camera_pose_payload = pending["camera_pose_payload"]
        try:
            ack = poll_randomize_ack(
                control_dir, generation, self._randomize_ack_timeout_s, proc_alive=self.render_proc_alive
            )
        except TimeoutError as e:
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: timed out after "
                f"{self._randomize_ack_timeout_s}s waiting for randomize "
                f"ack (generation={generation}): {e}"
            ) from e
        except RuntimeError as e:
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: render subprocess died "
                f"while waiting for randomize ack (generation={generation}): "
                f"{e}"
            ) from e

        if not ack.get("ok"):
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: render node randomize "
                f"failed (generation={generation}): {ack.get('error')!r}"
            )

        # The ack's readback — ok=True only means dispatch did not raise,
        # it does not mean the binding actually landed.
        #
        # ``applied`` being completely absent is NOT treated as "old
        # format, pass quietly" — the bridge already knows how many
        # material bindings/camera pose perturbations it sent this time
        # (``material_payload``/``camera_pose_payload``, no need to trust
        # the ack for that). If content really was sent (at least one env
        # has a binding or a perturbation) and the ack does not even have
        # an ``applied`` key, that is not "protocol version compatibility"
        # — it means the render node simply did not report a readback for
        # this command (wrong node version, or whoever handled that
        # branch forgot to collect the return value), and that must be a
        # loud failure. Only when this command genuinely sent neither (a
        # lights-only command, or material/camera_pose each with
        # ``requested == 0``) is the absence truly normal.
        applied_info = ack.get("applied")
        self._last_render_plans = plans
        if applied_info is None:
            total_material_requested = sum(len(b) for b in material_payload.values())
            total_camera_pose_requested = sum(len(b) for b in camera_pose_payload.values())
            if total_material_requested > 0 or total_camera_pose_requested > 0:
                raise RenderReloadError(
                    f"{type(self).__name__}.on_reset: sent "
                    f"{total_material_requested} material binding(s) across "
                    f"{len(material_payload)} env(s) and "
                    f"{total_camera_pose_requested} camera pose "
                    f"perturbation(s) across {len(camera_pose_payload)} "
                    f"env(s) (generation={generation}) but the ack has no "
                    f"'applied' readback field at all — the render node did "
                    f"not report whether they actually landed (version "
                    f"mismatch, or it forgot to collect the readback for "
                    f"this command); a randomization command must never be "
                    f"acked as done without proof"
                )
            return
        light_readback = applied_info.get("lights")
        material_info = applied_info.get("material") or {}
        camera_pose_info = applied_info.get("camera_pose") or {}
        # The mirror image of the material half's same discipline: the
        # ``applied`` key itself is present (not entirely absent — already
        # checked above), but it specifically has no ``"lights"`` sub-key,
        # even though this command really did send lights — the same
        # category of "the render node did not report" problem, so this
        # must not just quietly skip the three-tier check below merely
        # because ``light_readback is None``.
        if light_readback is None and render_payload.get("lights"):
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: sent a lights "
                f"randomization command but the ack's 'applied' field has "
                f"no 'lights' entry (generation={generation}) — the render "
                f"node did not report whether it landed (version mismatch, "
                f"or it forgot to collect the readback for this command)"
            )

        fail_env: Optional[int] = None
        fail_requested = 0
        fail_first_missing: Optional[str] = None
        fail_count = 0
        fail_cam_env: Optional[int] = None
        fail_cam_requested = 0
        fail_cam_first_missing: Optional[str] = None
        fail_cam_count = 0
        for env_id, plan in plans.items():
            if light_readback is not None and plan.render.get("lights"):
                plan.render["lights_readback"] = dict(light_readback)

            bindings = material_payload.get(str(env_id))
            if bindings:
                requested = len(bindings)
                info = material_info.get(str(env_id)) or {}
                applied_n = int(info.get("applied", 0))
                first_missing = info.get("first_missing_mesh")
                plan.render["material_readback"] = {
                    "requested": requested,
                    "applied": applied_n,
                    "first_missing_mesh": first_missing,
                }
                if applied_n == 0:
                    fail_count += 1
                    if fail_env is None:
                        fail_env = env_id
                        fail_requested = requested
                        fail_first_missing = first_missing or (f"/World/envs/env_{env_id}/{next(iter(bindings))}")
                elif applied_n < requested:
                    print(
                        f"[{type(self).__name__}] WARN: material "
                        f"randomization for env {env_id} only applied "
                        f"{applied_n}/{requested} binding(s) (generation="
                        f"{generation}) — some mesh/material prims were not "
                        f"found on the assembled multi-env stage; this env's "
                        f"material randomization is only PARTIALLY in effect "
                        f"(first missing mesh: {first_missing!r})",
                        flush=True,
                    )

            perturbations = camera_pose_payload.get(str(env_id))
            if perturbations:
                cam_requested = len(perturbations)
                cam_info = camera_pose_info.get(str(env_id)) or {}
                cam_applied_n = int(cam_info.get("applied", 0))
                cam_first_missing = cam_info.get("first_missing_camera")
                plan.render["camera_pose_readback"] = {
                    "requested": cam_requested,
                    "applied": cam_applied_n,
                    "first_missing_camera": cam_first_missing,
                }
                if cam_applied_n == 0:
                    fail_cam_count += 1
                    if fail_cam_env is None:
                        fail_cam_env = env_id
                        fail_cam_requested = cam_requested
                        fail_cam_first_missing = cam_first_missing or (f"env_{env_id}:{next(iter(perturbations))}")
                elif cam_applied_n < cam_requested:
                    print(
                        f"[{type(self).__name__}] WARN: camera pose "
                        f"randomization for env {env_id} only applied "
                        f"{cam_applied_n}/{cam_requested} camera(s) "
                        f"(generation={generation}) — some camera prims "
                        f"were not found on the assembled multi-env stage, "
                        f"or a stronger USD opinion overrode the pose "
                        f"write; this env's camera pose randomization is "
                        f"only PARTIALLY in effect (first missing/"
                        f"mismatched camera: {cam_first_missing!r})",
                        flush=True,
                    )

        # Lights get the same three-tier test material has: the earlier
        # readback only recorded the ack, it did not judge it. Lights are
        # a single shared value for the whole grid, so there is no need to
        # loop per env the way material does: ``light_readback`` is
        # already ``{"applied": int, "requested": int}``.
        # ``requested`` here is taken straight from the number the ack
        # reports (unlike material, the bridge itself has no other
        # independent source to check how many light prims actually exist
        # — that is a fact the render node only knows once it caches the
        # baseline), but when ``requested==0`` this naturally never enters
        # any branch below, so the red line is unchanged.
        if light_readback is not None:
            lights_requested = int(light_readback.get("requested", 0))
            lights_applied = int(light_readback.get("applied", 0))
            if lights_requested > 0 and lights_applied == 0:
                raise RenderReloadError(
                    f"{type(self).__name__}.on_reset: lights randomization "
                    f"requested {lights_requested} light prim(s) but the "
                    f"render node applied 0 (generation={generation}) — "
                    f"scene lighting this reset is not the planned one at "
                    f"all (dome/sun prims missing on the assembled stage?)"
                )
            elif lights_applied < lights_requested:
                print(
                    f"[{type(self).__name__}] WARN: lights randomization "
                    f"only applied {lights_applied}/{lights_requested} "
                    f"light prim(s) (generation={generation}) — this "
                    f"reset's lighting is only PARTIALLY as planned",
                    flush=True,
                )

        if fail_env is not None:
            extra = f" ({fail_count - 1} more env(s) also applied 0)" if fail_count > 1 else ""
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: material randomization "
                f"for env {fail_env} requested {fail_requested} binding(s) "
                f"but the render node applied 0{extra} (generation="
                f"{generation}) — the most likely cause is a mount-point "
                f"mismatch between the bridge's scanned mesh paths and the "
                f"assembled multi-env stage; first requested mesh: "
                f"{fail_first_missing!r}"
            )

        if fail_cam_env is not None:
            extra = f" ({fail_cam_count - 1} more env(s) also applied 0)" if fail_cam_count > 1 else ""
            raise RenderReloadError(
                f"{type(self).__name__}.on_reset: camera pose randomization "
                f"for env {fail_cam_env} requested {fail_cam_requested} "
                f"camera(s) but the render node applied 0{extra} (generation="
                f"{generation}) — the most likely cause is a mount-point "
                f"mismatch between the bridge's camera path list and the "
                f"assembled multi-env stage, or a stronger USD opinion "
                f"overriding the pose write; first requested camera: "
                f"{fail_cam_first_missing!r}"
            )

        for env_id in authored:
            info = material_info.get(env_id) or {}
            if int(info.get("applied", 0)) != len(material_payload[env_id]):
                raise RenderReloadError(f"instance material bindings only partially applied for env {env_id}: {info}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
