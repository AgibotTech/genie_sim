# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import json
import os

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from geniesim_benchmark.plugins.logger import Logger
import geniesim_benchmark.utils.system_utils as system_utils

logger = Logger()


def _nearest_dist(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """Distance from every row of `a` to the closest row of `b`.

    Blocked so the pairwise term stays `chunk x chunk` -- a scribbling policy
    can push the ink mesh to tens of thousands of vertices.
    """
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), np.inf)
    b2 = np.einsum("ij,ij->i", b, b)
    out = np.full(len(a), np.inf)
    for i in range(0, len(a), chunk):
        ablk = a[i : i + chunk]
        a2 = np.einsum("ij,ij->i", ablk, ablk)
        best = np.full(len(ablk), np.inf)
        for j in range(0, len(b), chunk):
            bblk = b[j : j + chunk]
            d2 = a2[:, None] + b2[None, j : j + len(bblk)] - 2.0 * (ablk @ bblk.T)
            np.minimum(best, d2.min(axis=1), out=best)
        out[i : i + len(ablk)] = best
    return np.sqrt(np.maximum(out, 0.0))


def _gauss_weight(d: np.ndarray, sigma: float) -> np.ndarray:
    """Per-point credit that decays smoothly instead of hitting a hard edge.

    ``exp(-d^2 / (2 sigma^2))``: 1 at d=0, never exactly 0. Used for
    precision, where the quantity being judged (how far ink drifted off the
    band) is naturally bounded and a smooth tail does not accumulate: it
    only ever applies to the ink actually laid down.
    """
    if sigma <= 0:
        return (d <= 0).astype(np.float64)
    return np.exp(-np.square(d) / (2.0 * sigma * sigma))


def _ramp(d: np.ndarray, full: float, zero: float) -> np.ndarray:
    """Per-point credit: 1 inside `full`, 0 past `zero`, linear between.

    Used for coverage, where a Gaussian's smooth tail is the wrong shape: a
    real policy's dashing repeats a gap every cycle along the *whole*
    stroke, not once, so any per-point credit still reachable several mm
    from the nearest dash gets added up over and over and the average stays
    high even when 90% of the stroke never got ink (confirmed empirically:
    sigma=3mm scored 0.52-0.59 at a true 10%-duty dash pattern, sigma=1.5mm
    still 0.37-0.44). A ramp's hard zero does not have that long tail, so it
    does not accumulate the same way -- yet a single occasional gap on one
    stroke of an otherwise-good multi-stroke letter still degrades smoothly
    (checked 0-25mm: 1.00 -> ~0.81), because that smoothness comes from
    *proportional averaging* over the whole stroke, not from the per-point
    function's shape.
    """
    if zero <= full:
        return (d <= full).astype(np.float64)
    return np.clip((zero - d) / (zero - full), 0.0, 1.0)


def _resample_by_arclength(points: np.ndarray, step: float) -> np.ndarray:
    """Even samples along a polyline, so a per-point ratio means per-length.

    The authored centrelines are not evenly sampled: their segments hold
    [1, 93, 3, 2, 26, 3] points for arc lengths of [0, 15.79, 0.17, 0.08,
    3.96, 0.17] cm, so counting raw points would weight a 0.8 mm pen-lift
    artefact the same as the 15.8 cm main stroke.
    """
    if len(points) < 2:
        return points
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return points[:1]
    n = max(2, int(total / step) + 1)
    q = np.linspace(0.0, total, n)
    return np.stack([np.interp(q, cum, points[:, k]) for k in range(points.shape[1])], axis=1)


class InkOverlap(EvaluateAction):
    """Evaluate how well generated ink traces the writing guide's pattern.

    Two one-sided scores, combined into one continuous total:

    * **precision** -- how well the ink sits on the guide band. Catches
      scribbling, and a trace that is offset or mis-scaled.
    * **coverage** -- how much of each stroke the ink actually reaches.
      Catches a partial trace, a dropped stroke, or a single dot.

    Each distance is turned into credit. Precision uses a Gaussian,
    ``exp(-d^2 / (2 sigma^2))``, which never hits exactly 0: the quantity it
    grades (how far ink drifted off the band) only ever applies to ink that
    was actually laid down, so a smooth tail does not accumulate into
    anything. Coverage uses a ramp with a hard zero past `cov_zero_dist`
    instead: a real policy's dashing repeats a gap every cycle along the
    *whole* stroke, so a Gaussian's smooth tail -- reachable several mm from
    the nearest dash -- gets summed over and over across every cycle, and
    the average stays high even when most of the stroke never got ink.
    Checked against the real failure case this was tuned on (two strokes,
    170mm + 30mm, dash=2mm/gap=18mm, a true 10%-duty dashed-but-correctly-
    shaped letter): a Gaussian scored 0.37-0.44 coverage at sigma=1.5-3mm and
    could not be pushed below about 0.15-0.2 at any sigma, because the
    "exactly on a dash" points alone already average to that floor. The ramp
    scores the same case at 0.28.

    ``sigma_precision`` and the ramp's ``cov_full_dist``/``cov_zero_dist``
    serve the same role the old single ramp's full/zero radii did for
    precision and coverage separately: precision judges *where* ink landed
    (a tight tolerance -- a trace drifted off the band is still wrong even
    if it never breaks), coverage judges *how much pen-lift is tolerable* (a
    tight-but-defined cutoff -- a real policy stutters, and a few-mm gap
    between dashes is not the same failure as a dropped stroke).

    Coverage is averaged **per stroke, then equally across strokes** -- not
    over all guide points pooled together. Pooling lets a long finished
    stroke dilute a short dropped one (the letter "B" is mostly its spine; a
    missing bowl barely moves a pooled average). Scoring each stroke on its
    own and averaging those scores keeps a one-stroke miss visible. This is
    also what keeps the ramp's hard zero from reproducing the old cliff: a
    single occasional gap on one stroke of an otherwise-good multi-stroke
    letter still degrades smoothly as the gap grows (checked 0-25mm: quality
    1.00 -> ~0.78), because that smoothness comes from the gap's *share of
    one stroke's arc length* growing gradually, not from the per-point
    function's shape.

    The two scores combine as ``(precision * coverage) ** quality_exponent``,
    the same shape the old ramp-based version used -- not the harmonic mean a
    first draft of this tried. The harmonic mean looked like a natural fit
    for "both sides need to be reasonable," but empirically it is too
    forgiving here: when precision sits near its ceiling (a real trace that
    stays on the band, dashed or not), ``2PR/(P+R)`` stayed above 0.85 even
    with a gap covering 60% of a stroke, because a per-point Gaussian average
    over a gap's distance profile decays only polynomially in gap length, not
    exponentially -- most points near a gap's edges are still close to *some*
    ink. The product does not have that inflation: it stays bounded by
    whichever side is worse, so a genuinely torn stroke still fails while a
    dashed-but-shaped one is graded continuously instead of cliffed.

    There is no separate "unbroken" gate. The old all-Gaussian design needed
    one because a Gaussian's long tail could not tell a few small gaps from
    one big tear at the same average distance; scoring each stroke on its
    own with a hard-zero ramp already reflects how broken that stroke is,
    continuously, so a second pass/fail switch on top of it would just
    reproduce a cliff this design avoids a different way.

    `pass_score`, `cov_full_dist`/`cov_zero_dist`, and the half-widths were
    first checked against synthetic single-stroke gap sweeps and one real
    dashed-letter case (see the verification plots generated alongside that
    change), not against a real policy rollout. They were then widened once:
    a real G2_omnipicker replay (write_uppercase_letter_with_pen_g2_omnipicker
    instance 1, letter B) that traced a recognizable, continuous B -- just
    consistently offset to one side of the centreline rather than on it --
    scored precision 0.35 / coverage 0.44 / quality 0.11 at the original
    sigma_precision=0.001 / cov_zero_dist=0. Both sub-scores were being
    dominated by sub-2mm positional noise, not a genuinely missing or
    scribbled stroke, which is what each tolerance is supposed to catch.
    Checked that a mean (arithmetic or harmonic) would have masked that
    distinction instead of fixing it: AM-GM-HM ordering makes any mean at
    least as forgiving as the harmonic mean already rejected above, so
    raising the two per-point tolerances -- not the combiner -- is what
    changed. sigma_precision=0.002 raised that sample's precision to 0.66-0.69;
    adding a cov_zero_dist=0.0025 ramp (cov_full_dist stays 0, so inside
    ink_half_width is still full credit) is a *small* reopening of the exact
    tail `cov_zero_dist`'s hard zero was introduced to close two paragraphs
    up -- 2.5mm, not the several-mm-reaching-every-cycle tail a Gaussian gave
    -- so a genuinely dropped stroke (tens of mm) still clears to ~0. Expect
    to keep revisiting all of these once more real rollouts exist.

    Args:
        env: The simulation environment.
        ink_prim: Mesh the pen's script appends to, e.g. ``Ink/strokes``. It
            does not exist until the first ink is laid down.
        guide_prim: The writing guide, used only for logging; the geometry comes
            from the instance's ``writing_guide.json``.
        pass_score: Product score at or above which the step passes.
        sigma_precision: Gaussian width for ink-to-band distance. Tight: this
            is a position/scale check, not a dashing-gap tolerance (see
            cov_zero_dist for that) -- but loosened from 0.001 to 0.002 after
            a real rollout showed sub-2mm positional noise alone could sink
            precision to 0.35 on an otherwise-recognizable trace.
        cov_full_dist: Guide-to-ink distance (after the ink half-width) below
            which a point scores full coverage credit.
        cov_zero_dist: Beyond this guide-to-ink distance, a point scores 0.
            A short ramp (not Gaussian): coverage must not accumulate a long
            tail across a repeating dash pattern the way a Gaussian would.
            0.0025 (2.5mm) tolerates the kind of small, consistent one-sided
            offset a real noisy rollout produces without reopening that
            problem -- a genuinely missing stroke is tens of mm, not ~2mm,
            past the cutoff.
        guide_half_width: Half the guide band's physical width, subtracted
            from raw point distance so a point inside the band reads as 0.
        ink_half_width: Coverage-side distance padding. Set to half the
            asset's ink-mesh sampling pitch (``MIN_POINT_SPACING`` / 2), not
            the ink's physical width: a perfectly continuous trace must still
            read as fully covered between consecutive ink samples. Wider
            than that (e.g. the true physical half-width) reopens the
            dashing-inflation problem `cov_zero_dist`'s hard zero exists to
            close, because it pads every dash's covered footprint outward.
        quality_exponent: Shapes partial credit; 1.0 is the raw product.
        resample_step: Guide centreline resampling pitch in metres.

    Limitations:
        * Guide points come from the authored world transform in
          ``writing_guide.json``; they do not follow the ``paper`` rigid body,
          so a paper knocked out of place invalidates the comparison frame.
        * Judges where ink landed, not stroke order or stroke count.
        * The half-width subtraction approximates "distance to a filled
          region" from point samples; it assumes the region is locally
          straight at that scale, which holds for a 1.5 mm pen stroke but
          would not for a sharp corner at the same radius.
    """

    GUIDE_SIDECAR = "writing_guide.json"

    def __init__(
        self,
        env,
        ink_prim: str,
        guide_prim: str,
        pass_score: float = 0.85,
        sigma_precision: float = 0.002,
        cov_full_dist: float = 0.0,
        cov_zero_dist: float = 0.0025,
        guide_half_width: float = 0.00075,
        ink_half_width: float = 0.0004,
        quality_exponent: float = 1.2,
        resample_step: float = 0.001,
    ):
        super().__init__(env)
        self.ink_prim = ink_prim
        self.guide_prim = guide_prim
        self.pass_score = float(pass_score)
        self.sigma_precision = float(sigma_precision)
        self.cov_full_dist = float(cov_full_dist)
        self.cov_zero_dist = float(cov_zero_dist)
        self.guide_half_width = float(guide_half_width)
        self.ink_half_width = float(ink_half_width)
        self.quality_exponent = float(quality_exponent)
        self.resample_step = float(resample_step)

        self._done_flag = False
        self._precision = 0.0
        self._coverage = 0.0
        self._score = 0.0
        self._best_score = 0.0
        self._stroke_coverages = []
        self._guide_segs = []
        self._guide_pts = self._load_guide_points()

        # Ink already on the mesh belongs to an earlier episode: the asset's
        # script has no clear-ink path (no on_stop, and it reloads existing
        # points from USD), and appends through the generatedGeometry:*End
        # cursors. Only vertices past this mark are ours to score.
        self._ink_offset = len(self._read_ink_points(slice_new=False))
        logger.info(
            f"[InkOverlap] guide={len(self._guide_pts) if self._guide_pts is not None else 'MISSING'} pts "
            f"over {len(self._guide_segs)} strokes, pre-existing ink={self._ink_offset} verts, "
            f"sigma_precision={self.sigma_precision * 1000:.2f}mm, "
            f"cov {self.cov_full_dist * 1000:.2f}->{self.cov_zero_dist * 1000:.2f}mm, "
            f"score=(precision*coverage)^{self.quality_exponent:.2f}, pass at >={self.pass_score:.2f}"
        )

    def _instance_dir(self) -> str:
        cfg = getattr(self.env, "init_task_config", {}) or {}
        # The rlinf checker runner builds init_task_config without a "scene"
        # key, and tasks run outside llm_task mode have no sub_task_name.
        sub_task_name = cfg.get("sub_task_name") or ""
        instance_id = (cfg.get("scene") or {}).get("scene_instance_id", 0)
        return os.path.join(
            system_utils.benchmark_conf_path(),
            "llm_task",
            sub_task_name,
            str(instance_id),
        )

    def _load_guide_points(self):
        """Guide centreline as evenly spaced world points, or None."""
        path = os.path.join(self._instance_dir(), self.GUIDE_SIDECAR)
        try:
            with open(path, "r") as f:
                guide = json.load(f)
        except Exception as e:
            # Scoring 0 here would be indistinguishable from a policy that drew
            # nothing, so say so loudly and let the step time out instead.
            logger.error(f"[InkOverlap] cannot read guide sidecar {path} ({e!r}); this step cannot be scored")
            return None

        origin = np.asarray(guide["origin_world_m"], dtype=np.float64)
        u_axis = np.asarray(guide["u_world"], dtype=np.float64)
        v_axis = np.asarray(guide["v_world"], dtype=np.float64)

        segments = []
        for uv in guide["stroke_centerlines_uv_m"]:
            uv = np.asarray(uv, dtype=np.float64)
            if len(uv) == 0:
                continue
            world = origin + uv[:, 0:1] * u_axis + uv[:, 1:2] * v_axis
            resampled = _resample_by_arclength(world, self.resample_step)
            if len(resampled) >= 2:
                # Kept per-stroke: coverage is scored per stroke, then
                # averaged equally, so a short dropped stroke is not diluted
                # by a long finished one.
                self._guide_segs.append(resampled)
            segments.append(resampled)
        if not segments:
            logger.error(f"[InkOverlap] guide sidecar {path} holds no centerlines")
            return None
        return np.vstack(segments)

    def _read_ink_points(self, slice_new: bool = True) -> np.ndarray:
        """This episode's ink vertices in world space; empty if none yet."""
        pts = self.api_core.get_prim_attribute(self._analyze_obj_name(self.ink_prim), "points", default=None)
        if pts is None or len(pts) == 0:
            return np.empty((0, 3))
        pts = np.asarray(pts, dtype=np.float64)
        return pts[self._ink_offset :] if slice_new else pts

    def update(self, delta_time: float) -> float:
        if self._done_flag or self._guide_pts is None:
            return super().update(delta_time)

        ink = self._read_ink_points()
        if len(ink) == 0:
            self._precision = self._coverage = self._score = 0.0
            self._stroke_coverages = [0.0] * len(self._guide_segs)
            return super().update(delta_time)

        # Precision: distance from each ink point to the guide *band*
        # (centreline distance minus the band's own half-width), weighted by
        # a tight Gaussian -- this is a position/scale check, not a gap
        # tolerance.
        d_ink_to_band = np.maximum(_nearest_dist(ink, self._guide_pts) - self.guide_half_width, 0.0)
        self._precision = float(_gauss_weight(d_ink_to_band, self.sigma_precision).mean())

        # Coverage: for each stroke, distance from its guide points to the
        # nearest ink *blob* (nearest-vertex distance minus the ink's own
        # half-width), weighted by a hard-zero ramp -- not a Gaussian -- so a
        # repeating dash pattern cannot accumulate a long tail into a high
        # average. Averaged per stroke, then equally across strokes, so a
        # dropped short stroke is not hidden by a long one.
        stroke_scores = []
        for seg in self._guide_segs:
            d_guide_to_ink = np.maximum(_nearest_dist(seg, ink) - self.ink_half_width, 0.0)
            stroke_scores.append(float(_ramp(d_guide_to_ink, self.cov_full_dist, self.cov_zero_dist).mean()))
        self._stroke_coverages = stroke_scores
        self._coverage = float(np.mean(stroke_scores)) if stroke_scores else 0.0

        self._score = (self._precision * self._coverage) ** self.quality_exponent
        self._best_score = max(self._best_score, self._score)

        if self._score >= self.pass_score:
            self._done_flag = True
            logger.info(
                f"[InkOverlap] SUCCESS -- score={self._score:.3f} "
                f"(precision={self._precision:.3f} coverage={self._coverage:.3f}) "
                f"over {len(ink)} ink verts"
            )
        else:
            logger.info(
                f"[InkOverlap] score={self._score:.3f}/{self.pass_score:.2f} "
                f"precision={self._precision:.3f} coverage={self._coverage:.3f} "
                f"min_stroke={min(self._stroke_coverages) if self._stroke_coverages else float('nan'):.3f} "
                f"ink_verts={len(ink)}"
            )

        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"
        self.progress_info["DETAILS"] = {
            "score": round(self._best_score, 4),
            "precision": round(self._precision, 4),
            "coverage": round(self._coverage, 4),
            "stroke_coverages": [round(v, 4) for v in self._stroke_coverages],
            "min_stroke_coverage": round(min(self._stroke_coverages), 4) if self._stroke_coverages else None,
            "pass_score": self.pass_score,
            "guide_unavailable": self._guide_pts is None,
        }

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info(f"Action [InkOverlap] {self.ink_prim} vs {self.guide_prim}, evt: {event.value}")

        if event == ActionEvent.FINISHED:
            # A pass scores a clean 1 so the E2E rollup -- which tests the last
            # step for ``== 1.0`` (output_system/eval_utils.py) -- can fire. A
            # failure reports the score it did reach, so a near miss is
            # distinguishable from a blank sheet.
            self.progress_info["SCORE"] = 1 if self._done_flag else round(self._best_score, 4)
