# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import numpy as np

from geniesim_benchmark.plugins.ader.action.common_actions import (
    EvaluateAction,
    ActionBase,
    ActionEvent,
)
from geniesim_benchmark.plugins.logger import Logger

logger = Logger()


class Erased(EvaluateAction):
    """Fraction of a chalk/ink surface that has been wiped off.

    Format: ``"<prim>|<threshold>"`` -- passes once the erased fraction
    reaches ``threshold`` (0..1).

    ``<prim>`` may be either end of the erasure mechanism; which attribute to
    read is auto-detected, so a scene can be judged from whichever prim it
    actually authors:

    * the **eraser's** ``entity`` prim, e.g.
      ``benchmark_blackboard_eraser_000/entity`` -- reads the scalar
      ``eraser:erasedFraction`` that the asset's OmniScripting behaviour
      (``erase_runtime.EraserController._metrics``) rewrites on EVERY physics
      step as ``1 - remaining/total``. One double per read; preferred.
    * the **board's** ``Chalk`` mesh, e.g.
      ``benchmark_whiteboard_000/entity/WritingSurface/Chalk`` -- reads the
      per-cell ``chalk:active`` array and counts it. This is the ground truth
      the scalar is derived from, but it is ~100k ints per read, so use it only
      to cross-check a scene whose eraser does not publish the scalar.

    Both are authored through the USD API by the behaviour script, so
    ``get_prim_attribute`` (a USD-layer read) sees them; neither is a
    rigid-body transform, so there is no PhysX-tensor staleness concern.

    No debounce: cells are only ever cleared within an episode (the controller
    re-arms them in ``reset()`` on play/stop), so the fraction is monotonic and
    a single frame over the threshold is not a glitch.

    SCORING IS PARTIAL ON PURPOSE. ``progress_info["SCORE"]`` is rewritten with
    the live fraction on every frame, not just 0/1 at the end, so a wipe that
    is cut short still earns what it achieved: half the cells gone is 0.5, and
    that 0.5 survives if the eraser is dropped or the step times out. This is
    why the write has to happen inside ``update`` -- ``ActionSetWaitAny`` stops
    its remaining children the moment another one finishes (here the Onfloor
    guard), and a stopped child never sees another FINISHED event, so anything
    reported only there would be lost. ``threshold`` therefore only decides
    when the step is declared done; the score tracks the fraction regardless.
    """

    _FRACTION_ATTR = "eraser:erasedFraction"
    _CELLS_ATTR = "chalk:active"

    def __init__(self, env, prim, threshold=0.95):
        super().__init__(env)
        self._holder_name, self._prim = self.placeholder_sparser(prim)
        self.threshold = float(threshold)
        self._done_flag = False
        self._last_fraction = 0.0
        self._source = None  # resolved on first successful read

    @property
    def prim(self) -> str:
        if self._holder_name:
            return getattr(self, self._prim)
        return self._prim

    def _read_fraction(self, path):
        value = self.api_core.get_prim_attribute(path, self._FRACTION_ATTR, default=None)
        if value is not None:
            return float(value), self._FRACTION_ATTR
        cells = self.api_core.get_prim_attribute(path, self._CELLS_ATTR, default=None)
        if cells is not None:
            cells = np.asarray(cells)
            if cells.size:
                return 1.0 - float(np.count_nonzero(cells)) / float(cells.size), self._CELLS_ATTR
        return None, None

    def update(self, delta_time: float) -> float:
        if not self.is_running() or self._done_flag:
            return super().update(delta_time)

        path = self._analyze_obj_name(self.prim)
        fraction, source = self._read_fraction(path)
        if fraction is None:
            logger.warning(
                f"[Erased] {path} publishes neither {self._FRACTION_ATTR} nor "
                f"{self._CELLS_ATTR}; cannot judge erasure"
            )
            return super().update(delta_time)

        if source != self._source:
            logger.info(f"[Erased] reading {source} off {path}")
            self._source = source
        self._last_fraction = fraction

        logger.info(
            f"[Erased] prim={self.prim} erased={fraction:.4f} "
            f"threshold={self.threshold:.4f} passed={fraction >= self.threshold}"
        )
        if fraction >= self.threshold:
            self._done_flag = True
        return super().update(delta_time)

    def _is_done(self) -> bool:
        return self._done_flag

    def update_progress(self):
        # Live fraction, every frame: this is the partial credit, and it is the
        # value that survives an abort (see the class docstring).
        self.progress_info["SCORE"] = round(self._last_fraction, 4)
        if self._done_flag:
            self.progress_info["STATUS"] = "SUCCESS"

    def handle_action_event(self, action: ActionBase, event: ActionEvent) -> None:
        logger.info("Action [Erased] evt: %d" % (event.value))
        # No SCORE here on FINISHED: update_progress owns it, and it runs right
        # after _update_state in the same tick, so writing 1 here would only be
        # overwritten -- and would be wrong for a threshold below 1.0.
