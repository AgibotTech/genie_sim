# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Read actual solved MJWarp hand/pan and pan/static contact constraints.

The chef MJXPBD path keeps its real SolverMuJoCo at ``sim._adapter._solver``;
``sim._robot_solver`` is the outer coupled solver. Its compact ModelView can
renumber bodies and shapes. The canonical MuJoCo-geom -> Newton-shape map is
therefore interpreted against *solver.model*, with full body labels aligning
the requested engine body ids. No ROS, Newton/Warp import or physics mutation
is needed here, and missing native data is never replaced by proximity.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.array(value, copy=True)


def _integers(value: Any, name: str) -> np.ndarray:
    result = _array(value)
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices")
    return result


def _numbers(value: Any, name: str, count: int) -> np.ndarray:
    result = _array(value)
    if result.ndim != 1 or len(result) < count:
        raise ValueError(f"{name} does not contain nacon complete records")
    result = result[:count]
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite numbers")
    return result


class WokContactSampler:
    """Sample active native constraints after a completed physics tick.

    A contact is active when any ``efc_address`` is nonnegative. If MJWarp
    supplies ``includemargin``, its signed ``dist`` must also be below that
    margin. Positive geometric gaps inside an active margin still count;
    candidate pairs with all constraint addresses -1 never count. This is
    solved-contact evidence, not a measured grip-force threshold.
    """

    def __init__(self, sim: Any, *, pan_id: int, hand_ids: np.ndarray) -> None:
        try:
            self._solver = sim._adapter._solver
            engine_model = sim._model
            model = self._solver.model
            mapping_source = self._solver.mjc_geom_to_newton_shape
        except AttributeError as exc:
            raise RuntimeError("wok contacts require the real MJWarp solver and geom mapping") from exc
        if self._solver is None or mapping_source is None:
            raise RuntimeError("wok contacts require the real MJWarp solver and geom mapping")
        if int(getattr(engine_model, "world_count", 1)) != 1 or int(getattr(model, "world_count", 1)) != 1:
            raise ValueError("wok contacts require a single world")
        global_labels = [str(name) for name in engine_model.body_label]
        labels = [str(name) for name in model.body_label]
        if len(set(global_labels)) != len(global_labels) or len(set(labels)) != len(labels):
            raise ValueError("wok contacts require unique full body labels")
        if (
            isinstance(pan_id, bool)
            or not isinstance(pan_id, (int, np.integer))
            or not 0 <= pan_id < len(global_labels)
        ):
            raise ValueError("invalid wok body id")
        hands = _integers(hand_ids, "hand_ids")
        if hands.ndim != 1 or not len(hands) or np.any(hands < 0) or np.any(hands >= len(global_labels)):
            raise ValueError("hand_ids must identify actual engine bodies")
        if len(np.unique(hands)) != len(hands) or pan_id in hands:
            raise ValueError("hand_ids must be unique and exclude the wok")
        pan_label = global_labels[int(pan_id)]
        hand_labels = {global_labels[int(index)] for index in hands}
        if pan_label not in labels or not hand_labels.issubset(labels):
            raise ValueError("MJWarp model view omits the requested wok or hand bodies")

        shape_body = _integers(model.shape_body, "MJWarp model shape_body")
        if shape_body.ndim != 1 or np.any(shape_body < -1) or np.any(shape_body >= len(labels)):
            raise ValueError("invalid MJWarp model view shape bodies")
        mapping = _integers(mapping_source, "mjc_geom_to_newton_shape")
        if mapping.ndim != 2 or mapping.shape[0] != 1 or not mapping.shape[1]:
            raise ValueError("geom mapping must have shape [1, ngeom]")
        self._geom_shapes = mapping[0]
        if np.any(self._geom_shapes < -1) or np.any(self._geom_shapes >= len(shape_body)):
            raise ValueError("geom mapping contains invalid model-view shape ids")
        self._pan = np.zeros(len(self._geom_shapes), dtype=bool)
        self._hand = np.zeros_like(self._pan)
        self._support = np.zeros_like(self._pan)
        for geom, shape in enumerate(self._geom_shapes):
            if shape < 0:
                continue
            body = int(shape_body[shape])
            if body == -1:
                self._support[geom] = True
            else:
                self._pan[geom] = labels[body] == pan_label
                self._hand[geom] = labels[body] in hand_labels
        if not self._pan.any() or not self._hand.any() or not self._support.any():
            raise ValueError("MJWarp geom mapping lacks wok, right-hand or static support collision shapes")
        # Validate the native data surface now, before scoring can start.
        self._data()

    def _data(self) -> Any:
        try:
            data = self._solver.mjw_data
            contact = data.contact
            for name in ("geom", "dist", "efc_address", "worldid"):
                if getattr(contact, name) is None:
                    raise AttributeError(name)
            if data.nacon is None:
                raise AttributeError("nacon")
        except AttributeError as exc:
            raise RuntimeError("wok contacts require native MJWarp nacon and solved contact data") from exc
        return data

    def sample(self) -> tuple[bool, bool]:
        data = self._data()
        count_array = _integers(data.nacon, "MJWarp nacon")
        if count_array.shape != (1,) or int(count_array[0]) < 0:
            raise ValueError("invalid MJWarp active contact count")
        count = int(count_array[0])
        if count == 0:
            return False, False
        contact = data.contact
        geoms = _integers(contact.geom, "contact.geom")
        addresses = _integers(contact.efc_address, "contact.efc_address")
        worlds = _integers(contact.worldid, "contact.worldid")
        distances = _numbers(contact.dist, "contact.dist", count)
        if (
            geoms.ndim != 2
            or geoms.shape[1] != 2
            or addresses.ndim != 2
            or addresses.shape[1] < 1
            or worlds.ndim != 1
            or distances.ndim != 1
            or any(len(array) < count for array in (geoms, addresses, worlds, distances))
        ):
            raise ValueError("MJWarp contact buffers do not contain nacon complete records")
        active = np.any(addresses[:count] >= 0, axis=1)
        # The source is one-world; unexpected world ids must not silently
        # turn a genuine support contact into a false 'unsupported' result.
        if np.any(worlds[:count] != 0):
            raise ValueError("single-world contact data contains another world id")
        margin_source = getattr(contact, "includemargin", None)
        if margin_source is not None:
            margins = _numbers(margin_source, "contact.includemargin", count)
            if margins.ndim != 1 or len(margins) < count:
                raise ValueError("contact margins do not contain nacon complete records")
            active &= distances[:count] < margins[:count]
        pairs = geoms[:count][active]
        if not len(pairs):
            return False, False
        if np.any(pairs < 0) or np.any(pairs >= len(self._geom_shapes)):
            raise ValueError("active contact references an invalid or non-rigid geom")
        if np.any(self._geom_shapes[pairs] < 0):
            raise ValueError("active contact geom has no Newton model-view shape mapping")
        left, right = pairs[:, 0], pairs[:, 1]
        hand = np.any((self._pan[left] & self._hand[right]) | (self._hand[left] & self._pan[right]))
        support = np.any((self._pan[left] & self._support[right]) | (self._support[left] & self._pan[right]))
        return bool(hand), bool(support)
