# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Native contact readback and compact-view mapping tests without Warp/ROS/GPU."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/wok_contacts.py"
spec = spec_from_file_location("tested_wok_contacts", SOURCE)
contacts = module_from_spec(spec)
sys.modules[spec.name] = contacts
spec.loader.exec_module(contacts)


class Array:
    def __init__(self, values, dtype=None):
        self.values = np.asarray(values, dtype=dtype)

    def numpy(self):
        return self.values


def simulation(pairs=(), *, dist=None, addresses=None, margins=True):
    # Parent body/shape ids intentionally differ from the compact solver view.
    # Engine bodies: left, right-finger, wok, right-palm, unrelated object.
    original = SimpleNamespace(
        world_count=1, body_label=["/robot/hand_l", "/robot/hand_r_finger", "chef_wok", "/robot/hand_r_palm", "other"]
    )
    view = SimpleNamespace(
        world_count=1,
        body_label=["chef_wok", "other", "/robot/hand_r_palm", "/robot/hand_r_finger", "/robot/hand_l"],
        shape_body=Array([2, -1, 0, 3, -1, 1, 4, 0]),
    )
    # MuJoCo geoms: support0, pan0, finger, unrelated, palm, support1, left, pan1.
    mapping = Array([[1, 2, 3, 5, 0, 4, 6, 7]])
    count = len(pairs)
    # Unused capacity contains junk; only nacon records may be interpreted.
    geom = np.full((count + 3, 2), -999, dtype=np.int32)
    geom[:count] = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
    efc = np.full((count + 3, 4), -1, dtype=np.int32)
    efc[:count, 0] = np.arange(count)
    if addresses is not None:
        efc[:count] = addresses
    dists = np.full(count + 3, np.nan)
    dists[:count] = -0.001 if dist is None else dist
    contact = SimpleNamespace(
        geom=Array(geom), efc_address=Array(efc), worldid=Array(np.zeros(count + 3, dtype=np.int32)), dist=Array(dists)
    )
    if margins:
        values = np.full(count + 3, np.nan)
        values[:count] = 0.0
        contact.includemargin = Array(values)
    solver = SimpleNamespace(
        model=view, mjc_geom_to_newton_shape=mapping, mjw_data=SimpleNamespace(nacon=Array([count]), contact=contact)
    )
    return SimpleNamespace(_model=original, _adapter=SimpleNamespace(_solver=solver))


def sampler(sim):
    return contacts.WokContactSampler(sim, pan_id=2, hand_ids=np.array([1, 3]))


@pytest.mark.parametrize(
    "pairs, expected",
    [
        ([], (False, False)),
        ([(1, 2)], (True, False)),
        ([(4, 7)], (True, False)),
        ([(1, 0)], (False, True)),
        ([(5, 7)], (False, True)),
        ([(2, 1), (7, 5)], (True, True)),
        ([(1, 6), (1, 3), (2, 5)], (False, False)),
    ],
)
def test_native_pairs_use_exact_compact_mapping_and_either_pair_order(pairs, expected):
    sim = simulation(pairs)
    assert sampler(sim).sample() == expected


def test_positive_gap_candidate_without_constraint_is_not_contact():
    sim = simulation([(1, 2), (1, 0)], dist=[0.001, 0.001], addresses=[[-1] * 4, [-1] * 4])
    sim._adapter._solver.mjw_data.contact.includemargin.values[:2] = 0.01
    assert sampler(sim).sample() == (False, False)


def test_positive_gap_active_margin_constraint_counts_without_force_gate():
    sim = simulation([(1, 2), (1, 0)], dist=[0.001, 0.002], addresses=[[-1, 8, -1, -1], [10, -1, -1, -1]])
    sim._adapter._solver.mjw_data.contact.includemargin.values[:2] = 0.01
    assert sampler(sim).sample() == (True, True)


@pytest.mark.parametrize("distance", [0.01, 0.02])
def test_active_address_at_or_outside_include_margin_does_not_count(distance):
    sim = simulation([(1, 2), (1, 0)], dist=[distance, distance])
    sim._adapter._solver.mjw_data.contact.includemargin.values[:2] = 0.01
    assert sampler(sim).sample() == (False, False)


def test_no_margin_field_still_uses_solved_constraint_addresses():
    sim = simulation([(1, 2), (1, 0)], addresses=[[1, -1, -1, -1], [-1] * 4], margins=False)
    assert sampler(sim).sample() == (True, False)


def test_unrelated_hand_to_static_contact_is_not_pan_support():
    assert sampler(simulation([(2, 0), (4, 5)])).sample() == (False, False)


def test_native_arrays_are_read_only_and_data_replacement_is_observed():
    sim = simulation([(1, 2)])
    observer = sampler(sim)
    before = sim._adapter._solver.mjw_data.contact.geom.values.copy()
    assert observer.sample() == (True, False)
    np.testing.assert_array_equal(sim._adapter._solver.mjw_data.contact.geom.values, before)
    sim._adapter._solver.mjw_data = simulation([(7, 0)])._adapter._solver.mjw_data
    assert observer.sample() == (False, True)


@pytest.mark.parametrize("missing", ["solver", "mapping", "data", "geom", "dist", "efc_address", "worldid", "nacon"])
def test_missing_native_data_fails_fast_instead_of_geometry_fallback(missing):
    sim = simulation()
    if missing == "solver":
        del sim._adapter._solver
    elif missing == "mapping":
        sim._adapter._solver.mjc_geom_to_newton_shape = None
    elif missing == "data":
        del sim._adapter._solver.mjw_data
    elif missing == "nacon":
        del sim._adapter._solver.mjw_data.nacon
    else:
        delattr(sim._adapter._solver.mjw_data.contact, missing)
    with pytest.raises(RuntimeError):
        sampler(sim)


@pytest.mark.parametrize("invalid", ["count", "world", "geom", "unmapped", "address_shape", "distance", "margin"])
def test_invalid_active_readback_fails_closed(invalid):
    sim = simulation([(1, 2)])
    solver = sim._adapter._solver
    observer = sampler(sim)
    if invalid == "count":
        solver.mjw_data.nacon.values[0] = 1000
    elif invalid == "world":
        solver.mjw_data.contact.worldid.values[0] = 1
    elif invalid == "geom":
        solver.mjw_data.contact.geom.values[0, 1] = 1000
    elif invalid == "unmapped":
        # A map may contain unused -1 slots, but active use is an error.
        solver.mjc_geom_to_newton_shape.values[0, 2] = -1
        observer = sampler(sim)
    elif invalid == "address_shape":
        solver.mjw_data.contact.efc_address.values = np.zeros((1, 0), dtype=np.int32)
    elif invalid == "distance":
        solver.mjw_data.contact.dist.values[0] = np.nan
    else:
        solver.mjw_data.contact.includemargin.values[0] = np.nan
    with pytest.raises(ValueError):
        observer.sample()


def test_unknown_or_duplicate_body_identity_cannot_silently_drop_hand_contacts():
    sim = simulation()
    sim._adapter._solver.model.body_label[2] = "different/hand_r_palm"
    with pytest.raises(ValueError, match="omits"):
        sampler(sim)
    sim = simulation()
    sim._model.body_label[3] = sim._model.body_label[1]
    with pytest.raises(ValueError, match="unique"):
        sampler(sim)
