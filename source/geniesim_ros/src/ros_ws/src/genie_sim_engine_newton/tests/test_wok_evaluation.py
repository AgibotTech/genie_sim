# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Deterministic physical scoring trajectories; no Newton, ROS or GPU runtime."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton/wok_evaluation.py"
spec = spec_from_file_location("tested_wok_evaluation", SOURCE)
evaluation = module_from_spec(spec)
sys.modules[spec.name] = evaluation
spec.loader.exec_module(evaluation)

REST = 1.2
LIFTED = 1.26
TABLE = 0.987
COUNT = 46


def geometry():
    angle = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    rim = np.column_stack((0.2 * np.cos(angle), 0.2 * np.sin(angle), np.zeros(len(angle))))
    return evaluation.WokEvaluationGeometry(
        table_top_z=TABLE,
        rest_rim_z=REST,
        rim_radius=0.2,
        bowl_depth=0.1,
        bowl_vertices=np.vstack((rim, [0, 0, -0.1])),
        handle_start=np.array([0.2, 0, 0]),
        handle_end=np.array([0.4, 0, 0]),
        handle_radius=0.02,
    )


def sample(
    sim_time,
    *,
    rim=REST,
    tilt=0,
    food=None,
    epoch=1,
    distant_hand=False,
    hand_contact=True,
    support_contact=False,
):
    radians = np.radians(tilt)
    rotation = np.array([[np.cos(radians), 0, np.sin(radians)], [0, 1, 0], [-np.sin(radians), 0, np.cos(radians)]])
    pose = np.array([0, 0, rim, 0, np.sin(radians / 2), 0, np.cos(radians / 2)])
    hand = np.array([[0.3, 0, 0]]) @ rotation.T + [0, 0, rim]
    if distant_hand:
        hand += [10, 0, 0]
    if food is None:
        food = foods(rim)
    return evaluation.WokEvaluationSample(
        sim_time=sim_time,
        reset_epoch=epoch,
        wok_pose=pose,
        hand_points=hand,
        food_objects=tuple(
            evaluation.FoodObjectSample(name=f"food_{i}", points=points) for i, points in enumerate(food)
        ),
        hand_contact=hand_contact,
        support_contact=support_contact,
    )


def foods(rim, airborne=0, *, airborne_z=0.12, outside=False, huge_first=False):
    result = []
    offsets = np.array([[-0.002, -0.002, -0.002], [0.002, 0.002, 0.002], [-0.002, 0.002, 0.002]])
    for i in range(COUNT):
        center = np.array(
            [0.4 if outside and i < airborne else (i % 5 - 2) * 0.01, 0, rim + (airborne_z if i < airborne else -0.05)]
        )
        points = offsets + center
        if huge_first and i == 0:
            points = np.tile(points, (1000, 1))
        result.append(points)
    return result


def update(scorer, value, *, dt=0.25):
    return scorer.update(value, dt=dt)


def stable_lift(scorer):
    update(scorer, sample(0))
    first = update(scorer, sample(0.25, rim=LIFTED))
    middle = update(scorer, sample(0.5, rim=LIFTED))
    final = update(scorer, sample(0.75, rim=LIFTED))
    assert first["score"] == middle["score"] == 0
    assert final["score"] == 1
    return final


def test_default_thresholds_and_exact_half_second_continuous_stable_lift():
    scorer = evaluation.WokEvaluator(geometry())
    result = stable_lift(scorer)
    assert result["max_score"] == 2 and not result["failed"]
    assert result["milestones"] == {"stable_lift": True, "food_toss": False}
    assert result["thresholds"]["lift_height_m"] == 0.05
    assert result["thresholds"]["stable_hold_s"] == 0.5
    assert result["thresholds"]["toss_height_m"] == 0.1
    assert result["thresholds"]["toss_object_fraction"] == 0.2
    assert (
        not {"max_linear_speed_m_s", "max_angular_speed_rad_s", "hand_handle_distance_m"} & result["thresholds"].keys()
    )


@pytest.mark.parametrize("initial", [REST - 0.15, REST + 0.3])
def test_lift_uses_actual_initial_rim_instead_of_configured_home_height(initial):
    scorer = evaluation.WokEvaluator(geometry())
    initial_result = update(scorer, sample(0, rim=initial), dt=0)
    assert initial_result["metrics"]["initial_rim_z"] == initial
    assert initial_result["metrics"]["lift_height_m"] == 0.0
    for sim_time in (0.25, 0.5, 0.75):
        result = update(scorer, sample(sim_time, rim=initial + 0.05 - 1e-6))
        assert result["score"] == 0
        assert result["metrics"]["initial_rim_z"] == initial
    for sim_time in (1.0, 1.25):
        assert update(scorer, sample(sim_time, rim=initial + 0.05 + 1e-6))["score"] == 0
    result = update(scorer, sample(1.5, rim=initial + 0.05 + 1e-6))
    assert result["score"] == 1
    assert result["metrics"]["lift_height_m"] == pytest.approx(0.05 + 1e-6)


def test_paused_refresh_cannot_move_the_initial_height_or_earn_a_lift():
    scorer = evaluation.WokEvaluator(geometry())
    initial = REST + 0.3
    update(scorer, sample(0, rim=initial), dt=0)
    for _ in range(100):
        result = update(scorer, sample(0, rim=initial + 0.1), dt=0)
        assert result["score"] == 0
        assert result["metrics"]["initial_rim_z"] == initial
        assert result["metrics"]["lift_height_m"] == pytest.approx(0.1)
    for sim_time in (0.25, 0.5):
        assert update(scorer, sample(sim_time, rim=initial + 0.1))["score"] == 0
    assert update(scorer, sample(0.75, rim=initial + 0.1))["score"] == 1


@pytest.mark.parametrize("motion", ["linear_speed", "angular_speed", "tilt", "distant_hand"])
def test_first_point_uses_contacts_without_absolute_motion_tilt_or_hand_distance_gates(motion):
    scorer = evaluation.WokEvaluator(geometry())
    update(scorer, sample(0))
    values = []
    for index, sim_time in enumerate((0.25, 0.5, 0.75)):
        tilt = (20, -20, 20)[index] if motion == "angular_speed" else 60 if motion == "tilt" else 0
        value = sample(sim_time, rim=LIFTED, tilt=tilt, distant_hand=motion == "distant_hand")
        if motion == "linear_speed":
            value.wok_pose[0] = 4 * sim_time
            value.hand_points[:, 0] += 4 * sim_time
            for food in value.food_objects:
                food.points[:, 0] += 4 * sim_time
        values.append(update(scorer, value))
    assert [value["score"] for value in values] == [0, 0, 1]
    assert not values[-1]["failed"]
    metric = values[-1]["metrics"]
    if motion == "linear_speed":
        assert metric["linear_speed_m_s"] > 0.25
    elif motion == "angular_speed":
        assert metric["angular_speed_rad_s"] > 1.5
    elif motion == "tilt":
        assert metric["tilt_deg"] > 45
    else:
        assert metric["hand_handle_distance_m"] > 0.1


@pytest.mark.parametrize("hand_contact,support_contact", [(False, False), (True, True), (False, True)])
def test_height_and_hand_proximity_cannot_replace_contact_and_support_release(hand_contact, support_contact):
    scorer = evaluation.WokEvaluator(geometry())
    update(scorer, sample(0))
    for sim_time in (0.25, 0.5, 0.75, 1, 2):
        result = update(
            scorer, sample(sim_time, rim=LIFTED, hand_contact=hand_contact, support_contact=support_contact)
        )
        assert result["score"] == 0 and not result["failed"]


def test_initial_hovering_high_above_table_never_earns_a_lift():
    scorer = evaluation.WokEvaluator(geometry())
    for sim_time in (0, 0.25, 0.5, 1, 2):
        result = update(scorer, sample(sim_time, rim=REST + 0.3))
        assert result["score"] == 0 and not result["failed"]


def test_pause_refreshes_and_repeated_simtime_cannot_accumulate_stable_hold():
    scorer = evaluation.WokEvaluator(geometry())
    update(scorer, sample(0))
    update(scorer, sample(0.25, rim=LIFTED))
    for _ in range(100):
        assert update(scorer, sample(0.25, rim=LIFTED), dt=0)["score"] == 0
    assert update(scorer, sample(0.5, rim=LIFTED))["score"] == 0
    assert update(scorer, sample(0.75, rim=LIFTED))["score"] == 1


def test_hold_uses_sample_simtime_instead_of_supplied_dt_or_callback_count():
    scorer = evaluation.WokEvaluator(geometry())
    update(scorer, sample(0), dt=50)
    update(scorer, sample(0.5, rim=LIFTED), dt=50)
    for sim_time in (0.6, 0.7, 0.8, 0.9):
        assert update(scorer, sample(sim_time, rim=LIFTED), dt=50)["score"] == 0
    assert update(scorer, sample(1.0, rim=LIFTED), dt=50)["score"] == 1


@pytest.mark.parametrize("interruption", ["lost_hand_contact", "support_returned", "height_lost"])
def test_contact_or_height_interruption_resets_the_continuous_hold_window(interruption):
    scorer = evaluation.WokEvaluator(geometry())
    update(scorer, sample(0))
    update(scorer, sample(0.25, rim=LIFTED))
    update(
        scorer,
        sample(
            0.5,
            rim=REST if interruption == "height_lost" else LIFTED,
            hand_contact=interruption != "lost_hand_contact",
            support_contact=interruption == "support_returned",
        ),
    )
    assert update(scorer, sample(0.75, rim=LIFTED))["score"] == 0
    assert update(scorer, sample(1.0, rim=LIFTED))["score"] == 0
    assert update(scorer, sample(1.25, rim=LIFTED))["score"] == 1


@pytest.mark.parametrize("objects,expected", [(9, 1), (10, 2)])
def test_toss_requires_twenty_percent_of_46_whole_objects(objects, expected):
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    result = update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=objects)))
    assert result["score"] == expected
    assert not result["failed"]


def test_many_airborne_particles_in_only_one_food_object_are_not_twenty_percent_of_objects():
    scorer = evaluation.WokEvaluator(geometry())
    for t, rim in ((0, REST), (0.25, LIFTED), (0.5, LIFTED), (0.75, LIFTED)):
        result = update(scorer, sample(t, rim=rim, food=foods(rim, huge_first=True)))
    assert result["score"] == 1
    assert update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=1, huge_first=True)))["score"] == 1


def test_a_noodle_tip_above_rim_does_not_make_the_whole_object_airborne():
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    food = foods(LIFTED, airborne=10)
    food[9][0, 2] = LIFTED - 0.01
    assert update(scorer, sample(1.0, rim=LIFTED, food=food))["score"] == 1


def test_food_never_observed_in_pan_cannot_earn_toss_even_if_it_rises():
    scorer = evaluation.WokEvaluator(geometry())
    for t, rim in ((0, REST), (0.25, LIFTED), (0.5, LIFTED), (0.75, LIFTED)):
        result = update(scorer, sample(t, rim=rim, food=foods(rim, airborne=COUNT)))
    assert result["score"] == 1
    assert update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=COUNT, airborne_z=0.3)))["score"] == 1


def test_pause_only_food_membership_does_not_arm_a_toss():
    scorer = evaluation.WokEvaluator(geometry())
    update(scorer, sample(0), dt=0)
    for t, rim in ((0, REST), (0.25, LIFTED), (0.5, LIFTED), (0.75, LIFTED)):
        result = update(scorer, sample(t, rim=rim, food=foods(rim, airborne=COUNT)))
    assert result["score"] == 1
    assert update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=COUNT, airborne_z=0.3)))["score"] == 1


@pytest.mark.parametrize(
    "case", ["inverted_pan", "outside_rim", "falling_food", "sinking_pan", "below_highest_tilted_rim"]
)
def test_flip_spill_and_relative_height_illusions_do_not_earn_toss(case):
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    if case == "inverted_pan":
        value = sample(1.25, rim=LIFTED, tilt=180, food=foods(LIFTED, airborne=COUNT))
    elif case == "outside_rim":
        value = sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=COUNT, outside=True))
    elif case == "sinking_pan":
        value = sample(1.0, rim=LIFTED - 0.2, food=foods(LIFTED))
    elif case == "below_highest_tilted_rim":
        value = sample(1.25, rim=LIFTED, tilt=30, food=foods(LIFTED, airborne=COUNT))
    else:
        # Airborne before stable lift is established; it is falling by the
        # first scoring opportunity, so absolute height alone cannot score.
        scorer = evaluation.WokEvaluator(geometry())
        update(scorer, sample(0))
        update(scorer, sample(0.25, rim=LIFTED))
        update(scorer, sample(0.5, rim=LIFTED, food=foods(LIFTED, airborne=COUNT, airborne_z=0.3)))
        value = sample(0.75, rim=LIFTED, food=foods(LIFTED, airborne=COUNT, airborne_z=0.2))
    result = update(scorer, value)
    assert result["score"] == 1 and not result["failed"]


def test_food_toss_needs_prior_stable_lift():
    scorer = evaluation.WokEvaluator(geometry())
    update(scorer, sample(0))
    result = update(scorer, sample(0.25, rim=LIFTED, food=foods(LIFTED, airborne=COUNT)))
    assert result["score"] == 0


def test_drop_uses_actual_tabletop_and_checks_even_during_zero_dt_refresh():
    scorer = evaluation.WokEvaluator(geometry())
    assert not update(scorer, sample(0, rim=TABLE), dt=0)["failed"]
    result = update(scorer, sample(0, rim=TABLE - 1e-6), dt=0)
    assert result["failed"] and result["failure_reason"] == "wok_below_table"
    assert result["score"] == 0


@pytest.mark.parametrize("earned", [0, 1, 2])
def test_drop_is_latched_and_preserves_previously_earned_score(earned):
    scorer = evaluation.WokEvaluator(geometry())
    if earned:
        stable_lift(scorer)
    else:
        update(scorer, sample(0))
    if earned == 2:
        assert update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=10)))["score"] == 2
    result = update(scorer, sample(1.25, rim=TABLE - 0.01, food=foods(LIFTED, airborne=COUNT)))
    assert result["score"] == earned and result["failed"]
    later = update(scorer, sample(2.0, rim=LIFTED, food=foods(LIFTED, airborne=COUNT)))
    assert later["score"] == earned and later["failed"]


def test_new_reset_epoch_clears_earned_score_failure_and_stability_history():
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=10)))
    update(scorer, sample(1.25, rim=TABLE - 0.01))
    new_initial = REST + 0.3
    result = update(scorer, sample(0, rim=new_initial, epoch=2), dt=0)
    assert result["score"] == 0 and not result["failed"]
    assert result["milestones"] == {"stable_lift": False, "food_toss": False}
    assert result["metrics"]["initial_rim_z"] == new_initial
    assert result["metrics"]["lift_height_m"] == 0
    for sim_time in (0.25, 0.5):
        assert update(scorer, sample(sim_time, rim=new_initial + 0.06, epoch=2))["score"] == 0
    assert update(scorer, sample(0.75, rim=new_initial + 0.06, epoch=2))["score"] == 1


@pytest.mark.parametrize("reenter", [False, True])
def test_failed_flight_cannot_become_valid_on_a_later_rebound_without_reentering_pan(reenter):
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    assert update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=10, airborne_z=0.05)))["score"] == 1
    assert update(scorer, sample(1.25, rim=LIFTED, food=foods(LIFTED, airborne=10, airborne_z=0.04)))["score"] == 1
    if reenter:
        update(scorer, sample(1.5, rim=LIFTED))
    result = update(scorer, sample(1.75, rim=LIFTED, food=foods(LIFTED, airborne=10)))
    assert result["score"] == (2 if reenter else 1)


@pytest.mark.parametrize("height,expected", [(0.1019, 1), (0.1021, 2)])
def test_whole_object_bottom_must_clear_the_highest_rim_by_ten_centimeters(height, expected):
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    assert (
        update(scorer, sample(1.0, rim=LIFTED, food=foods(LIFTED, airborne=10, airborne_z=height)))["score"] == expected
    )


def test_real_food_rise_scores_while_pan_returns_downward_during_a_toss():
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    # The food rises 17 cm in world space from its previous resting bottom;
    # the wok's simultaneous 4 cm descent is normal toss motion.
    result = update(scorer, sample(1.0, rim=LIFTED - 0.04, food=foods(LIFTED, airborne=10)))
    assert result["score"] == 2 and not result["failed"]


@pytest.mark.parametrize(
    "change",
    [
        {"lift_height_m": 0},
        {"stable_hold_s": -1},
        {"toss_object_fraction": 0},
        {"toss_object_fraction": 1.1},
        {"toss_height_m": float("nan")},
        {"max_tilt_deg": 90},
        {"min_food_upward_speed_m_s": 0},
        {"lift_height_m": True},
        {"unknown_threshold": 1},
    ],
)
def test_invalid_scoring_thresholds_are_rejected(change):
    with pytest.raises(ValueError):
        evaluation.WokEvaluationConfig.from_mapping(change)


@pytest.mark.parametrize("key", ["max_linear_speed_m_s", "max_angular_speed_rad_s", "hand_handle_distance_m"])
def test_legacy_first_point_thresholds_are_accepted_but_removed_from_the_effective_config(key):
    config = evaluation.WokEvaluationConfig.from_mapping({key: 0.0, "stable_hold_s": 0.7})
    assert config.stable_hold_s == 0.7
    assert not hasattr(config, key)
    scorer = evaluation.WokEvaluator(geometry(), config)
    result = update(scorer, sample(0))
    assert key not in result["thresholds"]


def test_json_config_reads_old_launcher_thresholds_without_advertising_ignored_gates():
    config = evaluation.WokEvaluationConfig.from_json(
        '{"max_linear_speed_m_s":0.25,"max_angular_speed_rad_s":1.5,"hand_handle_distance_m":0.1,'
        '"stable_hold_s":0.7}'
    )
    result = update(evaluation.WokEvaluator(geometry(), config), sample(0))
    assert result["thresholds"]["stable_hold_s"] == 0.7
    assert (
        not {"max_linear_speed_m_s", "max_angular_speed_rad_s", "hand_handle_distance_m"} & result["thresholds"].keys()
    )


@pytest.mark.parametrize("key", ["max_linear_speed_m_s", "max_angular_speed_rad_s", "hand_handle_distance_m"])
@pytest.mark.parametrize("invalid", [True, -1, float("inf"), float("nan"), "0.25", None])
def test_ignored_legacy_thresholds_still_reject_invalid_values(key, invalid):
    with pytest.raises(ValueError):
        evaluation.WokEvaluationConfig.from_mapping({key: invalid})


@pytest.mark.parametrize("key", ["hand_contact", "support_contact"])
@pytest.mark.parametrize("invalid", [1, "true", None])
def test_contact_flags_must_be_explicit_booleans(key, invalid):
    with pytest.raises(ValueError):
        update(evaluation.WokEvaluator(geometry()), sample(0, **{key: invalid}))


def test_clock_regression_without_reset_epoch_cannot_reuse_score_history():
    scorer = evaluation.WokEvaluator(geometry())
    stable_lift(scorer)
    with pytest.raises(ValueError, match="backwards"):
        update(scorer, sample(0.5, rim=LIFTED))


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


def sampler_sim(*, prefix=7):
    # Match the chef's actual emission counts: 16*324 + 8*27 + 12*24 + 10*32.
    cfg = SimpleNamespace(
        wok=SimpleNamespace(wok_dynamic=True, home_z=1.09, rim_r=0.16, depth=0.09),
        noodle=SimpleNamespace(n_noodles=16),
        ham=SimpleNamespace(n_cubes=8, dim=2),
        pepper=SimpleNamespace(n=12, n_len=5),
        carrot=SimpleNamespace(n=10, n_len=7),
    )
    plugin = SimpleNamespace(
        cfg=cfg,
        pan_id=3,
        visual_shape=0,
        grip_shape=1,
        auto_flip=False,
        starts_np=prefix + np.arange(16) * 324,
        n_len_noodle=80,
        pepper_pstart=prefix + 5184 + 216,
        carrot_pstart=prefix + 5184 + 216 + 288,
    )
    model = SimpleNamespace(
        world_count=1,
        body_label=["/genie/base", "/genie/hand_r_base_link", "/genie/hand_r_index_link1", "/chef_wok"],
        shape_label=["chef_wok_visual", "chef_wok_grip_round", "chef_stove_base"],
        shape_body=Array([3, 3, -1]),
        shape_transform=Array(
            [
                [0.01, 0.02, -0.03, 0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                [-0.246, 0, -0.017, 0, np.sqrt(0.5), 0, np.sqrt(0.5)],
                [0, 0, 0.487, 0, 0, 0, 1],
            ]
        ),
        shape_scale=Array([[0.2, 0.3, 0.4], [0.0095, 0.036, 1], [0.22, 0.22, 0.5]]),
        shape_source=[SimpleNamespace(vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])), None, None],
        particle_count=prefix + 6008,
    )
    poses = np.array([[i, i + 0.1, i + 0.2, 0, 0, 0, 1] for i in range(4)])
    state = SimpleNamespace(body_q=Array(poses), particle_q=Array(np.arange(model.particle_count * 3).reshape(-1, 3)))
    return SimpleNamespace(_model=model, _scene_plugin=plugin, _state_0=state, _sim_time_acc=2.5, _reset_epoch=3)


def test_sampler_uses_actual_shape_geometry_capsule_axis_and_table_box_top():
    sim = sampler_sim()
    sampled = evaluation.WokEvaluationSampler(sim, contacts=SimpleNamespace(sample=lambda: (True, False)))
    g = sampled.geometry
    assert g.table_top_z == pytest.approx(0.987)
    assert g.rest_rim_z == 1.09
    np.testing.assert_allclose(
        g.bowl_vertices, [[0.01, 0.02, -0.03], [0.01, 0.22, -0.03], [-0.29, 0.02, -0.03]], atol=1e-8
    )
    np.testing.assert_allclose(g.handle_start, [-0.282, 0, -0.017], atol=1e-8)
    np.testing.assert_allclose(g.handle_end, [-0.210, 0, -0.017], atol=1e-8)
    assert g.handle_radius == 0.0095


def test_sampler_preserves_all_46_real_objects_and_particle_offset_without_mutating_model():
    sim = sampler_sim()
    sampled = evaluation.WokEvaluationSampler(sim, contacts=SimpleNamespace(sample=lambda: (True, False)))
    result = sampled.sample()
    assert result.sim_time == 2.5 and result.reset_epoch == 3
    assert result.hand_contact is True and result.support_contact is False
    np.testing.assert_array_equal(result.wok_pose, sim._state_0.body_q.values[3])
    np.testing.assert_array_equal(result.hand_points, sim._state_0.body_q.values[1:3, :3])
    foods_by_name = {food.name: food.points for food in result.food_objects}
    assert len(foods_by_name) == 46 and sum(len(points) for points in foods_by_name.values()) == 6008
    assert [len(foods_by_name[name]) for name in ("noodle_0", "ham_0", "pepper_0", "carrot_0")] == [324, 27, 24, 32]
    np.testing.assert_array_equal(foods_by_name["noodle_0"], sim._state_0.particle_q.values[7:331])
    np.testing.assert_array_equal(foods_by_name["ham_0"], sim._state_0.particle_q.values[7 + 5184 : 7 + 5184 + 27])
    np.testing.assert_array_equal(foods_by_name["carrot_9"], sim._state_0.particle_q.values[-32:])
    old_particles = sim._state_0.particle_q.values.copy()
    result.food_objects[0].points[:] = -99
    result.wok_pose[:] = -99
    np.testing.assert_array_equal(sim._state_0.particle_q.values, old_particles)
    np.testing.assert_array_equal(sampled.sample().wok_pose, [3, 3.1, 3.2, 0, 0, 0, 1])


@pytest.mark.parametrize("hand_contact,support_contact", [(True, False), (False, True), (False, False), (True, True)])
def test_sampler_preserves_contact_sampler_flags(hand_contact, support_contact):
    sampled = evaluation.WokEvaluationSampler(
        sampler_sim(), contacts=SimpleNamespace(sample=lambda: (hand_contact, support_contact))
    )
    result = sampled.sample()
    assert result.hand_contact is hand_contact
    assert result.support_contact is support_contact


@pytest.mark.parametrize(
    "problem",
    [
        "missing_shape",
        "missing_hand",
        "no_bowl_mesh",
        "wrong_table_body",
        "wrong_bowl_body",
        "bad_capsule",
        "inconsistent_particle_start",
    ],
)
def test_sampler_rejects_incomplete_or_incompatible_scene_instead_of_guessing(problem):
    sim = sampler_sim()
    if problem == "missing_shape":
        sim._model.shape_label[1] = "wrong_grip"
    elif problem == "missing_hand":
        sim._model.body_label[1:3] = ["left_only", "another_left"]
    elif problem == "no_bowl_mesh":
        sim._model.shape_source[0] = None
    elif problem == "wrong_table_body":
        sim._model.shape_body.values[2] = 0
    elif problem == "wrong_bowl_body":
        sim._model.shape_body.values[0] = 0
    elif problem == "bad_capsule":
        sim._model.shape_scale.values[1, 1] = 0
    else:
        sim._scene_plugin.pepper_pstart += 1
    with pytest.raises(ValueError):
        evaluation.WokEvaluationSampler(sim, contacts=SimpleNamespace(sample=lambda: (True, False)))
