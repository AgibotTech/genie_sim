# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""CPU-only progress protocol/ownership checks; no ROS, GPU or services."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "genie_sim_engine_newton/engine/newton"
PACKAGE = ModuleType("tested_wok_ros")
PACKAGE.__path__ = [str(DIRECTORY)]
sys.modules[PACKAGE.__name__] = PACKAGE
SPEC = spec_from_file_location("tested_wok_ros.wok_evaluation_ros", DIRECTORY / "wok_evaluation_ros.py")
ros = module_from_spec(SPEC)
sys.modules[SPEC.name] = ros
SPEC.loader.exec_module(ros)


class FakeNode:
    def __init__(self, fail=False):
        self.fail = fail
        self.created = []
        self.destroyed = []
        self.messages = []

    def create_publisher(self, message_type, topic, qos):
        if self.fail:
            raise RuntimeError("publisher construction failed")
        publisher = SimpleNamespace(publish=lambda message: self.messages.append(json.loads(message.data)))
        self.created.append((publisher, message_type, topic, qos))
        return publisher

    def destroy_publisher(self, publisher):
        self.destroyed.append(publisher)


@pytest.fixture(autouse=True)
def fake_ros(monkeypatch):
    qos = SimpleNamespace(
        DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL="latched"),
        ReliabilityPolicy=SimpleNamespace(RELIABLE="reliable"),
        QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "rclpy", ModuleType("rclpy"))
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)
    monkeypatch.setitem(sys.modules, "std_msgs", ModuleType("std_msgs"))
    monkeypatch.setitem(sys.modules, "std_msgs.msg", SimpleNamespace(String=type("String", (), {})))


def status(*, score=0, failed=False):
    return dict(
        score=score,
        max_score=2,
        milestones=dict(stable_lift=score >= 1, food_toss=score >= 2),
        failed=failed,
        failure_reason="wok_below_table" if failed else None,
        metrics={"wok_z": 1.0},
        thresholds={"stable_hold_s": 0.5},
    )


class Sampler:
    def __init__(self, sim):
        self.sim = sim
        self.geometry = object()
        self.calls = 0
        self.error = None

    def sample(self):
        self.calls += 1
        if self.error:
            raise self.error
        return SimpleNamespace(reset_epoch=self.sim.reset_epoch, sim_time=self.sim._sim_time_acc)


class Evaluator:
    def __init__(self):
        self.calls = []
        self.value = status()

    def update(self, sample, *, dt):
        self.calls.append((sample.reset_epoch, dt))
        return self.value.copy()


def make_producer(*, node=None):
    node = node or FakeNode()
    sim = SimpleNamespace(reset_epoch=0, _sim_time_acc=10.0)
    sampler, evaluator = Sampler(sim), Evaluator()
    producer = ros.WokEvaluationProducer(
        node,
        sim,
        ros.WokEvaluationRosConfig("/wuji_eval/task_progress", {"stable_hold_s": 0.5}),
        dt=1 / 60,
        session_id="test-session",
        sampler=sampler,
        evaluator=evaluator,
    )
    return producer, node, sim, sampler, evaluator


def test_publisher_uses_existing_node_and_owns_only_its_publisher():
    producer, node, *_ = make_producer()
    _, _, topic, qos = node.created[0]
    assert topic == "/wuji_eval/task_progress"
    assert vars(qos) == dict(depth=1, reliability="reliable", durability="latched")
    producer.close()
    producer.close()
    producer.refresh()
    assert node.destroyed == [node.created[0][0]]
    assert node.messages == []


def test_publisher_construction_failure_leaves_no_owned_handle():
    node = FakeNode(fail=True)
    with pytest.raises(RuntimeError, match="publisher construction failed"):
        make_producer(node=node)
    assert node.created == node.destroyed == []


def test_progress_schema_tick_epoch_and_pause_refresh_without_scoring_time():
    producer, node, sim, sampler, evaluator = make_producer()
    try:
        producer.refresh()
        initial = node.messages[-1]
        assert initial == dict(
            **status(),
            version=1,
            task_name="wok_flip",
            session_id="test-session",
            reset_epoch=0,
            tick=0,
            sim_time=10.0,
            published_ns=initial["published_ns"],
            terminal=False,
        )
        sim._sim_time_acc += 1 / 60
        producer.after_step(-100.0)  # core's callback argument is not the completed time
        stepped = node.messages[-1]
        assert stepped["tick"] == 1 and stepped["sim_time"] == sim._sim_time_acc
        image_stamp = stepped["published_ns"]
        producer.refresh()
        assert node.messages[-1]["tick"] == 1
        assert node.messages[-1]["published_ns"] >= image_stamp
        assert evaluator.calls == [(0, 0.0), (0, 1 / 60), (0, 0.0)]
        sim.reset_epoch = 1
        producer.refresh()
        assert node.messages[-1]["reset_epoch"] == 1
        assert node.messages[-1]["tick"] == 0
        producer.after_step(0)
        assert node.messages[-1]["tick"] == 1
        assert sampler.calls == 5
    finally:
        producer.close()


def test_two_points_continue_then_physical_failure_retains_earned_score():
    producer, node, _, _, evaluator = make_producer()
    try:
        evaluator.value = status(score=2)
        producer.after_step(0)
        assert node.messages[-1]["score"] == 2
        assert node.messages[-1]["terminal"] is False
        evaluator.value = status(score=2, failed=True)
        producer.after_step(0)
        assert node.messages[-1]["score"] == 2
        assert node.messages[-1]["terminal"] is True
        assert node.messages[-1]["failure_reason"] == "wok_below_table"
    finally:
        producer.close()


def test_sampling_failure_latches_score_error_but_tracks_reset_epoch():
    producer, node, sim, sampler, evaluator = make_producer()
    try:
        evaluator.value = status(score=1)
        producer.after_step(0)
        sampler.error = RuntimeError("missing particles")
        producer.after_step(0)
        error = node.messages[-1]
        assert error["terminal"] and error["failed"] and error["score"] == 1
        assert error["failure_reason"] == "evaluation_error"
        assert error["error"] == "RuntimeError: missing particles"
        calls = sampler.calls
        sim.reset_epoch = 2
        sim._sim_time_acc = 20.0
        producer.refresh()
        assert node.messages[-1]["reset_epoch"] == 2 and node.messages[-1]["tick"] == 0
        assert node.messages[-1]["sim_time"] == 20.0
        producer.after_step(0)
        assert node.messages[-1]["tick"] == 1
        assert sampler.calls == calls  # scoring stays poisoned, identity does not
        assert evaluator.calls == [(0, 1 / 60)]
    finally:
        producer.close()


def test_sampler_construction_failure_can_publish_and_cleanup(monkeypatch):
    def fail(_):
        raise RuntimeError("unsupported chef geometry")

    monkeypatch.setitem(sys.modules, "tested_wok_ros.wok_evaluation", SimpleNamespace(WokEvaluationSampler=fail))
    node = FakeNode()
    sim = SimpleNamespace(reset_epoch=0, _sim_time_acc=0.0)
    producer = ros.WokEvaluationProducer(node, sim, ros.WokEvaluationRosConfig("/test", {}), dt=1 / 60)
    try:
        sim.reset_epoch = 1
        producer.refresh()
        assert node.messages[-1]["reset_epoch"] == 1
        assert node.messages[-1]["terminal"] is True
        assert node.messages[-1]["failure_reason"] == "evaluation_error"
    finally:
        producer.close()
    assert node.destroyed == [node.created[0][0]]


def test_config_is_launcher_json_and_absent_or_disabled_is_noop(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "tested_wok_ros.wok_evaluation",
        SimpleNamespace(WokEvaluationConfig=SimpleNamespace(from_mapping=lambda values: dict(values))),
    )
    assert ros.WokEvaluationRosConfig.from_params({}) is None
    assert ros.WokEvaluationRosConfig.from_params({"wok_evaluation_json": ""}) is None
    assert ros.WokEvaluationRosConfig.from_params({"wok_evaluation_json": '{"enabled":false}'}) is None
    config = ros.WokEvaluationRosConfig.from_params(
        {"wok_evaluation_json": '{"topic":"/test/task","thresholds":{"stable_hold_s":0.7}}'}
    )
    assert config.topic == "/test/task" and config.thresholds == {"stable_hold_s": 0.7}
    with pytest.raises(ValueError, match="absolute ROS topic"):
        ros.WokEvaluationRosConfig.from_params({"wok_evaluation_json": '{"topic":"relative"}'})
