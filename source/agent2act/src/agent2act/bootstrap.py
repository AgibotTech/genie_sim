# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Composition root: the only place that binds ports to concrete implementations."""

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent2act.adapters.kinematics.urdf import UrdfKinematics
from agent2act.adapters.motion.urdf_joint import UrdfJointMotion
from agent2act.adapters.planners.anthropic import AnthropicPlanner
from agent2act.adapters.planners.chat_completions import ChatCompletionsPlanner
from agent2act.adapters.planners.embodied import EmbodiedPlanner
from agent2act.adapters.planners.gemini import GeminiPlanner
from agent2act.adapters.planners.motion_probe import MotionProbePlanner
from agent2act.adapters.planners.ranked_motion import RankedMotionExecutor, RankedMotionPlanner
from agent2act.adapters.planners.responses import ResponsesPlanner
from agent2act.adapters.planners.scripted import ScriptedPlanner
from agent2act.adapters.policies.pi05 import Pi05Adapter
from agent2act.adapters.simulators.corobot import CoRobotCodec
from agent2act.adapters.simulators.corobot_ik_frame import CoRobotIKFrameCodec
from agent2act.application.routing import ExecutionRouter
from agent2act.application.runtime import AgentRuntime
from agent2act.configuration import AppSettings, parse_settings
from agent2act.domain.errors import ConfigurationError, ContractError
from agent2act.domain.models import RobotSpec
from agent2act.domain.ports import ActionExecutor, EventSink, PolicyBackend
from agent2act.domain.timing import MotionTiming
from agent2act.execution.fixed_steps import FixedStepsMotion
from agent2act.execution.motion import MotionCompiler
from agent2act.execution.policy import PolicyExecutor
from agent2act.registry import ComponentRegistry
from agent2act.service.observability import LoggingEventSink
from agent2act.service.observer import Observer
from agent2act.service.observer.process import ObserverProcess
from agent2act.service.session import AgentSession


def _local_motion(spec: RobotSpec, timing: MotionTiming, options: dict[str, Any]) -> ActionExecutor:
    if options:
        raise ContractError("local_interpolation accepts no backend options; use motion timing settings")
    return MotionCompiler(spec, timing)


def _joint_motion(spec: RobotSpec, timing: MotionTiming, options: dict[str, Any]) -> ActionExecutor:
    # Ownership transfers at entry; the adapter rolls back even if its FK fails.
    return UrdfJointMotion(spec, timing, options, MotionCompiler(spec, timing))


def default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    registry.planners.register("responses", lambda options: ResponsesPlanner(**options))
    registry.planners.register("embodied", lambda options: EmbodiedPlanner(**options))
    registry.planners.register("anthropic", lambda options: AnthropicPlanner(**options))
    registry.planners.register("gemini", lambda options: GeminiPlanner(**options))
    registry.planners.register("chat_completions", lambda options: ChatCompletionsPlanner(**options))
    registry.planners.register("scripted", lambda options: ScriptedPlanner(**options))
    registry.planners.register("motion_probe", lambda options: MotionProbePlanner(**options))
    registry.policies.register("pi05", Pi05Adapter)
    registry.kinematics.register("urdf", UrdfKinematics)
    registry.motions.register(
        "fixed_steps", lambda spec, timing, options: FixedStepsMotion(spec, timing, **options)
    )
    registry.motions.register("local_interpolation", _local_motion)
    registry.motions.register("urdf_joint", _joint_motion)
    registry.codecs.register("corobot", CoRobotCodec)
    registry.codecs.register("corobot_ik_frame", CoRobotIKFrameCodec)
    registry.observers.register("local", ObserverProcess)
    return registry


def create_observer(trace_dir: Path, port: int = 0, registry: ComponentRegistry | None = None) -> Observer:
    """Construct the optional viewer without creating planners or action backends."""
    return (registry or default_registry()).observers.resolve("local")(trace_dir, port)


class ApplicationFactory:
    def __init__(self, settings: AppSettings, registry: ComponentRegistry | None = None) -> None:
        self._settings = settings.model_copy(deep=True)
        self.registry = registry if registry is not None else default_registry()

    def create_session(self, event_sink: EventSink | None = None) -> AgentSession:
        # Each session owns independent mutable components and robot configuration.
        settings = self._settings.model_copy(deep=True)
        spec = settings.robot
        session_id = uuid4().hex
        with ExitStack() as rollback:
            try:
                codec = self.registry.codecs.resolve(settings.simulator_codec.type)(
                    spec, settings.simulator_codec.options
                )
                rollback.callback(codec.close)
                planner = self.registry.planners.resolve(settings.planner.type)(settings.planner.options)
                rollback.callback(planner.close)
                motion = self.registry.motions.resolve(settings.motion_backend.type)(
                    spec,
                    settings.motion.to_timing(settings.execution.execute_steps),
                    settings.motion_backend.options,
                )
                rollback.callback(motion.close)
                if settings.ranked_motion_candidates:
                    motion = RankedMotionExecutor(motion, spec, settings.ranked_motion_candidate_count)
                    rollback.callback(motion.close)
                    planner = RankedMotionPlanner(planner, motion)
                    rollback.callback(planner.close)
                executors: list[ActionExecutor] = [motion]
                if settings.policy is not None:
                    policies: dict[str, PolicyBackend] = {}
                    for name, plugin in settings.policy.backends.items():
                        policy = self.registry.policies.resolve(plugin.type)(spec, plugin.options)
                        rollback.callback(policy.close)
                        policies[name] = policy
                    executors.append(PolicyExecutor(policies, settings.policy.default))
                router = ExecutionRouter(executors, settings.execution)
                kinematics = None
                if settings.kinematics is not None:
                    kinematics = self.registry.kinematics.resolve(settings.kinematics.type)(
                        spec, settings.kinematics.options
                    )
                    rollback.callback(kinematics.close)
                runtime = AgentRuntime(
                    spec, planner, router, event_sink or LoggingEventSink(session_id), kinematics
                )
                session = AgentSession(
                    runtime,
                    session_id,
                    trace_dir=settings.logging.trace_dir,
                    codec=codec,
                    run_metadata=settings.hardware_snapshot(),
                )
            except (ContractError, TypeError, KeyError) as exc:
                raise ConfigurationError(f"component configuration: {exc}") from exc
            # After construction, runtime -> router -> executors own component cleanup.
            rollback.pop_all()
        return session

    def describe(self) -> dict[str, Any]:
        with self.create_session() as session:
            return {
                "robot": session.runtime.spec.describe(),
                "execution": session.runtime.router.describe(),
                "note": "Configuration only; model connectivity and robot calibration not tested.",
                **({"hardware": self._settings.hardware_snapshot()} if self._settings.embodiment else {}),
            }


def build_session(
    config: AppSettings | dict[str, Any],
    *,
    registry: ComponentRegistry | None = None,
    event_sink: EventSink | None = None,
) -> AgentSession:
    settings = config if isinstance(config, AppSettings) else parse_settings(config)
    return ApplicationFactory(settings, registry).create_session(event_sink)
