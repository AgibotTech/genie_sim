# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Exercise assembly contracts without installing any physics backend."""

from pathlib import Path

import pytest

from genie_sim_engine_core.assemble import (
    AssembleArtifacts,
    AssembleContext,
    AssembleInputs,
    AssembleStage,
    InvariantMiddleware,
    Manifest,
    Pipeline,
    PipelineRunner,
    RobotSource,
    StageResult,
)


@pytest.fixture
def ctx(tmp_path):
    return AssembleContext(
        inputs=AssembleInputs(
            scene_path=tmp_path / "scene.yaml",
            output_dir=tmp_path,
            base_path=tmp_path,
            physics_engine="test_backend",
            scene_cfg={},
            robot_source=RobotSource(kind="usd", config={}),
        ),
        artifacts=AssembleArtifacts(),
    )


class ArtifactStage(AssembleStage):
    name = "artifact"
    produces = frozenset({"artifact.json"})

    def run(self, ctx):
        path = ctx.inputs.output_dir / "artifact.json"
        path.write_text("{}")
        return StageResult(
            files_added={"artifact.json": path},
            manifest_slice={"physics": {"test_backend": {"ready": True}}},
        )


def test_missing_prerequisite_fails_before_execution():
    class MissingInput(ArtifactStage):
        requires = frozenset({"missing.json"})

    with pytest.raises(ValueError, match="no earlier stage produces"):
        Pipeline(name="invalid", stages=[MissingInput]).validate()


def test_pipeline_artifact_and_manifest_round_trip(ctx):
    runner = PipelineRunner(
        Pipeline(name="test", stages=[ArtifactStage]),
        middleware=[InvariantMiddleware()],
    )
    manifest = runner.run(ctx)
    path = ctx.inputs.output_dir / "manifest.json"
    manifest.write(path)
    restored = Manifest.read(path)
    assert restored.physics == {"test_backend": {"ready": True}}
    assert ctx.artifacts.files["artifact.json"].is_file()


def test_declared_artifact_must_exist(ctx):
    class MissingArtifact(ArtifactStage):
        def run(self, ctx):
            return StageResult(files_added={"artifact.json": Path("missing.json")})

    runner = PipelineRunner(
        Pipeline(name="invalid", stages=[MissingArtifact]),
        middleware=[InvariantMiddleware()],
    )
    with pytest.raises(RuntimeError, match="file does not exist"):
        runner.run(ctx)


@pytest.mark.parametrize("explicit", [False, True])
def test_skipped_stage_balances_middleware_hooks(ctx, explicit):
    events = []

    class Skipped(ArtifactStage):
        def skip_reason(self, ctx):
            events.append("skip_reason")
            return "not needed"

        def run(self, ctx):
            pytest.fail("skipped stage executed")

    class Recorder:
        def before(self, stage, ctx):
            events.append("before")

        def after(self, stage, ctx, result, err):
            assert result is None and err is None
            events.append("after")

    runner = PipelineRunner(Pipeline(name="skip", stages=[Skipped]), middleware=[Recorder()])
    if explicit:
        runner.explicit_skips.add("artifact")
    runner.run(ctx)
    assert events == (["before", "after"] if explicit else ["before", "skip_reason", "after"])


def test_stage_cannot_overwrite_core_manifest_version():
    with pytest.raises(ValueError, match="version"):
        Manifest().merge_slice({"version": 999}, "bad_stage")
