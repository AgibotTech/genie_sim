# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0
"""IPC scoring keeps raw state and frame indexes after success and failure."""

import json
from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from geniesim_benchmark.libuipc import runner, materials, streaming


def test_unsealed_archive_is_not_deleted(tmp_path):
    archive = tmp_path / "physical_state"
    archive.mkdir()
    (archive / "manifest.json").write_text('{"recording_closed": false}')
    (archive / "particle_q.bin").write_bytes(b"native writer still owns this")
    runner.discard_physical_state(tmp_path)
    assert (archive / "particle_q.bin").exists()


def test_sealed_archive_does_not_request_file_removal(tmp_path, monkeypatch):
    archive = tmp_path / "physical_state"
    archive.mkdir()
    (archive / "manifest.json").write_text('{"recording_closed": true}')
    (archive / "particle_q.bin").write_bytes(b"verified original state")
    (archive / "frames.jsonl").write_text("verified frame index")

    def forbid(*args, **kwargs):
        pytest.fail("IPC archive retention must not request deletion")

    monkeypatch.setattr(Path, "unlink", forbid)
    runner.discard_physical_state(tmp_path)
    assert (archive / "particle_q.bin").read_bytes() == b"verified original state"
    assert (archive / "frames.jsonl").read_text() == "verified frame index"


@pytest.mark.parametrize("failure", [None, "inference", "scoring", "scorer_stop"])
def test_episode_keeps_physical_state_after_scoring_and_failures(tmp_path, monkeypatch, failure):
    episode = tmp_path / "episode"
    archive = episode / "physical_state"
    archive.mkdir(parents=True)
    (archive / "manifest.json").write_text('{"recording_closed": false}')
    (archive / "particle_q.bin").write_bytes(b"temporary physical state")
    (archive / "frames.jsonl").write_text("frame index")
    (episode / "material_generalization.json").write_text("{}")
    (episode / "head.webm").write_bytes(b"video is not a cleanup target")
    events = []

    class Client:
        identity = {"epoch": 0, "source_id": "engine"}

        def observe(self, **kwargs):
            return {"snapshot": {"episode_index": 0}, "epoch": 0}

        def begin_episode(self):
            events.append("begin")

        def finish_episode(self):
            events.append("native_close")
            (archive / "manifest.json").write_text('{"recording_closed": true}')
            return {}

    class Scorer:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append("scorer_start")

        def finish(self, verification):
            assert (archive / "particle_q.bin").exists()
            assert events[-1] == "verify"
            events.append("score")
            if failure == "scoring":
                raise RuntimeError("scoring failed")
            return {"score": 25, "scoring_status": "scored"}

        def stop(self):
            assert (archive / "particle_q.bin").exists()
            events.append("scorer_stop")
            if failure == "scorer_stop":
                raise RuntimeError("scorer still running")

    def verify(directory):
        assert events[-1] == "native_close"
        assert (archive / "particle_q.bin").exists()
        events.append("verify")
        return {"physical_frames": 2}

    def run(*args, **kwargs):
        if failure == "inference":
            raise TimeoutError("model unavailable")
        return {"execution_status": "completed", "completed": True}

    for name in ("validate_identity", "validate_task_snapshot"):
        monkeypatch.setattr(runner, name, lambda *args: None)
    monkeypatch.setattr(materials, "validate_material_capture", lambda *args, **kwargs: {})
    monkeypatch.setattr(materials, "wait_for_material_receipt", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "load_fixed_joints", lambda *args: {})
    monkeypatch.setattr(runner, "InferenceClient", lambda *args, **kwargs: NS(close=lambda: None))
    monkeypatch.setattr(streaming, "StreamingScoringJob", Scorer)
    monkeypatch.setattr(runner, "verify_physical_archive", verify)
    monkeypatch.setattr(runner, "run_episode", run)
    monkeypatch.setattr(runner, "merge_score", lambda result, report: {**result, **report})
    monkeypatch.setattr(runner, "common_result", lambda task, result, *args, **kwargs: {"execution": result})
    settings = dict(
        task="fold_cloth",
        wait_timeout=1,
        infer_timeout=1,
        scoring_execution="streaming",
        scoring_drain_timeout_s=1,
        instruction="Fold",
        max_steps=1,
        action_chunk_steps=1,
        fps=30,
    )
    session = NS(
        episodes=[dict(directory=episode, token="test", instance_id=0, attempt_index=0)],
        config={"benchmark": {}, "libuipc": settings},
        client=Client(),
        process=NS(pid=1),
        assets={},
        assembled={"policy_frame_bindings": {}},
        scene=tmp_path / "scene.yaml",
        manifest=tmp_path / "manifest.json",
        repo_root=tmp_path,
        endpoint="test",
        robot_usd_path="robot.usda",
        directory=tmp_path,
    )
    evaluations = []
    result = runner.run_one_episode(session, 0, evaluations=evaluations)
    assert events == ["scorer_start", "begin", "native_close", "verify", "score", "scorer_stop"]
    assert (archive / "manifest.json").is_file()  # Still required by shutdown handshake.
    assert (archive / "particle_q.bin").exists()
    assert (archive / "frames.jsonl").exists()
    assert (episode / "head.webm").read_bytes() == b"video is not a cleanup target"
    assert json.loads((episode / "result.json").read_text()) == result
    assert evaluations == [{"execution": result}]
    if failure is None:
        assert result["execution_status"] == "completed" and result["score"] == 25
    else:
        assert result["execution_status"] == "failed"
