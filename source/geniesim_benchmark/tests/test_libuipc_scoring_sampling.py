# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Sampled ADER replay keeps final states, full verification and success identity."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from geniesim_benchmark.libuipc import scoring, streaming

physical = pytest.importorskip("genie_sim_engine_libuipc.physical_state_recording")
PhysicalStateArchiveFollower = physical.PhysicalStateArchiveFollower
PhysicalStateRecorder = physical.PhysicalStateRecorder
PhysicalStateRecordingConfig = physical.PhysicalStateRecordingConfig

REPO = Path(__file__).resolve().parents[3]


def make_episode(directory, count):
    profile = scoring.load_profile("fold_cloth")
    mesh = profile["mesh"]
    directory.mkdir()
    scoring.write_json(
        directory / "asset_identity.json",
        {
            "meshes": [
                {
                    "name": "tshirt",
                    "prim_path": mesh["path"],
                    "vertices": mesh["vertex_count"],
                    "face_counts_sha256": mesh["face_counts_sha256"],
                    "face_indices_sha256": mesh["face_indices_sha256"],
                }
            ]
        },
    )
    recorder = PhysicalStateRecorder(
        PhysicalStateRecordingConfig(str(directory / "physical_state")),
        metadata={
            "source_id": "sampling-test",
            "epoch": 0,
            "objects": [
                {
                    "name": "tshirt",
                    "particle_start": 0,
                    "particle_count": mesh["vertex_count"],
                    "mesh_prim_path": mesh["path"],
                }
            ],
        },
    )
    state = SimpleNamespace(
        particle_q=np.zeros((mesh["vertex_count"], 3), dtype=np.float32),
        particle_qd=np.zeros((mesh["vertex_count"], 3), dtype=np.float32),
        body_q=np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float64),
        body_qd=np.zeros((1, 6), dtype=np.float64),
        joint_q=np.zeros(1, dtype=np.float64),
        joint_qd=np.zeros(1, dtype=np.float64),
    )
    for tick in range(count):
        recorder.record(state, tick=tick, sim_time=tick * 0.01, dt=0.01 if tick else 0.0, initial=tick == 0)
    recorder.close()
    return directory


@pytest.mark.parametrize("follow", [False, True])
@pytest.mark.parametrize(
    "count,expected", [(2, [0, 1]), (10, [0, 9]), (11, [0, 10]), (12, [0, 10, 11]), (26, [0, 10, 20, 25])]
)
def test_real_archive_replay_samples_frames_and_preserves_final_state(tmp_path, follow, count, expected):
    episode = make_episode(tmp_path / "episode", count)
    output = episode / "scoring"
    result = scoring.score_physical_episode("fold_cloth", episode, output, REPO, follow=follow)
    rows = [json.loads(line) for line in (output / "checker_frames.jsonl").read_text().splitlines()]
    assert [row["tick"] for row in rows] == expected
    assert [row["physical_frame_index"] for row in rows] == expected
    assert result["observed_frames"] == len(expected)
    assert result["physical_frames"] == count
    assert result["scoring_frame_interval"] == 10
    assert result["scoring_status"] == "scored"
    assert result["score"] == 0


@pytest.mark.parametrize("follow", [False, True])
def test_unsampled_frame_corruption_still_rejects_score(tmp_path, follow):
    episode = make_episode(tmp_path / "episode", 12)
    manifest = json.loads((episode / "physical_state/manifest.json").read_text())
    with (episode / "physical_state/particle_q.bin").open("r+b") as stream:
        stream.seek(5 * manifest["arrays"]["particle_q"]["frame_bytes"])
        stream.write(b"\xff")
    output = episode / "scoring"
    with pytest.raises(ValueError, match="checksum mismatch"):
        scoring.score_physical_episode("fold_cloth", episode, output, REPO, follow=follow)
    assert not (output / "checker_result.json").exists()


@pytest.mark.parametrize("task", ["fold_cloth", "cloth_hanger", "drawstring_bag"])
def test_named_body_states_use_the_same_sampled_clock(task):
    labels = ["hanger"] if task == "cloth_hanger" else ["bottle_a", "bottle_b"]
    name = "drawstring_bag" if task == "drawstring_bag" else "tshirt"
    profile = {"mesh": {"path": "/cloth", "vertex_count": 2}, "bottles": [{"name": label} for label in labels]}

    class Archive:
        manifest = {
            "metadata": {
                "objects": [{"name": name, "particle_start": 0, "particle_count": 2, "mesh_prim_path": "/cloth"}],
                "body_labels": labels,
            }
        }

        def iter_frames(self):
            for tick in range(13):
                yield {
                    "tick": tick,
                    "sim_time": tick * 0.01,
                    "state": {
                        "particle_q": np.full((2, 3), tick),
                        "body_q": np.array([[tick, 0, 0, 0, 0, 0, 1]] * len(labels)),
                    },
                }

    frames, _ = scoring.named_frames(Archive(), task, profile)
    rows = list(frames)
    assert [row[1] for row in rows] == [0, 10, 12]
    for index, tick, sim_time, points, *poses in rows:
        assert index == tick
        np.testing.assert_array_equal(points, np.full((2, 3), tick))
        if poses:
            transforms = poses[0][None] if task == "cloth_hanger" else poses[0]
            np.testing.assert_array_equal(transforms[:, 0, 3], np.full(len(labels), tick))


def test_streaming_worker_publishes_sampled_report_after_full_archive_verification(tmp_path):
    episode = make_episode(tmp_path / "episode", 26)
    follower = PhysicalStateArchiveFollower(episode / "physical_state").initialize()
    for _ in follower.iter_frames():
        pass
    scoring.write_json(episode / "recording_verified.json", follower.verification)
    job = streaming.StreamingScoringJob("fold_cloth", episode, REPO)
    assert job.drain_timeout_s == 3600
    try:
        job.start()
        report = job.finish(follower.verification)
        assert report["observed_frames"] == 4
        assert report["physical_frames"] == 26
        assert (episode / "scoring/checker_result.json").is_file()
    finally:
        job.stop()


def test_success_evidence_matches_sample_number_and_original_physical_frame(tmp_path, monkeypatch):
    episode = make_episode(tmp_path / "episode", 26)
    job = streaming.StreamingScoringJob("fold_cloth", episode, REPO)
    scoring.score_physical_episode("fold_cloth", episode, job.staging, REPO)
    path = job.staging / "checker_frames.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[2]["score"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    evidence = {"tick": 20, "sim_time_s": 0.2, "observed_frames": 3, "physical_frame_index": 20}
    monkeypatch.setattr(job, "success_notification", lambda: evidence)
    report = {"success": True, "score": 1, "observed_frames": 4}
    job._validate_success_report(report, 26)
    evidence["physical_frame_index"] = 2
    with pytest.raises(ValueError, match="does not include the success notification frame"):
        job._validate_success_report(report, 26)


@pytest.mark.parametrize("task", ["cloth_hanger", "fold_cloth", "drawstring_bag"])
def test_public_tasks_allow_one_hour_scoring_drain(task):
    package = scoring.PACKAGE
    for folder in (package / "config",):
        config = yaml.safe_load((folder / f"g2crsbwuji_dexmanip_{task}.yaml").read_text())
        assert config["libuipc"]["scoring_drain_timeout_s"] == 3600
