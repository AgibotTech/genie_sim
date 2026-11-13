#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Explicit GPU/ROS probe: successive material/base-XY layouts, deterministic hold policy.

This is a lifecycle diagnostic, never a model evaluation result. It runs the
real benchmark owner/scorer/renderer and checks all initial state arrays and
process identities across reset. Invoke directly with a fresh --output-dir.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def state_differences(a, b):
    """Raw archives stay unchanged; q and -q encode the same physical rotation."""
    differences = {}
    for name in a:
        x, y = a[name], b[name]
        assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all(), name
        differences[name] = float(np.max(np.abs(x - y))) if x.size else 0.0
        if name == "body_q":
            assert np.allclose(x[:, :3], y[:, :3], rtol=0, atol=1e-6), "body positions"
            q, r = x[:, 3:].astype(float), y[:, 3:].astype(float)
            assert np.all(np.linalg.norm(q, axis=1) > 0.9) and np.all(np.linalg.norm(r, axis=1) > 0.9)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            r /= np.linalg.norm(r, axis=1, keepdims=True)
            distance = np.minimum(np.linalg.norm(q - r, axis=1), np.linalg.norm(q + r, axis=1))
            angle = float(np.max(4 * np.arcsin(np.clip(distance / 2, 0, 1)))) if len(q) else 0.0
            # Float32 FK -> native affine transform -> normalized quaternion
            # roundtrip measured below 4e-6 rad on these three 83-link robots.
            assert angle <= 1e-5, f"body rotation changed: {angle} rad"
            differences["body_rotation_max_error_rad"] = angle
            differences["body_position_max_abs_diff"] = float(np.max(np.abs(x[:, :3] - y[:, :3]))) if x.size else 0.0
        else:
            assert np.allclose(x, y, rtol=0, atol=1e-6), f"reset failed to restore {name}: {differences[name]}"
    return differences


def verify_run(directory):
    from genie_sim_engine_libuipc.physical_state_recording import PhysicalStateArchive

    episodes = sorted(Path(directory).glob("episode_*"))
    assert len(episodes) >= 2
    initial, receipts, identities, metadata = [], [], [], []
    for episode in episodes:
        archive = PhysicalStateArchive(episode / "physical_state")
        frames = archive.iter_frames()
        initial.append(next(frames)["state"])
        metadata.append(archive.manifest["metadata"])
        for _ in frames:
            pass
        assert archive.manifest["metadata"]["epoch"] == len(initial) - 1
        receipts.append(json.loads((episode / "material_generalization_applied.json").read_text()))
        result = json.loads((episode / "result.json").read_text())
        assert result["execution_status"] == "completed" and result["scoring_status"] == "scored"
        identities.append(result["source_identity"])
    poses = [m["scene_config"]["robot"].get("init_base_pose", {}) for m in metadata]
    robot_mask = np.array(
        [
            str(label).startswith("/" + metadata[0]["robot_prefix"].strip("/") + "/")
            for label in metadata[0]["body_labels"]
        ]
    )
    assert robot_mask.any()
    differences = []
    for state, pose in zip(initial[1:], poses[1:]):
        # Compare copies; authored base motion is allowed only for the robot.
        normalized = {key: value.copy() for key, value in state.items()}
        translation = [pose.get("x", 0) - poses[0].get("x", 0), pose.get("y", 0) - poses[0].get("y", 0), 0]
        normalized["body_q"][robot_mask, :3] -= np.array(translation)
        differences.append(state_differences(initial[0], normalized))
    assert all(isinstance(i["engine_pid"], int) and isinstance(i["cpu_worker_pid"], int) for i in identities)
    for key in ("source_id", "engine_pid", "cpu_worker_pid"):
        assert len({i[key] for i in identities}) == 1, key
    assert [i["epoch"] for i in identities] == list(range(len(episodes)))
    assert len({r["renderer_pid"] for r in receipts}) == 1
    assert len({r["plan_sha256"] for r in receipts}) == len(receipts)
    scores = Path(directory) / "evaluate_ret_00.json"
    assert len(json.loads(scores.read_text())["details"]) == len(episodes)
    return dict(
        diagnostic_only=True,
        layouts=len(episodes),
        preloaded_base_poses=poses,
        identities=identities,
        initial_state_differences=differences,
        quaternion_comparison="normalized q/-q rotation equivalence; raw archives preserved",
        renderer_pid=receipts[0]["renderer_pid"],
        material_plans=[r["plan_sha256"] for r in receipts],
        video_inputs=[str(e / "observations") for e in episodes],
        scores=str(scores),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--layouts", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        report = verify_run(args.output_dir)
        (args.output_dir / "reset-verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return
    if not args.config or not args.repo_root or args.layouts < 2:
        parser.error("running the GPU probe requires --config, --repo-root and at least two layouts")
    from geniesim_benchmark.libuipc import runner
    from geniesim_benchmark.libuipc.config import load_config, validate_settings, write_json
    from geniesim_benchmark.plugins.output_system.dexhand2 import write_summary

    class HoldPolicy:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

        def infer(self, payload, *, archive=None):
            state = payload["params"]["states"]
            arms, fingers = state["arm_joint_states"], state["gripper_states"]
            return {
                "left_arm": {"kind": "JOINT_ABS", "values": [arms[:7]] * 4},
                "right_arm": {"kind": "JOINT_ABS", "values": [arms[7:]] * 4},
                "left_effector": [fingers[:20]] * 4,
                "right_effector": [fingers[20:]] * 4,
            }

    config = load_config(args.config, repo_root=args.repo_root)
    config["benchmark"].update(num_instances=args.layouts, num_episode=1)
    offsets = config["libuipc"].get("instance_base_xy_offsets")
    if offsets is not None:
        if len(offsets) < args.layouts:
            parser.error("diagnostic --layouts exceeds the authored base pose count")
        # Scope only this explicit diagnostic, preserving the original file.
        config["libuipc"]["instance_base_xy_offsets"] = offsets[: args.layouts]
    config["libuipc"].update(max_steps=4, stop_on_success=False)
    validate_settings(config)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner.InferenceClient = HoldPolicy
    evaluations, results = [], []
    with runner.LibuipcRunSession(
        config, "diagnostic-hold-policy", args.repo_root, args.output_dir, headless=True
    ) as session:
        for index in range(args.layouts):
            result = runner.run_one_episode(session, index, evaluations=evaluations)
            results.append(result)
            write_summary(args.output_dir, evaluations)
            write_json(args.output_dir / "probe-progress.json", results)
            if result["execution_status"] != "completed":
                raise RuntimeError(str(result))
    report = verify_run(args.output_dir)
    report["session_shutdown_completed"] = True
    write_json(args.output_dir / "reset-verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
