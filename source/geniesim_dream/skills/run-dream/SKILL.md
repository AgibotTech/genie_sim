---
name: run-dream
description: >
  Run Genie Sim Dream G1/G2 world-model rollouts with recorded actions or the
  pi0.5 policy, then review the generated video with a VLM. Use for Dream
  inference, example replay, or closed-loop checks after its three Python
  environments and required checkpoints have been prepared.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "1.0"
prerequisites: []
inputs:
  - name: asset_root
    desc: Directory containing fd_cosmos/, policy/ and vlm/
    required: false
    default: checkpoints/
  - name: example_id
    desc: G1 or G2 example bundle to run from examples/
    required: false
  - name: mode
    desc: Recorded action replay or external pi0.5 policy
    required: false
    default: external
outputs:
  - desc: Session output under DREAM_OUTPUT_DIR, including vision.mp4, actions.npy and an evaluation JSON after VLM review
---

## When to Use

- Run a public G1/G2 Dream example with recorded actions or the pi0.5 policy.
- Run a 20-chunk pi0.5 policy rollout and review its video and visual report.

## Critical Patterns

1. Use the three separate environments in the [installation guide](../../docs/setup.md).
   The [asset guide](../../docs/assets.md) lists the required checkpoints.
   `DREAM_ASSET_ROOT` defaults to `source/geniesim_dream/checkpoints/`.
2. The launcher starts four services and loads the G1 bread policy and Qwen3.5
   evaluator. Other task policies load on first use.
3. Replay uses saved actions and cannot exceed the example's recorded chunks.
   The pi0.5 loop predicts actions from generated observations; run 20 chunks
   for a full policy rollout. Both modes default to 32 action rows per chunk.
   Select an example matching the robot and task.

## Workflow

1. Prepare the environments and assets using the guides above.
2. Start `./tools/start_services.sh` from `source/geniesim_dream/`; keep it
   running until the launcher reports all services ready.
3. Open the UI, choose the requested mode and an example, then create a session.
   For the policy loop, run 20 chunks; for replay, stay within the example's
   recorded length. Generate the full video and optionally run VLM evaluation.
4. Inspect `vision.mp4`, `actions.npy` and any report under `evaluations/` in
   the session output. Record the model, example, settings and session ID.

## Commands

From the repository root:

```bash
cd source/geniesim_dream
./tools/start_services.sh
```

Set `DREAM_ASSET_ROOT` before launching if assets are elsewhere. Keep the first
terminal open; Ctrl+C stops the services. To inspect status from a second
terminal:

```bash
cd source/geniesim_dream
./tools/start_services.sh status
```

The UI is at `http://127.0.0.1:8971/`. Choose `g1_bread_ep0009` or
`g2_cloth_ep0020`, select the pi0.5 closed loop, keep the default 32 action
rows and 20 chunks, then select **Run continuously**.

## Notes

- The world model predicts visual changes; it does not return measured robot
  state. The VLM report estimates task completion from generated video.

## Resources

- [Package README](../../README.md) — overview and getting started.
- [Installation guide](../../docs/setup.md) — environments and install checks.
- [Running guide](../../docs/running.md) — launcher and browser workflow.
- [Asset guide](../../docs/assets.md) — downloads and paths.
- [Agent guide](../../AGENTS.md) — architecture rules.
