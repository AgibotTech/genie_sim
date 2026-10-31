# Choosing a run configuration

`examples/` keeps one runnable deployment configuration, for the G2 omnipicker, and one offline demo
that does not call a model. Other capabilities (switching planner provider, toggling
depth/geometry/five candidates, and so on) are changed on top of this configuration as described in
the "Configuration options" section below, rather than shipping a separate JSON for every
combination.

## Overview

| Configuration | Embodiment | planner.type | Notes |
|---|---|---|---|
| [g2_llm_recipe.json](g2_llm_recipe.json) | `G2_omnipicker` | `embodied` | Flagship: five-candidate IK, depth queries, grasp geometry, gripper center, three-camera calibration, terminal diagnostics |
| [demo.toml](demo.toml) | `demo_dual_arm` | `scripted` | Offline demo without model calls, used with `observation.json` |

`g2_llm_recipe.json` loads its gripper description from the built-in `g2_omnipicker` embodiment
preset, so no extra file is needed:

```bash
AGENT2ACT_LIVE_LLM_KEY=... uv run --locked agent2act check --config examples/g2_llm_recipe.json
```

`observation.json` is a sample request for `demo.toml`, not a run configuration:

```bash
uv run --locked agent2act replay --config examples/demo.toml --request examples/observation.json
```

See [URDF sources and licenses](urdf/README.md) for robot-description provenance
and redistribution status.

## Configuration options

All configurations share one structure; override the fields you need for a deployment.

### Switching planner and model provider

`planner.type` selects the protocol implementation and `execution.mode` selects motion/policy/auto;
the two are independent. Built-in types: `responses` (OpenAI Responses protocol), `embodied`
(Responses plus embodied geometry tools), `anthropic` (native Messages), `gemini` (native
GenerateContent), `chat_completions` (OpenAI-compatible endpoints), and `scripted` / `motion_probe`
(no model calls).

`g2_llm_recipe.json` defaults to `embodied`. To switch to Claude or Gemini, replace the whole
`planner` block with:

```jsonc
// Claude: native Messages
"planner": {
  "type": "anthropic",
  "model": "claude-opus-5-5",
  "endpoint": "https://api.anthropic.com/v1/messages",
  "api_key_env": "ANTHROPIC_API_KEY",
  "json_mode": false,
  "reasoning_effort": "medium",
  "reasoning_summary": null,
  "max_output_tokens": 32768,
  "timeout_s": 600
}

// Gemini: native GenerateContent
"planner": {
  "type": "gemini",
  "model": "gemini-4-argon",
  "endpoint": "https://generativelanguage.googleapis.com/v1beta/models/gemini-4-argon:generateContent",
  "api_key_env": "GOOGLE_API_KEY",
  "json_mode": true,
  "reasoning_effort": "medium",
  "max_output_tokens": 32768,
  "timeout_s": 600
}

// Alibaba Bailian Qwen: Chat Completions compatible endpoint
"planner": {
  "type": "chat_completions",
  "model": "qwen3-vl-plus",
  "endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
  "api_key_env": "DASHSCOPE_API_KEY",
  "enable_thinking": false
}
```

Parameter support differs between protocols. `check` validates only the local configuration and
component construction, not whether the remote service accepts these parameters:

- `responses` supports `reasoning_effort`, `reasoning_summary`, `json_mode` and HTTP retries.
- `embodied` adds read-only geometry tools (depth queries, grasp pose synthesis, etc.) on top of that.
- `anthropic` does not support `json_mode`, streaming or `reasoning_summary`; `max_tokens` includes thinking.
- `gemini` expresses the reasoning level through `thinkingConfig`; `reasoning_summary` supports only `auto`.
- `chat_completions` does not support `reasoning_effort` / `reasoning_summary`; `enable_thinking` is a service extension.

The API key is read only from the environment variable named by `planner.api_key_env`; set it to
`null` for local services that need no key. Switching provider only changes the `planner` block;
`robot`, `execution`, `motion`, `motion_backend` and `simulator_codec` stay as they are.

### Location service (cloud)

To let external workers connect, override the listen address:

```bash
uv run --locked --no-sync python scripts/serve_llm.py \
  --config examples/g2_llm_recipe.json --host 0.0.0.0 --port 8000
```

The cloud image must contain the configuration file, the embodiment JSON and the URDF; the
`embodiment` path is relative to the configuration file, and URDF paths are relative to the project
root.

### Switching embodiment

One configuration serves exactly one embodiment. When switching embodiment, replace all of the
following together; changing only `robot_type` is not enough:

| Field | Notes |
|---|---|
| `robot.robot_type` | Embodiment identifier in the protocol; must match the simulator mapping |
| `robot.left_joints` / `right_joints` / `body_joints` | Joint name order |
| `robot.gripper_bounds` | Gripper command range (`-0.785` to `0` for omnipicker) |
| `motion_backend` | URDF path and IK configuration |
| `simulator_codec` | `corobot_ik_frame` (omnipicker) or `corobot` (no conversion) |
| `planner.use_gripper_center` + `gripper_center_offset_m` | Gripper-center conversion; must be consistent with the embodiment |

The gripper description (TCP offset, command polarity, grasp geometry) is supplied separately through
the `embodiment` block (a built-in `preset` or a `path` to a version 2 definition file).
Kinematics, limits and gripper range always come from the configuration file, so startup fails when
the embodiment's `robot_type` does not match the configuration. Only the G2 omnipicker configuration,
embodiment and URDFs are bundled; another robot needs its own configuration, embodiment file and URDF.

## Feature switches

All of the following are changed directly in `g2_llm_recipe.json`; restart the service afterwards.

### Five-candidate IK ranking

```jsonc
"ranked_motion_candidates": true,
"ranked_motion_candidate_count": 5
```

The model returns five targets ordered from best to worst; they are checked locally in order against
IK and action validation, and the first one that passes is executed. Set the count to `10` for ten
candidates; remove the field or set it to `false` to return to a single target. See
[Configuration reference: motion behavior](../docs/configuration-reference.md#motion-behavior).

### Depth and height prior

Depth, gripper center and height prior are independent and can also be overridden from the CLI:

```bash
--depth / --no-depth                  # sets both execution.need_depth and planner.use_depth
--gripper-center / --no-gripper-center
--height-prior / --no-height-prior
--height-calibration <file.json>      # supplies calibration values only; does not enable the height prior
```

`g2_llm_recipe.json` enables depth and disables the height prior. Enabling the height prior instead
uses the static table height in `planner.height_calibration` rather than depth measurements. A static
calibration is valid only for a specific fixed scene and must not be treated
as a measured contact height. See
[observation switches](../docs/configuration-reference.md#observation-switches).

### Motion continuation and confirmation wait

`execution.max_motion_continuations` lets the same target be recompiled from a new observation when
tracking is on target, reducing the number of LLM calls:

```jsonc
"execution": { "max_motion_continuations": 1 }
```

Combined with `complete_hold_steps`, which lengthens the hold window after completion, this reduces
premature completion decisions. See
[Configuration reference: motion behavior](../docs/configuration-reference.md#motion-behavior).

### Waist control

When `motion_backend.waist_yaw_joint` is present, the model may issue `waist_yaw`, and
`execution.body_execute_steps` sets how many steps of waist motion are issued. Removing the field
disables waist control, and waist actions from the model are rejected. See
[Configuration reference: waist control](../docs/configuration-reference.md#waist-control).

### Action diagnostics without the action model

`planner.type="motion_probe"` executes fixed actions directly, to test how the same target responds
under different action durations and continued holds; it involves neither the LLM nor the action
model. See
[Configuration reference: fixed-action diagnostics](../docs/configuration-reference.md#fixed-action-diagnostics).

## π0.5 policy

When `execution.mode` is `auto` and `allowed_modes` contains `policy`, the `policy` backend can be
used for execution:

```jsonc
"execution": { "mode": "auto", "allowed_modes": ["motion", "policy"] },
"policy": {
  "default": "pi05",
  "backends": {
    "pi05": { "type": "pi05", "host": "127.0.0.1", "port": 8000, "checkpoint": "...", "...": "..." }
  }
}
```

π0.5 is a policy backend, not a planner; the LLM still chooses between motion and policy and
requests budget through `policy_chunks`. The checkpoint, input/output mapping, `dt_s` and camera keys
must be calibrated per deployment, and the example parameters do not mean they have been verified in
Isaac Sim. The π0.5 dependencies come from the `pi05` extra of `uv`. See
[Policy backend contract](../docs/architecture.md#policy-backend-contract) and
[Proposal review contract](../docs/architecture.md#proposal-review-contract).
