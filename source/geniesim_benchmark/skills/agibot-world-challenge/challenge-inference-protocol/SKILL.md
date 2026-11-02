---
name: challenge-inference-protocol
description: >
  Reference for the Simulation Challenge inference wire protocol — the exact obs (input) and
  action (output) message format exchanged between the gateway/genie-sim simulator and the
  contestant's inference agent over the reverse WebSocket tunnel. Covers the JSON-RPC envelope,
  image encoding, per-board state/action joint layout, the plain-msgpack response caveat, and the
  authoritative source files.
  Trigger: When the user asks about "推理接口协议", "inference protocol", "obs/action format",
  "观测/动作格式", "网关下发什么", "agent 返回什么", "what does the gateway send", "result envelope",
  "state layout", "动作怎么拆", or needs to implement/debug the obs→model→action adapter in a tunnel agent.
metadata:
  author: zy
  version: "1.0"
---

# challenge-inference-protocol — Inference wire protocol (obs in / action out)

This documents the **application-layer payload** the Simulation Challenge gateway (driven by the
genie-sim simulator) exchanges with a contestant inference agent. The transport (reverse WebSocket
tunnel, control frames, data-frame framing) is specified in the **appendix of
`challenge-run-agent`**; this skill covers only what goes **inside** the data-frame payload.

> The gateway treats the payload as opaque bytes. The schema below is defined by the genie-sim
> client, not the gateway. **Authoritative source:**
> `source/geniesim_benchmark/src/geniesim_benchmark/benchmark/policy/corobotpolicy.py`
> — `get_payload()` builds the request, `_parse_result()` / `infer()` parse the response.
> **Reference agent implementation:** `ACoT-VLA/scripts/tunnel_agent.py` — `_adapt_obs()` (input),
> `_build_action_response()` (output).

---

## 1. Input — observation (gateway → agent)

A msgpack-encoded JSON-RPC envelope. Decode with `msgpack_numpy.unpackb` (str keys).

```jsonc
{
  "method": "infer",
  "params": {
    "timestamps": { "head": <ns int>, "states": <ns int> },
    "images": {
      "head":       { "encoding": "JPEG", "image_data": <bytes>, "height": 400,  "width": 640  },
      "hand_left":  { "encoding": "JPEG", "image_data": <bytes>, "height": 1056, "width": 1280 },
      "hand_right": { "encoding": "JPEG", "image_data": <bytes>, "height": 1056, "width": 1280 }
    },
    "states": {
      "head_joint_states":  [],          // 0 dims on G2_omnipicker
      "arm_joint_states":   [ ...14 ],   // left_arm(7) + right_arm(7)
      "waist_joint_states": [ ...5  ],
      "gripper_states":     [ ...2  ]    // [left, right]
    },
    "prompt":        "<natural-language task instruction>",
    "robot_type":    "G2_omnipicker",
    "task_name":     "pick_block_color",
    "episode_idx":   0,
    "episode_done":  false,
    "task_progress": [ ... ]
  }
}
```

**Image decode:** each camera is JPEG bytes under `image_data`. Decode with
`cv2.imdecode(IMREAD_COLOR)` → HWC **BGR**, then convert to **RGB** (the model expects RGB HWC
uint8). Depth fields (`*_depth`) exist in the schema but are commented out / not sent today.

**Camera rename** to the model's expected names: `head→top_head`, `hand_left→hand_left`,
`hand_right→hand_right`.

**State assembly** for the model (matches training `state_keys` order
`joint, left_effector, right_effector, waist`):

```
state = concat(arm_joint_states[14], gripper_states[2], waist_joint_states[5])  # = 21 on G2
```

One assembly works for all boards: configs with `include_waist=False` (instruction / spatial) mask
the trailing waist dims to zero, so the extra 5 dims are harmless; `include_waist=True` (manip)
needs waist in dims 16–20. `info.json` is absent at inference, so `state_indices=None` (no
remapping) — the agent must hand the model the already-ordered vector.

---

## 2. Output — action (agent → gateway)

A msgpack-encoded `result` envelope:

```jsonc
{
  "result": {
    "left_arm":       { "kind": "JOINT_ABS", "values": [[...7], ...H] },
    "right_arm":      { "kind": "JOINT_ABS", "values": [[...7], ...H] },
    "left_effector":  [[...1], ...H],
    "right_effector": [[...1], ...H],
    "waist":          { "kind": "JOINT_ABS", "values": [[...5], ...H] }   // optional; manip waist tasks only
  }
}
```

- `H` = action horizon (instruction/spatial: 50, manip: 30). The sim buffers the whole chunk and
  replans when it drains.
- `kind`: `JOINT_ABS` (absolute joint positions, what this model emits) or `EEF_ABS` (end-effector
  pose, IK-solved sim-side). `left_arm.kind` and `right_arm.kind` must match.
- The sim reads `result["result"]`; a top-level `{"error": "..."}` is treated as a fatal server error.

**Map a model action chunk `acts[H, D]` → the envelope:**

| slice | field |
|-------|-------|
| `acts[:, 0:7]`   | `left_arm.values` |
| `acts[:, 7:14]`  | `right_arm.values` |
| `acts[:, 14:15]` | `left_effector` |
| `acts[:, 15:16]` | `right_effector` |
| `acts[:, 16:]`   | `waist.values` (only when `D > 16`) |

instruction/spatial emit `D=16` (no waist key); manip emits `D=21` only on waist tasks
(e.g. `sorting_packages`), else 16.

> ### ⚠️ The plain-msgpack caveat (most common output bug)
> genie-sim unpacks the response with **plain `msgpack` (`raw=False`), NOT `msgpack_numpy`**.
> If you pack numpy arrays they arrive as ext-encoded garbage and `np.array(values)` breaks.
> **Convert every array to native Python lists (`.tolist()`) before packing.** Packing the
> list-only dict with either `msgpack` or `msgpack_numpy` is fine.

---

## 3. Per-board model layout

| Board (`config.board`) | train config | ckpt dir | horizon | state/action dims |
|------------------------|--------------|----------|---------|-------------------|
| `instruction`, `robust` | `pi05_genie_sim_instruction_and_robust_20260526` | `checkpoints/instruction_and_robust` | 50 | 16 (no waist) |
| `spatial` | `pi05_genie_sim_spatial_20260511` | `checkpoints/spatial` | 50 | 16 (no waist) |
| `manip` | `pi05_genie_sim_manip_20260526` | `checkpoints/manipulation` | 30 | 21 (`include_waist=True`) |

`ACOT_BOARD=<board> ./scripts/tunnel.sh <gpu> <job_uuid> <gateway>` selects config+ckpt. The board
the agent serves **must match** the `config.board` of the submitted job.

---

## 4. Quick checklist when debugging the adapter

- `KeyError: 'state'` in the policy → you fed raw `params` to the model; run `_adapt_obs` first
  (gateway no longer sends a flat top-level `state`).
- Sessions open/close rapidly with no stepping, job stuck at 0 → response format wrong (sim can't
  parse `result["result"]`); check the envelope keys and the `.tolist()` caveat.
- Black/garbled images → forgot BGR→RGB, or fed CHW where HWC expected.
- manip arm flailing / waist ignored → waist not included in state (dims 16–20) or not emitted in
  the action envelope.
