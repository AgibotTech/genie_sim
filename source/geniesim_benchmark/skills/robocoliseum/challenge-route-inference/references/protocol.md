# Simulator inference protocol

This is the application payload
inside a tunnel data frame; the gateway transports opaque bytes. The simulator
runtime defines observation fields and action semantics. For tunnel dialing,
control frames, and reconnect, read the
[transport appendix](../../challenge-run-agent/SKILL.md#appendix--tunnel-protocol-for-a-self-built-client).
For connecting an existing service, read [adapter integration](adapters.md).

## Source of truth

The following paths are relative to the **simulator code** repository. They are
source references, not dependencies bundled with this skill:

| Contract | Source path under `source/geniesim_benchmark/src/geniesim_benchmark/` |
| --- | --- |
| Main Isaac request, response, history, depth, and termination | `benchmark/policy/corobotpolicy.py` |
| Embodiment tags, joint order, and gripper conversion | `utils/name_utils.py`, `utils/infer_post_process.py` |
| Camera intrinsics/extrinsics | `camera_calibration.py` |
| Chassis observations and execution | `benchmark/envs/pi_env.py` |
| EEF pose conversion and IK | `utils/eef_control.py` |
| Native Newton Wuji request/response | `newton/corobot.py` |
| Native libuipc Wuji request/response | `libuipc/adapter.py` |
| Native MuJoCo Rubik request/response | `backends/mujoco/rubik/policy.py`, `backends/mujoco/rubik/native.py` |
| Local inference probe | `scripts/check_inference.py` |

The main Isaac client and native clients share the CoRobot envelope, but their
optional capabilities differ. Check the runtime's contract before adding depth,
history, chassis actions, or terminal-only replies.

## 1. Encoding and request/response exchange

Use a binary WebSocket message containing MessagePack, with string map keys:

```python
request_bytes = msgpack.packb(request, use_bin_type=True)
response = msgpack.unpackb(response_bytes, raw=False)
```

The direct CoRobot exchange sends one request and receives one binary response,
without a server metadata greeting. Native OpenPI's greeting and flat observation
schema require an adapter. Do not send the tunnel's session framing to a direct
inference endpoint: the relay removes it before forwarding the payload.

Convert NumPy arrays and scalars to native lists/numbers before serialization.
The simulator response readers use plain MessagePack; NumPy-specific extension
encodings are not part of this action contract. Images remain binary bytes,
not base64 strings, on the inference wire.

## 2. Complete observation examples

### Main Isaac request with all conditional fields enabled

This structural example includes every `params` field emitted by the main Isaac
request builder, including its conditional fields. It illustrates a
`G2_crsB_90d_wbc` request with depth/calibration enabled, available EEF poses,
nonempty instruction segments, and history enabled by a previous response.

`image_data` strings stand for **binary bytes**, and prim paths, joint values,
poses, intrinsics, and capture IDs are illustrative. Replace them with actual
runtime data; this is not a readiness fixture. A depth buffer here contains
`height * width * 2` bytes. The three current RGB/depth/calibration groups share
one capture ID; history belongs to earlier captures and has no current-frame
calibration attached by this builder.

```json
{
  "method": "infer",
  "params": {
    "timestamps": {"head": 1700000000000000000, "states": 1700000000000000000},
    "images": {
      "head": {
        "encoding": "JPEG",
        "image_data": "<binary head JPEG bytes>",
        "height": 400,
        "width": 640,
        "calibration": {
          "available": true,
          "snapshot_id": "capture-0007",
          "camera_path": "<actual head camera prim path>",
          "base_path": "<actual robot base_link prim path>",
          "source": "same render product camera_params + USD",
          "model": "pinhole",
          "width": 640,
          "height": 400,
          "fx": 400.0,
          "fy": 400.0,
          "cx": 320.0,
          "cy": 200.0,
          "distortion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          "frame": "base_link",
          "units": "m",
          "optical_to_frame": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          "depth_kind": "distance_to_image_plane",
          "depth_scale_m": 0.001,
          "rgb_depth_aligned": true,
          "pose_checked_against_render": true,
          "pixel_convention": "u right, v down; intrinsics at source resolution",
          "optical_axes": "+x right, +y down, +z forward",
          "transform_convention": "column vectors: p_base = optical_to_frame @ [x,y,z,1]"
        },
        "snapshot_id": "capture-0007"
      },
      "hand_left": {
        "encoding": "JPEG",
        "image_data": "<binary hand_left JPEG bytes>",
        "height": 400,
        "width": 640,
        "calibration": {
          "available": true,
          "snapshot_id": "capture-0007",
          "camera_path": "<actual hand_left camera prim path>",
          "base_path": "<actual robot base_link prim path>",
          "source": "same render product camera_params + USD",
          "model": "pinhole",
          "width": 640,
          "height": 400,
          "fx": 400.0,
          "fy": 400.0,
          "cx": 320.0,
          "cy": 200.0,
          "distortion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          "frame": "base_link",
          "units": "m",
          "optical_to_frame": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          "depth_kind": "distance_to_image_plane",
          "depth_scale_m": 0.001,
          "rgb_depth_aligned": true,
          "pose_checked_against_render": true,
          "pixel_convention": "u right, v down; intrinsics at source resolution",
          "optical_axes": "+x right, +y down, +z forward",
          "transform_convention": "column vectors: p_base = optical_to_frame @ [x,y,z,1]"
        },
        "snapshot_id": "capture-0007"
      },
      "hand_right": {
        "encoding": "JPEG",
        "image_data": "<binary hand_right JPEG bytes>",
        "height": 400,
        "width": 640,
        "calibration": {
          "available": true,
          "snapshot_id": "capture-0007",
          "camera_path": "<actual hand_right camera prim path>",
          "base_path": "<actual robot base_link prim path>",
          "source": "same render product camera_params + USD",
          "model": "pinhole",
          "width": 640,
          "height": 400,
          "fx": 400.0,
          "fy": 400.0,
          "cx": 320.0,
          "cy": 200.0,
          "distortion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          "frame": "base_link",
          "units": "m",
          "optical_to_frame": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
          "depth_kind": "distance_to_image_plane",
          "depth_scale_m": 0.001,
          "rgb_depth_aligned": true,
          "pose_checked_against_render": true,
          "pixel_convention": "u right, v down; intrinsics at source resolution",
          "optical_axes": "+x right, +y down, +z forward",
          "transform_convention": "column vectors: p_base = optical_to_frame @ [x,y,z,1]"
        },
        "snapshot_id": "capture-0007"
      },
      "head_depth": {
        "encoding": "RAW_UINT16",
        "image_data": "<binary head uint16 depth bytes>",
        "height": 400,
        "width": 640,
        "snapshot_id": "capture-0007"
      },
      "hand_left_depth": {
        "encoding": "RAW_UINT16",
        "image_data": "<binary hand_left uint16 depth bytes>",
        "height": 400,
        "width": 640,
        "snapshot_id": "capture-0007"
      },
      "hand_right_depth": {
        "encoding": "RAW_UINT16",
        "image_data": "<binary hand_right uint16 depth bytes>",
        "height": 400,
        "width": 640,
        "snapshot_id": "capture-0007"
      }
    },
    "states": {
      "head_joint_states": [0.0, 0.0, 0.0],
      "arm_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "waist_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0],
      "gripper_states": [0.0, 0.0],
      "end_pose": {
        "base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        },
        "arm_base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        }
      },
      "chassis_states": [0.0, 0.0, 0.0]
    },
    "prompt": "<full task instruction>",
    "robot_type": "g2a_sim",
    "ee_type": "zhiyuan_gripper_ctek90d",
    "task_name": "<current WBC sub-task name>",
    "episode_idx": 0,
    "episode_done": false,
    "segments": ["<first instruction segment>", "<next instruction segment>"],
    "history": {
      "interval": 5,
      "images": [
        {
          "head": {"encoding": "JPEG", "image_data": "<binary head JPEG bytes>", "height": 200, "width": 320},
          "hand_left": {"encoding": "JPEG", "image_data": "<binary hand_left JPEG bytes>", "height": 200, "width": 320},
          "hand_right": {"encoding": "JPEG", "image_data": "<binary hand_right JPEG bytes>", "height": 200, "width": 320}
        }
      ]
    }
  }
}
```

### Field presence in the main Isaac client

"Always" below describes the non-preview request built by this client, not a
requirement that every native runtime implements the same optional features.

| Field in `params` | Presence | Meaning |
| --- | --- | --- |
| `timestamps.head`, `timestamps.states` | Always | Integer wall-clock nanoseconds; the main client assigns the same timestamp to both. |
| `images.head`, `images.hand_left`, `images.hand_right` | Always | Current JPEG image records with `encoding`, binary `image_data`, `height`, and `width`. |
| `images.<camera>.calibration`, `images.<camera>.snapshot_id` | While depth is enabled | Calibration record and capture ID. Calibration can have `available: false` with a `reason`; unavailable data must not be used for backprojection. |
| `images.head_depth`, `images.hand_left_depth`, `images.hand_right_depth` | While depth is enabled and each sensor has a nonempty depth map | `RAW_UINT16` records, dimensions, binary data, and capture ID. |
| `states.head_joint_states`, `states.arm_joint_states`, `states.waist_joint_states`, `states.gripper_states` | Always | Named joint-state vectors in the selected embodiment's order/convention; unsupported groups may be empty. |
| `states.end_pose` | Always | Both arms' positions/XYZW orientations under `base_link` and `arm_base_link`, or null when the required FK/transform is unavailable. |
| `states.chassis_states` | Only when chassis state is nonempty | WBC chunk-relative `[dx,dy,dyaw]`, not world odometry. Omitted for fixed-base robots. |
| `prompt` | Always | Complete task instruction, preserved even when `segments` is present. |
| `robot_type`, `ee_type` | Always | Explicit embodiment and end-effector routing tags. |
| `task_name` | Always | Current sub-task name. |
| `episode_idx` | Always | Episode index supplied by the caller. |
| `episode_done` | Always | Simulator notification that the episode has ended; distinct from a server stop response. |
| `segments` | Only when nonempty segments are supplied | List of instruction segments. |
| `history.interval`, `history.images` | Only when history interval is positive | Samples captured while replaying the preceding chunk. The image list can be empty; `last_n` can limit the transmitted samples. |

History is not present until enabled by a prior response. Depth starts enabled
for a main-client episode, then follows each response's `need_depth` setting.
`simulation_time`, `episode_start`, `task_progress`, and `action_contract` are not
main-client additions; native-runtime examples below show their actual scope.

### Native Newton and libuipc requests

Native Newton uses the same `method`/`params` structure with the fields below;
its Wuji embodiment has 40 effector-state values. It does not attach the main
client's depth/calibration or WBC extensions. This example enables segments and
caller-owned history and includes the available dual-frame EEF observations.

```json
{
  "method": "infer",
  "params": {
    "timestamps": {"head": 1700000000000000000, "states": 1700000000000000000},
    "images": {
      "head": {"encoding": "JPEG", "image_data": "<binary head JPEG bytes>", "height": 400, "width": 640},
      "hand_left": {"encoding": "JPEG", "image_data": "<binary hand_left JPEG bytes>", "height": 400, "width": 640},
      "hand_right": {"encoding": "JPEG", "image_data": "<binary hand_right JPEG bytes>", "height": 400, "width": 640}
    },
    "states": {
      "head_joint_states": [0.0, 0.0, 0.0],
      "arm_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "waist_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0],
      "gripper_states": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "end_pose": {
        "base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        },
        "arm_base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        }
      }
    },
    "prompt": "<full task instruction>",
    "robot_type": "g2a_sim",
    "ee_type": "wuji",
    "task_name": "<current native Wuji task name>",
    "episode_idx": 0,
    "episode_done": false,
    "segments": ["<first instruction segment>", "<next instruction segment>"],
    "history": {
      "interval": 5,
      "images": [
        {
          "head": {"encoding": "JPEG", "image_data": "<binary head JPEG bytes>", "height": 200, "width": 320},
          "hand_left": {"encoding": "JPEG", "image_data": "<binary hand_left JPEG bytes>", "height": 200, "width": 320},
          "hand_right": {"encoding": "JPEG", "image_data": "<binary hand_right JPEG bytes>", "height": 200, "width": 320}
        }
      ]
    }
  }
}
```

The native libuipc builder additionally emits `task_progress: []` for
compatibility. Its complete request is shown separately below. `episode_done` is
false, timestamps come from the synchronized capture, and head/waist
observations include their scene-fixed holds. Neither builder publishes live
checker trees or score progress.

```json
{
  "method": "infer",
  "params": {
    "timestamps": {"head": 1700000000000000000, "states": 1700000000000000000},
    "images": {
      "head": {"encoding": "JPEG", "image_data": "<binary head JPEG bytes>", "height": 400, "width": 640},
      "hand_left": {"encoding": "JPEG", "image_data": "<binary hand_left JPEG bytes>", "height": 400, "width": 640},
      "hand_right": {"encoding": "JPEG", "image_data": "<binary hand_right JPEG bytes>", "height": 400, "width": 640}
    },
    "states": {
      "head_joint_states": [0.0, 0.0, 0.0],
      "arm_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "waist_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0],
      "gripper_states": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "end_pose": {
        "base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        },
        "arm_base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        }
      }
    },
    "prompt": "<full task instruction>",
    "robot_type": "g2a_sim",
    "ee_type": "wuji",
    "task_name": "<current native Wuji task name>",
    "episode_idx": 0,
    "episode_done": false,
    "segments": ["<first instruction segment>", "<next instruction segment>"],
    "history": {
      "interval": 5,
      "images": [
        {
          "head": {"encoding": "JPEG", "image_data": "<binary head JPEG bytes>", "height": 200, "width": 320},
          "hand_left": {"encoding": "JPEG", "image_data": "<binary hand_left JPEG bytes>", "height": 200, "width": 320},
          "hand_right": {"encoding": "JPEG", "image_data": "<binary hand_right JPEG bytes>", "height": 200, "width": 320}
        }
      ]
    },
    "task_progress": []
  }
}
```

### Native MuJoCo Rubik request

Rubik adds `simulation_time`, `episode_start`, `task_progress`, and a full
`action_contract`. It does not add main-client depth/calibration, chassis, or
history fields. Below, typed string placeholders represent arrays/values
published by the actual environment; they are not the wire types. For example,
`bounds` is a numeric `[54,2]` array on the wire, not a string. Do not substitute
invented joint limits or controller settings for the runtime's contract.

```json
{
  "method": "infer",
  "params": {
    "timestamps": {"head": 1700000000000000000, "states": 1700000000000000000},
    "images": {
      "head": {"encoding": "JPEG", "image_data": "<binary head JPEG bytes>", "height": 400, "width": 640},
      "hand_left": {"encoding": "JPEG", "image_data": "<binary hand_left JPEG bytes>", "height": 400, "width": 640},
      "hand_right": {"encoding": "JPEG", "image_data": "<binary hand_right JPEG bytes>", "height": 400, "width": 640}
    },
    "states": {
      "head_joint_states": [0.0, 0.0, 0.0],
      "arm_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      "waist_joint_states": [0.0, 0.0, 0.0, 0.0, 0.0],
      "gripper_states": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
      ],
      "end_pose": {
        "base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        },
        "arm_base_link": {
          "left_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]},
          "right_arm": {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
        }
      }
    },
    "prompt": "<full task instruction>",
    "robot_type": "g2a_sim",
    "ee_type": "wuji",
    "task_name": "rubik_cube_solve_1turn",
    "episode_idx": 0,
    "episode_done": false,
    "segments": ["<first instruction segment>", "<next instruction segment>"],
    "simulation_time": 0.0,
    "episode_start": true,
    "task_progress": [],
    "action_contract": {
      "version": 1,
      "kind": "JOINT_ABS",
      "arm_target_limit_policy": "clip_to_joint_limits",
      "units": "radians",
      "dimension": 54,
      "order": ["left_arm[7]", "right_arm[7]", "left_effector[20]", "right_effector[20]"],
      "joint_names": "<54 exact runtime joint-name strings in published order>",
      "bounds": "<54 numeric [minimum, maximum] rows>",
      "control_hz": "<runtime numeric control frequency>",
      "action_time_scale": "<runtime numeric time scale>",
      "interpolation": "<quintic smoothstep or expert filter, according to the controller>",
      "hand_target_tolerance": "<runtime numeric tolerance in radians>",
      "physics_hz": 1000,
      "waypoint_timing": "nominal control interval stretched to satisfy motor reference speed limits",
      "motor_velocity_limits_radians_per_second": "<numeric array in motor_joint_names order>",
      "motor_joint_names": "<exact runtime motor-joint-name array>",
      "motor_command_speed_limits_radians_per_second": "<numeric array for the native position controller, otherwise null>",
      "fingertip_force_feedforward": false,
      "arm_bias_compensation": "<boolean: true for the natural hand-controller configuration>",
      "arm_velocity_limits_radians_per_second": "<numeric runtime arm-speed-limit array>",
      "arm_speed_limit_enforcement": "substep_interpolation_with_actual_speed_check",
      "motor_target_semantics": "<speed-bounded substep interpolation of position actuator waypoints, or expert filtered joint poses>",
      "hand_controller": "<runtime controller class name>",
      "hand_joint_limits": "<runtime hand-controller configuration name>"
    }
  }
}
```

| Native-only field | Scope and meaning |
| --- | --- |
| `simulation_time` | Rubik: floating-point `env.data.time` in simulation seconds. |
| `episode_start` | Rubik: boolean `env.steps == 0`; not a generic main-client field. |
| `task_progress` | libuipc and Rubik: empty compatibility list, not live checker progress. Native Newton and the main Isaac builder omit it. |
| `action_contract` | Rubik: runtime-published action layout, exact joint names, bounds, control/timing information, and controller limits. All keys returned by the reviewed `action_contract()` are shown above. |

Do not combine all runtime extensions into one universal payload. Preserve the
complete request from the selected simulator client. Checker requirements,
`task_evaluation`, live score progress, and step budgets remain local to evaluation
and must not be added to these inference requests.

### Embodiment routing and dimensions

These are registered simulator configurations, not platform board names:

| Simulator robot configuration | Wire `robot_type` | Wire `ee_type` | Arm joints per side | Effector values per side |
| --- | --- | --- | ---: | ---: |
| `G1_omnipicker` | `a2d_sim` | `zhiyuan_gripper_omnipicker` | 7 | 1 |
| `G2_omnipicker`, `G2_crsB_omnipicker` | `g2a_sim` | `zhiyuan_gripper_omnipicker` | 7 | 1 |
| `G2_crsB_90d`, `G2_crsB_90d_wbc` | `g2a_sim` | `zhiyuan_gripper_ctek90d` | 7 | 1 |
| `G2_crsB_wuji` | `g2a_sim` | `wuji` | 7 | 20 |
| `arx` | `arx_sim` | `arx_gripper` | 6 | 1 |
| `agilex` | `agilex_sim` | `agilex_gripper` | 6 | 1 |

The field is **`ee_type`**, not `eef_type`. Several G2 embodiments share
`robot_type="g2a_sim"`; use `ee_type` as well to distinguish grippers and hands.
For real-robot probe mode, the main policy removes a trailing `_sim` from the
robot tag. Other legacy/custom robot configurations can fall back to their raw
configuration name; inspect their simulator registration rather than guessing.

`arm_joint_states` concatenates left then right arm joints. `gripper_states`
concatenates left then right effector values. G2 waist/head groups have 5/3 joints;
G1 has 2/2 in its registration, while ARX/Agilex register neither group. Actual
scene groups can be empty. ARX/Agilex retain a left-arm block for compatibility
while commanding the right arm; do not treat the decorative left arm as movable.

Wuji uses native radians and 20 joints per hand, ordered thumb, index, middle,
ring, pinky, then joints 1–4 within each finger. Parallel-gripper observations and
actions use the selected robot configuration's label/scale conversion; units and
scales are not universally interchangeable even when routing tags match.

There is no universal 16- or 21-dimensional policy vector. Flattening, camera
renaming such as `head -> top_head`, and checkpoint normalization belong to the
model adapter and must match its training contract.

### EEF observations

`states.end_pose` is null when unavailable. Otherwise it has this shape:

```json
{
  "base_link": {
    "left_arm": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
    "right_arm": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
  },
  "arm_base_link": {
    "left_arm": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
    "right_arm": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]}
  }
}
```

Positions are in the named robot frame; wire quaternions are **`[qx,qy,qz,qw]`**.
Internal renderer/body poses can use WXYZ, so preserve the adapter's conversion.

### Chassis observations

WBC requests can add `states.chassis_states: [dx, dy, dyaw]`. In the main
simulator this is a delta relative to the current chunk-start anchor and is
reported as `[0,0,0]` at that boundary, rather than a global odometry pose. The
field is omitted for robots without a chassis.

## 3. RGB, depth, and calibration

RGB records contain `encoding: "JPEG"`, binary `image_data`, `height`, and `width`.
OpenCV decodes BGR; convert to RGB only if the model expects RGB. Preserve the
wire camera names unless the model adapter explicitly maps them.

In the main Isaac policy, an episode starts with depth enabled. Each ordinary
response independently requests depth for the **next** request with
`result.need_depth: true`; omission or false disables it. Reset enables it again.
Depth is sent when the sensor data is available, under `head_depth`,
`hand_left_depth`, and `hand_right_depth`:

| Depth field | Value |
| --- | --- |
| `encoding` | `RAW_UINT16` |
| `image_data` | Raw row-major uint16 bytes, not PNG/JPEG |
| `height`, `width` | Depth map dimensions |
| `snapshot_id` | Capture identity shared with the RGB/calibration record |

The encoder stores millimetres (`metres * 1000`), replaces nonfinite input with
zero, clips to uint16 range, and adds per-pixel multiplicative noise: +/-2% at
input depth <=1 m and +/-3% farther away. The implementation uses NumPy's native
uint16 byte order; the deployed Linux encoding is little-endian. It does not add
a byte-order field. Native runtime clients do not all implement this depth path.

Calibration is attached to each current RGB record under `calibration`, with a
matching `snapshot_id`. Available calibration contains:

- `available: true`, `model` (`pinhole` or `opencv_pinhole`), `width`, `height`,
  `fx`, `fy`, `cx`, `cy`, and `distortion` in the order
  `[k1,k2,p1,p2,k3,k4,k5,k6]`.
- `frame: "base_link"`, `units: "m"`, and `optical_to_frame`, a 4x4 transform for
  column vectors: `p_base = optical_to_frame @ [x,y,z,1]`.
- Optical axes `+x right, +y down, +z forward`,
  `depth_kind: "distance_to_image_plane"`, `depth_scale_m: 0.001`,
  `rgb_depth_aligned: true`, and `pose_checked_against_render: true`.
- Capture/source metadata, including `snapshot_id`, `camera_path`, and `base_path`.

Before backprojection, require available/aligned calibration and matching capture
IDs. Unavailable calibration includes `available: false` and `reason`; camera
noise, occlusion, or drop-frame augmentation can invalidate alignment. Do not use
an invalid transform. Intrinsics apply to the source resolution; adjust them if
the model resizes pixels. Distorted images need the matching lens model.

## 4. Ordinary action response

Return a binary MessagePack envelope with `result`. A nonempty top-level `error`
or a text WebSocket response is an inference error. This one-row example shows
the structure for a G2 parallel-gripper embodiment; it is not a task policy:

```json
{
  "result": {
    "left_arm": {"kind": "JOINT_ABS", "values": [[0, 0, 0, 0, 0, 0, 0]]},
    "right_arm": {"kind": "JOINT_ABS", "values": [[0, 0, 0, 0, 0, 0, 0]]},
    "left_effector": [[0]],
    "right_effector": [[0]]
  }
}
```

For ordinary chunks, both arms and both effectors are required. Every supplied
action channel must have the same nonzero horizon `H`, rectangular finite numeric
rows, and the dimension required by that embodiment. `H` comes from the actual
model/runtime, not the board name; native clients can impose a maximum chunk
size. The simulator buffers the chunk and requests another when it drains.

| Field in `result` | Shape and meaning |
| --- | --- |
| `left_arm`, `right_arm` | `{kind, values}`. `JOINT_ABS` gives absolute joint positions in radians, with the embodiment's arm width. `EEF_ABS` gives `[H,7]` pose rows `[x,y,z,qx,qy,qz,qw]`, position in metres. |
| `left_effector`, `right_effector` | `[H,E]` lists in the embodiment's effector order/convention; `E=20` for Wuji, usually 1 for parallel grippers. |
| `waist` | Optional `{kind: "JOINT_ABS", values: [H,W]}` in radians; the main Isaac parser also accepts the legacy `ABS_JOINT` alias. |
| `head` | Optional `[H,N]` list or `{values: [H,N]}`. Native clients may require `kind: "JOINT_ABS"` on a supplied mapping. |
| `wheels` | WBC-only `[H,3]` list or `{kind: "POSE_DELTA", values: [H,3]}`; see below. |
| `base_link` | Optional EEF coordinate-frame selector. The historical default is `"base_link"`. |
| `need_depth` | Main Isaac client's next-request depth opt-in. |
| `history`, `hist_frame_interval` | History capture settings, described below. |
| `episode_done` | Server stop/termination indication; semantics depend on the runtime. It does not assert checker success. |

Arm kinds must match. The main Isaac parser defaults an omitted arm kind to
`JOINT_ABS`, but some native paths require it explicitly, so include it.
For `EEF_ABS`, each arm may override the frame with its own `base_link` field;
otherwise the top-level selector applies. Both arms must resolve to the same
supported frame, `base_link` or `arm_base_link`. IK uses measured state and the
runtime's actual robot transform. Do not mix frames or pass WXYZ quaternions.
EEF support also depends on the configured robot's IK; the ARX/Agilex registrations
are for joint replay and do not establish EEF control support.

For WBC, each wheel row is `[delta_x_m, delta_y_m, delta_yaw_rad]` relative to the
chunk-start robot anchor. It is a pose target, not velocity or an incremental
per-tick displacement. Only `POSE_DELTA` is supported. The main Isaac policy
broadcasts the **last row's target across the whole chunk**, then the chassis
servo executes toward that fixed end target. Keep equal row counts even though
earlier wheel rows are not executed independently. Fixed-base native clients
reject chassis control.

## 5. History feedback

The main Isaac client reads these fields from each ordinary response's `result`:

```json
{
  "history": {
    "interval": 5,
    "last_n": 4,
    "resolution": {"head": [640, 400], "hand_left": [320, 264], "hand_right": [320, 264]}
  },
  "need_depth": false
}
```

`interval` is the action-step sampling interval; zero disables capture. A zero or
missing nested interval falls back to legacy `hist_frame_interval`. `last_n > 0`
limits transmission to the newest N samples; zero sends all captured samples.
Resolution is `[width,height]` per recognized wire camera, clamped to the source
size (no upscaling). Without a per-camera setting, hand history is half resolution
and head history keeps its source resolution.

The next request can contain:

```json
{"history": {"interval": 5, "images": [{"head": {"encoding": "JPEG", "image_data": "<binary JPEG bytes>", "height": 400, "width": 640}}]}}
```

Each frame is a camera-to-JPEG-record dictionary. The main policy captures all
three cameras; caller-owned native history can contain a smaller camera set.
An empty `images` list is valid if no frame was collected. History comes from
replaying the preceding chunk, not from future steps. Sending a request clears
the previous buffer; reset also clears it. Omission of both history settings in
a reply disables main-policy history. Native clients can support only the legacy
interval or no history; see the runtime differences below.

## 6. Episode lifecycle and terminal replies

Keep `episode_idx`, optional `episode_start`, request `episode_done`, and tunnel
session lifecycle distinct. The main Isaac client clears buffered actions when
the simulator reports episode completion, allowing a final `episode_done: true`
request to reach the service. Its reset clears history, terminal state, and depth
preferences. Do not invent a universal wire `reset` method: isolate adapter state
using the runtime's episode/session signals and the route's close hook.

In the main Isaac policy, an explicit no-action termination is accepted only with
all of the following fields and **no action channel keys at all**:

```json
{
  "result": {
    "episode_done": true,
    "agent2act": {
      "protocol_version": 1,
      "status": "planning_exhausted",
      "termination_reason": "no_valid_hold",
      "action_issued": false,
      "episode_terminal": true,
      "error": "<nonempty explanation>"
    }
  }
}
```

`protocol_version` must be integer 1 and the boolean fields must be actual
booleans. Missing actions alone are not this protocol. This response terminates
the episode at its last native state without issuing an action or incrementing
the policy dispatch count. It does not award success. Ordinary accepted action
chunks increment the main client's dispatch count unless the request was already
an episode-end notification; replaying chunk rows is not a new dispatch.

Native Newton accepts a boolean `result.episode_done: true` before parsing action
channels. Native Rubik also accepts a terminal-only result when action fields
are absent. These behaviors do not generalize to every runtime; the libuipc
parser still requires ordinary action channels.

## 7. Runtime differences

| Runtime | Additional requirements and capabilities |
| --- | --- |
| Main Isaac `CoRobotPolicy` | Registered embodiments; optional chassis, depth/calibration, nested or legacy history, EEF frames, and the explicit no-action termination above. |
| Native Newton Wuji | 7+7 arm and 20+20 hand values, native radians, plus optional waist/head. Fixed or absent joints must remain at scene-initial holds. Supports EEF through hard IK. Rejects wheels. Uses legacy `hist_frame_interval`; the request builder can attach caller-owned history. A boolean terminal reply ends the episode without requiring actions. |
| Native libuipc Wuji | Same 54 controlled arm/hand joints. Optional 5-value waist and 3-value head channels are validated/retained but cannot move scene welds. Supports EEF through verified robot transforms and hard IK. Rejects nonempty wheels. Uses legacy `hist_frame_interval`; current requests retain `task_progress: []`. |
| Native MuJoCo Rubik | Adds `simulation_time`, `episode_start`, and `action_contract`. Consumes 54-dimensional arm/hand targets and rejects nonempty head/waist/wheels and nonzero legacy history interval. Supports EEF via native IK. Terminal-only replies are accepted. |

Rubik's `action_contract` carries version, units, dimension, channel order, exact
`joint_names`, per-joint `bounds`, control/timing, and actuator constraints. It
accepts either ordinary CoRobot channels or
`result: {kind: "JOINT_ABS", joint_targets: [H,54]}` in the published order; mixing
both forms is rejected. The runtime enforces maximum chunk size and finite
values, clips arm targets to joint limits, and validates hand bounds with its
configured tolerance. Do not apply this flat 54-value shortcut to other clients.

Capabilities absent from a native request must not be fabricated to match the
main Isaac example. Use real payloads for each target runtime and robot family.

## 8. Integration checks and helper limits

- Validate the actual request and response against the selected runtime before
  spending an evaluation submission. If the simulator CLI is available, use
  `geniesim benchmark check-inference --infer-host=<HOST>:<PORT>` with a trusted
  embodiment-appropriate payload; a default payload need not represent every robot.
- Decode one real RGB/depth capture and verify byte format, color order, dimensions,
  calibration availability, and capture IDs. Do not drop depth/history merely
  because a particular baseline does not consume them.
- Verify arm kinds, EEF frames, effector widths/order, horizon consistency, and
  model normalization. A 16-value parallel-gripper adapter cannot serve Wuji.
- Preserve extra request fields and response metadata through the transparent relay.
  The reference baseline's model layout belongs to
  [challenge-baseline-model](../../challenge-baseline-model/SKILL.md), not this wire
  contract; checkpoint names and horizons are not protocol constants.
- The bundled route helper's `corobot` validator checks ordinary rectangular
  finite chunks and equal horizons, but does not validate exact embodiment widths,
  matching arm kinds/frames, scene bounds, or the special terminal alternatives.
  It rejects terminal-only results and Rubik's flat `joint_targets` form. For those
  forms, use a trusted adapter that validates the target-runtime contract and
  `validator: "bytes"`; that setting alone proves only nonempty binary output.
- A socket connection, a well-shaped action chunk, or a server terminal indication
  does not prove task success. Official runtime checkers and platform results
  determine the outcome.
