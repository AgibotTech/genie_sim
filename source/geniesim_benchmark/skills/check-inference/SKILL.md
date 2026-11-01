---
name: check-inference
description: >
  Probe a model inference WebSocket server (e.g. `serve_policy`) and validate
  the response — using the `geniesim benchmark check-inference` CLI verb,
  which wraps the benchmark package's `check_inference.py`.
  Trigger: When the user asks to "check inference", "校验模型推理",
  "test inference server", "verify policy server", "ping the model", or
  provides an IP/port and wants to confirm a serve_policy / WebSocket
  inference server is working before running benchmarks.
license: MPL-2.0
metadata:
  author: genie-sim
  version: "2.0"
prerequisites:
  - geniesim_cli:fresh-machine-setup   # see source/geniesim_cli/AGENTS.md § 0
inputs:
  - name: payload
    desc: Path to a corobot `.pkl` payload (defaults to the bundled corobot_payload.pkl)
    required: false
  - name: infer_host
    desc: "IP:PORT of the inference WebSocket server"
    required: true
  - name: iters
    desc: Number of consecutive requests to send (catch flakiness)
    required: false
    default: "1"
outputs:
  - desc: "Pass/fail verdict with per-dim min/max/mean/std/NaN/Inf checks and round-trip latency; non-zero exit on failure"
---

## When to Use

- Sanity-check whether a running inference server actually accepts requests and returns valid actions, before launching a full task.
- User provides `ip:port` and a payload, and asks to verify connectivity / output validity.
- Quick smoke test in CI / pre-deploy.
- Diagnosing "the benchmark hangs / outputs garbage" — this probe surfaces protocol mismatches and NaN/Inf in actions before you sink time into a full simulator launch.

Do **not** use for:
- Running the benchmark itself → use the `run-benchmark` skill.
- Submitting jobs to the Challenge platform → `challenge-submit-job`.

## What This Skill Does

Sends a saved corobot `.pkl` payload to `ws://<HOST>:<PORT>` and validates the reply:

1. Loads the payload.
2. Connects to the WebSocket server using msgpack-numpy.
3. Sends one request, receives one action chunk.
4. Validates: schema (key presence), per-dim min/max/mean/std, NaN/Inf flags, out-of-range checks (kind-aware: JOINT_ABS in radians, EEF_ABS in meters+quat, gripper in [0,1]), large jumps from the input state.

The payload is a corobot JSON-RPC envelope — `{"method": "infer", "params": {...}}` — and the server replies with `{"result": {"left_arm": …, "right_arm": …, …}}` (or `{"error": …}`).

Required actions match the benchmark's corobot consumer: `left_arm` and
`right_arm` contain numeric `values` chunks and matching `kind` values
(`JOINT_ABS` by default, or `EEF_ABS`); `left_effector` and `right_effector`
are numeric chunks. Each chunk has shape `[steps, dimensions]`, is non-empty,
and has the same step count. `EEF_ABS` arm poses contain seven values
(xyz + xyzw quaternion). Head and waist actions are optional. Numeric metadata
such as scores or history settings cannot substitute for actions.

## Required Input

The user must provide:
- **`HOST`** — server IP (e.g. `127.0.0.1`).
- **`PORT`** — server port (e.g. `8999`).

`PAYLOAD` is optional — it defaults to the bundled `corobot_payload.pkl`. Pass a
path only to override it (see *Generating a payload* below). If host/port are
missing, ask the user before running.

## How to Run

```bash
# Bundled payload — just point it at the server
geniesim benchmark check-inference --infer-host=<HOST>:<PORT>

# Override with your own payload (positional)
geniesim benchmark check-inference debug_preview/debug_0001.pkl \
    --host <HOST> --port <PORT>
```

If `geniesim` isn't on `$PATH` (the launcher wasn't installed), substitute
`python3 -m geniesim_cli benchmark check-inference …` — same args, same behaviour.

### Optional flags (forwarded to `check_inference.py`)

| Flag | Effect |
|---|---|
| `--iters N` | Send N consecutive requests (default 1). Use 5–10 to catch flakiness. |
| `--max-dims N` | Max idx rows printed per array (default 64). |

## Generating a payload

A canonical `corobot_payload.pkl` ships next to the script and is used by
default, so you usually don't need to supply one. To probe with a fresh /
task-specific observation, run a benchmark task with the corobot policy's debug
dump enabled — it writes `debug_preview/debug_NNNN.pkl` (a `{"payload": …,
"obs": …}` wrapper the probe unwraps automatically) — then pass that path.

## Interpreting Output

The script prints structured sections:

| Section | What it tells you |
|---|---|
| `📦 Payload` | Payload loaded and recognised as corobot. |
| `🔌 Connecting` | Connected to ws server (or refused). |
| `📡 …response` | Server returned a reply, schema check. |
| `📊 <key>` | Per-dim min/max/mean/std + flags, per output (left_arm/right_arm/…). |
| `⏱ latency` | Round-trip latency. |

| Outcome | Meaning |
|---|---|
| `✅ PASS — …` | Server is up and returning sane actions. |
| `❌ Connection refused / timed out` | Server isn't listening on that host:port. Verify it's running and the firewall is open. |
| `❌ … NaN / Inf …` | Server responded but model output is broken. Check the policy checkpoint and normalization stats — not a network problem. |
| `❌ response missing 'result' dict` | Server schema mismatch — it isn't speaking the corobot JSON-RPC protocol the probe expects. |
| `❌ server error: …` | The server returned a JSON-RPC error; read the message. |
| `⚠️  OOB[…]` flags | Action is finite but outside the kind's expected range. Could be a units bug (radians vs degrees) or an unnormalized output. |

## Dependencies

The script needs (in the Python env that runs `python3`):
- `msgpack`
- `numpy`
- `websockets`

It does **not** need Isaac Sim — pure-Python deps only. The CLI deliberately uses `python3` instead of `omni_python` here so the probe is snappy.

## Resources

- **Script source**: `source/geniesim_benchmark/src/geniesim_benchmark/scripts/check_inference.py` (resolved via the `geniesim_benchmark` package)
- **CLI dispatcher**: `source/geniesim_cli/src/geniesim_cli/commands/benchmark.py` (`_do_check_inference`)
- **Payload dump hook**: `source/geniesim_benchmark/src/geniesim_benchmark/benchmark/policy/corobotpolicy.py`
