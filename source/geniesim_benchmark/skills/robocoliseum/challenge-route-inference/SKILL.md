---
name: challenge-route-inference
description: >
  Connect existing local or remote GenieSim-compatible inference services to
  official evaluation through one or more tunnel routes. Use for new or resumed
  jobs, shared-service concurrency, route management, and simulator observation/
  action protocol or adapter questions. Includes the CoRobot protocol reference;
  does not train models or download checkpoints.
---

# Route existing inference services into official evaluation

Reuse running model processes. Routes forward requests and do not load weights:
**model processes, route processes, and platform simulation GPUs are separate
resources**. A service can serve multiple routes. Each route has an independent
connection and persistent `agent_id`. Each job accepts up to its `PARALLELISM` live
connections; every job of the account has its own limit.

## Identify the inputs

Use the conversation, service configuration, and platform state before asking for
information that is still missing:

- Inference endpoints, protocols, and loaded weights. The service must already
  exchange valid actions with the simulator client.
- A real request's **raw binary payload** for readiness checks. Export `.msgpack`
  or `.bin` with the trusted client; placeholder text, empty images, and unknown
  pickle files are not valid substitutes.
- The HTTP API base. The tunnel WebSocket endpoint is a different host and comes
  from the job response's `tunnel_endpoint` (saved as `TUNNEL_ENDPOINT` in the job's
  state file). Do not guess it from the website domain or hard-code an address.
- Whether to create or resume a job; numeric `JOB_ID`, UUID `JOB_UUID`, private
  credentials file, board, and route counts per service.
- Routes already running for other jobs and the services they share. State the
  allocation, such as 3 routes for an existing job plus 5 for a new one; do not
  silently disconnect an active job.

A user's explicit request to create/start with a specified concurrency authorizes
that work; do not ask again for the same authorization. Planning or inspection
alone does not authorize submission. Resuming an existing job does not create a
new one. Job creation, including the live quota report, follows
[challenge-submit-job](../challenge-submit-job/SKILL.md).

Reuse the existing weights. Changing checkpoints, restarting or stopping models,
and downloading a replacement baseline require a user request for that work.

## Files and environment

This skill supports Linux and Python >=3.10. Set `SKILL_DIR` to the directory
containing this file and `PYTHON` to an interpreter with the bundled requirements:

```bash
python3 -m venv ./route-venv
./route-venv/bin/python -m pip install -r "$SKILL_DIR/scripts/requirements.txt"
PYTHON="$(pwd)/route-venv/bin/python"
```

An existing compatible environment also works. The tunnel client
(`tunnel_client.py`) and agent identity helper (`agent_identity.py`) are bundled;
no separate official SDK package is required.

Store credentials in private, mode-0600, per-job `.env` or JSON files. Keep tokens
and passwords out of command arguments, route JSON, logs, and skill documents.
For login, use [challenge-login](../challenge-login/SKILL.md); reuse a valid login.

## Protocol and adapters

Read [the simulator inference protocol](references/protocol.md) when implementing
or checking observation/action fields, embodiment tags, joint ordering, EEF frames,
depth/calibration, history, chassis control, or episode termination. This reference
is part of this inference skill.

The default relay expects a **CoRobot MessagePack WebSocket service with no
unsolicited greeting and one response per request**. It forwards the entire
payload and response without changing `robot_type`, `ee_type`, prompt, waist,
chassis, hand joints, horizon, or response metadata.

For HTTP, RPC, native OpenPI, greetings, compression, or state-reset handling,
read [adapter integration](references/adapters.md) and reuse the client's existing
simulation adapter. A model running locally does not establish wire compatibility.

## Configure and probe routes

Create `routes.json`. Relative paths resolve against that JSON file's directory.
This example allocates four routes to each of two existing services:

```json
{
  "state_file": "./job-a.env",
  "api_base": "https://robocoliseum.ai",
  "models": [
    {"name": "model_a", "endpoint": "ws://127.0.0.1:9000", "routes": 4,
     "warmup_payload": "./real-observation.msgpack", "validator": "corobot"},
    {"name": "model_b", "endpoint": "ws://127.0.0.1:9001", "routes": 4,
     "warmup_payload": "./real-observation.msgpack", "validator": "corobot"}
  ]
}
```

`state_file` is a per-job copy of the state file written by
[challenge-submit-job](../challenge-submit-job/SKILL.md) (see "Create or attach a job"
below). `gateway_url`, `job_id`, and `job_uuid` default to its values; overrides must
match the job to which the credentials belong. That file has no `BASE_URL`, so set
`api_base`; a `BASE_URL` recorded in a state file must match it. An optional
`auth_file` supplies the current login `CHALLENGE_TOKEN` for HTTP requests, while
tunnel connections still prefer each job's `JOB_TOKEN`. Refreshing login then
need not rewrite every job's identity file.

Use `state_dir` to select identity storage and each model's optional
`action_horizon` to check its row count. The jobs list is per season (`1` for the
Season I boards, `2` for `v2`) and defaults to Season II; `start` uses `season`
from the config, else `JOB_SEASON` from the state file, else search both. A
50-step horizon is not assumed. `probe` does not require job credentials, so
check the model before submitting:

```bash
"$PYTHON" "$SKILL_DIR/scripts/route_inference.py" probe --config ./routes.json
```

Probes issue real fixtures concurrently at the requested route count. The bundled
CoRobot validator checks nonempty rectangular finite numeric chunks and matching
horizons for arms, effectors, and supplied optional channels. It does not verify
embodiment-specific widths, action semantics, or task success. Its handling of
terminal responses is described in the protocol reference.

Probe each relevant robot/task family with its own real fixture. For shared
models, also check episode/history isolation under interleaved requests. Measure
actual VRAM use, JIT costs, and concurrent latency; routes do not isolate global
model state or impose a model memory limit.

## Create or attach a job

This skill does not submit jobs. Create one only when requested, with
[challenge-submit-job](../challenge-submit-job/SKILL.md): it checks profile and
quota, confirms before the POST, and saves `JOB_ID`, `JOB_UUID`, `PARALLELISM`,
`TUNNEL_ENDPOINT`, `JOB_TOKEN`, and `JOB_SEASON` to `~/.simubotix-challenge.env`.
A failed or timed-out POST is not retried; check the job list first, as that skill
describes. Review submission and leaderboard publication are separate user actions.

That file holds one job and the next submission overwrites it. Right after each
submission, copy it to a per-job state file and point `routes.json` at the copy:

```bash
cp ~/.simubotix-challenge.env ./job-a.env && chmod 600 ./job-a.env
```

For simultaneous old and new jobs, keep one copy, configuration, and run directory
per job. Never overwrite a live job's copy to connect another job.

### Resume

Point `state_file` at the intended job's copy. Prefer its nonempty `JOB_TOKEN`;
a new job's token cannot resume an old job. `start` checks numeric ID and UUID
against the jobs list and rejects completed, failed, or stopping jobs. If the copy
is lost, recover `id` and `uuid` from `GET /api/challenge/jobs` (see
[challenge-poll-result](../challenge-poll-result/SKILL.md)); without `JOB_TOKEN`,
the login `CHALLENGE_TOKEN` also dials the tunnel.

For HTTP 500 `not login`, check credentials first. If a job is missing, check API
host, pagination, and season semantics; do not hard-code a season label or create
a replacement job without a request.

## Start, scale, and inspect

```bash
"$PYTHON" "$SKILL_DIR/scripts/route_inference.py" start --config ./routes.json --run-dir ./run-job
"$PYTHON" "$SKILL_DIR/scripts/route_inference.py" status --run-dir ./run-job
```

Job progress and results come from [challenge-poll-result](../challenge-poll-result/SKILL.md)
with that job's `JOB_ID` and `JOB_SEASON`.

`start` probes every new route's actual model response before launching detached
processes. A listening TCP port is not model readiness. Each route has its own
log and health JSON; `routes.json` records PID, process identity, and persistent
`agent_id`. Repeat the real-request probe after a model restart.

Starting the same configuration again is idempotent. Increase `routes` and start
with the same run directory to add missing routes without interrupting live ones.
Changing a live route's model configuration, shrinking the allocation, or changing
the job is rejected; stop the intended slots explicitly first.

For simultaneous old and new jobs, use separate configurations, state-file copies,
and run directories, such as 3 routes for one job and 5 for another. `start`
checks a configuration's route count against that job's `PARALLELISM` and the
account's `max_concurrent_cases`, but cannot see connections that other tools or
hosts opened for the same job. Inspect known runs; do not retry forever or kill
unrelated processes to free capacity.

Distinguish the following in reports:

1. `process_alive`: the route process exists and its PID identity matches.
2. `model_warmed_at`: a real model request passed validation.
3. `gateway_connected` / `phase`: the latest observed tunnel state, updated on
   disconnect/reconnect; process liveness alone is insufficient. A new connection
   receives `warmup` and moves to `RUNNING` (or `QUEUED` when no GPU is free).
   `CONNECTED` means the tunnel has stayed open past a short grace period without
   a control frame on this connection, which is the case for a reattach inside
   the reconnect window. `start` treats `CONNECTED`, `QUEUED`, and `RUNNING` as
   ready.
4. `actions_generated` / `last_action_at`: valid action responses were generated.
   Use session and worker logs to check simulation stepping; generation does not
   acknowledge platform receipt or scoring.
5. Platform `progress.done/total`: settled cases. Episodes can advance before
   these counters change. Platform results determine the final score and failure.

For `queued/no_gpu_available`, keep routes connected while platform simulation
resources become available. The bundled client sends unsolicited pong frames
every 15 seconds; the documented gateway ping interval is 20 seconds. Reconnect
with the same `agent_id` within the approximately 30-second session window.
Recovery beyond that window depends on platform retries/checkpoints; do not
promise arbitrary-frame resumption. The client exits after `drain` without
reconnecting. Fetch terminal results with
[challenge-poll-result](../challenge-poll-result/SKILL.md).

## Stop or switch

```bash
"$PYTHON" "$SKILL_DIR/scripts/route_inference.py" stop --run-dir ./run-job --slot route_model_a_0
"$PYTHON" "$SKILL_DIR/scripts/route_inference.py" stop --run-dir ./run-job
```

`stop` targets only recorded processes whose PID identities match. It preserves
models, weights, platform jobs, scores, and persistent agent identities.
Disconnecting an active session can interrupt its episode; prefer idle or
finished slots, and record any user-authorized interruption. Do not use global
`pkill -f` matching, take over unknown cases, or fabricate actions/scores to skip
tasks. After a model memory or endpoint change, pass the real probe again before
reattaching routes.

Local mock regression tests are in `scripts/test_routes.py`. Run them in isolation;
they do not require official jobs or access to ongoing evaluations.
