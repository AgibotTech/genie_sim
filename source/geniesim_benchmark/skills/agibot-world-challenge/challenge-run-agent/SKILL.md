---
name: challenge-run-agent
description: Use when the contestant wants to launch the inference Agent that connects to the Simulation gateway via WebSocket. Wraps the inference repo's ./scripts/tunnel.sh and scales to PARALLELISM processes. This is side-effecting (consumes GPUs and a parallelism slot) — confirm before launching.
---

# challenge-run-agent — Launch the inference Agent

The Agent is the contestant's long-running process that:

1. Reverse-connects to the gateway over WebSocket using `JOB_UUID` and a per-process `agent_id`.
2. Receives observation frames, runs inference, returns action bytes.
3. Stays online until the platform sends a `drain` control frame (job complete).

## Which client to run

**There is no official published Simulation SDK package.** Do not tell the user to `pip install`
one, and do not point them at any client inside the platform's own repository — the platform repo
ships only internal test clients, which are not supported for contest use and can change without
notice.

Exactly two supported paths:

| Path | When | How |
|------|------|-----|
| **Inference repo's `./scripts/tunnel.sh`** (recommended) | The contestant has cloned an inference repo — the baseline one, or their own fork | `./scripts/tunnel.sh <gpu_index> <job_uuid> <gateway_url>`. If they haven't cloned it, run `challenge-baseline-model` first. |
| **Their own WebSocket client** | They wrote their own inference stack and want to talk to the gateway directly | Implement the tunnel protocol — the full contract is inlined in the appendix at the bottom of this skill. Nothing else is needed; there is no library to install. |

The rest of this skill assumes `tunnel.sh`. If the contestant is on their own client, the
concurrency rules, launch plan, and lifecycle sections below still apply verbatim — only the
command changes.

## Preconditions

| Variable | Source |
|----------|--------|
| `JOB_TOKEN` | `challenge-submit-job` (job response `.job_token`); preferred credential for the tunnel dial — scoped to `JOB_UUID`, never expires. If unset (older platform builds), falls back to `CHALLENGE_TOKEN` |
| `CHALLENGE_TOKEN` | `challenge-login`; used as the tunnel-dial fallback when `JOB_TOKEN` is unset, and still required for all normal HTTP API calls (submit/poll/result/etc.) |
| `JOB_UUID` | `challenge-submit-job` (job response `.uuid`) — the tunnel dial takes the **UUID**, never the numeric `JOB_ID` |
| `PARALLELISM` | `challenge-submit-job` |
| `TUNNEL_ENDPOINT` | `challenge-submit-job` (job response), else fixed default `ws://120.92.88.78/api/challenge/tunnel` |
| inference repo cloned | user-side; `./scripts/tunnel.sh` must exist (→ `challenge-baseline-model`), unless they run their own client |

If any are missing, jump back to the producing skill instead of guessing.

`JOB_TOKEN` does not expire, so long-running agent processes won't hit a mid-run `401` from token
expiry the way a `CHALLENGE_TOKEN`-based dial could.

Every Bash call in this skill should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

(Cross-shell state lives in `~/.simubotix-challenge.env`; AI assistants spawn each command in a new subshell. See README "State file".)

## Concurrency rules (read these before launching)

- The platform enforces **at most `$PARALLELISM` concurrent agent processes** per user. Excess connections are closed immediately after the WS upgrade.
- Each tunnel carries **at most one active session at a time** (1:1 contract). To run cases in parallel, run multiple agent processes — that's the whole point of `parallelism`.
- Each process must use a **different GPU index** if launched on the same host. If the user's box has only `N` GPUs, `min(N, PARALLELISM)` is the real cap.
- Each process must use a **different `agent_id`**. `tunnel.sh` derives it from the job and GPU index and persists it, so one process per GPU needs nothing extra. If you start **more than one process on the same GPU**, give each a distinct `AGENT_SLOT=0,1,2…` in its environment; otherwise they share an `agent_id` and the gateway rejects all but the first.

## Step 1 — Confirm the launch plan with the user

Before running anything, summarise:

> About to launch **K** agent processes against `JOB_UUID=…`, gateway `TUNNEL_ENDPOINT=…`. K must be ≤ PARALLELISM (`$PARALLELISM`) and ≤ available GPUs. Each process pins one GPU. Proceed?

Wait for explicit confirmation. Default K to `$PARALLELISM` only if the user said so — otherwise ask.

## Step 2 — Launch a single agent (sanity check)

```bash
cd <inference-repo>
CHALLENGE_TOKEN="${JOB_TOKEN:-$CHALLENGE_TOKEN}" ./scripts/tunnel.sh 0 "$JOB_UUID" "$TUNNEL_ENDPOINT"
```

The reference `tunnel.sh` takes no token argument — it reads the credential from
`$CHALLENGE_TOKEN` in its environment. The inline `CHALLENGE_TOKEN=...` prefix therefore makes it
dial with the non-expiring `job_token` for **this tunnel process only**; it does not overwrite the
shell's own `CHALLENGE_TOKEN`, so other HTTP API calls (submit/poll/result) in the same session
keep using the login token. Falls back to the login token if `JOB_TOKEN` is unset (older
platforms). If the user's fork reads a differently-named variable, check the top of their
`tunnel.sh` and adjust the prefix.

Argument order:

| Position | Value | Notes |
|----------|-------|-------|
| `$1` | GPU index | e.g. `0`. Must exist on the host. |
| `$2` | `$JOB_UUID` | The `uuid` from `POST /api/challenge/job`, not the numeric `id`. |
| `$3` | `$TUNNEL_ENDPOINT` | Full WS URL. Defaults to the fixed `ws://120.92.88.78/api/challenge/tunnel`; prefer the job response's `tunnel_endpoint` if it carries one. |

The baseline repo also needs a board selector env var so the loaded checkpoint matches the job's
`config.board` — see `challenge-baseline-model` Step 4 for the exact variable it expects.

Verify the agent reaches the **WARMUP → RUNNING** lifecycle before scaling out. If it disconnects right after handshake, see `challenge-troubleshoot`.

## Step 3 — Scale to PARALLELISM processes

First decide the real launch count `K = min(PARALLELISM, NUM_GPUS)`. Detect available GPUs (one of):

```bash
NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
# fallback: if nvidia-smi is unavailable, ask the user how many GPUs to use.
K=$(( NUM_GPUS < PARALLELISM ? NUM_GPUS : PARALLELISM ))
echo "launching K=$K processes (parallelism=$PARALLELISM, gpus=$NUM_GPUS)"
```

If `K < PARALLELISM`, tell the user explicitly — they can either accept reduced concurrency or add more GPUs. Do **not** double-pin a GPU.

Map GPU `i` to agent process `i`:

```bash
PIDS=()
for i in $(seq 0 $((K - 1))); do
  CHALLENGE_TOKEN="${JOB_TOKEN:-$CHALLENGE_TOKEN}" ./scripts/tunnel.sh "$i" "$JOB_UUID" "$TUNNEL_ENDPOINT" &
  PIDS+=($!)
done
trap 'kill "${PIDS[@]}" 2>/dev/null || true' EXIT
wait
```

Same scoped `CHALLENGE_TOKEN="${JOB_TOKEN:-$CHALLENGE_TOKEN}"` prefix as Step 2, applied per-process — each backgrounded `tunnel.sh` gets the job token in its own environment without touching the parent shell's `CHALLENGE_TOKEN`.

`trap` ensures `Ctrl-C` (exit 130) cleans up children. The `wait` at the end blocks the foreground shell until every agent exits — see "Driving from an AI assistant" below if you're orchestrating this from a non-interactive session.

### Driving from an AI assistant

The launch loop is **long-lived**: it stays alive until every process gets `drain` (exit 0) or the user hits Ctrl-C. If you're an AI assistant orchestrating this from a single shell:

- Tell the user to run the loop in a **dedicated terminal** (or under `tmux` / `nohup`), then come back to your session for `challenge-poll-result`.
- Do NOT background the whole loop with `&` and continue polling from the same shell — when the assistant's session ends, the agents go with it.

If you have to launch and poll from the same automation, run the loop with `nohup` redirected to a log file and treat its PID as opaque until poll reaches a terminal status.

## Lifecycle and exit codes

| Code | Meaning |
|------|---------|
| 0 | `drain` received → graceful shutdown. Job dispatch is finished; do **not** reconnect. |
| 1 | Bad arguments / model load error / inference kept failing / could not reconnect for 180 s. Relaunching the same command keeps the `agent_id`. |
| 2 | The gateway refused the dial (HTTP 401/403/404): token invalid, or the job is no longer active. Do not relaunch — check job status first. |
| 130 | Ctrl-C. |

If a process dies abnormally, **relaunch the same command** — `tunnel.sh` keeps the `agent_id` in
`~/.simubotix/agent_<job_uuid>_<gpu>[-s<AGENT_SLOT>].id`, so a restart inside the **30-second reconnect window**
reattaches the open session. Past 30 s the open case is failed and one of its 3 retries is
consumed, but while the job is still `Ready`/`Running` a new dial is still accepted and the case is
assigned again. Only when a case has no retries left does the job fail (dials then return 401) and
the user must resubmit — which consumes a **daily submission slot**; confirm before suggesting it
and re-probe `/api/challenge/submission/quota` first. Do not start a second copy while the first is
still running: the gateway refuses the duplicate dial.

## After launch

Cases queue and run as agents stay online. Hand off to `challenge-poll-result` to track status.

- **Don't kill agents** until the job's `detailed_status` reaches `completed` / `failed` — early termination loses in-flight cases.
- Once dispatch is done the platform sends `drain` and the agent exits 0 on its own. Wait for that natural exit; do not Ctrl-C preemptively.
- `detailed_status: queued` means an agent is attached but no case is running yet (usually no free
  GPU cluster-wide) — **not** that something is wrong. Keep the agents online; killing them
  discards the warmed-up slot.

## Failure shortlist

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Handshake `401` | Token / job mismatch, or job already terminal | `challenge-login`; verify `JOB_UUID` belongs to this account and the job is still active. |
| Connects then drops | Per-user parallelism cap exceeded | Reduce K or wait for old agents to finish. |
| `drain` received | Job is wrapping up | Stop launching, wait for graceful exit. |
| Stuck in `WARMUP` | Inference handler errored on the warmup empty frame | Check the inference process's logs. |

For more, jump to `challenge-troubleshoot`.

---

## Appendix — Tunnel protocol (for a self-built client)

Everything needed to write your own agent. The wire format is two WebSocket message types and six
JSON control frames. No Protobuf, no gRPC. For what goes *inside* the binary payload
(observation / action schema), see `challenge-inference-protocol`.

### A.1 Dial

```
GET <TUNNEL_ENDPOINT>?job=<JOB_UUID>&agent=<agent_id>
Authorization: Bearer <JOB_TOKEN or CHALLENGE_TOKEN>
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Version: 13
```

- `agent_id` — a UUIDv4 you generate (≤ 64 bytes UTF-8). **Every process must use its own.**
- Prefer `JOB_TOKEN`: scoped to that one job, no expiry, unaffected by password changes.
  `CHALLENGE_TOKEN` is accepted for back-compat but expires in 12h.
- `101` = handshake OK. `401` = bad/missing token, job not found or not owned by you, job in a
  terminal state, or the `?job=` doesn't match the token's job. `500` = backend lookup failed,
  retry with backoff.
- **Capacity exhaustion still returns `101`** (HTTP can't 429 after `Upgrade`) and the server
  immediately sends a close frame. Treat an instant close after upgrade as "over the parallelism
  cap".
- Dialing with a `(job_uuid, agent_id)` pair that already has a live connection closes the *new*
  socket; the existing one survives.

After upgrade the connection sits in `QUEUED`, waiting for the server's first control frame.

### A.2 Control frames — `TextMessage`, JSON

```json
{ "type": "<control-type>", "session_id": "<optional>", "reason": "<optional>" }
```

| Type | Direction | Purpose |
|------|-----------|---------|
| `warmup` | server → agent | Load model weights / CUDA contexts now |
| `ready` | agent → server | Warmup complete; ready for sessions |
| `queued` | server → agent | `ready` received, but no GPU free cluster-wide; stay connected |
| `session_open` | server → agent | Allocate a fresh inference context for `session_id` |
| `session_close` | either | Release the context for `session_id` |
| `drain` | server → agent | No more sessions; wind down and close. **Do not reconnect.** |

- `session_id` is required on `session_open` / `session_close`; ignored on the rest.
- Unknown `type` values are logged and ignored by the gateway, not fatal — **do the same** so
  future server-side frames don't break your client.
- Wrong-direction frames are tolerated but have no effect: an agent sending `warmup` or
  `session_open` is silently ignored. Session lifecycle is server-driven.
- Do **not** invent a `{"type":"ping"}` — there is no application-level heartbeat (see A.6).

`queued` (`{"type":"queued","reason":"no_gpu_available"}`) is informational and one-shot: at most
one per agent per job, only after `ready` and before the first `session_open`. Keep the socket
open and the model resident; there is no "queue cancelled" frame and no server-side queue
timeout. You leave the state on `session_open` or `drain`.

### A.3 Data frames — `BinaryMessage`

```
┌──────────────────┬───────────────────┬─────────────────────┐
│ sidLen (4 bytes) │ session_id (UTF-8)│ payload (arbitrary) │
│ big-endian uint32│ sidLen bytes      │ rest of the frame   │
└──────────────────┴───────────────────┴─────────────────────┘
```

- `session_id` must be ≤ 256 bytes. Reply with the **same** `session_id`.
- `payload` may be zero-length — the warmup round-trip uses an empty payload, and your handler
  **must tolerate it** (short-circuit and return a no-op action). This is the single most common
  contest bug.
- The gateway is payload-agnostic: bytes in, same bytes out on the simulator side.
- No framing checksum; WebSocket's per-message CRC is the integrity guarantee.
- A frame shorter than 4 bytes, or whose `sidLen` overruns the frame, **drops the connection**.
  A frame for an unregistered `session_id` is silently dropped.

### A.4 State machine

```
QUEUED ──warmup──▶ WARMUP ──ready──▶ RUNNING ──drain──▶ DRAINING
```

All other transitions are rejected. Sending `ready` before receiving `warmup` is ignored — don't
try to short-circuit warmup. Data frames are accepted in any state as long as the `session_id` is
known.

### A.5 Sessions and reconnect

One active session per tunnel, sequentially reused across cases: `session_open` → data frames →
`session_close`, then the scheduler may open the next one on the same tunnel. Echoing
`session_close` back is optional. To run cases concurrently, run more processes (up to
`parallelism`), not more sessions.

On an unclean close you get a **30-second reconnect window**: redial with the same
`(job_uuid, agent_id)` and Bearer token and the open session is restored. There's no explicit
"I'm reconnecting" handshake — the server matches on the pair. Frames the server tried to send
while you were down are **dropped, not queued**. Past 30 s the tunnel is gone and the in-flight case loses one retry; a new dial is still accepted while the job is active. Never reconnect after `drain`.

### A.6 Heartbeat

Liveness is handled at the WebSocket library layer — **no application-level heartbeat.** The
gateway pings every 20 s and enforces a 45 s read deadline; 45 s with no inbound bytes opens the
reconnect window. While a multi-MB observation frame is downloading the gateway's ping is stuck
behind it, so send an **unsolicited pong every ~15 s** (`await ws.pong()`) — it rides your upstream
path and the gateway counts it as liveness. Do not let the client close on a missing pong; prefer
no client ping at all: `websockets.connect(url, ping_interval=None, ping_timeout=None)`.
The simulator waits at most 120 s per answer, and that includes downloading the frame (~6.5 MB with
depth on): a slow downstream link fails cases even when the tunnel stays up.

### A.7 Minimal Python skeleton

Handshake, warmup, one session round-trip. Omits reconnect (A.5), per-session concurrency, and
bounded buffering — all three are required in a real client.

```python
import asyncio, json, struct, uuid, websockets

async def run_agent(tunnel_url: str, token: str, job_uuid: str, agent_id: str | None = None):
    agent_id = agent_id or str(uuid.uuid4())
    ws_url = f"{tunnel_url}?job={job_uuid}&agent={agent_id}"
    headers = {"Authorization": f"Bearer {token}"}

    async with websockets.connect(ws_url, additional_headers=headers,
                                  ping_interval=None, ping_timeout=None) as ws:
        async for msg in ws:
            if isinstance(msg, str):                       # control frame (A.2)
                ctl = json.loads(msg)
                t = ctl.get("type")
                if t == "warmup":
                    await load_model()                     # your code
                    await ws.send(json.dumps({"type": "ready"}))
                elif t == "session_open":
                    open_context(ctl["session_id"])        # your code
                elif t == "session_close":
                    close_context(ctl["session_id"])       # your code
                elif t == "drain":
                    return                                 # do NOT reconnect
                # any other type: log and ignore
            else:                                          # data frame (A.3)
                sid_len, = struct.unpack(">I", msg[:4])
                sid = msg[4:4 + sid_len].decode("utf-8")
                payload = msg[4 + sid_len:]
                action = infer(sid, payload)               # must handle payload == b""
                await ws.send(struct.pack(">I", sid_len) + sid.encode("utf-8") + action)
```

Run `parallelism` of these as separate OS processes (each with its own `agent_id` and GPU), not as
asyncio tasks in one process — one process per GPU is the supported topology.
