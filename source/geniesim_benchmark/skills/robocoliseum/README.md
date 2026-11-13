# Simulation Challenge — AI Agent Skills

A set of stage-by-stage skills that let an AI assistant (Claude Code, Cursor, Copilot, Cline, …) walk a contestant through the full submission flow: **login → submit job → launch inference agent → poll result → check ranking**.

`challenge-submit-job` returns a `job_token`; `challenge-run-agent` uses it to dial the tunnel — it is scoped to the job and does not expire. The login token `CHALLENGE_TOKEN` is also accepted for the dial.

Already have an inference service that works with your simulator? Use [challenge-route-inference](./challenge-route-inference/SKILL.md) to reuse it through multiple tunnel routes, with separate state for each job. Its route helpers and protocol client are bundled in that skill folder.

Two job identifiers come out of a submission and are **not** interchangeable: the numeric `JOB_ID`
(`.id`) addresses every HTTP endpoint (`/job/{id}/result`, `/log`), while `JOB_UUID` (`.uuid`) is
used only for the WebSocket tunnel dial.

## How to use

1. Copy each skill folder you need, including its `scripts/` and `references/` resources, into your
   AI tool's skill directory. Keep related skill folders together so cross-skill links resolve.
   The simulator inference protocol is included under `challenge-route-inference/references/`.
2. Tell the assistant the stage you want, e.g. *"use challenge-submit-job to submit a model"*.
3. The assistant follows the skill, gathers missing parameters, and runs the authorized workflow.
   It explains submission costs and route allocations before taking external actions, and requests
   any missing authorization without repeating approval already given for the concrete plan.

All skills assume the following environment variables exist; the assistant should help you set them up if missing:

```bash
export BASE_URL="https://robocoliseum.ai"    # platform host
export EMAIL="you@example.com"
export PASSWORD="your-password"
```

## State file (cross-shell persistence)

`CHALLENGE_TOKEN` and the job-scoped variables (`JOB_ID` / `JOB_UUID` / `PARALLELISM` / `TUNNEL_ENDPOINT` / `JOB_TOKEN` / `JOB_SEASON`) are **persisted to disk** at:

```
~/.simubotix-challenge.env       # mode 0600
```

Why: AI assistants (Claude Code's Bash tool, Cursor agent, etc.) frequently spawn each command in a **new subshell**, so plain `export` does not survive across calls. Writing to a file and `source`-ing it at the top of every command keeps state coherent without forcing the user to manually re-export anything.

**Producer skills** (`challenge-login`, `challenge-submit-job`) write to this file using the helper below. **Consumer skills** all start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

When chaining commands in one Bash call, `&&` still works — the file is the durable backup, not a replacement.

### `challenge_save_var` helper

Idempotent upsert of `KEY=VALUE` lines. Drop this in a terminal session or have the AI inline it:

```bash
challenge_save_var() {
  # usage: challenge_save_var KEY VALUE
  local f=~/.simubotix-challenge.env key="$1" val="$2"
  touch "$f" && chmod 600 "$f"
  # remove any prior line for this key, then append
  if [ -s "$f" ]; then
    awk -v k="$key" '$0 !~ "^" k "="' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
  printf '%s=%q\n' "$key" "$val" >> "$f"
  chmod 600 "$f"
  export "$key=$val"
}
```

The format is plain `KEY=VALUE` (shell-quoted via `%q`), so `. ~/.simubotix-challenge.env` works directly.

### Tearing it down

The state file persists across shells and across days; nothing in the skills auto-deletes it. Two ways to reset:

```bash
# Nuke everything (forces a fresh challenge-login)
rm -f ~/.simubotix-challenge.env

# Drop only the job vars, keep the token
challenge_reset_job() {
  local f=~/.simubotix-challenge.env
  [ -f "$f" ] || return 0
  awk '$0 !~ /^(JOB_ID|JOB_UUID|JOB_TOKEN|PARALLELISM|TUNNEL_ENDPOINT|JOB_SEASON)=/' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  chmod 600 "$f"
  unset JOB_ID JOB_UUID JOB_TOKEN PARALLELISM TUNNEL_ENDPOINT JOB_SEASON
}
```

Skills will detect the missing file (or missing keys) and re-run the producer skill (`challenge-login` / `challenge-submit-job`).

### Multi-job caveat (single-file design)

The state file holds **one job at a time**. `challenge-submit-job` rewrites `JOB_ID` / `JOB_UUID` / `PARALLELISM` / `TUNNEL_ENDPOINT` / `JOB_TOKEN` / `JOB_SEASON` on every submission, so the previous job's vars are lost. `challenge-route-inference` keeps a per-job copy of this file for each job it connects.

If you need to track an older job after submitting a new one, fetch it back from history rather than expecting the state file to remember it:

```bash
curl -fsS "$BASE_URL/api/challenge/jobs?season=1&page=1&per-page=20" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
# Season II jobs: season=2. The list is per season; omitted = 2.
```

`challenge-submit-job` Step 3 will explicitly call this out and ask for confirmation before overwriting unfinished job state.

## Skills

| Directory | When to use |
|-----------|-------------|
| [`challenge-help/`](./challenge-help/SKILL.md) | Entry point — pick the right skill for the user's intent. |
| [`challenge-login/`](./challenge-login/SKILL.md) | Get / refresh the JWT (`CHALLENGE_TOKEN`) and inspect the current user. |
| [`challenge-submit-job/`](./challenge-submit-job/SKILL.md) | Build the request body, check daily quota, POST `/api/challenge/job`, capture `JOB_ID` / `JOB_UUID` / `PARALLELISM` / `TUNNEL_ENDPOINT` / `JOB_TOKEN` / `JOB_SEASON`. Also submits a finished Season II job for review. |
| [`challenge-run-agent/`](./challenge-run-agent/SKILL.md) | Launch the inference agent. The supported invocation is `./scripts/tunnel.sh <gpu> <job_uuid> <endpoint>` from the **inference repo** (see `challenge-baseline-model`); contestants running their own client implement the tunnel protocol, which is inlined in this skill's appendix. Scale to `PARALLELISM` processes. Dials the tunnel with `JOB_TOKEN` (the login token `CHALLENGE_TOKEN` is also accepted). |
| [`challenge-route-inference/`](./challenge-route-inference/SKILL.md) | Reuse an already working simulation inference service: one or multiple services → multiple transparent routes → official tunnel evaluation. Includes real-request readiness probes, per-job state, new/resumed jobs, route scaling/status/stop, custom adapters, and the simulator inference protocol. Router processes do not load models; one model can serve several routes. |
| [`challenge-poll-result/`](./challenge-poll-result/SKILL.md) | Poll `GET /api/challenge/jobs` for `detailed_status` until terminal; then `/job/$JOB_ID/result` for scores, `/log` on failure. |
| [`challenge-ranking/`](./challenge-ranking/SKILL.md) | Fetch `best-score` and the leaderboards viewed by model type or combined (Season I, Season II, top 5). |
| [`challenge-troubleshoot/`](./challenge-troubleshoot/SKILL.md) | Map a symptom to the likely cause and the next command to run. |
| [`challenge-diagnose-connection/`](./challenge-diagnose-connection/SKILL.md) | Agent keeps disconnecting / `failure_reason: 9` / `tunnel lost`: read-only local checks (event-loop lag, link loss, large-frame throughput) to separate "my machine/network" from "the platform", plus what evidence to send the organizers. |
| [`challenge-baseline-model/`](./challenge-baseline-model/SKILL.md) | Clone the inference repo, pull checkpoints, install deps, run the baseline agent. |
| [`challenge-download-datasets/`](./challenge-download-datasets/SKILL.md) | Pull seven LeRobot v2.1 training suites from ModelScope, including cross-embodiment transfer, dexterous manipulation, precision manipulation, and whole-body coordination. |
| [`challenge-download-assets/`](./challenge-download-assets/SKILL.md) | Obtain the simulator code, install its CLI, then selectively download and verify runtime assets by task or track. |

## Operating mode (hybrid)

- **Read-only** calls (login, current-user-info, jobs/result/log, best-score, ranking, quota check) — the assistant runs them directly and reports back.
- **Side-effecting** calls (creating/reviewing jobs, launching/stopping agents or routes, refreshing a token) — establish the concrete plan and consequences (submission quota or concurrency slots) before acting. Honor explicit authorization already given; request approval only when it is missing for the intended action.

The **daily submission cap is a platform setting, not a constant to memorize.** Every skill that
mentions it reads `GET /api/challenge/submission/quota` and quotes the returned
`limit` / `used` / `remaining`. Never state a number from prose or from a previous session.

## Self-contained by design

These skills are meant to be copied out of this repo into a contestant's AI-tool skill directory,
so **they do not link to any file outside `skills/`**. Everything needed is inlined:

- Tunnel wire protocol (dial, control frames, binary framing, state machine, reconnect, heartbeat)
  → appendix of `challenge-run-agent`.
- Observation/action payload schema, embodiment routing, depth/calibration, history, EEF/chassis
  control, and runtime-specific replies →
  [`challenge-route-inference/references/protocol.md`](./challenge-route-inference/references/protocol.md).
- Existing-service adapter hooks →
  [`challenge-route-inference/references/adapters.md`](./challenge-route-inference/references/adapters.md).
- API response shapes and status enums → `challenge-poll-result`.
- Accepted `config.board` values → `challenge-submit-job` Step 2.

There is **no official published Simulation SDK package**. Model-loading agents use the inference
repo's `./scripts/tunnel.sh`; existing services can use the helpers bundled with
`challenge-route-inference`. Custom clients implement the documented tunnel and application protocols. Do not direct users at clients inside the platform's own repository — those are internal
test tooling and unsupported.
