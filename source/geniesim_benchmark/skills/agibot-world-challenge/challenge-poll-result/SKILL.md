---
name: challenge-poll-result
description: Use to track a Simulation Challenge job's progress — list jobs, watch a job's status until terminal, fetch its per-task scores, or pull execution logs when it failed. Read-only; safe to run without confirmation.
---

# challenge-poll-result — Track job status, scores, and logs

All endpoints here are **read-only** — run them directly and report findings to the user.

## Preconditions

- `CHALLENGE_TOKEN` set (else → `challenge-login`).
- `JOB_ID` set (else → `challenge-submit-job`, or list jobs first to find one).

Every Bash call should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

(See README "State file" — AI assistants spawn each command in a new subshell, so file-backed state is the only reliable handoff.)

## Two job identifiers — do not mix them up

| Variable | Shape | Used for |
|----------|-------|----------|
| `JOB_ID` | **integer** (e.g. `42`), from the submit response's `.id` | every HTTP path here: `/job/{JOB_ID}/result`, `/log`, `/episodes/videos` |
| `JOB_UUID` | UUIDv4 string, from the submit response's `.uuid` | **only** the WebSocket tunnel dial (`challenge-run-agent`) |

The path segment is parsed as an unsigned integer. Passing the UUID returns
`400 {"status":"error","error":"invalid job_id: <uuid>"}` — under `curl -fsS` that surfaces
only as exit code 22, which reads like a server fault. If a call fails, check you used the
numeric id first.

## Which endpoint answers which question

| You want to know | Call | Read |
|------------------|------|------|
| Is it finished? Which state? How far along? | `GET /api/challenge/jobs` | `detailed_status`, `progress`, `score` on your job's item |
| Per-task / per-step scores | `GET /api/challenge/job/{JOB_ID}/result` | `tasks`, `total` |
| Why did it fail? | `GET /api/challenge/job/{JOB_ID}/log` | `failure_reason`, `result` |

**Status is on the job list; `/result` is the scores endpoint and deliberately has no status
field.** Don't tell the user "the API doesn't expose status" — it does, just not on `/result`.
There is no single-job status endpoint, so fetch the list and pick your job by `id` (below).

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
curl -fsS "$BASE_URL/api/challenge/jobs?page=1&per-page=20" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

> **Query param is `per-page` with a hyphen.** The server reads `per-page`; `per_page` is
> silently ignored and you get the default page size of 20. The *response* echoes it back as
> `per_page` (underscore) — the asymmetry is real, don't "fix" it in either direction.

Response envelope:

```json
{ "items": [ … ], "total": 7, "page": 1, "per_page": 20, "total_pages": 1 }
```

Each item, with the fields that matter for polling:

```json
{
  "id": 42,
  "uuid": "3f7a18c2-8e1d-4bd9-9a42-9a14c0b7a7b1",
  "name": "challenge_test",
  "model_name": "test",
  "board": "instruction",
  "status": 1,
  "detailed_status": "evaluating",
  "progress": { "done": 3, "total": 12 },
  "score": 0,
  "time_cost": 0,
  "created_at": "…",
  "list_on_leaderboard": true
}
```

Other query params: `model_name` (fuzzy match), `order` / `sort-by` (`col:dir`, comma-separated).
There is **no id filter** — select client-side with `jq`.

### `status` is an integer, not a string

| `status` | Name | Meaning |
|---------:|------|---------|
| `0` | Ready | Created, nothing dispatched yet |
| `1` | Running | At least one case dispatched |
| `2` | Succeed | All cases landed successfully |
| `3` | Stopped | Stopped by user or system |
| `4` | Failed | A case exhausted its retries → the whole job is dead |
| `5` | WaitingForStop | Stop requested, teardown in flight |

Do not compare it against `"Finished"` / `"Pending"` / `"Cancelled"` — those strings exist
nowhere in this API.

### `detailed_status` is the string you should actually branch on

| `detailed_status` | Meaning | Next action |
|-------------------|---------|-------------|
| `pending` | No agent has ever connected for this job | Launch agents (`challenge-run-agent`). Cases sit in queue until one is online. |
| `queued` | An agent tunnel is live (connected, warming up, or idle) but no case is executing yet — usually waiting for a free GPU cluster-wide | Keep the agents alive and keep polling. Nothing to fix. |
| `evaluating` | At least one case is executing | Keep polling. |
| `inference_disconnected` | An agent connected earlier but none is live now | Check the agent processes; see `challenge-troubleshoot`. |
| `completed` | Terminal — all cases landed, or job reached Succeed/Stopped | Stop polling. Fetch `/result`, then suggest `challenge-ranking`. |
| `failed` | Terminal — job is dead | Stop polling. Pull `/log` and route to `challenge-troubleshoot`. |

Terminal set is exactly `{completed, failed}`. `failed` is checked before the "all cases done"
rule, so a job whose cases all landed but which was judged failed still reports `failed` — trust
`detailed_status`, not `progress`.

`progress` is case-level `X/N`. On a failed job it freezes at the real count (aborted in-flight
cases count as done; never-started ones don't), so `done < total` on a `failed` job is expected,
not a bug.

## Polling loop

Poll every **2–5 seconds** until `detailed_status` is terminal. Don't poll faster — the platform
doesn't change state that quickly and you'll burn rate budget.

```bash
while true; do
  [ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
  JOB=$(curl -fsS "$BASE_URL/api/challenge/jobs?page=1&per-page=100" \
    -H "Authorization: Bearer $CHALLENGE_TOKEN" \
    | jq -c --argjson id "$JOB_ID" '.items[] | select(.id == $id)')
  if [ -z "$JOB" ]; then
    echo "job $JOB_ID not found on page 1 — widen per-page or check the id"; break
  fi
  DS=$(echo "$JOB" | jq -r '.detailed_status // "?"')
  PROG=$(echo "$JOB" | jq -r '"\(.progress.done)/\(.progress.total)"')
  SCORE=$(echo "$JOB" | jq -r '.score')
  printf "detailed_status=%-22s progress=%-8s score=%s\n" "$DS" "$PROG" "$SCORE"
  case "$DS" in
    completed|failed) break ;;
  esac
  sleep 3
done
```

`--argjson` (not `--arg`) matters: `.id` is a JSON number, and `select(.id == "42")` never
matches a string.

If the job might be off page 1 (many submissions), either raise `per-page` or add
`&sort-by=id:desc`.

When you call this from an AI assistant, do not loop indefinitely without checking in — every
~30 seconds report progress (`detailed_status`, `progress`, `score`) so the user can interrupt
if something looks wrong (e.g. stuck at `pending` because no agent is connected).

## Fetch a job's scores

```bash
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/result" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

Exact response shape — **two keys (`tasks`, `total`); status is on the job list, not here**:

```json
{
  "tasks": {
    "pick_block_color": { "score": [95.0, 92.0, 88.0], "total": 91.67 },
    "place_on_plate":   { "score": [100.0, 100.0, 98.0], "total": 99.33 }
  },
  "total": 191.0
}
```

- Keys of `tasks` are **task names** (`EmuTask.Name`), which vary by board — don't hard-code them.
- `tasks[…].score` is a per-step average across the task's latest episodes (element *i* is the
  mean of step *i*), **not** a per-case score list.
- `tasks[…].total` is the mean of those episodes' average scores.
- Top-level `total` is the job's stored score. It is `0` until scoring lands, so a `0` here on a
  still-running job means "not scored yet", not "scored zero".
- A job with no episodes yet returns `{"tasks":{},"total":0}` — that is a success, not an error.

`/result` answers "what are the scores", not "is it done", so **never** use it to decide whether
to keep polling. Check `detailed_status` on the job list first, and call `/result` once it
reaches `completed`.

## Fetch failed-job log

Only meaningful when `detailed_status == "failed"`:

```bash
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/log" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

Returns the newest failed case's `EmuCaseExecLog`:

```json
{
  "id": 913,
  "job_id": 42,
  "task_id": 88,
  "case_id": 401,
  "run_instance_id": 7,
  "worker_id": 3,
  "status": 1,
  "failure_reason": 1,
  "result": { "stdout": "https://…presigned…" },
  "created_at": "…"
}
```

- If the job has **no** failed case log, the endpoint errors (HTTP 500, body
  `{"status":"error","error":"failed to get case exec log…"}`). Check `detailed_status` first
  rather than probing.
- `status`: `0` success, `1` failed, `2` running. This endpoint only serves `1`.
- `failure_reason` is the platform's own classification — read it before reading any log text:

  | Value | Meaning |
  |------:|---------|
  | `0` | none |
  | `1` | execution error (your code or the sim crashed) |
  | `2` | submit error |
  | `3` | timeout |
  | `4` | terminated by signal |
  | `5` | startup timeout — the pod never reported started |
  | `6` | stale placement — the pod started then hung |
  | `7` | manual reap by an operator |
  | `8` | job already dead; this case was aborted as an in-flight sibling |
  | `9` | tunnel lost — your agent disconnected and did not return within the 30 s reconnect window |

  `5`/`6`/`7`/`8` are **platform-side teardowns, not your bug** — for `8` in particular the real
  root cause is a *different* case, so don't debug this log. `1`/`3`/`4` point at your inference
  code or the case itself. `9` is **usually on your side**: the agent dropped the tunnel (process
  exit, OOM, or a long inference call starving the heartbeat). Read the sub-reason in
  `result.error` — only `startup reconcile` is platform-side; `reconnect window expired` and
  `liveness probe` mean your agent went away.
- `result` is a free-form JSON map written by the worker. There is no guaranteed `exit_code` or
  `stderr` key — inspect what's actually there (`jq '.result | keys'`) instead of assuming.
- **`result.stdout` is usually a URL, not the log text.** When the worker archived stdout to
  object storage, the platform rewrites it into a 1-hour HTTP presigned link, so you must fetch
  it separately:

  ```bash
  LOG=$(curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/log" \
    -H "Authorization: Bearer $CHALLENGE_TOKEN")
  echo "$LOG" | jq '{failure_reason, status, case_id, run_instance_id}'
  URL=$(echo "$LOG" | jq -r '.result.stdout // empty')
  case "$URL" in
    http*) curl -fsS "$URL" | tail -100 ;;
    error) echo "platform failed to sign the stdout URL — report to organizers" ;;
    *)     echo "$LOG" | jq -r '.result' ;;
  esac
  ```

  The literal string `"error"` means presigning failed platform-side. Platform-generated logs
  (e.g. a manual reap) put a plain message under `result.error` instead.

When the log is large, **don't dump everything**. Surface the load-bearing tail:

1. `failure_reason` and `status` (always include).
2. The last `Traceback (most recent call last):` block, if present, in full — that's the actual failure.
3. The last ~20 lines around it for context.

A common failure pattern: a `Traceback` in the observation-decode / handler path during a case
that started with an empty frame is the **warmup empty-frame bug** — see `challenge-troubleshoot`
row #11. It can surface as `failed` mid-job, not only as "stuck in WARMUP".

## Episode videos (optional)

```bash
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/episodes/videos" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

Returns `{job_id, status, expires_in, episodes:[{task_name, episode_uuid, task_id, case_id, score, head}]}`.
Here `status` is a *video-availability* code, unrelated to job status: `0` normal, `1` past the
72-hour retention window (`head` is an empty string), `2` evaluation didn't finish successfully.
Links expire after `expires_in` seconds (8h).

## Hand-off

- `completed` → fetch `/result`, then suggest `challenge-ranking` to see where the score lands.
- `failed` → pull `/log`, read `failure_reason` first, then `challenge-troubleshoot`.
- `pending` / `inference_disconnected` for more than a few minutes → `challenge-troubleshoot`
  (no agent connected, or all dropped past the gateway's 30 s reconnect window).
- `queued` → nothing is wrong; an agent is attached but no case is running yet (usually no free
  GPU). Keep agents online and wait.
