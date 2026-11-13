---
name: challenge-troubleshoot
description: Use when something in the Simulation Challenge pipeline is misbehaving — auth errors, agent disconnects, jobs stuck in Pending, jobs ending in Failed, drain frames. Maps a symptom to its likely cause and the next command to run.
---

# challenge-troubleshoot — Symptom → cause → next command

When the user reports a problem, do this in order:

1. **Reproduce / verify the symptom** with a read-only call (`/jobs`, `/result`, `/log`, `/current-user-info`) before doing anything destructive. For "what's the status" questions, ask `/jobs` (`detailed_status`); `/result` is for scores only.
2. Match the symptom to the table below.
3. Hand the user the next command from the **Action** column. Do NOT auto-resubmit jobs (quota cost) or auto-kill agents (loses in-flight cases) without explicit confirmation.

## Diagnostic heuristic: 4xx is never a network blip

A structured `{"status":"error", ...}` body with HTTP **4xx** (400/401/403/404) is a **semantic rejection** by the platform — the request reached the server and was refused on its merits. Retrying it without changing the request gets the same answer. A rejected request writes nothing, so it does not consume a daily submission slot — but the retry loop still wastes the user's time and the assistant's turns.

Only these qualify as transient and may be retried with backoff:

- TCP connection reset / timeout / DNS failure (curl exits non-zero before getting a response)
- HTTP `5xx` (server-side, signals a transient backend failure)

If the user says "maybe a network blip" but you have a 4xx body in hand, push back: name the actual error and point at the table row.

## Diagnostic heuristic: 5xx is often a missing / wrong token, not a real platform outage

`curl -fsS` collapses any HTTP 5xx to a terse `(22) The requested URL returned error: 5XX` and hides the body. Before concluding the platform is down, check this in order:

1. **State file sourced?** Each Bash call should start with `[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env` — AI assistants spawn every command in a new subshell, so plain `export` from a previous call is gone.
2. **Token presence after sourcing.** `echo "len=${#CHALLENGE_TOKEN}"` — if it's 0, either the file doesn't exist (re-run `challenge-login` Step 1) or the file exists but doesn't contain the key (the previous login silently failed — see `challenge-login` Step 1's status check).
3. **Token shape.** Should be a 3-segment JWT (two dots). `echo "$CHALLENGE_TOKEN" | head -c 20; echo` — gibberish or `null` means a previous step persisted a bad value.
4. **Body of the actual response.** Drop `-f` so curl prints the body even on non-2xx:
   ```bash
   [ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
   curl -sS -i "$BASE_URL/api/challenge/current-user-info" \
     -H "Authorization: Bearer $CHALLENGE_TOKEN" | tail -15
   ```
   The body usually says exactly what's wrong (often a 401-style message returned with a 500 status code).

Only after the token is confirmed valid (e.g. the same value just succeeded against `/login`) should you treat the 5xx as a real platform issue and retry with backoff.

## Symptom table

| # | Symptom | Likely cause | Action |
|---|---------|--------------|--------|
| 1 | `POST /api/challenge/job` returns **HTTP 400** `invalid board` / `no task templates for board` | `config.board` is missing or not in the accepted set | The accepted values are `instruction` (default), `spatial`, `manip`, `robust`, `v2`. Use one of those and resubmit. **Do not retry the same wrong board** — 400 is a semantic rejection, so the retry gets the identical error. It does not burn a slot (nothing is written), but it gets you nowhere. This holds even when the user says "maybe a network blip" — see the heuristic above. If a contestant names any other board, verify with the organizers first; `challenge-submit-job` Step 2 is the only list to trust. |
| 2 | `POST /api/challenge/job` returns an "upload limit" error / `/api/challenge/submission/quota` returns `remaining: 0` | Today's submission quota is used up | Wait until Beijing midnight (UTC+8), or work with an existing job. Report the actual `used` / `limit` from the endpoint — don't quote a cap from memory. |
| 3 | Any plain HTTP `/api/challenge/*` call (login, result, log, job, tunnel-endpoint) returns `401` | Invalid / expired `CHALLENGE_TOKEN` | `challenge-login` Step 3 to refresh; if refresh also 401s, fall back to Step 1 (email + password). This only affects the plain HTTP APIs — it does not touch a `JOB_TOKEN`-based tunnel dial. |
| 3b | The WS handshake (`GET /api/challenge/tunnel`) returns `401` | When dialing with `JOB_TOKEN` (the default per `challenge-run-agent`), the handshake does **not** 401 on token expiry — `job_token`s don't expire. A `401` here means the `?job=` UUID is not owned by this account, the job is in a terminal state, **or** the `?job=` doesn't match the job the token was issued for. If dialing with `CHALLENGE_TOKEN` (no `JOB_TOKEN` in the state file), a `401` can additionally mean the login token itself expired | Verify `JOB_UUID` / `?job=` came from this account's `POST /api/challenge/job` and that the job isn't terminal. Only if dialing with `CHALLENGE_TOKEN`, also try `challenge-login` Step 3 to refresh. |
| 4 | Agent connects, then immediately disconnects | The job's `parallelism` cap is reached — that job already has `PARALLELISM` live agent connections | Reduce K in `challenge-run-agent`, or wait for older agents of that job to finish. |
| 5 | Job stuck at `detailed_status: pending` / `inference_disconnected`, or `progress` not advancing for minutes | No agent online, or all agents dropped past the 30 s window | `ps`/check the agent processes; relaunch via `challenge-run-agent` if needed. **`queued` is different** — it means the cluster has no free GPU; the agent is fine and you must keep it online. |
| 6 | Agent disconnected unexpectedly mid-run | Network blip | Within the gateway-side **30 s reconnect window**, `tunnel.sh` redials **automatically** with the same `agent_id` to resume the open session. **Do not manually relaunch** in that window — you'd race the automatic redial and the gateway will reject the duplicate. Past 30 s the in-flight case loses one of its retries and is scheduled again; while the job is active, relaunching the same command reconnects. A new submission is needed only once the job has `failed`, and costs a daily submission slot — confirm before suggesting it. If it keeps happening, or the job failed with `failure_reason` 9, run `challenge-diagnose-connection` instead of assuming a blip. |
| 7 | Received `drain` control frame | Platform finished dispatching cases and is asking for graceful shutdown | Let in-flight sessions finish, close the socket, and **do NOT reconnect**. |
| 8 | Job ends at `detailed_status: failed` (list `status` = `4`) | A case exhausted its retries, which kills the whole job | `GET /api/challenge/job/$JOB_ID/log` (see `challenge-poll-result`). **Read `failure_reason` before any log text**: `5`/`6`/`7`/`8` are platform-side teardowns, not your bug — `8` in particular means a *different* case was the root cause. `9` (tunnel lost) is the opposite — it usually means your agent dropped the WebSocket; check `result.error` for the sub-reason and see rows #5/#6 above. For `1`/`3`/`4`, fetch `result.stdout` (usually a presigned URL, not inline text) and read the traceback. Common causes: handler exception during warmup (row #11), OOM, action-encoding mismatch. **Fix the bug locally before resubmitting** — resubmission costs a daily submission slot. |
| 9 | `/api/challenge/tunnel/endpoint` returns empty string | Gateway address not configured yet | Retry after a few seconds. Do not substitute a hard-coded or guessed address; if it stays empty, report it to the organizers. |
| 10 | `parallelism` in the job response is `0` | User's `MaxConcurrentCases` is misconfigured (not "exhausted" — exhaustion is a runtime gateway check, not a response field) | Stop and surface to organizers; do not launch agents. |
| 11 | Agent stuck in `WARMUP`, never reaches `RUNNING` — **or** the job ends `Failed` with a traceback referencing `frame_bytes=b''` / zero-length decode in the `/log` output | Inference handler errored on the warmup empty frame | The warmup call passes an **empty frame** — handlers must tolerate that (e.g. short-circuit when `len(frame_bytes) == 0` and return a no-op action). Fix in the inference repo, then resubmit (mind quota). |
| 12 | `tunnel.sh` exits with code 1 immediately | Bad arguments / handler import failure / reconnect retries exhausted | Re-check `<gpu_index> <job_uuid> <gateway_url>` argument order; check inference-repo handler import path. |
| 13 | `tunnel.sh` exits with code 130 | Ctrl-C | Expected; user-driven. |
| 14 | `POST /api/challenge/job` returns **HTTP 400** `paper_link is not a valid URL` / `paper_link too long` | Optional `paper_link` field is malformed or > 512 chars | Fix the URL or omit the field entirely; resubmit. The rejected request consumed no slot. |
| 15 | `POST /api/challenge/job` returns **HTTP 400** `model type is not set yet` | The account profile has no model type (`policy` / `agent`), or the profile change that sets it is still awaiting review | Set it via `POST /api/challenge/profile` (`model_type`) and wait for admin review (`current-user-info` shows `model_type` and `pending_change`), then resubmit. Rejected before the quota check — no slot consumed. Ask the user which type their model is; don't pick one on their behalf. |
| 16 | `POST /api/challenge/job/{id}/review` returns **HTTP 400** `invalid score review request: …` | Suffix says why: `job not found` (wrong id or not this account's job), `only v2 jobs can be reviewed`, `only succeeded jobs can be reviewed` (job not `completed` yet, or failed), `job already approved` | Fix the cause; don't retry the same request. Use the numeric `JOB_ID`, not `JOB_UUID`. A `rejected` job *can* be re-submitted. |
| 17 | Season II job is `completed` but the user isn't on `GET /api/challenge/leaderboard/v2` | Not submitted for review, still `pending`, `rejected`, the approval hasn't reached the cached leaderboard yet, the account profile has no model type, or the leaderboard being viewed is for the other model type | Check `review_status` / `review_rejected_reason` on the job list (`challenge-poll-result`). Not submitted → `challenge-submit-job` Step 6 (confirm first). `pending` → wait for the organizers. `rejected` → read the reason, fix, re-submit. `approved` but missing → the leaderboard is cached; wait and re-check before escalating. Also: only the user's **highest-scoring approved** job is listed, and it appears on the board of the type in their profile (`current-user-info` → `model_type`; query that leaderboard with the matching `model_type`) as well as on the combined `model_type=all` board. A user whose profile has no model type is on no leaderboard. |
| 18 | `GET /api/challenge/leaderboard…` (any leaderboard or top5 request) returns **HTTP 400** `invalid model_type: … (want policy, agent or all)` | `model_type` is missing or not `policy` / `agent` / `all` | Add `model_type=policy`, `model_type=agent`, or `model_type=all` to the request. |

## Diagnostics shortlist

Run these in order when triaging an unclear failure:

```bash
# 1. Token still valid?
curl -fsS "$BASE_URL/api/challenge/current-user-info" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq

# 2. Job exists, what status? (read detailed_status from the job LIST; /result only returns scores)
curl -fsS "$BASE_URL/api/challenge/jobs?season=${JOB_SEASON:-2}&page=1&per-page=100" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" \
  | jq -c --argjson id "$JOB_ID" '.items[] | select(.id == $id)
          | {id, status, detailed_status, progress, score}'

# 3. If detailed_status == failed, get the log (read failure_reason first)
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/log" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq

# 4. Scores, once terminal
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/result" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq

# 5. Gateway endpoint sane?
curl -fsS "$BASE_URL/api/challenge/tunnel/endpoint" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN"
```

`$JOB_ID` must be the **numeric** `id`; the UUID belongs only to the tunnel dial. Passing a UUID
to these paths yields `400 invalid job_id`, which `curl -fsS` hides behind a bare exit code 22.

## Things NOT to do

- The website/API domain is `robocoliseum.ai`, but the tunnel gateway is a separate host. Use the `tunnel_endpoint` returned when the job was created (`TUNNEL_ENDPOINT`). **Do not** "fix" the tunnel URL by swapping in the website domain or by hard-coding an address.
- **Do not** reconnect after a `drain` — the platform is finishing up, and the connection will be rejected.
- **Do not** spam `POST /api/challenge/job` to "retry" — a *successful* call consumes one of the day's submission slots, and a failing one just repeats the same rejection. **Remaining quota is not authorization to guess** (e.g. guessing a `board` because "quota remains"). Skill rules forbid the guess itself, independent of remaining slots.
- **Do not** state the daily cap from memory. Read `GET /api/challenge/submission/quota` and quote its `limit` / `used` / `remaining`; the limit is a platform setting.
- **Do not** kill running agents to "free a slot" without checking the job list first; you may be killing the agent that's about to finish your last case.
- **Do not** retry a 4xx as if it were a network blip — see the diagnostic heuristic at the top of this file.
- **Do not** poll `/result` for status — it is the scores endpoint and returns only `{tasks, total}`. A loop that branches on `.status` there never terminates. Read `detailed_status` from `GET /api/challenge/jobs`.

## Reference

For the WS wire protocol (dial, control frames, binary frame layout, state machine, reconnect,
heartbeat), see the **appendix of `challenge-run-agent`** — it is inlined there in full. Most
contestants don't need it; `./scripts/tunnel.sh` from the inference repo already implements it.
For the observation/action payload schema inside the data frames, see
the [inference protocol](../challenge-route-inference/references/protocol.md).
