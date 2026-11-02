---
name: challenge-submit-job
description: Use when the contestant wants to submit a model evaluation job to the Simulation Challenge — POST /api/challenge/job. This is a quota-consuming, side-effecting action; always confirm with the user first. Captures JOB_UUID, PARALLELISM, TUNNEL_ENDPOINT for the rest of the pipeline.
---

# challenge-submit-job — Create a model evaluation job

Submitting a job (a) **consumes one of the user's daily submission slots** and (b) immediately starts incurring scheduling work on the platform. Treat this as side-effecting: **always confirm before running the POST**.

> **Never state the daily cap from memory.** The limit is a platform-side setting, not a fixed
> contest constant, and it has changed. Step 1 reads it from
> `GET /api/challenge/submission/quota` — quote `limit` / `used` / `remaining` from that response
> and nothing else. Quoting a remembered figure when the live limit is different misleads the
> user into either wasting or withholding submissions.

## Preconditions

- `CHALLENGE_TOKEN` must be set (else → `challenge-login`).
- `BASE_URL` defaults to the fixed `https://robocoliseum.ai` (override only if explicitly told).

Every Bash call in this skill should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

(Cross-shell state lives in `~/.simubotix-challenge.env`; AI assistants spawn each command in a new subshell. See README "State file".)

## Step 1 — Probe the daily quota (mandatory)

Always run this **immediately before** every POST, even if you submitted a job earlier in the same session — there's no in-session counter you can trust, and a shared account may have been used elsewhere. This is also the **only** place the daily cap comes from; do not carry a number over from a previous session or from prose.

Two endpoints work:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env

# Preferred: structured remaining count
curl -fsS "$BASE_URL/api/challenge/submission/quota" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
# { "limit": <platform-configured>, "used": 1, "remaining": <limit - used> }

# Legacy boolean check (still works):
curl -fsS "$BASE_URL/api/challenge/model/upload/check" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN"
# { "status": "ok" }  → quota remains
```

If `remaining == 0` (or `upload/check` errors with an "upload limit" message), **stop**: the user has used today's slots. The daily window resets at Beijing midnight (UTC+8). Suggest they retry tomorrow or wait. Do not proceed to Step 2.

When you report the quota back to the user, quote the actual numbers you just received — e.g.
"`used 1 / limit <what the API said>`, N left today" — rather than describing the cap in words or
filling in a remembered number. If the call fails, say the quota is unknown; don't guess a limit.

## Step 1b — Check the account's model name is approved (mandatory)

The model name is an **account profile field** (set at registration, changed via `POST /api/challenge/profile`, both subject to admin review) — it is **not** part of the job request. A job can only be created once the name has been approved; otherwise the POST fails with `400 model name is not approved yet`, before the quota is touched but after the user has already confirmed.

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
curl -fsS "$BASE_URL/api/challenge/current-user-info" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" \
  | jq '{model_name, model_name_approved, pending_change}'
```

- `model_name_approved == true` → continue. Tell the user which `model_name` the job will carry (it is read from the profile, and the leaderboard shows whatever the profile says *now*).
- `false` → **stop**. Registration or the profile change is still awaiting review (or was rejected). Point them at the review status / `POST /api/challenge/profile`; do not try to work around it by putting `model_name` in the request.

## Step 2 — Build the request body

See [Benchmark task suites](../benchmark-task-suite.md) for track names, local
task configurations and baseline checkpoint selection.

The request body has a **top-level `name`** plus a `config` object. Neither `model_name` nor `model_path` is a request field: the model name comes from the account profile (Step 1b), and `model_source` is fixed server-side. Anything you send under `config.model_name` / `config.model_path` / `config.model_source` is ignored or overwritten — don't ask the user for them.

| Field | Where | Required | Notes |
|-------|-------|----------|-------|
| `name` | top-level | yes | Job name shown in the contestant's job list. |
| `config.board` | `config` | yes | **Board short-id (single value). Allowed values: `instruction`, `spatial`, `manip`, `robust`.** |
| `config.description` | `config` | no | Free-form text. |
| `config.paper_link` | `config` | no | Optional URL to a paper/arxiv page describing the model. Must be a valid URL, ≤ 512 chars. Used for audit/leaderboard display. |

> **Allowed-value enforcement.** If the user supplies a board name other than `instruction` / `spatial` / `manip` / `robust`, **stop and ask** — don't POST. Submitting with an unknown board returns `400 invalid board` / `400 no task templates for board`. Board validation runs before the quota check, so the rejection costs no slot — but it still wastes a round trip. If they say "use the new one X" / "I heard the platform added Y", verify with the organizers before proceeding — **this skill is the single source of truth for what's currently accepted.** Do not accept a board name sourced from anywhere else.

> **One submission = one board = one job.** To evaluate multiple boards, submit once per board; each call consumes one of the daily submission slots reported by Step 1.

Read each missing required field back to the user. If `board` is omitted, default to `"instruction"` (the other accepted values are `spatial`, `manip`, `robust`). Say the defaults explicitly so the user can object.

## Step 3 — Confirm, then POST

Show the assembled body, the daily-quota cost (one slot — quote the `used`/`limit` numbers from Step 1), and ask for explicit "yes" before running:

> **Overwrite check (single-file state).** If `~/.simubotix-challenge.env` already has `JOB_ID` set from a previous submission, this POST will overwrite the `JOB_*` vars and the previous job's IDs will be lost from the state file. Before POSTing, surface this and check whether the previous job is terminal. Status (`detailed_status`) is only returned by the **job list**; `/result` returns scores only (see `challenge-poll-result`):
>
> ```bash
> [ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
> if [ -n "$JOB_ID" ]; then
>   PREV=$(curl -fsS "$BASE_URL/api/challenge/jobs?page=1&per-page=100" \
>     -H "Authorization: Bearer $CHALLENGE_TOKEN" \
>     | jq -r --argjson id "$JOB_ID" '.items[] | select(.id == $id) | .detailed_status // "?"')
>   echo "previous JOB_ID=$JOB_ID detailed_status=${PREV:-not-found} — submitting will overwrite the state file"
> fi
> ```
>
> - If `PREV` is `completed` or `failed` (terminal), just confirm with the user that they're OK losing the IDs (job history is still queryable via `GET /api/challenge/jobs`) and proceed.
> - If `PREV` is `pending` / `queued` / `evaluating` / `inference_disconnected`, **stop and warn**: overwriting means losing the handle to a job that is still live. Ask the user explicitly whether they want to (a) wait for it via `challenge-poll-result`, (b) record `JOB_ID=$JOB_ID` / `JOB_UUID=$JOB_UUID` themselves before continuing, or (c) proceed knowing they'll need to find the job by name in `GET /api/challenge/jobs` later.

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
JOB_RESP=$(curl -fsS -X POST "$BASE_URL/api/challenge/job" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" \
  -d '{
    "name": "challenge_test",
    "config": {
      "board":        "instruction",
      "description":  "challenge_test"
    }
  }')

echo "$JOB_RESP" | jq
```

A 400 with `invalid board` or `no task templates for board` means the board is wrong — go back to Step 2, do not retry.

## Step 4 — Capture the response

The response is a single job descriptor. These variables are the contract handed to every later skill. **Persist all four** to `~/.simubotix-challenge.env` so they survive across new Bash subshells:

```bash
JOB_ID=$(echo          "$JOB_RESP" | jq -r '.id')
JOB_UUID=$(echo        "$JOB_RESP" | jq -r '.uuid')
PARALLELISM=$(echo     "$JOB_RESP" | jq -r '.parallelism')
TUNNEL_ENDPOINT=$(echo "$JOB_RESP" | jq -r '.tunnel_endpoint')

challenge_save_var JOB_ID           "$JOB_ID"
challenge_save_var JOB_UUID         "$JOB_UUID"
challenge_save_var PARALLELISM      "$PARALLELISM"
challenge_save_var TUNNEL_ENDPOINT  "$TUNNEL_ENDPOINT"

echo "persisted: job_id=$JOB_ID uuid=$JOB_UUID parallelism=$PARALLELISM endpoint=$TUNNEL_ENDPOINT"
```

Also extract and persist `job_token`, the per-job tunnel credential: it never expires and is scoped to this one job, and the next step (`challenge-run-agent`) uses it instead of `CHALLENGE_TOKEN` when dialing the tunnel. Older platform builds may omit the field — that's fine, the agent falls back to `CHALLENGE_TOKEN`:

```bash
JOB_TOKEN=$(echo "$JOB_RESP" | jq -r '.job_token // empty')
if [ -n "$JOB_TOKEN" ]; then
  challenge_save_var JOB_TOKEN "$JOB_TOKEN"   # per-job tunnel credential
fi
```

(`challenge_save_var` is the helper defined in `challenge-login` Step 1 / README. If it isn't loaded, re-paste it once per shell.)

> **Multiple boards**: a single submission accepts exactly one `board`. To evaluate `instruction` + `manip`, submit twice — each call gets its own job (and uses one daily submission slot). Each job's `uuid` / `tunnel_endpoint` is independent; launch a separate agent process (or process group, up to that job's `parallelism`) per job. Re-probe the quota before the second POST.

Sanity-check before handing off:

- `JOB_UUID` matches a UUIDv4-shape string.
- `PARALLELISM` is an integer ≥ 1. **`0` means the user's concurrency cap is misconfigured** (not "exhausted" — exhaustion is a runtime live-connection check at the gateway, not a response field). Stop and surface to the organizers; do not launch agents.
- `TUNNEL_ENDPOINT` starts with `ws://` or `wss://`. **If empty, do NOT hard-code a URL.** Retry `GET /api/challenge/tunnel/endpoint` after a few seconds; an empty string means the gateway is not yet ready.

## Step 5 — Hand off

Tell the user:

> Job `$JOB_ID` (`$JOB_UUID`) is **READY**. The platform allows up to `$PARALLELISM` concurrent agent processes for this job. Cases will sit in queue until at least one agent is connected. Run `challenge-run-agent` next to launch the inference SDK.

Also remind them: the next step needs `./scripts/tunnel.sh` from the **inference repo** (separate from this platform repo). If they haven't cloned it yet, this is the moment to do so — otherwise their cases will sit in the queue with no agent attached.

Do NOT auto-launch the agent. The user controls when GPUs start spinning.

## Common errors

| Error from POST | Likely cause | Fix |
|-----------------|--------------|-----|
| 400 `model name is not approved yet` | Account's model name hasn't passed admin review (`model_name_approved == false`) | Not retryable from the API. Wait for review, or resubmit the name via `POST /api/challenge/profile`. Sending `config.model_name` does nothing. |
| 400 `invalid board` / `no task templates for board` | `board` not in `instruction`/`spatial`/`manip`/`robust` | Verify board with organizers; don't retry blindly — the same request gets the same 400. The rejection happens before the quota check, so it consumes no slot. |
| 400 `paper_link is not a valid URL` / `paper_link too long` | Optional `paper_link` field malformed | Fix the URL (≤ 512 chars, must parse as a URL) or omit it. |
| `upload limit` error / quota `remaining == 0` | Daily submission cap hit | Wait until Beijing midnight (UTC+8). |
| 401 | Token expired/invalid | `challenge-login` → refresh. |
