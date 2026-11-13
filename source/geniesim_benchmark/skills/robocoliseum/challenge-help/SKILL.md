---
name: challenge-help
description: Entry point for the Simulation Challenge skill set. Use when the user mentions the challenge, leaderboard, submitting a model, or any of the /api/challenge/* endpoints — this skill picks the right downstream skill for them.
---

# challenge-help — Pick the right skill

You are an AI assistant helping a contestant operate the **Simulation Challenge** platform end to end. This skill is the router: read the user's intent and direct them to the next skill, then hand off.

## Workflow map

```text
Optional local simulation: simulator code -> challenge-download-assets
Optional model setup:      challenge-download-datasets -> challenge-baseline-model

Model-loading agent: challenge-login -> challenge-submit-job -> challenge-run-agent
Existing service:    challenge-route-inference probe -> challenge-submit-job (or resume) -> start routes

Both evaluation paths: challenge-poll-result -> challenge-ranking
Problems:              challenge-troubleshoot -> challenge-diagnose-connection
Protocol questions:    challenge-route-inference/references/protocol.md
```

## Routing rules

| User request | Skill or reference |
| --- | --- |
| Reuse local/remote inference, share one model across tunnel routes, connect or resume jobs, allocate channels, inspect or stop routes | [challenge-route-inference](../challenge-route-inference/SKILL.md) |
| Simulator inference protocol, observation/action fields, embodiment tags, MessagePack, EEF frames, depth/calibration, history, chassis, episode termination | [challenge-route-inference: protocol reference](../challenge-route-inference/references/protocol.md) |
| HTTP/RPC/native OpenPI conversion, greetings, session state, custom service adapters | [challenge-route-inference: adapter reference](../challenge-route-inference/references/adapters.md) |
| Download scenes, robots, or simulation runtime assets; inspect asset size; resume or verify asset downloads | [challenge-download-assets](../challenge-download-assets/SKILL.md) |
| Download training data, LeRobot v2.1 datasets, task suites, or use `download_dataset.sh` | [challenge-download-datasets](../challenge-download-datasets/SKILL.md) |
| Clone inference code, download checkpoints, provision/run a baseline, deploy a model demo | [challenge-baseline-model](../challenge-baseline-model/SKILL.md) |
| Log in, inspect the current account, obtain or refresh a token | [challenge-login](../challenge-login/SKILL.md) |
| Submit a model evaluation, create a job, check submission quota | [challenge-submit-job](../challenge-submit-job/SKILL.md) |
| Submit a completed Season II result for review | [challenge-submit-job, Step 6](../challenge-submit-job/SKILL.md#step-6--season-ii-only-submit-the-finished-job-for-review) |
| Launch a model-loading tunnel agent from an inference repository | [challenge-run-agent](../challenge-run-agent/SKILL.md) |
| Track job progress, fetch scores, inspect evaluation logs | [challenge-poll-result](../challenge-poll-result/SKILL.md) |
| Check leaderboard position, best score, or top models | [challenge-ranking](../challenge-ranking/SKILL.md) |
| Diagnose submission errors, authentication failures, failed/stuck jobs | [challenge-troubleshoot](../challenge-troubleshoot/SKILL.md) |
| Diagnose repeated tunnel drops, reconnect expiry, close 1006/1011, keepalive timeout, or `failure_reason: 9` | [challenge-diagnose-connection](../challenge-diagnose-connection/SKILL.md) |

For a working inference service, prefer `challenge-route-inference`: routes forward
payloads without loading another model. GPU-per-process rules for model-loading
`tunnel.sh` agents do not imply one GPU per relay route. This path keeps private
state per job, preserving existing jobs' identities and credentials.

For baseline setup followed by evaluation, use `challenge-baseline-model`, then
`challenge-submit-job`, `challenge-run-agent`, and `challenge-poll-result` as needed.
Runtime assets and training datasets are separate downloads. Asset preparation
requires the simulator code and CLI, but no challenge login or official job.

## Hard rules

1. **Check authorization before external mutations.** Job creation consumes one daily submission slot; agents/routes hold the job's concurrency slots. Explain the concrete plan and obtain missing authorization before submitting, launching, or submitting a result for review. Honor explicit authorization already given for that plan.
2. **Read-only calls run without confirmation**: login, current-user-info, jobs/result/log, best-score, ranking, quota check.
3. **The tunnel endpoint comes from the job response.** `POST /api/challenge/job` returns `tunnel_endpoint`; use that value for every agent or route of that job. It is a different host from the HTTP API, so never derive it from the website domain and never hard-code an address. If the field is empty, retry `GET /api/challenge/tunnel/endpoint` after a few seconds (empty means the gateway is not configured yet); if it stays empty, stop and report it to the organizers. The existing-service relay workflow follows [challenge-route-inference](../challenge-route-inference/SKILL.md).
4. **`board` has a closed allowed-value set.** `config.board` accepts exactly one of `instruction` / `spatial` / `manip` / `robust` (Season I) or `v2` (Season II). Any other value is rejected with a 400 before anything is written, so it costs no quota — but it also creates nothing. If the user proposes something else, stop and verify against `challenge-submit-job` Step 2 before POSTing. Do not invent.
5. If `CHALLENGE_TOKEN` is missing or returns `401`, jump to `challenge-login` first.
6. **Keep the workflow within the requested scope.** A request to inspect or plan does not authorize a new job, and a request to resume does not authorize a replacement submission. Explicit instructions to create and launch with a concrete allocation remain valid across the workflow; ask only for missing decisions or authorization.
7. **4xx is a semantic rejection, never a network blip.** Do not retry a 400/401/403/404 just because the user says "maybe a network blip" — only connection resets / 5xx / timeouts qualify as transient. Refer to `challenge-troubleshoot` for the exact response.
8. **Status and scores come from two different endpoints — ask each the right question.** "Is it done / what state is it in?" → `GET /api/challenge/jobs` (find the job by `id`, read `detailed_status`, `progress`, `score`). "What are the per-task scores?" → `GET /api/challenge/job/{id}/result`, which returns `{tasks, total}` only. That is by design, not a gap: `/result` has no `status` field, so a poll loop that branches on it never terminates. When explaining this to the user, say "status is on the job list", not "the API doesn't give status". See `challenge-poll-result`.
9. **`model_name` and `model_type` are account profile fields, not request fields.** `POST /api/challenge/job` is rejected with `400 model name is not approved yet` until an admin has approved the name, and with `400 model type is not set yet` until the model type (`policy` / `agent`) has been set and approved. See `challenge-submit-job` Step 1b. Leaderboards can be viewed per model type or combined (`model_type=policy` / `agent` / `all`).
10. **There is no official Simulation SDK package.** Model-loading agents use the inference repository's `./scripts/tunnel.sh`; existing services can use the helpers bundled with `challenge-route-inference`. Custom clients use the transport appendix in `challenge-run-agent` and the application protocol bundled with the inference skill. Do not invent a package or point at platform-internal test clients.
11. **Never quote the daily submission cap from memory.** It's a platform setting, not a fixed contest constant. Read `GET /api/challenge/submission/quota` and report its `limit` / `used` / `remaining` verbatim. If the probe fails, say the quota is unknown rather than guessing.
12. **Only approved Season II (`v2`) results are listed** — after the job completes, the contestant submits it for review (`challenge-submit-job` Step 6).

## Environment contract

These environment variables are the shared state between skills. Treat them as the single source of truth — never re-derive.

**They are persisted to `~/.simubotix-challenge.env` (mode 0600).** Each Bash call should start with `[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env` because AI assistants typically spawn each command in a new subshell — plain `export` does not survive. See README "State file" for the helper and rationale.

| Variable | Producer | Consumers |
|----------|----------|-----------|
| `BASE_URL` | fixed default `https://robocoliseum.ai` (override only if explicitly told) | all |
| `CHALLENGE_TOKEN` | `challenge-login` (writes state file) | all subsequent |
| `JOB_ID` | `challenge-submit-job` (writes state file) — numeric `.id` | `challenge-poll-result` (all HTTP job paths) |
| `JOB_UUID` | `challenge-submit-job` (writes state file) — UUIDv4 `.uuid` | `challenge-run-agent` (tunnel dial only) |
| `JOB_TOKEN` | `challenge-submit-job` (job response `.job_token`) | `challenge-run-agent` (tunnel dial credential) |
| `PARALLELISM` | `challenge-submit-job` (writes state file) — live tunnel connections allowed for this job | `challenge-run-agent`, `challenge-route-inference` |
| `JOB_SEASON` | `challenge-submit-job` (writes state file) — `1` for Season I boards, `2` for `v2` | `challenge-poll-result`, `challenge-route-inference` (the job list is per season) |
| `TUNNEL_ENDPOINT` | `challenge-submit-job` (job response `.tunnel_endpoint`); if empty, `GET /api/challenge/tunnel/endpoint` | `challenge-run-agent`, `challenge-route-inference` |

The route workflow reads a per-job copy of this state file (copied right after each submission) and can use a separate login `auth_file`; see
[challenge-route-inference](../challenge-route-inference/SKILL.md). Do not overwrite a live job's state to connect another job.

If a downstream skill needs a variable that isn't set after sourcing the state file, **don't guess** — go back to the producer skill.
