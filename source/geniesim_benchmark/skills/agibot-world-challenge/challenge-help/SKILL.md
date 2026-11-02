---
name: challenge-help
description: Entry point for the Simulation Challenge skill set. Use when the user mentions the challenge, leaderboard, submitting a model, or any of the /api/challenge/* endpoints — this skill picks the right downstream skill for them.
---

# challenge-help — Pick the right skill

You are an AI assistant helping a contestant operate the **Simulation Challenge** platform end to end. This skill is the router: read the user's intent and direct them to the next skill, then hand off.

## Stage map

```
   ── prep (before any job) ──────────────────────────────────────
   ┌──────────────────────────┐     ┌──────────────────────────┐
   │ challenge-download-      │     │ challenge-baseline-       │
   │     datasets             │ ──▶ │     model                 │
   │ LeRobot v2.1 training    │     │ clone code + ModelScope   │
   │ data from ModelScope     │     │ ckpts → install → run     │
   └──────────────────────────┘     └────────────┬─────────────┘
   ───────────────────────────────────────────────┼──────────────
                                                   ▼
        ┌─────────────────┐
        │ challenge-login │  get/refresh CHALLENGE_TOKEN
        └────────┬────────┘
                 ▼
       ┌────────────────────┐
       │ challenge-submit-  │  POST /api/challenge/job  →  JOB_ID (numeric), JOB_UUID,
       │       job          │  PARALLELISM, TUNNEL_ENDPOINT, JOB_TOKEN
       └────────┬───────────┘  (consumes a daily submission slot!)
                ▼
        ┌────────────────────┐
        │ challenge-run-     │  ./scripts/tunnel.sh <gpu> <job_uuid> <endpoint>
        │     agent          │  scale to PARALLELISM processes  (uses JOB_UUID)
        └────────┬───────────┘
                 ▼
       ┌────────────────────┐         ┌──────────────────────┐
       │ challenge-poll-    │ ──────▶ │ challenge-           │
       │     result         │         │     troubleshoot     │  ← if failed / stuck
       └────────┬───────────┘         └──────────────────────┘
                ▼   (uses JOB_ID)
        ┌────────────────────┐
        │ challenge-ranking  │  best-score + leaderboard
        └────────────────────┘
```

## Routing rules

| User says / intends | Use this skill |
|---------------------|----------------|
| "下载训练数据/数据集", "download training data", "get the lerobot v2.1 data", "download task suite", mentions `download_dataset.sh` / GenieSim3.0-Dataset | `challenge-download-datasets` |
| "拉取/搭一个 baseline", "下载推理代码/权重", "clone the inference repo", "download ckpts", "set up/run the baseline model", "部署/跑个 demo" | `challenge-baseline-model` |
| "log in", "I have credentials", "token expired" | `challenge-login` |
| "submit my model", "create a job", "what's my quota" | `challenge-submit-job` |
| "start the agent", "run my SDK", "use that GPU", "launch inference" | `challenge-run-agent` |
| "is my job done", "check the score", "what's the status of $JOB_ID" | `challenge-poll-result` |
| "where am I on the leaderboard", "best score" | `challenge-ranking` |
| "agent disconnected", "401", "stuck in Pending", "Failed why" | `challenge-troubleshoot` |
| "keeps disconnecting", "tunnel lost", "reconnect window expired", "close 1006 / 1011", "keepalive ping timeout", `failure_reason` 9 | `challenge-diagnose-connection` |
| End-to-end "submit and run" — `challenge-submit-job` → `challenge-run-agent` → `challenge-poll-result` | daisy-chain in order |

## Hard rules

1. **Always confirm before** `challenge-submit-job` (each submission consumes one daily submission slot) and before launching agent processes (they hold a `parallelism` slot).
2. **Read-only calls run without confirmation**: login, current-user-info, jobs/result/log, best-score, ranking, quota check.
3. **The gateway host is fixed at `120.92.88.78` and does not change.** Defaults: `BASE_URL=https://robocoliseum.ai` (HTTP API base for all `curl "$BASE_URL/api/challenge/..."` calls) and `TUNNEL_ENDPOINT=ws://120.92.88.78/api/challenge/tunnel` (WebSocket tunnel for `tunnel.sh`/run-agent). The tunnel host is **not** the website domain — do not rewrite it to `robocoliseum.ai`. If the job response carries a `tunnel_endpoint`, prefer it; otherwise fall back to the default above — do not block on it being absent.
4. **`board` has a closed allowed-value set today.** `config.board` accepts exactly one of `instruction` / `spatial` / `manip` / `robust`. Any other value is rejected with a 400 before anything is written, so it costs no quota — but it also creates nothing. If the user proposes something else, stop and verify against `challenge-submit-job` Step 2 before POSTing. Do not invent.
5. If `CHALLENGE_TOKEN` is missing or returns `401`, jump to `challenge-login` first.
6. **Auto-pilot does not skip confirmation gates.** If the user says "全程做完" / "do it end to end", you may merge the gates into a single up-front confirmation ("I'll submit job + launch K agents + poll until terminal — OK?"), but you may not run them silently. The daily submission quota and the GPU-holding agents are user-visible costs that must be approved.
7. **4xx is a semantic rejection, never a network blip.** Do not retry a 400/401/403/404 just because the user says "可能是网络抖动" — only connection resets / 5xx / timeouts qualify as transient. Refer to `challenge-troubleshoot` for the exact response.
8. **Status and scores come from two different endpoints — ask each the right question.** "Is it done / what state is it in?" → `GET /api/challenge/jobs` (find the job by `id`, read `detailed_status`, `progress`, `score`). "What are the per-task scores?" → `GET /api/challenge/job/{id}/result`, which returns `{tasks, total}` only. That is by design, not a gap: `/result` has no `status` field, so a poll loop that branches on it never terminates. When explaining this to the user, say "status is on the job list", not "the API doesn't give status". See `challenge-poll-result`.
9. **`model_name` is not a request field.** It is the account's profile model name, and `POST /api/challenge/job` is rejected with `400 model name is not approved yet` until an admin has approved it. See `challenge-submit-job` Step 1b.
10. **There is no official Simulation SDK package.** The supported client is the inference repo's `./scripts/tunnel.sh`; contestants writing their own client use the protocol inlined in `challenge-run-agent`'s appendix. Don't invent a `pip install` or point at platform-internal test clients.
11. **Never quote the daily submission cap from memory.** It's a platform setting, not a fixed contest constant, and it has changed. Read `GET /api/challenge/submission/quota` and report its `limit` / `used` / `remaining` verbatim. If the probe fails, say the quota is unknown rather than guessing.

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
| `PARALLELISM` | `challenge-submit-job` (writes state file) | `challenge-run-agent` |
| `TUNNEL_ENDPOINT` | `challenge-submit-job` (job response), else fixed default `ws://120.92.88.78/api/challenge/tunnel` | `challenge-run-agent` |

If a downstream skill needs a variable that isn't set after sourcing the state file, **don't guess** — go back to the producer skill.
