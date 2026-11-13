---
name: challenge-ranking
description: Use to fetch the contestant's best score and the leaderboards (Season I per-board leaderboards, the Season II leaderboard, the top-5 overview) for the Simulation Challenge. Read-only; safe to run without confirmation.
---

# challenge-ranking — Best score and leaderboard

All endpoints here are read-only. There are two tracks:

- **Season I** — four independent boards `instruction` / `spatial` / `manip` / `robust`. Scores and ranks are **per board**; never silently average them.
- **Season II** — board `v2`, one leaderboard per model type. Each row carries an overall score plus four sub-board scores (`finemanip` / `crsemb` / `wbc` / `dexmanip`), all from the same job. Only results approved by the organizers are listed.

Every leaderboard request carries `model_type=policy`, `model_type=agent`, or `model_type=all` (case-insensitive); a missing or invalid value returns `400 invalid model_type: <v> (want policy, agent or all)`. `policy` and `agent` rank separately, each starting at rank 1. `all` is a combined board: every user who has a model type in their profile is ranked together, regardless of type. A user appears on the one type board matching the type in their account profile, and on the `all` board whenever their profile has a model type; a user whose profile has no model type is on no board.

If the user doesn't say which track or which model type, ask or report both.

## Preconditions

- `CHALLENGE_TOKEN` set for `best-score` (else → `challenge-login`).
- The leaderboard endpoints are **public** — they work without a token, but include `Authorization` when you have one to stay symmetric with the rest of the skill set.
- Leaderboards are cached and refreshed periodically, so a just-finished (or just-approved) result may take a while to appear on a leaderboard, on the `rank` returned by `best-score`, and on `/leaderboard/overall/me`. The `score` returned by `best-score` and the job list are read live.

Every Bash call should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

## My best score and rank (Season I, per board)

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
curl -fsS "$BASE_URL/api/challenge/best-score?board=instruction" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
# { "status": "ok", "score": 573.0, "rank": 5 }
```

Replace `instruction` with `spatial` / `manip` / `robust` as needed. `rank` is the caller's position on that board among users whose profile has the same model type — the number the leaderboard shows for that user. A user whose profile has no model type is on no board and therefore has no rank, which returns the same `{"status":"error","score":0,"rank":0}` as a board with no submission. If `status` is `"error"` (or `rank` is `0`), the user has no scored submission on that board for their model type yet — suggest `challenge-submit-job`.

All four at once:

```bash
for b in instruction spatial manip robust; do
  printf '%-12s ' "$b"
  curl -fsS "$BASE_URL/api/challenge/best-score?board=$b" \
    -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq -c '{score, rank}'
done
```

Season II has no `best-score` endpoint — find the user's row on the Season II leaderboard instead.

## Season I leaderboard (per board)

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard?board=instruction&model_type=policy&page=1&per-page=20" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

> **Query param is `per-page` with a hyphen** (`per_page` is silently ignored and you get the
> default 20). The response envelope echoes it back as `per_page` — same asymmetry as
> `/api/challenge/jobs`.

Each row is one user's best job for that board: `rank`, `user_uid`, `username`, `organization`, `model_name`, `logo_url`, `model_type`, `score`, `submit_time`, `paper_link`. `model_type` is the type in that user's account profile; on a `policy` or `agent` board it always equals the `model_type` you queried, on the `all` board it varies row by row. `rank` is always the position within the board you queried. Optional params: `sort` (default `score`), `order` (`asc` / `desc`, default `desc`), `q` (case-insensitive substring search over `username`).

Per-task breakdown for one user (`{board}` is one of the four Season I boards; takes no `model_type`; `rank` is the position within that user's own profile type and `all_rank` within the combined board, both `0` when they aren't listed):

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/instruction/detail?user_uid=<uid>" | jq
```

## Season II leaderboard

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/v2?model_type=policy&page=1&per-page=20" | jq
```

Same envelope. Each row:

```json
{
  "rank": 1, "user_uid": "…", "username": "…", "organization": "…",
  "model_name": "…", "logo_url": "…", "model_type": "policy",
  "score": 62.5,
  "paper_link": "",
  "boards": [
    { "board": "finemanip", "score": 70.0 },
    { "board": "crsemb",    "score": 55.0 },
    { "board": "wbc",       "score": 60.0 },
    { "board": "dexmanip",  "score": 65.0 }
  ],
  "submit_time": "…"
}
```

- One row per user: their highest-scoring approved Season II job. `score` is the overall score. `paper_link` is the paper link the organizers attached at approval; it may be empty. `rank` is the position within the board you queried — with `model_type=all` it is the position among all users who have a model type.
- Params: `q`, `order`, and `sort` ∈ `score` (default) / `time` / `finemanip` / `crsemb` / `wbc` / `dexmanip` (anything else → 400). Sorting by a sub-board only reorders rows; `rank` stays the overall rank.
- A user missing from the board usually means no approved Season II job yet — check `review_status` on their job list (`challenge-poll-result`).

Per-task breakdown for one user, grouped by sub-board (takes no `model_type`; the breakdown is for the job that user has on the board for their own profile type):

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/v2/detail?user_uid=<uid>" | jq
# { "user_uid", "rank", "all_rank", "job_uuid", "model_type", "score", "paper_link",
#   "boards": [ { "board", "score", "tasks": [ { "name", "success_rate", "score" } ] } ] }
```

`rank` is the user's overall position on the Season II leaderboard of their model type (same as on the leaderboard row; `0` when not listed). `all_rank` is that position on the combined `all` board (always `>= rank`; `0` when not listed). `paper_link` is the same approval-time link as on the leaderboard row (may be empty). `"boards": []` means the user isn't on the Season II leaderboard (not an error).

## Top-5 overview

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/top5?season=2&model_type=policy" | jq
# { "boards": [ { "board": "finemanip", "items": [...] }, …, { "board": "v2", "items": [...] } ] }
```

`season` is `2` (Season II, the default when omitted) or `1` (Season I); digits only, no other value is accepted (anything else → `400 invalid season: <v> (want 1 or 2)`). `boards` is an array. `season=1` returns the four Season I boards (`instruction` / `spatial` / `manip` / `robust`). `season=2` returns the four sub-boards (`finemanip` / `crsemb` / `wbc` / `dexmanip`, each ranked by that sub-board's score, with the sub-board score as `score`) followed by `v2` (the overall score). Select with `jq '.boards[] | select(.board == "v2")'` rather than by position. Each item: `rank`, `user_uid`, `username`, `model_name`, `logo_url`, `model_type`, `score`.

## Citation (BibTeX)

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/citation" | jq -r '.citation // .'
```

Returns the platform-configured BibTeX string verbatim — surface as-is.

## Notes

- `GET /api/challenge/leaderboard/overall?model_type=…` rows carry the user's profile `model_type` at the top level (alongside `model_name` and `logo_url`); on the `all` board it varies row by row.
- `GET /api/challenge/leaderboard/overall/me` takes no `model_type`; it returns the caller's rank among users of the type in the caller's own profile (`entry.rank`), plus `all_rank` — the same caller's rank on the combined `all` board, always `>= entry.rank`. If the profile has no model type, `qualified` is `false` and both ranks are absent.
- There is **no `rank_change` field** — only the current rank/score. Do not invent deltas.
- To find the user's own row past page 1, walk pages until you hit their `username`. Keep `per-page` ≤ 100.

## Hand-off

- Improve a Season I rank → `challenge-submit-job` with that board (one submission per board; mind the daily quota).
- Get onto the Season II leaderboard → `challenge-submit-job` with `board: "v2"`, then submit the finished job for review (Step 6).
- Score or review status of a specific submission → `challenge-poll-result`.
