---
name: challenge-ranking
description: Use to fetch the contestant's best score and the per-board leaderboard for the Simulation Challenge. Read-only; safe to run without confirmation.
---

# challenge-ranking — Best score and leaderboard

All endpoints here are read-only.

> **Board scoping.** Best-score, rank, and leaderboard are all **per-board**: there is no global "total" score, and the same user can have four independent best-score / rank tuples (one per board: `instruction` / `spatial` / `manip` / `robust`). Always ask the user which board they care about, or report all four — never silently average.

## Preconditions

- `CHALLENGE_TOKEN` set for `best-score` (else → `challenge-login`).
- The leaderboard endpoints (`/leaderboard`, `/leaderboard/top5`, `/leaderboard/citation`) are **public** — they work without a token, but include `Authorization` when you have one to stay symmetric with the rest of the skill set.

Every Bash call should start with:

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
```

## My best score and rank (per board)

`best-score` requires a `board` query parameter (it filters the user's per-board best):

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
curl -fsS "$BASE_URL/api/challenge/best-score?board=instruction" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
# { "status": "ok", "score": 573.0, "rank": 5 }
```

Replace `instruction` with `spatial` / `manip` / `robust` as needed. Report `score` and `rank` plainly. If `status` is `"error"` (or `rank` is `0`), the user has no scored submission on that board yet — suggest they run `challenge-submit-job` against it.

To get a quick all-board summary, fan out in parallel:

```bash
for b in instruction spatial manip robust; do
  printf '%-12s ' "$b"
  curl -fsS "$BASE_URL/api/challenge/best-score?board=$b" \
    -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq -c '{score, rank}'
done
```

## Per-board leaderboard

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard?board=instruction&page=1&per-page=20" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

> **Query param is `per-page` with a hyphen** (`per_page` is silently ignored and you get the
> default 20). The response envelope echoes it back as `per_page` — same asymmetry as
> `/api/challenge/jobs`.

Each row is one user's best job for that board. Optional query params: `sort` (default `score`), `order` (`asc` / `desc`, default `desc`), `q` (substring search over user / organization name).

There is **no `rank_change` field** — only the current rank/score. Do not invent deltas.

## Top-5 hero (all boards at once)

For a homepage-style overview without four separate calls:

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/top5" | jq
# { "boards": { "instruction": [...top 5], "spatial": [...], "manip": [...], "robust": [...] } }
```

Public, no auth needed.

## Per-board detail (one user's per-task breakdown)

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/instruction/detail?user_uid=<uid>" | jq
```

The `{board}` path segment must be one of `instruction` / `spatial` / `manip` / `robust`. `user_uid` is the target user's UID (find it via the leaderboard row).

## Citation (BibTeX)

```bash
curl -fsS "$BASE_URL/api/challenge/leaderboard/citation" | jq -r '.citation // .'
```

Returns the platform-configured BibTeX string verbatim — surface as-is.

## Pagination

If the user wants their own row and they're past page 1, walk pages until you hit their `name`. Keep `per-page` ≤ 100 to stay polite.

## Hand-off

- User wants to improve their rank on a specific board → `challenge-submit-job` (mind the daily submission quota — check `/api/challenge/submission/quota`; each board needs its own submission).
- User wants their score for a specific recent submission → `challenge-poll-result`.
