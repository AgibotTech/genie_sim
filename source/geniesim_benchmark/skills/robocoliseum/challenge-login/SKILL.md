---
name: challenge-login
description: Use when the contestant needs to obtain or refresh their Simulation Challenge JWT (CHALLENGE_TOKEN), or wants to inspect the current logged-in user. Trigger words include "log in", "token", "401", "current user", "name", "refresh".
---

# challenge-login — Acquire and refresh the JWT

The challenge API is gated by a JWT delivered as `Authorization: Bearer <token>`. Almost everything else fails with `401` until this is set.

> Note: the challenge API uses **email + password** auth, NOT the Casdoor OAuth path used elsewhere on the platform.

## Inputs

| Variable | Required | Notes |
|----------|----------|-------|
| `BASE_URL` | no | fixed default `https://robocoliseum.ai` (override only if explicitly told) |
| `EMAIL` | for first login | contestant email |
| `PASSWORD` | for first login | contestant password |
| `CHALLENGE_TOKEN` | for refresh / current-user | existing JWT |

If any are missing, ask before proceeding. **Never** print the password back to the user.

## Step 1 — Login (read-only on the platform; safe to run)

When echoing this command back to the user, **redact the password** (show `[REDACTED]` or just its length). Never include the literal `$PASSWORD` value in your transcript or log output.

```bash
# Helper (define once per shell; or inline at the top of each Bash call)
challenge_save_var() {
  local f=~/.simubotix-challenge.env key="$1" val="$2"
  touch "$f" && chmod 600 "$f"
  if [ -s "$f" ]; then
    awk -v k="$key" '$0 !~ "^" k "="' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  fi
  printf '%s=%q\n' "$key" "$val" >> "$f"
  chmod 600 "$f"
  export "$key=$val"
}

LOGIN_RESP=$(curl -fsS -X POST "$BASE_URL/api/challenge/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"$PASSWORD\"}")

STATUS=$(echo "$LOGIN_RESP" | jq -r '.status // "error"')
TOKEN=$(echo  "$LOGIN_RESP" | jq -r '.token  // empty')

if [ "$STATUS" != "ok" ] || [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
  echo "login failed: $LOGIN_RESP" >&2
  exit 1
fi

challenge_save_var CHALLENGE_TOKEN "$TOKEN"
echo "CHALLENGE_TOKEN persisted to ~/.simubotix-challenge.env (length=${#TOKEN})"
```

Always check `.status == "ok"` AND `.token` is non-null/non-empty before persisting — a malformed response would otherwise silently leave `CHALLENGE_TOKEN=null` on disk, breaking every later call with a confusing 401.

`challenge_save_var` writes `CHALLENGE_TOKEN=<jwt>` to `~/.simubotix-challenge.env` (mode 0600) **and** exports it in the current shell. Subsequent Bash calls (even fresh subshells) recover it via `. ~/.simubotix-challenge.env`.

Anything other than `{"status":"ok","token":"<JWT>"}` → stop and report the body verbatim to the user.

## Step 2 — Inspect the current user (recommended after login)

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
curl -fsS "$BASE_URL/api/challenge/current-user-info" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq
```

Read these fields back to the user:

- `name` — what shows up on the leaderboard.
- `max_concurrent_cases` — the `parallelism` each of this user's jobs gets: the number of live agent connections one job accepts. Remember this for `challenge-run-agent`.

> **If this returns HTTP 5xx (e.g. `curl: (22) ... error 500`)**, do NOT assume the platform is down. The most common cause is `$CHALLENGE_TOKEN` being unset / empty / not exported into the current shell — the server may surface that as a 500 instead of a clean 401. Drop `-f` to see the body, and confirm the state file is being sourced:
>
> ```bash
> [ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
> echo "token length=${#CHALLENGE_TOKEN}"   # 0 → still missing; re-run Step 1
> curl -sS -i "$BASE_URL/api/challenge/current-user-info" \
>   -H "Authorization: Bearer $CHALLENGE_TOKEN" | tail -15
> ```
>
> Only treat the 5xx as a real platform issue after the token is confirmed valid (e.g. login itself just succeeded with the same value).

## Step 3 — Refresh an expired token

`/login/refresh` works even when the JWT has expired, **as long as it can still be parsed**. Once it's truly malformed, fall back to Step 1.

> **Confirm before running.** Refresh is read-only on the platform but **mutates the user's shell `$CHALLENGE_TOKEN`** — that's a side effect they should approve. Print the command, name the consequence, and wait for "yes". Exception: if the user has already said "just refresh" / "auto-refresh on 401", treat that as standing consent.

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env

REFRESH_RESP=$(curl -fsS "$BASE_URL/api/challenge/login/refresh" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN")
NEW=$(echo "$REFRESH_RESP" | jq -r '.token // empty')

if [ -z "$NEW" ] || [ "$NEW" = "null" ]; then
  echo "refresh failed: $REFRESH_RESP" >&2
  echo "fall back to Step 1 (email + password)" >&2
else
  challenge_save_var CHALLENGE_TOKEN "$NEW"   # rewrites ~/.simubotix-challenge.env
  echo "refreshed (length=${#NEW})"
fi
```

If refresh returns null (token unparseable, account disabled, etc.), do NOT loop — go to Step 1 and ask the user for `EMAIL` / `PASSWORD` if they aren't in the environment.

## Failure modes

| Symptom | Action |
|---------|--------|
| `curl` exits non-zero / HTTP 4xx | Show the response body. Do NOT retry blindly. |
| `.token` is `null` / empty | Auth credentials wrong; ask the user to verify; never auto-retry with guesses. |
| Subsequent calls still return `401` | Token may be for a different `BASE_URL`. Confirm the host. |

## Resetting state

The state file `~/.simubotix-challenge.env` is **not** auto-cleaned. To reset:

```bash
# Forget everything (forces a fresh login + submit-job)
rm -f ~/.simubotix-challenge.env

# Or: keep token, drop only job vars (e.g. before working on a different job)
challenge_reset_job() {
  local f=~/.simubotix-challenge.env
  [ -f "$f" ] || return 0
  awk '$0 !~ /^(JOB_ID|JOB_UUID|JOB_TOKEN|PARALLELISM|TUNNEL_ENDPOINT|JOB_SEASON)=/' "$f" > "$f.tmp" && mv "$f.tmp" "$f"
  chmod 600 "$f"
  unset JOB_ID JOB_UUID JOB_TOKEN PARALLELISM TUNNEL_ENDPOINT JOB_SEASON
}
```

Use `challenge_reset_job` when the user wants a clean slate before `challenge-submit-job` and the previous job's `JOB_ID` is no longer needed (or already terminal).

## Hand-off

After login, suggest the next skill based on the user's stated goal:

- "I want to submit a model" → `challenge-submit-job`
- "I want to know my rank" → `challenge-ranking`
- "Just checking my account" → done.

Do NOT proceed automatically; wait for the user to confirm.
