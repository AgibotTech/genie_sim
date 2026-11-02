---
name: challenge-diagnose-connection
description: Use when the contestant's agent keeps disconnecting from the gateway, a job fails with failure_reason 9 / "tunnel lost: reconnect window expired", or the log shows close 1006 / 1011 / "keepalive ping timeout". Runs read-only local checks (event-loop lag, link loss, throughput during a large frame) to tell "my machine/network" apart from "the platform", and says what evidence to send the organizers.
---

# challenge-diagnose-connection — Why does my agent keep disconnecting?

`failure_reason: 9` plus `tunnel lost: reconnect window expired` only says **the platform gave up
waiting** — your agent was gone for more than 30 s. It does **not** say why the socket dropped.
This skill finds the why, from your side, using read-only checks.

Everything here is read-only. Do not resubmit (costs a daily slot) or kill running agents while diagnosing.

## Step 0 — Get the platform's view

```bash
[ -f ~/.simubotix-challenge.env ] && . ~/.simubotix-challenge.env
curl -fsS "$BASE_URL/api/challenge/job/$JOB_ID/log" \
  -H "Authorization: Bearer $CHALLENGE_TOKEN" | jq '{failure_reason, status, case_id, result}'
```

- `failure_reason` must be `9`. If it is `1`/`3`/`4`, this is **not** a connection problem —
  go to `challenge-troubleshoot` row 8.
- Newer platform builds append the last disconnect type and the number of disconnects to
  `result.error`, and put the simulator's last output in `result.stdout`. Older ones only say
  `reconnect window expired`. Quote what is actually there; don't assume the newer format.

## Step 1 — Classify the disconnects from YOUR logs

Find the close events in your agent's log and bucket them. The same drop looks different from each
side:

| What you see in your agent | Meaning | Go to |
|---|---|---|
| `close 1006` / `no close frame received or sent` / `unexpected EOF` / connection reset | The connection ended **without a close frame**. 1006 is a placeholder the WebSocket library invents for that situation — it is **not** a code the other side sent and it carries **no cause** | Step 2, 3, and "Step 1b" below |
| `close 1011` / `keepalive ping timeout` | Your client closed the connection because a ping it sent got no pong within its `ping_timeout` (that is what the message says; the library docs do not document the close code, so go by the message text) | Step 2, 3 |
| `close 1000` / `1001` you did not intend | Your code (or a wrapper/supervisor) closed the connection | Step 4 |
| Platform side shows `server_write_error` (in `result.error`: "the gateway could not write to your agent in time") | The gateway could not push a frame to you within its write window — the downstream link stalled while a large observation frame was in flight | Step 3 |
| Process exit (`returncode`, supervisor restart) | Your process died or was killed | Step 4 |

Two facts about the gateway that matter (full detail: appendix A.5/A.6 of `challenge-run-agent`):
it pings every 20 s and drops the connection after **45 s without a pong**; after any drop you have
**30 s** to reconnect with the same `agent_id`. Short blips that reconnect inside 30 s are harmless —
a failure means one outage (or a reconnect that came late) exceeded the window.

Count them and note how long each took to recover. How many drops there were, and how many came
back inside 30 s versus not, tells the organizers more than the last error alone.

## Step 1b — A 1006: what to check before blaming anything

A 1006 only tells you the connection vanished. Rule things out in this order; each is something you
can read off your own machine:

1. **Did your process die or restart?** Look at the exit status / supervisor log around the
   timestamp, and `dmesg | grep -i -E "killed process|oom"` (Linux). A killed process closes its
   sockets without a close frame — it looks exactly like a network cut.
2. **Does the drop line up with something your own code does on a timer?** Idle keep-alive sends,
   periodic "hold" responses, log rotation, checkpoint saves, GC pauses. Put the disconnect
   timestamps next to those timestamps and compute the gap for each drop.
   - If the gap is **always the same** (say, every drop happens N seconds after the same action),
     suspect your own logic first — a fixed interval is just as consistent with your code as with
     anything on the network, so do not conclude "the network has an N-second timeout" from the
     number alone.
   - If the gaps are scattered, that points away from a timer.
3. **Does it reproduce from a different network** (e.g. a phone hotspot) with the same code? If the
   drops disappear, the cause is in the path from your usual network; if they stay, it is in your
   host or code.
4. **Is anything between you and the gateway configured?** A proxy, VPN, corporate firewall, or
   `HTTP(S)_PROXY` in the environment can reset or time out long-lived connections. Check
   `env | grep -i proxy` and what your VPN/firewall does with idle or long-lived TCP.

Only if all four come back clean is it worth sending the evidence in Step 5. Do not pick a cause
from the 1006 alone.

## Step 2 — Is it the machine or the network? Two cheap discriminators

**(a) Is your event loop stalled?** If inference runs on the same asyncio loop that owns the
WebSocket, a long step stops pongs. Run a lag monitor next to the agent (or add it in-process):

```python
import asyncio, time

async def loop_lag_monitor(threshold=1.0):
    last = time.monotonic()
    while True:
        await asyncio.sleep(0.5)
        now = time.monotonic()
        lag = now - last - 0.5
        if lag > threshold:
            print(f"[loop-lag] event loop blocked {lag:.1f}s at {time.strftime('%X')}")
        last = now
```

Lag spikes that line up with the disconnect times → something is blocking the loop. If that is
synchronous inference, moving it off the loop (`asyncio.to_thread(...)` for a blocking call, or a
separate process) lets the loop keep answering pings. No spikes → this is not your cause.

**(b) Is the link lossy?** Run the bundled probe from the machine that runs the agent:

```bash
# From the Genie Sim checkout root; for a copied skill, use its installation directory.
SKILL_DIR="$(pwd)/source/geniesim_benchmark/skills/agibot-world-challenge/challenge-diagnose-connection"
bash "$SKILL_DIR/scripts/probe_gateway.sh" # ICMP loss/rtt, 20x GET /health, mtr if installed
```

Loss, or `/health` max far above avg, suggests a lossy or jittery path. A clean result is **not**
exoneration — a short probe doesn't load the pipe. Go to Step 3.

## Step 3 — Large frames: does a big observation starve the heartbeat?

Observation frames can be **several MB each**. A WebSocket is a single TCP stream, so bytes queued
behind a large message — including a ping or a pong — wait until the bytes ahead of them have been
sent. If a big frame is slow enough that a pong arrives later than a peer's deadline, that peer
drops the connection even though both processes are healthy. This is one possible explanation for
"disconnects only while a request is in flight"; the measurements below tell you whether it applies
to you. Check your own frame sizes first — do not assume them.

Measure it instead of guessing. Log, per inbound binary frame: byte size and time from the previous
send to receipt. And watch the link while a frame is in flight:

```bash
# Linux: live per-connection TCP stats to the gateway (rtt, retrans, delivery_rate, cwnd)
watch -n 1 "ss -ti dst 120.92.88.78"

# whole-NIC throughput while a frame is arriving (is something else eating the uplink/downlink?)
sar -n DEV 1        # or: ifstat 1 / nload
```

Read it like this (these are things to suspect, not proof — confirm with a second measurement):

| Observation | Suspect |
|---|---|
| Frame size ÷ transfer time is far below your line speed, and `ss` shows `retrans` growing | Loss or congestion on the path — Step 5 |
| Throughput is low and **steady** (e.g. a few Mbit/s) | A bandwidth limit somewhere between you and the gateway (shared uplink, proxy/VPN, Wi-Fi, an egress limit) — this table cannot tell which side it is on |
| Several agents on one host, drops when two receive at once | They are competing for the same uplink |
| A frame takes longer than the gateway's read deadline, then the drop follows | Heartbeat starvation; a client-side `ping_timeout` change cannot lengthen the gateway's deadline, so a transfer that slow needs a faster path |
| Frames arrive in a few seconds, drops still happen | Bandwidth is unlikely to be the cause — back to Step 1b, Step 2(a) or Step 4 |

What **you** can change in the client: if you set `ping_timeout` to a small value (the appendix
example uses `10`), your client will itself abort whenever a pong is late by that much — including
while a big frame is still arriving. For Python `websockets`, its own documentation says setting
`ping_timeout=None` "disables only timeouts. This enables keepalive, to keep idle connections open,
and disables heartbeat, to support large latency spikes", and suggests raising it if traffic spikes
cause unwanted timeouts. That removes your client's own aborts; it does not change the gateway's
read deadline, so it cannot rescue a transfer that is too slow for that.

## Step 4 — The close was yours (1000/1001, process exit)

- A `close 1000`/`1001` in your log with no `drain` before it: first check whether **your own side**
  closed the socket — leaving an `async with ws_connect(...)` block (including via an unhandled
  exception), a supervisor restart, a health-check killing the process, `Ctrl-C`/`SIGTERM`
  (`returncode -15`). Your agent log usually shows whether your code initiated it
  (`sent_close_code` vs `received_close_code`, if your client logs them). If it clearly did not,
  say so in Step 5 — that is something only the organizers can check on their side.
- Check for restarts: `returncode`, supervisor logs, `dmesg | grep -i -E "killed process|oom"`.
- **Reconnect speed:** after a drop, redial promptly with the same `agent_id`. A backoff that grows
  toward 30 s, or a restart that has to reload a model before it redials, can miss the 30 s window.
  Keep the connection logic separate from the model load.

## Step 5 — What to send the organizers

If Steps 2–4 point at the path to the gateway (or you can't explain it), send:

1. `JOB_ID`, and the UTC+8 times of the failures.
2. The Step 0 output (`failure_reason`, `result.error`).
3. Disconnect log excerpts with timestamps (close code per event) from **your** agent.
4. Frame size and transfer time for a few frames, plus the `ss -ti` / `sar` output captured during
   one.
5. `probe_gateway.sh` output, ideally from your normal network **and** from another network (e.g. a
   phone hotspot) — a difference between the two separates your LAN/ISP from the platform.
6. The machine's region / provider and how many agents share its uplink.

Items 3–5 give the organizers something concrete to compare with the platform's own connection
records.

## Things NOT to do

- **Do not** resubmit to "try again" — every resubmission costs a daily slot and hits the same path.
- **Do not** conclude "the platform is down" from `reconnect window expired` alone; check `/health`
  and whether other things work first.
- **Do not** kill a running agent to "free the connection" — you may be killing the one about to
  finish its case.
- **Do not** add an application-level heartbeat frame: there isn't one (appendix A.6) and unknown
  control frames are not part of the contract.
- **Do not** state a bandwidth figure for the platform from memory. If you need it, measure from
  your side (Step 3) and report what you measured.
