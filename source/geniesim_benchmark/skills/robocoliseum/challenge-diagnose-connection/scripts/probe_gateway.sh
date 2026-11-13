#!/usr/bin/env bash
# probe_gateway.sh — read-only link probe from THIS machine to the gateway.
#
# Measures what the platform can't see from its side: latency, jitter and packet loss on the
# path between the machine that runs your agent and the gateway. It sends nothing but ICMP
# echoes and `GET /health` — no job state is touched, no quota is used.
#
# It does NOT measure bulk throughput (the platform exposes no download endpoint for that).
# For throughput, use the frame timings your own agent logs — see SKILL.md Step 3.
#
# usage: probe_gateway.sh [gateway_host] [count]
#   gateway_host  default: host part of $TUNNEL_ENDPOINT (the job response's tunnel_endpoint)
#   count         default: 20 (samples per probe)
set -u

HOST="${1:-}"
if [ -z "$HOST" ] && [ -n "${TUNNEL_ENDPOINT:-}" ]; then
  HOST="${TUNNEL_ENDPOINT#*://}"; HOST="${HOST%%/*}"; HOST="${HOST%%:*}"
fi
if [ -z "$HOST" ]; then
  echo "usage: $0 [gateway_host] [count]  (or export TUNNEL_ENDPOINT from the job response)" >&2
  exit 2
fi
COUNT="${2:-20}"

echo "== probe $HOST ($(date '+%F %T'))"

echo
echo "-- ICMP (loss / rtt) --"
if command -v ping >/dev/null 2>&1; then
  # -i 0.5 works unprivileged on both macOS and Linux.
  ping -c "$COUNT" -i 0.5 "$HOST" 2>&1 | grep -E "packet loss|min/avg|round-trip|rtt" || echo "(ping produced no summary; ICMP may be blocked — not conclusive)"
else
  echo "(ping not installed)"
fi

echo
echo "-- HTTP GET /health x$COUNT (connect / total, seconds) --"
fail=0
: > /tmp/.probe_gateway.$$
for _ in $(seq 1 "$COUNT"); do
  out=$(curl -s -m 10 -o /dev/null -w "%{http_code} %{time_connect} %{time_total}" "http://$HOST/health") || { fail=$((fail+1)); continue; }
  echo "$out" >> /tmp/.probe_gateway.$$
  sleep 0.5
done
awk -v fail="$fail" '
  { n++; if ($1!=200) bad++; c+=$2; t+=$3; if ($3>mx) mx=$3; if (mn==""||$3<mn) mn=$3 }
  END {
    if (n==0) { printf "no successful request (%d failed)\n", fail; exit }
    printf "ok=%d failed=%d non200=%d  connect_avg=%.3f  total min/avg/max=%.3f/%.3f/%.3f\n", n, fail, bad+0, c/n, mn, t/n, mx
  }' /tmp/.probe_gateway.$$
rm -f /tmp/.probe_gateway.$$

echo
echo "-- path (mtr, if installed) --"
if command -v mtr >/dev/null 2>&1; then
  mtr -r -c 20 -n "$HOST" 2>&1 | tail -20
else
  echo "(mtr not installed; 'brew install mtr' / 'apt install mtr-tiny' for a per-hop loss view)"
fi

cat <<'NOTE'

How to read this
  - Loss > 0% on ICMP or any failed /health: the path to the gateway is lossy. Run the same probe
    from a different network (e.g. a phone hotspot) to tell your LAN apart from your ISP.
  - Large max vs avg (spikes of seconds): jitter/queueing — typical of a saturated uplink.
  - All clean here does NOT rule the link out: a short probe doesn't load the pipe. Large
    observation frames (several MB each) are what stress it; check frame timings, SKILL.md Step 3.
NOTE
