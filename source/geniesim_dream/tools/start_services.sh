#!/usr/bin/env bash
# Start the four local inference services in their separate Python environments.
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
policy_python="${DREAM_POLICY_PYTHON:-$root/.venv-policy/bin/python}"
eval_python="${DREAM_EVAL_PYTHON:-$root/.venv-evaluation/bin/python}"
world_python="${DREAM_WORLD_PYTHON:-$root/.venv-world/bin/python}"

port_open() {
  local port_hex slot local_address remote_address state tail
  printf -v port_hex '%04X' "$1"
  while read -r slot local_address remote_address state tail; do
    if [[ ${local_address##*:} == "$port_hex" && $state == 0A ]]; then
      return 0
    fi
  done < /proc/net/tcp
  return 1
}
service_running() { pgrep -f -- "^$1 -m $2( |$)" >/dev/null; }
ready() {
  python3 - "$1" <<'PY'
import json
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        sys.exit(0 if json.load(response).get("ready") is True else 1)
except Exception:
    sys.exit(1)
PY
}

status() {
  for entry in "8712 pi0.5 $policy_python geniesim_dream_policy.server" \
               "8972 bridge $policy_python geniesim_dream_policy.bridge" \
               "8981 VLM $eval_python geniesim_dream_evaluation.dream_service" \
               "8971 Dream $world_python geniesim_dream"; do
    read -r port name python module <<< "$entry"
    if port_open "$port"; then
      if service_running "$python" "$module"; then
        echo "$name: release environment listening on 127.0.0.1:$port"
      else
        echo "$name: other process listening on 127.0.0.1:$port"
      fi
    elif service_running "$python" "$module"; then
      echo "$name: starting (process running; port $port not listening yet)"
    else
      echo "$name: stopped (port $port)"
    fi
  done
  ready http://127.0.0.1:8971/health/ready && echo 'Dream: ready' || true
  ready http://127.0.0.1:8981/health/ready && echo 'VLM: ready' || true
}

if [[ ${1:-start} == status ]]; then status; exit 0; fi
if [[ ${1:-start} != start ]]; then echo 'Usage: tools/start_services.sh [start|status]' >&2; exit 2; fi

if service_running "$policy_python" geniesim_dream_policy.server && \
   service_running "$policy_python" geniesim_dream_policy.bridge && \
   service_running "$eval_python" geniesim_dream_evaluation.dream_service && \
   service_running "$world_python" geniesim_dream && \
   port_open 8712 && port_open 8972 && \
   ready http://127.0.0.1:8981/health/ready && ready http://127.0.0.1:8971/health/ready; then
  echo 'All services are already running.'
  status
  exit 0
fi
for port in 8712 8972 8981 8971; do
  if port_open "$port"; then
    echo "Port $port is occupied; stop the partial stack before starting all services." >&2
    status
    exit 1
  fi
done

export DREAM_PACKAGE_ROOT="$root"
export COSMOS_TRAINING="${COSMOS_TRAINING:-0}"
export DREAM_ASSET_ROOT="${DREAM_ASSET_ROOT:-$root/checkpoints}"
export DREAM_OUTPUT_DIR="${DREAM_OUTPUT_DIR:-$root/outputs}"
export GENIESIM_DREAM_POLICY_ASSETS="${GENIESIM_DREAM_POLICY_ASSETS:-$DREAM_ASSET_ROOT/policy}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$DREAM_ASSET_ROOT/policy/openpi_cache}"
export GENIESIM_DREAM_COSMOS_FRAMEWORK="${GENIESIM_DREAM_COSMOS_FRAMEWORK:-$root/third_party/cosmos-framework}"
export GENIESIM_DREAM_G2_URDF="${GENIESIM_DREAM_G2_URDF:-$root/assets/robot/urdf/G2_t2v2_crsB_ctek90d.urdf}"
export DREAM_QWEN3_VL_CHECKPOINT="${DREAM_QWEN3_VL_CHECKPOINT:-$DREAM_ASSET_ROOT/vlm/qwen3-vl}"
export DREAM_QWEN3_5_CHECKPOINT="${DREAM_QWEN3_5_CHECKPOINT:-$DREAM_ASSET_ROOT/vlm/qwen3.5}"

for python in "$policy_python" "$eval_python" "$world_python"; do
  if [[ ! -x $python ]]; then echo "Missing Python environment: $python" >&2; exit 1; fi
done
if [[ ! -d $DREAM_ASSET_ROOT ]]; then echo "Missing checkpoint directory: $DREAM_ASSET_ROOT" >&2; exit 1; fi

world_checkpoint="$DREAM_ASSET_ROOT/fd_cosmos"
world_tokenizer="$world_checkpoint/tokenizer/Qwen3-VL-8B-Instruct"
world_missing=()
for required in "$world_checkpoint/config.json" \
                "$world_checkpoint/world_model_stats_pin.json" \
                "$world_checkpoint/vae/Wan2.2_VAE.pth" \
                "$world_tokenizer/vocab.json" \
                "$world_tokenizer/merges.txt" \
                "$world_tokenizer/tokenizer_config.json" \
                "$world_tokenizer/tokenizer.json" \
                "$world_tokenizer/chat_template.json"; do
  if [[ ! -f $required ]]; then world_missing+=("$required"); fi
done
if (( ${#world_missing[@]} )); then
  echo "Preparing missing world-model files: ${world_missing[*]}"
  "$world_python" "$root/tools/materialize_world_model.py" \
    --checkpoint "$world_checkpoint" \
    --vae "$world_checkpoint/vae/Wan2.2_VAE.pth" \
    --tokenizer "$world_tokenizer" \
    --g1-stats "$root/assets/stats/agibotworld_g1_stateend.json" \
    --g2-stats "$root/assets/stats/agibotworld_g2_stateend.json" \
    --replace --download-missing >/dev/null
  echo 'World-model files ready.'
else
  echo "Using world-model files from $world_checkpoint"
fi
policy_tokenizer="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
if [[ -f $policy_tokenizer ]]; then
  echo "Policy tokenizer available: $policy_tokenizer"
else
  echo "Policy tokenizer missing: $policy_tokenizer"
  echo 'OpenPI will fetch gs://big_vision/paligemma_tokenizer.model; policy startup needs network access.'
fi

policy_path="$root/services/policy/src:$root/third_party/openpi/src:$root/third_party/openpi/packages/openpi-client/src${DREAM_POLICY_EXTRA_PYTHONPATH:+:$DREAM_POLICY_EXTRA_PYTHONPATH}"
eval_path="$root/services/evaluation/src${DREAM_EVAL_EXTRA_PYTHONPATH:+:$DREAM_EVAL_EXTRA_PYTHONPATH}"
world_path="$root/src${DREAM_WORLD_EXTRA_PYTHONPATH:+:$DREAM_WORLD_EXTRA_PYTHONPATH}"
logs="$root/.local/logs"
mkdir -p "$logs" "$DREAM_OUTPUT_DIR"
pids=()
service_names=(pi0.5 bridge VLM Dream)
service_logs=("$logs/policy.log" "$logs/bridge.log" "$logs/evaluator.log" "$logs/world.log")
cleanup() {
  for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env PYTHONUNBUFFERED=1 PYTHONPATH="$policy_path" "$policy_python" -m geniesim_dream_policy.server --task-profiles --default-profile bread --port 8712 >"$logs/policy.log" 2>&1 & pids+=("$!")
env PYTHONUNBUFFERED=1 PYTHONPATH="$policy_path" "$policy_python" -m geniesim_dream_policy.bridge --policy-url ws://127.0.0.1:8712 --port 8972 >"$logs/bridge.log" 2>&1 & pids+=("$!")
env PYTHONUNBUFFERED=1 PYTHONPATH="$eval_path" "$eval_python" -m geniesim_dream_evaluation.dream_service --models-config services/evaluation/configs/models.example.json --shared-output-root "$DREAM_OUTPUT_DIR" --port 8981 >"$logs/evaluator.log" 2>&1 & pids+=("$!")
env PYTHONUNBUFFERED=1 PYTHONPATH="$world_path" "$world_python" -m geniesim_dream serve --config configs/dream.example.json --host 127.0.0.1 --port 8971 >"$logs/world.log" 2>&1 & pids+=("$!")

echo "Starting services; logs: $logs"
echo 'Waiting for pi0.5:8712, bridge:8972, VLM:8981, Dream:8971.'
deadline=$((SECONDS + ${DREAM_START_TIMEOUT:-1800}))
started=$SECONDS
last_report=-30
last_state=''
last_policy_phase_count=0
state='no readiness checks completed'
while (( SECONDS < deadline )); do
  for i in "${!pids[@]}"; do
    if ! kill -0 "${pids[$i]}" 2>/dev/null; then
      echo "${service_names[$i]} exited during startup; see ${service_logs[$i]}" >&2
      tail -n 20 "${service_logs[$i]}" >&2
      exit 1
    fi
  done
  policy_state=waiting
  bridge_state=waiting
  vlm_state=loading
  world_state=loading
  if port_open 8712; then policy_state=listening; fi
  if port_open 8972; then bridge_state=listening; fi
  if ready http://127.0.0.1:8981/health/ready; then vlm_state=ready; fi
  if ready http://127.0.0.1:8971/health/ready; then world_state=ready; fi
  mapfile -t policy_phases < <(grep '^DREAM_POLICY_STARTUP ' "$logs/policy.log" || true)
  for ((i = last_policy_phase_count; i < ${#policy_phases[@]}; i++)); do
    echo "${policy_phases[$i]}"
  done
  last_policy_phase_count=${#policy_phases[@]}
  if [[ $policy_state == listening && $bridge_state == listening && $vlm_state == ready && $world_state == ready ]]; then
    echo 'All services ready: http://127.0.0.1:8971/'
    echo 'Keep this terminal open; Ctrl+C stops the stack.'
    wait -n "${pids[@]}" || true
    echo "A service stopped. Check $logs" >&2
    exit 1
  fi
  state="pi0.5=$policy_state bridge=$bridge_state VLM=$vlm_state Dream=$world_state"
  if [[ $state != "$last_state" ]] || (( SECONDS - last_report >= 30 )); then
    echo "Startup $((SECONDS - started))s: $state (logs: $logs)"
    last_state=$state
    last_report=$SECONDS
  fi
  sleep 2
done
echo "Startup timed out after $((SECONDS - started))s: $state; see $logs" >&2
exit 1
