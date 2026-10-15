# Sourced by job scripts ON the instance. Bash 5.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# Never re-resolve/sync inside runs: `uv run` otherwise re-checks the lock and refetches
# direct-URL wheels (flash-attn from GitHub) on every call. vast_setup.sh did the sync.
export UV_NO_SYNC=1
export PRL_OUTPUT_DIR="$HOME/wordle-rl/outputs"    # every prime-rl run writes to $PRL_OUTPUT_DIR/<run.name>
# On GPUs without NVLink (e.g. 2x A100 PCIe) prime-rl turns NCCL P2P/SHM off inside each process,
# but vLLM's worker has already initialized NCCL by then and keeps them on: trainer and inference
# disagree on transports and the first weight broadcast hangs. Fixing both up front keeps every
# process on the same transports (NCCL's defaults; 3.5 GB broadcast in 0.26 s on 2x A100 PCIe).
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}" NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
LOGS="$HOME/wordle-rl/logs"
RESULTS="$HOME/wordle-rl/results"
CONFIGS="$HOME/wordle-rl/configs"
mkdir -p "$PRL_OUTPUT_DIR" "$LOGS" "$RESULTS"
cd "$HOME/prime-rl"

stage() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

# Should this stage run? STAGES="a b c" (from the job's args) limits stages; empty runs all.
want() { [ -z "${STAGES:-}" ] || [[ " $STAGES " == *" $1 "* ]]; }

# with_server <model> <log-name> -- <command...>
# Start `uv run inference` for <model> on :8000, wait until it serves, run the command,
# then stop the server (whole process group) and wait for GPU memory to free up.
with_server() {
  local model="$1" name="$2"; shift 3
  stage "inference server: $model (log: logs/$name.log)"
  # shellcheck disable=SC2086  # SERVER_ARGS: optional extra inference flags
  setsid uv run inference --vllm.model "$model" ${SERVER_ARGS:-} > "$LOGS/$name.log" 2>&1 &
  local pid=$!
  local ok=0
  for _ in $(seq 1 120); do
    if curl -sf localhost:8000/v1/models >/dev/null 2>&1; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 5
  done
  if [ "$ok" != 1 ]; then
    echo "inference server failed to start; tail of logs/$name.log:"; tail -30 "$LOGS/$name.log"
    kill -TERM -- "-$pid" 2>/dev/null || true
    return 1
  fi
  local rc=0
  "$@" || rc=$?
  stage "stopping inference server"
  kill -TERM -- "-$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -KILL -- "-$pid" 2>/dev/null || true
  wait_gpus_free
  return "$rc"
}

wait_gpus_free() {
  for _ in $(seq 1 60); do
    local used; used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)"
    [ "${used:-0}" -lt 1000 ] && return 0
    sleep 2
  done
  echo "warning: GPU memory still in use:"; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

# export_hf <checkpoint step dir>  -> <dir>/weights (full fine-tunes only)
export_hf() {
  stage "export DCP -> HF safetensors: $1"
  [ -d "$1/weights" ] && { echo "already exported"; return 0; }
  uv run python tools/convert_dcp_to_bf16.py "$1"
  ls -la "$1/weights"
}

latest_step_dir() { ls -d "$PRL_OUTPUT_DIR/$1"/checkpoints/step_* 2>/dev/null | sort -t_ -k2 -n | tail -1; }
