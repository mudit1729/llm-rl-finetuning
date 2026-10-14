#!/usr/bin/env bash
# Runs ON the vast.ai instance (started by `make vast-setup` inside tmux, logged to
# ~/wordle-rl/logs/setup.log). Follows prime-rl's README "Manual Setup" at the pinned SHA:
# uv, clone, submodules, uv sync; then checks python, flash_attn, torch/CUDA, nvidia-smi.
set -euo pipefail

PRIME_RL_SHA="1e5318108da29d10960e91e54b80305f597de39d"
PRIME_RL_DIR="$HOME/prime-rl"
CHECK_FILE="$HOME/wordle-rl/results/setup_check.txt"

echo "== apt packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends git curl ca-certificates tmux rsync htop build-essential python3.12-dev >/dev/null  # Python.h: Triton JIT-compiles a C helper at startup

echo "== uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== prime-rl @ $PRIME_RL_SHA"
if [ ! -d "$PRIME_RL_DIR/.git" ]; then
  git clone --filter=blob:none https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_RL_DIR"
fi
cd "$PRIME_RL_DIR"
git fetch -q origin "$PRIME_RL_SHA" 2>/dev/null || git fetch -q origin
git checkout -q --detach "$PRIME_RL_SHA"
git log -1 --format='   %H %cd %s'

echo "== submodules (README list; git@github.com: URLs routed to HTTPS, no GitHub key here)"
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0="url.https://github.com/.insteadOf" GIT_CONFIG_VALUE_0="git@github.com:"
git submodule update --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config
git submodule status -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config

echo "== uv sync (all extras + the wordle and reverse-text envs, which are opt-in workspace members)"
time uv sync --all-extras --package prime-rl --package wordle --package reverse-text

echo "== project patch (turn_grpo algorithm) + wordle-turns env"
PATCH="$HOME/wordle-rl/patches/prime-rl-turn-grpo.patch"
if git apply --reverse --check "$PATCH" 2>/dev/null; then echo "   turn_grpo patch already applied"
else git apply "$PATCH" && echo "   applied $(basename "$PATCH")"; fi
uv pip install --quiet --no-deps -e "$HOME/wordle-rl/envs/wordle_turns"

echo "== checks"
export UV_NO_SYNC=1   # use the env just synced; no network re-resolution
mkdir -p "$(dirname "$CHECK_FILE")"
{
  echo "date: $(date -Is)"
  echo "prime-rl: $(git rev-parse HEAD)"
  echo "verifiers: $(git -C deps/verifiers rev-parse HEAD)"
  echo "python: $(uv run python -V 2>&1)"
  uv run python - <<'PY'
import torch, flash_attn, vllm
print(f"torch: {torch.__version__} (CUDA {torch.version.cuda}), cuda available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
print(f"flash_attn: {flash_attn.__version__}")
print(f"vllm: {vllm.__version__}")
import wordle, reverse_text, wordle_turns  # noqa: F401  (the tasksets this project trains on)
from prime_rl.orchestrator.algo import ALGORITHM_CLASSES
assert "turn_grpo" in ALGORITHM_CLASSES, "turn_grpo patch missing"
print("tasksets: wordle, reverse_text, wordle_turns import ok; turn_grpo registered")
PY
  echo "nvidia-smi:"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  echo "disk: $(df -h / | tail -1)"
} 2>&1 | tee "$CHECK_FILE"

grep -q "cuda available: True" "$CHECK_FILE" || { echo "FAIL: torch cannot see CUDA"; exit 1; }
grep -q "^flash_attn: " "$CHECK_FILE" || { echo "FAIL: flash_attn import"; exit 1; }
echo "SETUP OK"
