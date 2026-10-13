#!/usr/bin/env bash
# Serve Qwen3-1.7B (4-bit MLX) on 127.0.0.1:8080 with an OpenAI-compatible API.
# mlx-lm runs in an isolated uvx tool env so it never touches the project venv.
set -euo pipefail
MODEL="${MODEL:-mlx-community/Qwen3-1.7B-4bit}"
PORT="${PORT:-8080}"
exec uvx --python 3.12 --from "mlx-lm" mlx_lm.server \
  --model "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --temp 1.0 --max-tokens 1024
