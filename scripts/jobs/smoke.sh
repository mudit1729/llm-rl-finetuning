#!/usr/bin/env bash
# Phase 3 smoke test: prime-rl's examples/basic/reverse-text end to end on 2 GPUs.
#   base eval -> SFT (our own) -> export -> RL from our SFT weights -> export -> eval
# Args (optional): stage names to run, e.g. `rl export-rl eval-rl` (default: all).
source "$HOME/wordle-rl/scripts/jobs/lib.sh"
STAGES="$*"
EX=examples/basic/reverse-text

if want eval-base; then
  with_server Qwen/Qwen3-0.6B smoke-infer-base -- \
    uv run eval @ $EX/eval.toml --run.name smoke-eval-base --clean
fi

if want sft; then
  stage "SFT (trainer GPU + inference GPU for online evals)"
  uv run sft @ $EX/sft.toml --run.name smoke-sft --clean
fi

if want export-sft; then
  export_hf "$(latest_step_dir smoke-sft)"
fi

if want rl; then
  SFT_WEIGHTS="$(latest_step_dir smoke-sft)/weights"
  stage "RL from $SFT_WEIGHTS"
  uv run rl @ $EX/rl.toml --model.name "$SFT_WEIGHTS" --run.name smoke-rl --no-monitors.wandb --clean
fi

if want export-rl; then
  export_hf "$(latest_step_dir smoke-rl)"
fi

if want eval-rl; then
  RL_WEIGHTS="$(latest_step_dir smoke-rl)/weights"
  with_server "$RL_WEIGHTS" smoke-infer-rl -- \
    uv run eval @ $EX/eval.toml -m "$RL_WEIGHTS" --run.name smoke-eval-rl --clean
fi

stage "smoke done"
