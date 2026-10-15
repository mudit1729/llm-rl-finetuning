#!/usr/bin/env bash
# Phase 4: Wordle on 2 GPUs. Stages (run in this order; pass one or more as args):
#   data       filter the SFT data (drop games whose answer is an eval word)
#   eval-base  eval Qwen/Qwen3-1.7B                          -> outputs/wordle-eval-base
#   sft        SFT warmup (configs/wordle/sft.toml)          -> outputs/wordle-sft
#   eval-sft   export SFT to HF weights, eval it             -> outputs/wordle-eval-sft
#   rl         full RL from our SFT weights (rl.toml)        -> outputs/wordle-rl
#   eval-rl    export RL weights, eval them                  -> outputs/wordle-eval-rl
#   rl-lora    LoRA RL from our SFT weights (rl_lora.toml)   -> outputs/wordle-rl-lora
#   eval-lora  serve SFT base + final LoRA adapter, eval     -> outputs/wordle-eval-lora
#   eval-lora-seen / eval-rl-seen  diagnostic: same eval on 20 TRAINING games (eval_seen.toml)
#   rl-turns   run 2: full RL with per-turn credit (rl_turns.toml, wordle-turns env, turn_grpo)
#   eval-turns export run 2 and eval it twice: example eval, and with the reasoning parser on
#   sft-cot    run 3: SFT on structured-thinking games (sft_cot.toml; data/cot-wordle from the Mac)
#   eval-sft-cot  export run 3's SFT, drop its optimizer state, eval it (example + parser on)
#   rl-cot     run 3: per-turn credit reweighted toward clue use (rl_cot.toml) from the CoT SFT
#   eval-cot   export run 3's RL and eval it (example + parser on)
#   eval-parsed  diagnostic: eval.toml for all 4 models with vLLM's qwen3 reasoning parser on the
#                server, so the game sees only the answer (as in training, where prime-rl's renderer
#                splits the thinking out). The example eval passes <think> text through to the game.
source "$HOME/wordle-rl/scripts/jobs/lib.sh"
[ $# -gt 0 ] || { echo "usage: wordle.sh <stage...>"; exit 2; }
STAGES="$*"
C="$CONFIGS/wordle"
eval_run() { uv run eval @ "$C/${3:-eval}.toml" -m "$1" --run.name "$2" --clean; }
sft_weights() { echo "$(latest_step_dir wordle-sft)/weights"; }

if want data; then
  stage "filter SFT data"
  uv run python "$HOME/wordle-rl/scripts/filter_sft_data.py"
fi

if want eval-base; then
  with_server Qwen/Qwen3-1.7B wordle-infer-base -- eval_run Qwen/Qwen3-1.7B wordle-eval-base
fi

if want sft; then
  stage "SFT"
  [ -f "$HOME/wordle-rl/data/v3-wordle-heldout/train.parquet" ] || { echo "run the data stage first"; exit 1; }
  uv run sft @ "$C/sft.toml" --run.name wordle-sft --clean
fi

if want eval-sft; then
  export_hf "$(latest_step_dir wordle-sft)"
  with_server "$(sft_weights)" wordle-infer-sft -- eval_run "$(sft_weights)" wordle-eval-sft
fi

if want rl; then
  stage "RL (full fine-tune) from $(sft_weights)"
  uv run rl @ "$C/rl.toml" --model.name "$(sft_weights)" --run.name wordle-rl --clean
fi

if want eval-rl; then
  export_hf "$(latest_step_dir wordle-rl)"
  W="$(latest_step_dir wordle-rl)/weights"
  with_server "$W" wordle-infer-rl -- eval_run "$W" wordle-eval-rl
fi

if want rl-lora; then
  stage "RL (LoRA) from $(sft_weights)"
  uv run rl @ "$C/rl_lora.toml" --model.name "$(sft_weights)" --run.name wordle-rl-lora --clean
fi

if want eval-lora || want eval-lora-seen; then
  # LoRA runs broadcast the PEFT adapter to <run>/broadcasts/step_N (only the last 2 kept);
  # keep the final one as lora_adapter_step_N (pulled to the Mac with --checkpoints).
  last="$(ls -d "$PRL_OUTPUT_DIR"/wordle-rl-lora/broadcasts/step_* | sort -t_ -k2 -n | tail -1)"
  A="$PRL_OUTPUT_DIR/wordle-rl-lora/lora_adapter_$(basename "$last")"
  [ -d "$A" ] || cp -r "$last" "$A"
  ls -la "$A"
  # Serve the SFT base with LoRA enabled (base is served as "<model>-base"), hot-load the
  # adapter on the vLLM engine (:8100, behind the router on :8000), eval under its name.
  serve_lora() {
    curl -sf -X POST localhost:8100/load_lora_adapter -H 'Content-Type: application/json' \
      -d "{\"lora_name\": \"wordle-lora\", \"lora_path\": \"$A\"}" && echo " adapter loaded"
    curl -s localhost:8000/v1/models | python3 -c "import json,sys; print([m['id'] for m in json.load(sys.stdin)['data']])"
    if want eval-lora; then eval_run wordle-lora wordle-eval-lora; fi
    if want eval-lora-seen; then eval_run wordle-lora wordle-eval-lora-seen eval_seen; fi
  }
  SERVER_ARGS="--vllm.enable-lora --vllm.max-lora-rank 32" \
    with_server "$(sft_weights)" wordle-infer-lora -- serve_lora
fi

if want eval-rl-seen; then
  W="$(latest_step_dir wordle-rl)/weights"
  with_server "$W" wordle-infer-rl-seen -- eval_run "$W" wordle-eval-rl-seen eval_seen
fi

if want rl-turns; then
  # Export SFT if no earlier stage did, then drop its 21 GB optimizer state: RL's own
  # checkpoints (21 GB each, two at once while rotating) need the disk.
  export_hf "$(latest_step_dir wordle-sft)"
  rm -rf "$(latest_step_dir wordle-sft)/trainer"
  stage "RL with per-turn credit (turn_grpo) from $(sft_weights)"
  uv run rl @ "$C/rl_turns.toml" --model.name "$(sft_weights)" --run.name wordle-rl-turns --clean
fi

if want eval-turns; then
  export_hf "$(latest_step_dir wordle-rl-turns)"
  W="$(latest_step_dir wordle-rl-turns)/weights"
  with_server "$W" wordle-infer-turns -- eval_run "$W" wordle-eval-turns
  SERVER_ARGS="--vllm.reasoning-parser qwen3" with_server "$W" wordle-infer-turns-parsed -- eval_run "$W" wordle-eval-turns-parsed
fi

if want eval-parsed; then
  P="--vllm.reasoning-parser qwen3"
  SERVER_ARGS="$P" with_server Qwen/Qwen3-1.7B wordle-infer-base-parsed -- eval_run Qwen/Qwen3-1.7B wordle-eval-base-parsed
  SERVER_ARGS="$P" with_server "$(sft_weights)" wordle-infer-sft-parsed -- eval_run "$(sft_weights)" wordle-eval-sft-parsed
  W="$(latest_step_dir wordle-rl)/weights"
  SERVER_ARGS="$P" with_server "$W" wordle-infer-rl-parsed -- eval_run "$W" wordle-eval-rl-parsed
  A="$(ls -d "$PRL_OUTPUT_DIR"/wordle-rl-lora/lora_adapter_step_* | tail -1)"
  serve_lora_parsed() {
    curl -sf -X POST localhost:8100/load_lora_adapter -H 'Content-Type: application/json' \
      -d "{\"lora_name\": \"wordle-lora\", \"lora_path\": \"$A\"}" && echo " adapter loaded"
    eval_run wordle-lora wordle-eval-lora-parsed
  }
  SERVER_ARGS="--vllm.enable-lora --vllm.max-lora-rank 32 $P" \
    with_server "$(sft_weights)" wordle-infer-lora-parsed -- serve_lora_parsed
fi

cot_weights() { echo "$(latest_step_dir wordle-sft-cot)/weights"; }
# eval_both <weights> <name>: the example eval, then the same eval with vLLM's reasoning parser on
eval_both() {
  with_server "$1" "wordle-infer-$2" -- eval_run "$1" "wordle-eval-$2"
  SERVER_ARGS="--vllm.reasoning-parser qwen3" with_server "$1" "wordle-infer-$2-parsed" -- eval_run "$1" "wordle-eval-$2-parsed"
}

if want sft-cot; then
  stage "SFT on structured-thinking games (run 3)"
  [ -f "$HOME/wordle-rl/data/cot-wordle/train.jsonl" ] || {
    echo "no data/cot-wordle/train.jsonl: run scripts/make_cot_data.py on the Mac (vast_sync push copies it)"; exit 1; }
  uv run sft @ "$C/sft_cot.toml" --run.name wordle-sft-cot --clean
fi

if want eval-sft-cot || want rl-cot; then
  # Export, then drop the 21 GB optimizer state: RL's own checkpoints need the disk.
  export_hf "$(latest_step_dir wordle-sft-cot)"
  rm -rf "$(latest_step_dir wordle-sft-cot)/trainer"
fi

if want eval-sft-cot; then
  eval_both "$(cot_weights)" sft-cot
fi

if want rl-cot; then
  stage "RL, per-turn credit reweighted toward clue use (run 3), from $(cot_weights)"
  uv run rl @ "$C/rl_cot.toml" --model.name "$(cot_weights)" --run.name wordle-rl-cot --clean
fi

if want eval-cot; then
  export_hf "$(latest_step_dir wordle-rl-cot)"
  eval_both "$(latest_step_dir wordle-rl-cot)/weights" cot
fi

stage "wordle stages done: $STAGES"
