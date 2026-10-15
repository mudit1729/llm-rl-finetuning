#!/usr/bin/env bash
# Run 3 end to end in one job, so the instance never idles between stages:
# structured-thinking SFT, its eval, reweighted per-turn-credit RL, the final eval.
exec bash "$HOME/wordle-rl/scripts/jobs/wordle.sh" sft-cot eval-sft-cot rl-cot eval-cot "$@"
