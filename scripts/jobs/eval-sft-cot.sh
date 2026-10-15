#!/usr/bin/env bash
# Run 3 stage: see scripts/jobs/wordle.sh
exec bash "$HOME/wordle-rl/scripts/jobs/wordle.sh" eval-sft-cot "$@"
