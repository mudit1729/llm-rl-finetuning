#!/usr/bin/env bash
# Phase 4 diagnostic stage: see scripts/jobs/wordle.sh
exec bash "$HOME/wordle-rl/scripts/jobs/wordle.sh" eval-lora-seen "$@"
