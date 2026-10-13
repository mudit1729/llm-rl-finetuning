#!/usr/bin/env bash
# Runs ON the instance inside tmux session job-<step>: executes scripts/jobs/<step>.sh [args],
# appends output to ~/wordle-rl/logs/<step>.log and writes the exit code to logs/<step>.exit.
# Started by scripts/remote_job.sh; the tmux session ends when the job does.
step="$1"; shift
cd "$HOME/wordle-rl" || exit 1
mkdir -p logs
rm -f "logs/$step.exit"
export PATH="$HOME/.local/bin:$PATH"
echo "=== $step start $(date -Is) args: $*" | tee -a "logs/$step.log"
bash "scripts/jobs/$step.sh" "$@" 2>&1 | tee -a "logs/$step.log"
rc=${PIPESTATUS[0]}
echo "=== $step end $(date -Is) rc=$rc" | tee -a "logs/$step.log"
echo "$rc" > "logs/$step.exit"
