#!/usr/bin/env bash
# Run long jobs on the instance inside tmux, logging to ~/wordle-rl/logs/<step>.log.
#
#   scripts/remote_job.sh start <step> [args]  # tmux session job-<step> runs scripts/jobs/<step>.sh [args]
#   scripts/remote_job.sh wait <step>     # block until it finishes; exit with its exit code
#   scripts/remote_job.sh tail <step> [N] # last N log lines (default 40)
#   scripts/remote_job.sh status          # running sessions + finished exit codes
set -euo pipefail
source "$(dirname "$0")/vast_lib.sh"

step="${2:-}"
need_step() { [ -n "$step" ] || die "usage: remote_job.sh $1 <step>"; }

case "${1:-}" in
  start)
    need_step start
    [ -f "$ROOT/scripts/jobs/$step.sh" ] || die "no job script scripts/jobs/$step.sh"
    args=""; [ $# -gt 2 ] && args="$(printf '%q ' "${@:3}")"
    "$ROOT/scripts/vast_sync.sh" push >/dev/null
    rssh "command -v tmux >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null; }
          tmux has-session -t job-$step 2>/dev/null && { echo 'job-$step is already running'; exit 1; }
          tmux new-session -d -s job-$step 'bash $REMOTE_ROOT/scripts/jobs/run.sh $step $args'"
    info "started job-$step on instance $(instance_id); log: $REMOTE_ROOT/logs/$step.log"
    ;;
  wait)
    need_step wait
    while :; do
      rc="$(rssh "cat $REMOTE_ROOT/logs/$step.exit 2>/dev/null" || true)"
      if [ -n "$rc" ]; then info "job $step finished rc=$rc"; exit "$rc"; fi
      if ! rssh "tmux has-session -t job-$step 2>/dev/null"; then
        die "job-$step is not running and left no exit code; see: scripts/remote_job.sh tail $step"
      fi
      sleep 30
    done
    ;;
  tail)
    need_step tail
    rssh "tail -n ${3:-40} $REMOTE_ROOT/logs/$step.log"
    ;;
  status)
    rssh "tmux ls 2>/dev/null || echo 'no tmux sessions'; for f in $REMOTE_ROOT/logs/*.exit; do [ -e \"\$f\" ] && echo \"\$(basename \"\$f\" .exit): rc=\$(cat \"\$f\")\"; done; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader"
    ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
