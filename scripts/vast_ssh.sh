#!/usr/bin/env bash
# SSH helpers for the recorded instance.
#
#   scripts/vast_ssh.sh                 # interactive shell
#   scripts/vast_ssh.sh '<command>'     # run a command
#   scripts/vast_ssh.sh --dashboard     # start prime-rl's dashboard (tmux) and forward it to
#                                       # http://localhost:7788 (foreground; Ctrl-C to stop)
set -euo pipefail
source "$(dirname "$0")/vast_lib.sh"

if [ "${1:-}" = "--dashboard" ]; then
  rssh "tmux has-session -t dashboard 2>/dev/null || {
          mkdir -p $REMOTE_ROOT/outputs $REMOTE_ROOT/logs
          tmux new-session -d -s dashboard 'cd ~/prime-rl && PATH=\$HOME/.local/bin:\$PATH UV_NO_SYNC=1 uv run dashboard $REMOTE_ROOT/outputs 2>&1 | tee -a $REMOTE_ROOT/logs/dashboard.log'
        }"
  info "forwarding instance 127.0.0.1:7788 -> http://localhost:7788 (Ctrl-C to stop)"
  # shellcheck disable=SC2046
  exec ssh $(ssh_opts) -p "$(ssh_port)" -N -L 7788:127.0.0.1:7788 "root@$(ssh_host)"
fi

if [ $# -eq 0 ]; then
  # shellcheck disable=SC2046
  exec ssh $(ssh_opts) -t -p "$(ssh_port)" "root@$(ssh_host)"
fi
rssh "$@"
