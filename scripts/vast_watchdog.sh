#!/usr/bin/env bash
# Idle guard, run on the Mac while an instance exists:
#
#   scripts/vast_watchdog.sh [--idle-min 15] [--phase N]
#
# Every minute: the instance is busy if a tmux session job-* is running or any GPU is
# above 5% utilization. After --idle-min idle minutes it pulls everything and destroys the
# instance (scripts/vast_down.sh --yes). `touch .vast/keep` pauses it (keep the instance).
# Exits when no instance is recorded.
set -uo pipefail
source "$(dirname "$0")/vast_lib.sh"

idle_min=15 phase=""
while [ $# -gt 0 ]; do
  case "$1" in --idle-min) idle_min="$2"; shift 2;; --phase) phase="$2"; shift 2;; *) die "bad arg $1";; esac
done

idle_since="" unreachable=0
while [ -s "$VAST_DIR/instance_id" ]; do
  now="$(date +%s)"
  if [ -e "$VAST_DIR/keep" ]; then idle_since=""; sleep 60; continue; fi
  state="$(rssh -o BatchMode=yes "tmux ls -F '#S' 2>/dev/null | grep -c '^job-' ; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sort -n | tail -1" 2>/dev/null)"
  if [ -z "$state" ]; then
    unreachable=$((unreachable + 1))
    echo "[$(date +%H:%M)] instance unreachable ($unreachable min)"
    [ "$unreachable" -ge "$idle_min" ] && { echo "WATCHDOG: unreachable for ${idle_min} min; check it and run make vast-down"; exit 2; }
    sleep 60; continue
  fi
  unreachable=0
  jobs="$(echo "$state" | sed -n 1p)"; util="$(echo "$state" | sed -n 2p)"
  if [ "${jobs:-0}" -gt 0 ] || [ "${util:-0}" -gt 5 ]; then
    idle_since=""
  else
    [ -n "$idle_since" ] || idle_since="$now"
    idle=$(( (now - idle_since) / 60 ))
    echo "[$(date +%H:%M)] idle ${idle} min (no job-* session, max GPU util ${util}%)"
    if [ "$idle" -ge "$idle_min" ]; then
      echo "WATCHDOG: idle ${idle_min} min; pulling and destroying"
      exec "$ROOT/scripts/vast_down.sh" --yes --phase "$phase" --notes "auto-destroyed after ${idle_min} min idle"
    fi
  fi
  sleep 60
done
echo "no instance recorded; watchdog exiting"
