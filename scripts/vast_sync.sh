#!/usr/bin/env bash
# Move files between the Mac and the instance (rsync over the instance's SSH port).
#
#   scripts/vast_sync.sh push                 # configs/, scripts/, envs/, patches/ -> instance:~/wordle-rl/
#   scripts/vast_sync.sh pull                 # results/, logs/, run outputs (no checkpoints) -> artifacts/<label>/
#   scripts/vast_sync.sh pull --dry-run       # list what a pull would copy, and its total size
#   scripts/vast_sync.sh pull --checkpoints   # also final exported weights / LoRA adapters (checks Mac disk first)
#   scripts/vast_sync.sh pull --adapters      # also the final LoRA adapters only (~140 MB each)
#
# Remote layout: ~/wordle-rl/{configs,scripts,logs,results,outputs}; prime-rl runs write to
# ~/wordle-rl/outputs/<run>/ (output_dir set by the job scripts).
set -euo pipefail
source "$(dirname "$0")/vast_lib.sh"

label() { python3 -c "import json; d=json.load(open('$VAST_DIR/instance.json')); print(f\"{d['label']}-{d['instance_id']}\")" 2>/dev/null || echo "instance-$(instance_id)"; }

# Heavy training state stays on the instance unless --checkpoints. RL training-rollout traces
# (~23 MB/step, ~4.6 GB per 200-step run) stay too; metrics.jsonl (the reward curves) and the
# eval runs' traces (a few MB each) come back.
HEAVY_EXCLUDES=(--exclude 'checkpoints/' --exclude 'broadcasts/' --exclude 'weights/' --exclude 'rollouts/'
                --exclude '*.safetensors' --exclude '*.distcp' --exclude '*.pt' --exclude '*.bin'
                --exclude 'wordle-rl*/monitors/file/traces/' --exclude 'smoke-rl/monitors/file/traces/')

case "${1:-}" in
  push)
    # The bare CUDA image has no rsync until vast_setup.sh runs.
    rssh "command -v rsync >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync >/dev/null; }
          mkdir -p $REMOTE_ROOT/logs $REMOTE_ROOT/results $REMOTE_ROOT/outputs"
    rsync -az --delete -e "$(rsync_e)" "$ROOT/configs/" "$(remote):$REMOTE_ROOT/configs/"
    rsync -az --delete -e "$(rsync_e)" --exclude '__pycache__/' "$ROOT/scripts/" "$(remote):$REMOTE_ROOT/scripts/"
    rsync -az --delete -e "$(rsync_e)" --exclude '__pycache__/' --exclude '*.egg-info/' "$ROOT/envs/" "$(remote):$REMOTE_ROOT/envs/"
    rsync -az --delete -e "$(rsync_e)" "$ROOT/patches/" "$(remote):$REMOTE_ROOT/patches/"
    info "pushed configs/, scripts/, envs/, patches/ to $(remote):$REMOTE_ROOT"
    ;;
  pull)
    if [ "${2:-}" = "--dry-run" ]; then
      rsync -azn -v --stats -e "$(rsync_e)" "${HEAVY_EXCLUDES[@]}" \
        "$(remote):$REMOTE_ROOT/results" "$(remote):$REMOTE_ROOT/logs" "$(remote):$REMOTE_ROOT/outputs" "$ROOT/artifacts/$(label)/"
      exit 0
    fi
    dest="$ROOT/artifacts/$(label)"
    mkdir -p "$dest"
    rsync -az -e "$(rsync_e)" "${HEAVY_EXCLUDES[@]}" \
      "$(remote):$REMOTE_ROOT/results" "$(remote):$REMOTE_ROOT/logs" "$(remote):$REMOTE_ROOT/outputs" "$dest/"
    if [ "${2:-}" = "--checkpoints" ] || [ "${2:-}" = "--adapters" ]; then
      # --checkpoints: exported HF dirs (<ckpt>/weights) and LoRA adapter dirs; --adapters: adapters only.
      if [ "$2" = "--adapters" ]; then pat="-name 'lora_adapter*'"; else pat="-name weights -o -name 'lora_adapter*' -o -name adapter"; fi
      list="$(rssh "cd $REMOTE_ROOT/outputs 2>/dev/null && find . -type d \( $pat \) -prune -print" || true)"
      if [ -z "$list" ]; then
        info "no exported weights/adapters found under outputs/"
      else
        need_kb="$(rssh "cd $REMOTE_ROOT/outputs && du -sk $(echo "$list" | tr '\n' ' ') | awk '{s+=\$1} END {print s}'")"
        free_kb="$(df -k "$ROOT" | awk 'NR==2 {print $4}')"
        info "checkpoints to pull: $((need_kb / 1024)) MB; Mac free: $((free_kb / 1024)) MB"
        [ "$((need_kb + 2 * 1024 * 1024))" -lt "$free_kb" ] || die "not enough Mac disk (keeps 2 GB margin); free space or pull selectively"
        # One rsync per dir: macOS ships openrsync, so avoid --files-from.
        for d in $(echo "$list" | sed 's#^\./##'); do
          mkdir -p "$dest/outputs/$d"
          rsync -az -e "$(rsync_e)" "$(remote):$REMOTE_ROOT/outputs/$d/" "$dest/outputs/$d/"
          info "  pulled outputs/$d"
        done
      fi
    fi
    date +%s > "$VAST_DIR/last_pull"
    info "pulled to ${dest#$ROOT/} ($(du -sh "$dest" | cut -f1))"
    ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
