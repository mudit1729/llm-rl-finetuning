#!/usr/bin/env bash
# Pull everything, destroy the instance, confirm it is gone, log the cost.
#
#   scripts/vast_down.sh [--yes] [--no-pull] [--checkpoints|--adapters] [--phase N] [--notes "..."]
#
# The pull runs first (instance disks are lost on destroy); if it fails, nothing is destroyed
# unless --no-pull is given. Cost = runtime x $/hr, appended to COST.md (an estimate:
# reconcile with `vastai show invoices -c --instance_label <label>`).
set -euo pipefail
source "$(dirname "$0")/vast_lib.sh"

yes=0 pull=1 ckpt="" phase="" notes=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes) yes=1; shift ;;
    --no-pull) pull=0; shift ;;
    --checkpoints) ckpt="--checkpoints"; shift ;;
    --adapters) ckpt="--adapters"; shift ;;
    --phase) phase="$2"; shift 2 ;;
    --notes) notes="$2"; shift 2 ;;
    *) die "bad arg $1" ;;
  esac
done

id="$(instance_id)"
read -r gpu dph created label <<EOF
$(python3 -c "import json; d=json.load(open('$VAST_DIR/instance.json')); print(d['gpu'].replace(' ','_'), d['dph_total'], d['created_at'], d['label'])")
EOF
# Prefer the live $/hr (includes storage) if the API still reports it.
live_dph="$(instance_field dph_total 2>/dev/null || true)"
[ -n "$live_dph" ] && dph="$live_dph"
hours="$(python3 -c "import time; print(f'{(time.time()-$created)/3600:.3f}')")"
info "instance $id ($label, ${gpu//_/ }) up ${hours}h at \$$dph/hr => ~\$$(python3 -c "print(f'{$dph*$hours:.2f}')")"

if [ "$pull" = 1 ]; then
  "$ROOT/scripts/vast_sync.sh" pull $ckpt || die "pull failed; NOT destroying. Fix it, or rerun with --no-pull to destroy anyway."
fi

if [ "$yes" != 1 ]; then
  read -r -p "Destroy instance $id? Its disk is deleted. Type 'yes': " ans
  [ "$ans" = "yes" ] || die "aborted"
fi

vastai destroy instance "$id" --raw >/dev/null || vastai destroy instance "$id"
info "destroy requested; confirming with vastai show instances"
gone=0
for i in $(seq 1 18); do
  if vastai show instances --raw | python3 -c "import json,sys; sys.exit(0 if all(str(x['id'])!='$id' for x in json.load(sys.stdin)) else 1)"; then
    gone=1; break
  fi
  sleep 10
done
vastai show instances
[ "$gone" = 1 ] || die "instance $id still listed after 3 min; check the console"
info "instance $id is gone"

python3 "$ROOT/scripts/cost_log.py" add --phase "$phase" --instance "$id" --gpu "${gpu//_/ }" \
  --dph "$dph" --hours "$hours" --notes "${notes:-$label}"
mkdir -p "$VAST_DIR/history"
mv "$VAST_DIR/instance.json" "$VAST_DIR/history/$id.json"
rm -f "$VAST_DIR/instance_id" "$VAST_DIR/ssh_url" "$VAST_DIR/known_hosts" "$VAST_DIR/keep"
