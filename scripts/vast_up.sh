#!/usr/bin/env bash
# Find and (only after you pick one) create a vast.ai instance.
#
#   scripts/vast_up.sh search [smoke|smoke1|wordle] [--hours H] [--dl-gb 30]
#                                                          # top 5 offers; creates nothing
#   scripts/vast_up.sh create <offer_id> --hours H [--yes] # rent it, wait for SSH
#   scripts/vast_up.sh wait                                # wait for the recorded instance
#
# Profiles
#   smoke   2x RTX 4090 / RTX 3090 (24 GB): reverse-text end to end. prime-rl's `rl` needs
#           >= 1 inference GPU + 1 trainer GPU; there is no shared-GPU mode.
#   smoke1  1x RTX 4090 / RTX 3090: SFT / inference / eval only (no RL)
#   wordle  2x 80 GB A100 / H100 (1 inference GPU + 1 trainer GPU)
# Common filters: verified host, rentable, direct ports, >= 100 GB disk, driver with
# CUDA >= 13.0 (prime-rl's torch/flash-attn are cu130 builds), reliability > 0.98,
# >= 200 Mb/s download, not in mainland China (Hugging Face is unreachable there).
# Prices include 100 GB of storage.
set -euo pipefail
source "$(dirname "$0")/vast_lib.sh"

BASE='verified=true rentable=true rented=false direct_port_count>=1 disk_space>=100 cuda_vers>=13.0 reliability>0.98 inet_down>=200 geolocation notin [CN]'
profile_query() {
  case "$1" in
    smoke)  echo "$BASE num_gpus=2 gpu_name in [RTX_4090,RTX_3090] cpu_ram>=48" ;;
    smoke1) echo "$BASE num_gpus=1 gpu_name in [RTX_4090,RTX_3090] cpu_ram>=32" ;;
    # Peak GPU memory measured on A100 80GB: full-FT trainer 30.3 GiB (SFT 29.2), LoRA 9.4 GiB,
    # so 40 GB A100s fit the whole pipeline.
    wordle) echo "$BASE num_gpus=2 gpu_ram>=38 gpu_name in [A100_SXM4,A100_PCIE] cpu_ram>=64" ;;
    wordle80) echo "$BASE num_gpus=2 gpu_ram>=75 gpu_name in [A100_SXM4,A100_PCIE,H100_SXM,H100_PCIE,H100_NVL] cpu_ram>=96" ;;
    *) die "unknown profile '$1' (smoke|smoke1|wordle|wordle80)" ;;
  esac
}

cmd_search() {
  local profile="${1:-smoke}" hours="" dl_gb=30
  shift || true
  while [ $# -gt 0 ]; do
    case "$1" in --hours) hours="$2"; shift 2;; --dl-gb) dl_gb="$2"; shift 2;; *) die "bad arg $1";; esac
  done
  local q; q="$(profile_query "$profile")"
  info "query: $q"
  vastai search offers "$q" -o 'dph+' --storage "$DISK_GB" --limit 40 --raw > "$VAST_DIR/last_search_$profile.json"
  python3 - "$VAST_DIR/last_search_$profile.json" "$hours" "$dl_gb" <<'PY'
import json, sys
offers = json.load(open(sys.argv[1])); hours = float(sys.argv[2]) if sys.argv[2] else None; dl_gb = float(sys.argv[3])
# With --hours, rank by estimated total: rental + downloads (setup pulls ~dl_gb of wheels/image/models).
est = lambda o: o["dph_total"] * hours + o["inet_down_cost"] * dl_gb
offers.sort(key=est if hours else (lambda o: o["dph_total"]))
if not offers:
    sys.exit("no offers match; loosen filters in scripts/vast_up.sh")
hdr = f"{'offer':>9}  {'GPU':<16} {'VRAM':>5} {'$/hr':>6} {'reliab':>7} {'CUDA':>5} {'disk':>5} {'down':>6} {'$/GB dl':>7}  location"
if hours: hdr += f"  {'est $':>6}"
print(hdr)
for o in offers[:5]:
    gpu = f"{o['num_gpus']}x {o['gpu_name']}"
    line = (f"{o['id']:>9}  {gpu:<16} {o['gpu_ram']/1024:>4.0f}G {o['dph_total']:>6.3f} {o['reliability2']:>7.4f} "
            f"{o['cuda_max_good']:>5} {o['disk_space']:>4.0f}G {o['inet_down']:>5.0f}M {o['inet_down_cost']:>7.4f}  {o.get('geolocation','?')}")
    if hours: line += f"  {est(o):>6.2f}"
    print(line)
basis = f"est $ = {hours}h rental + {dl_gb:.0f} GB download" if hours else "sorted by $/hr"
print(f"({len(offers)} matching offers; cheapest 5, {basis}. $/hr includes 100 GB storage.)")
PY
}

cmd_wait() {
  local id; id="$(instance_id)"
  info "waiting for instance $id to be running (image pull can take several minutes)"
  local i status=""
  for i in $(seq 1 90); do
    status="$(instance_field actual_status || true)"
    [ "$status" = "running" ] && break
    printf '   [%s] status=%s\n' "$(date +%H:%M:%S)" "${status:-pending}" >&2
    sleep 20
  done
  [ "$status" = "running" ] || die "instance $id not running after 30 min (status=$status). Consider: make vast-down"
  info "running; waiting for ssh"
  rm -f "$VAST_DIR/ssh_url"
  for i in $(seq 1 30); do
    if rssh -o BatchMode=yes true 2>/dev/null; then
      info "ssh ok: $(ssh_url)"
      rssh 'nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; df -h / | tail -1'
      return 0
    fi
    sleep 10
  done
  die "ssh not reachable after 5 min; check 'vastai show instance $id'"
}

cmd_create() {
  local offer="${1:-}" hours="" yes=0
  [ -n "$offer" ] || die "usage: vast_up.sh create <offer_id> --hours H [--yes]"
  shift
  while [ $# -gt 0 ]; do
    case "$1" in --hours) hours="$2"; shift 2;; --yes) yes=1; shift;; *) die "bad arg $1";; esac
  done
  [ -n "$hours" ] || die "--hours is required (expected runtime, used for the budget check)"
  if [ -s "$VAST_DIR/instance_id" ]; then die "instance $(cat "$VAST_DIR/instance_id") is already recorded; run make vast-down first"; fi
  local running; running="$(vastai show instances --raw | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  [ "$running" = "0" ] || die "you already have $running instance(s) on vast.ai (vastai show instances); refusing to add another"

  # The offer must come from a search we showed you.
  local found
  found="$(python3 - "$offer" "$VAST_DIR"/last_search_*.json <<'PY'
import json, sys
offer = int(sys.argv[1])
for path in sys.argv[2:]:
    for o in json.load(open(path)):
        if o["id"] == offer:
            profile = path.rsplit("last_search_", 1)[1].removesuffix(".json")
            print(profile, f"{o['num_gpus']}x_{o['gpu_name'].replace(' ', '_')}", f"{o['dph_total']:.4f}", f"{o['reliability2']:.4f}")
            sys.exit(0)
sys.exit(1)
PY
)" || die "offer $offer is not in a saved search; run: scripts/vast_up.sh search <profile>"
  set -- $found
  local profile="$1" gpu="$2" dph="$3" rel="$4"
  local est; est="$(python3 -c "print(f'{$dph*$hours:.2f}')")"
  echo "Offer $offer: $gpu  \$$dph/hr  reliability $rel  expected ${hours}h  => ~\$$est"
  python3 "$ROOT/scripts/cost_log.py" check --planned "$est" || die "would exceed the budget cap in COST.md; not creating"
  if [ "$yes" != 1 ]; then
    read -r -p "Create this instance? Type 'yes': " ans
    [ "$ans" = "yes" ] || die "aborted"
  fi

  local label="wordle-rl-$profile"
  local out
  out="$(vastai create instance "$offer" --image "$IMAGE" --disk "$DISK_GB" --ssh --direct \
          --label "$label" --cancel-unavail --raw)"
  local id
  id="$(echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert d.get("success"), d; print(d["new_contract"])')" \
    || die "create failed: $out"
  echo "$id" > "$VAST_DIR/instance_id"
  python3 - "$VAST_DIR/instance.json" <<PY
import json, sys, time
json.dump({"instance_id": $id, "offer": $offer, "profile": "$profile", "label": "$label", "gpu": "$gpu".replace("_", " "),
           "dph_total": $dph, "expected_hours": $hours, "created_at": time.time()}, open(sys.argv[1], "w"), indent=2)
PY
  info "created instance $id (label $label). Billing has started."
  cmd_wait
}

case "${1:-}" in
  search) shift; cmd_search "$@" ;;
  create) shift; cmd_create "$@" ;;
  wait)   cmd_wait ;;
  *) sed -n '2,8p' "$0"; exit 1 ;;
esac
