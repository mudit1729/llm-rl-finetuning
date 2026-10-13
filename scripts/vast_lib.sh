# Shared helpers for the Mac-side vast.ai scripts. Source it; don't run it.
# Bash 3.2-compatible (macOS /bin/bash). JSON is parsed with python3.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAST_DIR="$ROOT/.vast"                     # gitignored state: instance id, ssh url, known_hosts
mkdir -p "$VAST_DIR"

# Pins (keep in sync with README.md)
PRIME_RL_SHA="1e5318108da29d10960e91e54b80305f597de39d"
# prime-rl @ PRIME_RL_SHA builds on CUDA 13 (torch cu130, flash-attn cu130, nixl-cu13);
# this is the base image of its own Dockerfile.cuda. Hosts need a driver with CUDA >= 13.0.
IMAGE="nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04"
DISK_GB=100

REMOTE_ROOT="/root/wordle-rl"               # configs/, scripts/, logs/, outputs/, results/
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

die()  { echo "error: $*" >&2; exit 1; }
info() { echo ">> $*" >&2; }

instance_id() {
  [ -s "$VAST_DIR/instance_id" ] || die "no active instance recorded in .vast/instance_id (run: make vast-up OFFER=<id>)"
  cat "$VAST_DIR/instance_id"
}

# Field of `vastai show instance <id> --raw`, e.g. instance_field actual_status
instance_field() {
  vastai show instance "$(instance_id)" --raw 2>/dev/null |
    python3 -c "import json,sys; d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d; v=d.get('$1'); print('' if v is None else v)"
}

# ssh://root@HOST:PORT, cached per instance
ssh_url() {
  local id; id="$(instance_id)"
  if [ ! -s "$VAST_DIR/ssh_url" ] || [ "$(head -1 "$VAST_DIR/ssh_url")" != "$id" ]; then
    local url; url="$(vastai ssh-url "$id" 2>/dev/null | tail -1)"
    case "$url" in ssh://*) ;; *) die "could not get ssh url for instance $id (got: $url)";; esac
    printf '%s\n%s\n' "$id" "$url" > "$VAST_DIR/ssh_url"
  fi
  sed -n 2p "$VAST_DIR/ssh_url"
}
ssh_host() { ssh_url | sed -E 's#ssh://([^@]+@)?([^:]+):([0-9]+).*#\2#'; }
ssh_port() { ssh_url | sed -E 's#ssh://([^@]+@)?([^:]+):([0-9]+).*#\3#'; }

# Fresh host keys per instance, kept out of ~/.ssh/known_hosts.
ssh_opts() {
  echo "-i $SSH_KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new" \
       "-o UserKnownHostsFile=$VAST_DIR/known_hosts -o ServerAliveInterval=30" \
       "-o ServerAliveCountMax=6 -o ConnectTimeout=15 -o LogLevel=ERROR"
}

# rssh [ssh args...] -- run on the instance (no args: interactive shell)
rssh() {
  # shellcheck disable=SC2046
  ssh $(ssh_opts) -p "$(ssh_port)" "root@$(ssh_host)" "$@"
}

# rsync over the same ssh transport
rsync_e() { echo "ssh $(ssh_opts) -p $(ssh_port)"; }
remote() { echo "root@$(ssh_host)"; }
