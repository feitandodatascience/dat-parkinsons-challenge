#!/usr/bin/env bash
# Bootstrap a freshly rented GPU instance for the 128^3 run.
#
#   bash scripts/cloud/bootstrap_gpu_box.sh
#
# Run this from the repo root ON THE INSTANCE, after the repo is cloned and BEFORE the data is
# uploaded. It is idempotent — safe to re-run if a step fails.
#
# The compliance-relevant part is the tmpfs mount. Everything data-shaped lives in RAM, so no
# copy of the competition data is ever written to the instance's disk. That turns "delete every
# copy at the end of the challenge" (docs/07 condition 3) from an audit of the provider's
# retention policy into a property of how the machine was set up: terminate the instance, or
# merely lose power, and the data is gone. See docs/07 and docs/08.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RAM_MOUNT="${RAM_MOUNT:-/mnt/ram}"
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}"

say() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mFAILED: %s\033[0m\n' "$*" >&2; exit 1; }

# --- 1. The GPU is real and CUDA is present -------------------------------------------------
say "GPU"
command -v nvidia-smi >/dev/null || die "no nvidia-smi — this is not a GPU instance"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- 2. RAM disk for everything data-shaped -------------------------------------------------
# Sized at 60% of total RAM. The 128^3 cache is ~11.4 GB at float32 for 1362 scans; the raw
# NIfTIs and the interim CSVs sit alongside it. On a 200 GB box that is ample headroom, and
# leaving 40% keeps the dataloader workers from competing with the page cache.
say "RAM disk at $RAM_MOUNT"
RAM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
TMPFS_GB=$(( RAM_KB * 6 / 10 / 1024 / 1024 ))
[ "$TMPFS_GB" -ge 32 ] || die "only ${TMPFS_GB}G of tmpfs available; rent a box with more RAM"

if mountpoint -q "$RAM_MOUNT"; then
  echo "already mounted:"
else
  sudo mkdir -p "$RAM_MOUNT"
  sudo mount -t tmpfs -o "size=${TMPFS_GB}G,mode=0700,uid=$(id -u),gid=$(id -g)" tmpfs "$RAM_MOUNT"
fi
df -h "$RAM_MOUNT"

# Point the repo's data/ and models/ at RAM. Checkpoints are derived from the Data (CLAUDE.md
# §0), so they are kept off the disk too and pulled back over scp before teardown.
say "Redirecting data/ and models/ into RAM"
for d in data models; do
  mkdir -p "$RAM_MOUNT/$d"
  if [ -L "$d" ]; then
    :
  elif [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null || true)" ]; then
    die "$d/ exists on disk and is not empty — refusing to shadow it. Inspect it first."
  else
    rm -rf "$d"
    ln -s "$RAM_MOUNT/$d" "$d"
  fi
done
mkdir -p "$RAM_MOUNT/data/raw" "$RAM_MOUNT/data/interim" "$RAM_MOUNT/data/processed/cache"
ls -la data models

# --- 3. Python 3.12, matching the competition runtime ---------------------------------------
say "Python 3.12 + venv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.12
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version; print(sys.version)'

# --- 4. Torch pinned to the runtime's version -----------------------------------------------
# The checkpoints this box produces get loaded by main.py under torch 2.12.1+cu129 on the
# platform. State dicts are usually portable across versions; matching costs nothing and
# removes the question.
say "torch $TORCH_VERSION (cu129)"
uv pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_INDEX"
uv pip install -e '.[dev]'
uv pip install lightgbm   # 09_blend.py; not a runtime dependency of the package

python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")
assert torch.cuda.is_available(), "torch cannot see the GPU — wrong wheel or wrong driver"
PY

# --- 5. Tests, on synthetic data only -------------------------------------------------------
say "pytest (synthetic data, no competition data needed)"
pytest -q

say "Bootstrap complete"
cat <<'NEXT'

Next, from your LOCAL machine (docs/08 step 3):

  scp data/raw/<archive>            <user>@<host>:/mnt/ram/data/raw/
  scp data/interim/folds.csv        <user>@<host>:/mnt/ram/data/interim/

folds.csv must be UPLOADED, not regenerated. The paired comparison in 11_compare_runs.py is
only valid if both runs used identical fold assignments.

NEXT
