#!/usr/bin/env bash
# Downloads the LiteFlow checkpoint + offline InternVL3-1B snapshot from the GitLab Package
# Registry into $CHECKPOINT_DIR/liteflow/ (default: ./checkpoints/liteflow/ at the repo root),
# where docker-compose.yml's checkpoints volume mounts them into the container at
# /checkpoints/liteflow/ (see config.yaml). Give CHECKPOINT_DIR as an absolute path;
# run_docker.sh already does.
#
# Usage:
#   ./fetch_weights.sh [checkpoint_version]   # default: step_20000_ema
#   (reads GITLAB_TOKEN from the env, or falls back to ~/.gitlab_token; scope: read_api)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT_DIR="${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints}/liteflow"

CKPT_VERSION="${1:-step_20000_ema}"
HF_VERSION="internvl3-1b"
PROJECT="mis-miis-robotics%2Fdelta%2Fros2_policy_inference"
API="http://miislab-gitlab.synology.me:32769/api/v4/projects/$PROJECT/packages/generic"

: "${GITLAB_TOKEN:=$(cat "$HOME/.gitlab_token" 2>/dev/null || true)}"
if [ -z "${GITLAB_TOKEN:-}" ]; then
    echo "[ERROR] No GitLab token found. Either export \$GITLAB_TOKEN or save one to"
    echo "        ~/.gitlab_token (scope: read_api)."
    exit 1
fi

fetch_and_extract() {
    local package="$1" version="$2" file="$3" dest="$4"
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "[fetch_weights] $dest already populated, skipping $package/$version"
        return
    fi
    mkdir -p "$dest"
    local tmp
    tmp="$(mktemp --suffix=.tar.gz)"
    trap 'rm -f "$tmp"' RETURN
    echo "[fetch_weights] downloading $package/$version..."
    curl --fail --location --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
        -o "$tmp" "$API/$package/$version/$file"
    tar -xzf "$tmp" -C "$dest"
    echo "[fetch_weights] extracted into $dest"
}

fetch_and_extract "liteflow-weights" "$CKPT_VERSION" "checkpoint.tar.gz" "$OUT_DIR/$CKPT_VERSION"
fetch_and_extract "liteflow-hf-cache" "$HF_VERSION" "hf_cache.tar.gz" "$OUT_DIR/hf_cache"

echo "[fetch_weights] done."
