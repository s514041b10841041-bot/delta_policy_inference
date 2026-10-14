#!/usr/bin/env bash
# Auto-pulls the environment image from the GitLab Package Registry if it
# isn't already present locally (and, for liteflow, fetches its checkpoint
# if missing too), then runs docker compose.
#
# --policy-name picks which policy to launch and, from that, which image:
# liteflow runs in its own image (ROS2 + CUDA + InternVL3/FLOWER — see
# Dockerfile.liteflow), everything else (act/depth_act/pi/dummy/diffusion/diffusion_rtc/depth_inference) in the shared
# lerobot one. Either way, code is mounted in and colcon-built fresh by
# entrypoint.sh on every container start, so pulling an older image is
# fine as long as its Dockerfile hasn't changed. Rebuild the image
# only when it has.
#
# Usage:
#   ./run_docker.sh --policy-name diffusion  # docker compose run, diffusion policy
#   ./run_docker.sh --policy-name liteflow   # docker compose run, liteflow policy
#   ./run_docker.sh --policy-name liteflow bash   # interactive shell instead
#   ./run_docker.sh --policy-name diffusion_rtc --checkpoint diffusion_50k
#       # another checkpoint of that policy: a folder under checkpoints/<policy folder>/, found as
#       # given, as checkpoint_<name>, or with a trailing "k" dropped (policies/__init__.py)
#   ./run_docker.sh --policy-name depth_act --gradcam
#       # also publish Grad-CAM overlays on /policy/gradcam/image (view with rqt_image_view)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

usage() {
    echo "Usage: $0 --policy-name <dummy|act|depth_act|pi|liteflow|diffusion|diffusion_rtc|depth_inference|depth_inference_rtc>" \
         "[--checkpoint <name>] [--gradcam] [command...]" >&2
    exit 1
}
POLICY_NAME=""
CHECKPOINT=""
GRADCAM=""
while [ $# -gt 0 ]; do
    case "$1" in
        --policy-name) [ -n "${2:-}" ] || usage; POLICY_NAME="$2"; shift 2 ;;
        --checkpoint)  [ -n "${2:-}" ] || usage; CHECKPOINT="$2"; shift 2 ;;
        --gradcam)     GRADCAM=1; shift ;;
        *) break ;;
    esac
done
[ -n "$POLICY_NAME" ] || usage
if { [ -n "$CHECKPOINT" ] || [ -n "$GRADCAM" ]; } && [ $# -gt 0 ]; then
    echo "[run_docker] --checkpoint / --gradcam only apply to the default launch command, not to: $*" >&2
    exit 1
fi

if [ "$POLICY_NAME" = "liteflow" ]; then
    IMAGE_NAME="liteflow_ros2:latest"
    PACKAGE_NAME="liteflow-ros2-docker"
    SERVICE="policy-inference-liteflow"
else
    IMAGE_NAME="lerobot_ros2:latest"
    PACKAGE_NAME="lerobot-ros2-docker"
    SERVICE="policy-inference"
fi
PACKAGE_URL="http://miislab-gitlab.synology.me:32769/api/v4/projects/mis-miis-robotics%2Fdelta%2Fros2_policy_inference/packages/generic/$PACKAGE_NAME/1.0.0/${IMAGE_NAME%%:*}.tar.gz"

# Falls back to a token saved on disk so you don't have to export it every
# time. Create it once with: printf '%s' 'glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token
: "${GITLAB_TOKEN:=$(cat "$HOME/.gitlab_token" 2>/dev/null || true)}"

if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    echo "[run_docker] $IMAGE_NAME not found locally, pulling from GitLab Package Registry..."
    if [ -z "${GITLAB_TOKEN:-}" ]; then
        echo "[ERROR] No GitLab token found. Either export \$GITLAB_TOKEN, save one to"
        echo "        ~/.gitlab_token (scope: read_api), or build locally instead:"
        echo "        docker compose -f docker/docker-compose.yml build $SERVICE"
        exit 1
    fi

    TMP_TAR="$(mktemp --suffix=.tar.gz)"
    trap 'rm -f "$TMP_TAR"' EXIT
    curl --fail --location --header "PRIVATE-TOKEN: $GITLAB_TOKEN" \
        -o "$TMP_TAR" "$PACKAGE_URL"
    gunzip -c "$TMP_TAR" | docker load
fi

# Compose's project directory is docker/ (where the compose file lives), so
# it would neither read a repo-root .env nor resolve a relative CHECKPOINT_DIR
# against the repo root. Pass the .env explicitly and make CHECKPOINT_DIR
# absolute here (a shell-set value wins over .env, same as compose).
COMPOSE=(docker compose -f docker/docker-compose.yml)
if [ -f .env ]; then
    COMPOSE+=(--env-file .env)
    if [ -z "${CHECKPOINT_DIR:-}" ]; then
        CHECKPOINT_DIR="$(sed -n 's/^[[:space:]]*CHECKPOINT_DIR[[:space:]]*=[[:space:]]*//p' .env \
            | tail -n 1 | sed -e 's/[[:space:]]*$//' -e 's/^["'\'']//' -e 's/["'\'']$//')"
    fi
fi
if [ -n "${CHECKPOINT_DIR:-}" ]; then
    export CHECKPOINT_DIR="$(realpath -m -- "$CHECKPOINT_DIR")"
fi

# fetch_weights.sh downloads into $CHECKPOINT_DIR/liteflow (default
# checkpoints/liteflow), i.e. the same folder compose mounts at /checkpoints.
if [ "$POLICY_NAME" = "liteflow" ] && [ ! -s "${CHECKPOINT_DIR:-checkpoints}/liteflow/step_20000_ema/config.json" ]; then
    echo "[run_docker] liteflow checkpoint not found in ${CHECKPOINT_DIR:-checkpoints}/liteflow, fetching..."
    ./ros2_policy_inference/policies/liteflow/fetch_weights.sh
fi

if [ $# -eq 0 ]; then
    "${COMPOSE[@]}" run --rm "$SERVICE" \
        ros2 launch ros2_policy_inference inference.launch.py "policy_name:=$POLICY_NAME" \
        ${CHECKPOINT:+"checkpoint:=$CHECKPOINT"} ${GRADCAM:+"gradcam_enable:=true"}
else
    "${COMPOSE[@]}" run --rm "$SERVICE" "$@"
fi
