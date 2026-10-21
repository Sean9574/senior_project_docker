#!/bin/bash
set -euo pipefail

IMAGE_NAME="senior_project:humble"
CONTAINER_NAME="senior_project"

cd "$(dirname "$0")"

if [ ! -f "Dockerfile" ]; then
  echo "ERROR: Dockerfile not found in $(pwd)"
  exit 1
fi

# Flags:
#   CLEAN=1     -> disables build cache
#   PULL=1      -> pulls latest base images
#   DETACH=1    -> run in background (container stays running after exit)
#   WITH_SAM3=0 -> sim-only image without SAM3 (default 1: SAM3 turns your prompt
#                  into goals; needs an approved HF token in ~/.hf.env)
#   DOMAIN=N    -> ROS_DOMAIN_ID inside the container (default 90, away from the host's 0)
CLEAN="${CLEAN:-0}"
PULL="${PULL:-0}"
DETACH="${DETACH:-0}"
WITH_SAM3="${WITH_SAM3:-1}"
DOMAIN="${DOMAIN:-90}"
HF_ENV_FILE="$HOME/.hf.env"

BUILD_FLAGS=()
if [ "$CLEAN" = "1" ]; then
  BUILD_FLAGS+=(--no-cache)
fi
if [ "$PULL" = "1" ]; then
  BUILD_FLAGS+=(--pull)
fi
if [ "$WITH_SAM3" = "1" ]; then
  # Token goes in as a BuildKit secret, so it never lands in the image history
  HF_TOKEN=$(grep -E "^(HUGGINGFACE_HUB_TOKEN|HF_TOKEN)=" "$HF_ENV_FILE" 2>/dev/null | head -1 | cut -d'=' -f2 || true)
  if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: SAM3 needs HF_TOKEN=... in $HF_ENV_FILE (approved access to facebook/sam3)."
    echo "       Or run WITH_SAM3=0 $0 for a sim-only image."
    exit 1
  fi
  export HF_TOKEN
  BUILD_FLAGS+=(--build-arg WITH_SAM3=1 --secret id=hf_token,env=HF_TOKEN)
else
  BUILD_FLAGS+=(--build-arg WITH_SAM3=0)
fi

echo "Building Docker image..."
DOCKER_BUILDKIT=1 docker build \
  "${BUILD_FLAGS[@]}" \
  -t "$IMAGE_NAME" \
  -f Dockerfile ..

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

echo ""
echo "=== Starting container with performance optimizations ==="
if [ "$DETACH" = "1" ]; then
  echo "=== DETACHED MODE: Container will stay running after exit ==="
fi
echo ""

# If DISPLAY is not set (headless), don't pass X11 env/mounts.
DISPLAY_VALUE="${DISPLAY:-}"

# Checkpoints, replay buffers, eval CSVs and dashboard CSVs persist here.
# Create it first so Docker doesn't create it owned by root.
mkdir -p "$HOME/rl_checkpoints"

DOCKER_RUN_ARGS=(
  --gpus all
  --ipc=host
  --shm-size=8g
  --network host
  --ulimit memlock=-1:-1
  --ulimit rtprio=99:99
  --ulimit nofile=65536:65536
  --cap-add=SYS_NICE
  --name "$CONTAINER_NAME"
  -e ROS_DOMAIN_ID="$DOMAIN"
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e CUDA_CACHE_DISABLE=0
  -e CUDA_CACHE_MAXSIZE=2147483648
  -e CUDNN_BENCHMARK=1
  -v "$HOME/rl_checkpoints:/home/stretch/rl_checkpoints"
)

if [ -f "$HF_ENV_FILE" ]; then
  DOCKER_RUN_ARGS+=(--env-file "$HF_ENV_FILE")
fi

if [ -n "$DISPLAY_VALUE" ]; then
  DOCKER_RUN_ARGS+=(
    -e DISPLAY="$DISPLAY_VALUE"
    -v /tmp/.X11-unix:/tmp/.X11-unix
    -v /dev/dri:/dev/dri
  )
else
  echo "NOTE: DISPLAY is not set; running headless (no X11 forwarding)."
fi

if [ "$DETACH" = "1" ]; then
  # Detached mode: container stays running
  DOCKER_RUN_ARGS+=(-d --restart unless-stopped)
  
  docker run "${DOCKER_RUN_ARGS[@]}" "$IMAGE_NAME" \
    bash -c "while true; do sleep 3600; done"
  
  echo ""
  echo "Container started in background!"
  echo ""
  echo "Commands:"
  echo "  docker exec -it $CONTAINER_NAME bash     # Open shell"
  echo "  docker stop $CONTAINER_NAME              # Stop container"
  echo "  docker logs $CONTAINER_NAME              # View logs"
  echo ""
else
  # Interactive mode: container removed on exit
  DOCKER_RUN_ARGS+=(-it --rm)
  
  docker run "${DOCKER_RUN_ARGS[@]}" "$IMAGE_NAME" bash
fi