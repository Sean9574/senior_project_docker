#!/bin/bash
set -euo pipefail

IMAGE_NAME="senior_project:humble"
CONTAINER_NAME="senior_project"

cd "$(dirname "$0")"

if [ ! -f "Dockerfile" ]; then
  echo "ERROR: Dockerfile not found in $(pwd)"
  exit 1
fi

# Extract HF token from env file (if present)
HF_TOKEN=$(grep -E "^(HUGGINGFACE_HUB_TOKEN|HF_TOKEN)=" "$HOME/.hf.env" 2>/dev/null | head -1 | cut -d'=' -f2 || true)

# Flags:
#   CLEAN=1  -> disables build cache
#   PULL=1   -> pulls latest base images
#   DETACH=1 -> run in background (container stays running after exit)
CLEAN="${CLEAN:-0}"
PULL="${PULL:-0}"
DETACH="${DETACH:-0}"

BUILD_FLAGS=()
if [ "$CLEAN" = "1" ]; then
  BUILD_FLAGS+=(--no-cache)
fi
if [ "$PULL" = "1" ]; then
  BUILD_FLAGS+=(--pull)
fi

echo "Building Docker image..."
docker build \
  "${BUILD_FLAGS[@]}" \
  --build-arg HF_TOKEN="$HF_TOKEN" \
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

DOCKER_RUN_ARGS=(
  --gpus all
  --pid=host
  --ipc=host
  --shm-size=8g
  --network host
  --ulimit memlock=-1:-1
  --ulimit rtprio=99:99
  --ulimit nofile=65536:65536
  --cap-add=SYS_NICE
  --env-file "$HOME/.hf.env"
  --name "$CONTAINER_NAME"
  -e ROS_DOMAIN_ID=10
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  -e NVIDIA_DRIVER_CAPABILITIES=all
  -e CUDA_CACHE_DISABLE=0
  -e CUDA_CACHE_MAXSIZE=2147483648
  -e CUDNN_BENCHMARK=1
  -v "$HOME/rl_checkpoints:/home/stretch/ament_ws/src/stretch_ros2/senior_project/parallel_training"
)

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