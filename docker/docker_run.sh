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
#   CLEAN=1   -> disables cache
#   PULL=1    -> pulls latest base images
# Examples:
#   CLEAN=1 ./build.sh
#   CLEAN=1 PULL=1 ./build.sh
CLEAN="${CLEAN:-0}"
PULL="${PULL:-0}"

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
echo ""

docker run -it --rm \
  --gpus all \
  --pid=host \
  --ipc=host \
  --shm-size=8g \
  --network host \
  --ulimit memlock=-1:-1 \
  --ulimit rtprio=99:99 \
  --ulimit nofile=65536:65536 \
  --cap-add=SYS_NICE \
  --env-file "$HOME/.hf.env" \
  --name "$CONTAINER_NAME" \
  -e DISPLAY="$DISPLAY" \
  -e ROS_DOMAIN_ID=10 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_CACHE_DISABLE=0 \
  -e CUDA_CACHE_MAXSIZE=2147483648 \
  -e CUDNN_BENCHMARK=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/dri:/dev/dri \
  "$IMAGE_NAME" bash
