#!/bin/bash

IMAGE_NAME="senior_project:humble"
CONTAINER_NAME="senior_project"

cd "$(dirname "$0")"

if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found in $(pwd)"
    exit 1
fi

# Extract HF token from env file
HF_TOKEN=$(grep -E "^(HUGGINGFACE_HUB_TOKEN|HF_TOKEN)=" "$HOME/.hf.env" 2>/dev/null | head -1 | cut -d'=' -f2)

echo "Building Docker image..."
docker build --build-arg HF_TOKEN="$HF_TOKEN" -t $IMAGE_NAME -f Dockerfile ..

if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

docker rm -f $CONTAINER_NAME 2>/dev/null

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
  --name $CONTAINER_NAME \
  -e DISPLAY=$DISPLAY \
  -e ROS_DOMAIN_ID=10 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e CUDA_CACHE_DISABLE=0 \
  -e CUDA_CACHE_MAXSIZE=2147483648 \
  -e CUDNN_BENCHMARK=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/dri:/dev/dri \
  $IMAGE_NAME bash