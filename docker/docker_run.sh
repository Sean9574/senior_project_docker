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

echo "Starting container..."
docker run -it --rm \
    --gpus all \
    --env-file "$HOME/.hf.env" \
    --name $CONTAINER_NAME \
    -p 8100:8100 \
    -p 8101:8101 \
    -p 8765:8765 \
    -e DISPLAY=$DISPLAY \
    -e ROS_DOMAIN_ID=10 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    $IMAGE_NAME \
    bash