#!/bin/bash

IMAGE_NAME="senior_project:humble"
CONTAINER_NAME="senior_project"

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found in $(pwd)"
    exit 1
fi

# Build from parent directory (ament_ws) but use Dockerfile in docker/
echo "Building Docker image..."
docker build -t $IMAGE_NAME -f Dockerfile ..

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "ERROR: Docker build failed"
    exit 1
fi

# Remove existing container with same name if it exists
docker rm -f $CONTAINER_NAME 2>/dev/null

# Run the container
echo "Starting container..."
docker run -it --rm \
    --gpus all \
    --env-file "$HOME/.hf.env" \
    --name $CONTAINER_NAME \
    -p 8100:8100 \
    -p 8101:8101 \
    -p 8765:8765 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    $IMAGE_NAME \
    bash