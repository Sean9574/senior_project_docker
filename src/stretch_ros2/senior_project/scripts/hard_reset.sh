#!/bin/bash

# Source ROS2 and workspace
source /opt/ros/humble/setup.bash
source ~/ament_ws/install/setup.bash

# Configuration
PACKAGE_NAME="senior_project"
LAUNCH_FILE="RL.launch.py"
MODELS_DIR="./models"
MAX_EPISODES=200
EPISODE_TIME=180

# Create models directory
mkdir -p "${MODELS_DIR}/current"

# Cleanup function for Ctrl+C
cleanup() {
    echo ""
    echo "=========================================="
    echo "MANUAL STOP - Cleaning up..."
    echo "=========================================="
    
    # Kill all ROS2 processes
    pkill -9 -f "ros2 launch" 2>/dev/null || true
    pkill -9 -f "senior_project" 2>/dev/null || true
    pkill -9 -f "gazebo" 2>/dev/null || true
    pkill -9 -f "gzserver" 2>/dev/null || true
    pkill -9 -f "gzclient" 2>/dev/null || true
    pkill -9 -f "ros2" 2>/dev/null || true
    
    sleep 2
    echo "✓ All processes killed"
    echo "Training stopped at episode ${episode:-0}"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

echo "Starting hard reset training..."
echo "Episodes: ${MAX_EPISODES}"
echo ""

# Main loop
for episode in $(seq 1 ${MAX_EPISODES}); do
    echo "=========================================="
    echo "EPISODE ${episode} / ${MAX_EPISODES}"
    echo "=========================================="
    
    # Launch ROS2 with timeout (runs in foreground, blocks until timeout)
    # --signal=SIGTERM gives graceful shutdown, --kill-after=30 force kills if not done in 30s
    echo "Starting ROS2..."
    echo "  Episode: ${episode}"
    echo "  Models dir: ${MODELS_DIR}"
    timeout --signal=SIGTERM --kill-after=30 ${EPISODE_TIME} ros2 launch ${PACKAGE_NAME} ${LAUNCH_FILE} \
        episode_num:=${episode} \
        models_dir:=${MODELS_DIR}
    
    echo "Timeout reached, ROS2 was killed"
    
    # Timeout already killed it, just cleanup any stragglers
    echo "Episode timeout reached, cleaning up..."
    pkill -9 -f "senior_project" 2>/dev/null || true
    pkill -9 -f "ros2" 2>/dev/null || true
    
    sleep 2
    
    echo "Episode ${episode} complete"
    echo ""
    sleep 3
done

echo "Training complete!"