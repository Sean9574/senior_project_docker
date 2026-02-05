#!/bin/bash
set -e

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f /home/stretch/ament_ws/install/setup.bash ]; then
    source /home/stretch/ament_ws/install/setup.bash
fi

# Execute the command passed to docker run
exec "$@"