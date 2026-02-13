#!/bin/bash
set -e

# =============================================================================
# PERFORMANCE OPTIMIZED Entrypoint for ROS2 in Docker
# 
# Default: Cyclone DDS (better Docker performance)
# For real robot: set USE_FASTRTPS=1 before running container
# =============================================================================

# Check if user wants FastRTPS (for real robot compatibility)
if [ "$USE_FASTRTPS" = "1" ]; then
    echo "=== Using FastRTPS (real robot mode) ==="
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
else
    # Create optimized Cyclone DDS config
    cat > /tmp/cyclonedds.xml << 'CYCLONE_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<CycloneDDS xmlns="https://cdds.io/config" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="https://cdds.io/config https://raw.githubusercontent.com/eclipse-cyclonedds/cyclonedds/master/etc/cyclonedds.xsd">
    <Domain id="any">
        <General>
            <Interfaces>
                <NetworkInterface autodetermine="true" priority="default" multicast="default"/>
            </Interfaces>
            <AllowMulticast>default</AllowMulticast>
            <EnableMulticastLoopback>true</EnableMulticastLoopback>
        </General>
        <Internal>
            <Watermarks>
                <WhcHigh>500kB</WhcHigh>
            </Watermarks>
            <SynchronousDeliveryPriorityThreshold>1</SynchronousDeliveryPriorityThreshold>
        </Internal>
        <Tracing>
            <Verbosity>warning</Verbosity>
        </Tracing>
    </Domain>
</CycloneDDS>
CYCLONE_EOF

    # Set Cyclone DDS as the RMW implementation (faster in Docker than FastRTPS)
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    export CYCLONEDDS_URI=file:///tmp/cyclonedds.xml
    echo "=== Using Cyclone DDS (simulation mode) ==="
fi

# Enable cuDNN benchmark mode for faster repeated operations
export CUDNN_BENCHMARK=1

# Set CUDA caching
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f /home/stretch/ament_ws/install/setup.bash ]; then
    source /home/stretch/ament_ws/install/setup.bash
fi

# Start ROS2 daemon for faster discovery
ros2 daemon start 2>/dev/null || true

# Print diagnostic info on startup
echo "===================================="
echo "RMW: $RMW_IMPLEMENTATION"
echo "MUJOCO_GL: $MUJOCO_GL"
echo "CUDNN_BENCHMARK: $CUDNN_BENCHMARK"
echo "===================================="

# Execute the command passed to docker run
exec "$@"