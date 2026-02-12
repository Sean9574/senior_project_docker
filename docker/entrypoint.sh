#!/bin/bash
set -e

# =============================================================================
# Optimized entrypoint for ROS2 in Docker
# =============================================================================

# Create optimized FastRTPS profile for Docker shared memory transport
# This is CRITICAL for ROS2 performance in containers
cat > /tmp/fastrtps_docker.xml << 'FASTRTPS_EOF'
<?xml version="1.0" encoding="UTF-8"?>
<dds xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <profiles>
        <!-- Shared Memory Transport for high-bandwidth, low-latency -->
        <transport_descriptors>
            <transport_descriptor>
                <transport_id>shm_transport</transport_id>
                <type>SHM</type>
                <!-- 20MB segment for large image messages -->
                <segment_size>20971520</segment_size>
                <!-- 5MB max message size (enough for HD images) -->
                <max_message_size>5242880</max_message_size>
            </transport_descriptor>
            <!-- UDP fallback for cross-machine communication if needed -->
            <transport_descriptor>
                <transport_id>udp_transport</transport_id>
                <type>UDPv4</type>
            </transport_descriptor>
        </transport_descriptors>
        
        <participant profile_name="default_participant" is_default_profile="true">
            <rtps>
                <!-- Disable default transports, use our optimized ones -->
                <useBuiltinTransports>false</useBuiltinTransports>
                <userTransports>
                    <!-- Prefer shared memory (fastest) -->
                    <transport_id>shm_transport</transport_id>
                    <!-- UDP as fallback -->
                    <transport_id>udp_transport</transport_id>
                </userTransports>
                
                <!-- Increase buffers for high-frequency topics -->
                <sendSocketBufferSize>1048576</sendSocketBufferSize>
                <listenSocketBufferSize>4194304</listenSocketBufferSize>
            </rtps>
        </participant>
        
        <!-- Optimized DataWriter for sensor data -->
        <data_writer profile_name="sensor_writer">
            <historyMemoryPolicy>PREALLOCATED_WITH_REALLOC</historyMemoryPolicy>
            <qos>
                <reliability>
                    <kind>BEST_EFFORT</kind>
                </reliability>
                <durability>
                    <kind>VOLATILE</kind>
                </durability>
            </qos>
        </data_writer>
        
        <!-- Optimized DataReader for sensor data -->
        <data_reader profile_name="sensor_reader">
            <historyMemoryPolicy>PREALLOCATED_WITH_REALLOC</historyMemoryPolicy>
            <qos>
                <reliability>
                    <kind>BEST_EFFORT</kind>
                </reliability>
                <durability>
                    <kind>VOLATILE</kind>
                </durability>
            </qos>
        </data_reader>
    </profiles>
</dds>
FASTRTPS_EOF

export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/fastrtps_docker.xml

# Verify the file was created
if [ -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
    echo "✓ FastRTPS SHM profile created at $FASTRTPS_DEFAULT_PROFILES_FILE"
else
    echo "✗ Warning: Failed to create FastRTPS profile"
fi

# Set performance-related environment variables
export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity}] [{name}]: {message}"

# Disable Python bytecode caching (can cause issues in containers)
export PYTHONDONTWRITEBYTECODE=1

# Set CUDA to cache compiled kernels
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648

# Source ROS2
source /opt/ros/humble/setup.bash

# Source workspace if it exists
if [ -f /home/stretch/ament_ws/install/setup.bash ]; then
    source /home/stretch/ament_ws/install/setup.bash
fi

# Print diagnostic info on startup
echo "=== Docker Entrypoint Diagnostics ==="
echo "RMW_IMPLEMENTATION: $RMW_IMPLEMENTATION"
echo "ROS_DOMAIN_ID: $ROS_DOMAIN_ID"
echo "FASTRTPS_DEFAULT_PROFILES_FILE: $FASTRTPS_DEFAULT_PROFILES_FILE"
echo "MUJOCO_GL: $MUJOCO_GL"
echo "CUDA available: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Shared memory: $(df -h /dev/shm | tail -1 | awk '{print $2}')"
echo "====================================="

# Execute the command passed to docker run
exec "$@"