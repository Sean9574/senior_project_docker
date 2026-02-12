#!/bin/bash
# =============================================================================
# Docker Performance Diagnostic Script for ROS2/MuJoCo
# Run this INSIDE your Docker container to diagnose performance issues
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=========================================="
echo "  Docker Performance Diagnostic Tool"
echo "=========================================="
echo ""

# -----------------------------------------------------------------------------
# 1. GPU/CUDA Diagnostics
# -----------------------------------------------------------------------------
echo -e "${YELLOW}1. GPU/CUDA Diagnostics${NC}"
echo "------------------------"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,temperature.gpu,utilization.gpu --format=csv
    echo ""
    
    # Check if GPU is being throttled
    THROTTLE=$(nvidia-smi --query-gpu=clocks_throttle_reasons.active --format=csv,noheader 2>/dev/null)
    if [ "$THROTTLE" != "0x0000000000000000" ] && [ -n "$THROTTLE" ]; then
        echo -e "${RED}⚠ GPU throttling detected: $THROTTLE${NC}"
    else
        echo -e "${GREEN}✓ No GPU throttling${NC}"
    fi
else
    echo -e "${RED}✗ nvidia-smi not found - GPU not accessible${NC}"
fi
echo ""

# PyTorch CUDA check
echo "PyTorch CUDA check:"
python3 << 'PYTHON_EOF'
import torch
import time

print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  cuDNN version: {torch.backends.cudnn.version()}")
    print(f"  cuDNN benchmark mode: {torch.backends.cudnn.benchmark}")
    
    # Quick benchmark
    print("\n  Running quick GPU benchmark...")
    x = torch.randn(2000, 2000, device='cuda')
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(50):
        y = torch.matmul(x, x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"  MatMul benchmark: {elapsed:.3f}s for 50 iterations")
    print(f"  Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  Memory cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
else:
    print("  CUDA not available!")
PYTHON_EOF
echo ""

# -----------------------------------------------------------------------------
# 2. Memory Diagnostics
# -----------------------------------------------------------------------------
echo -e "${YELLOW}2. Memory Diagnostics${NC}"
echo "---------------------"
free -h
echo ""

echo "Shared memory (/dev/shm):"
df -h /dev/shm
SHM_SIZE=$(df /dev/shm | tail -1 | awk '{print $2}')
SHM_SIZE_GB=$(echo "scale=2; $SHM_SIZE / 1048576" | bc 2>/dev/null || echo "N/A")
if [ "$SHM_SIZE" -lt 4194304 ] 2>/dev/null; then
    echo -e "${YELLOW}⚠ Shared memory < 4GB - consider increasing --shm-size${NC}"
else
    echo -e "${GREEN}✓ Shared memory size OK${NC}"
fi
echo ""

# -----------------------------------------------------------------------------
# 3. ROS2/DDS Diagnostics
# -----------------------------------------------------------------------------
echo -e "${YELLOW}3. ROS2/DDS Diagnostics${NC}"
echo "-----------------------"
echo "RMW_IMPLEMENTATION: ${RMW_IMPLEMENTATION:-not set}"
echo "ROS_DOMAIN_ID: ${ROS_DOMAIN_ID:-0}"
echo "FASTRTPS_DEFAULT_PROFILES_FILE: ${FASTRTPS_DEFAULT_PROFILES_FILE:-not set}"

if [ -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
    echo -e "${GREEN}✓ FastRTPS profile file exists${NC}"
    # Check if SHM transport is configured
    if grep -q "SHM" "$FASTRTPS_DEFAULT_PROFILES_FILE"; then
        echo -e "${GREEN}✓ Shared Memory transport configured${NC}"
    else
        echo -e "${YELLOW}⚠ SHM transport not found in FastRTPS config${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No FastRTPS profile file - using defaults (may use UDP)${NC}"
fi
echo ""

# Check ROS2 daemon
echo "ROS2 daemon status:"
ros2 daemon status 2>/dev/null || echo "Unable to check daemon status"
echo ""

# -----------------------------------------------------------------------------
# 4. CPU Diagnostics
# -----------------------------------------------------------------------------
echo -e "${YELLOW}4. CPU Diagnostics${NC}"
echo "------------------"
echo "CPU cores: $(nproc)"
cat /proc/cpuinfo | grep "model name" | head -1
echo ""

# Check CPU governor
GOVERNOR=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "not accessible")
echo "CPU governor: $GOVERNOR"
if [ "$GOVERNOR" = "powersave" ]; then
    echo -e "${YELLOW}⚠ CPU in powersave mode - may affect performance${NC}"
fi
echo ""

# Check for RT capabilities
echo "Real-time capabilities:"
if [ -f /sys/fs/cgroup/cpu/cpu.rt_runtime_us ]; then
    RT_RUNTIME=$(cat /sys/fs/cgroup/cpu/cpu.rt_runtime_us)
    echo "  RT runtime: $RT_RUNTIME us"
else
    echo "  RT scheduling not available in container"
fi

# Check nice limits
NICE_LIMIT=$(ulimit -e)
echo "  Nice limit: $NICE_LIMIT"
echo ""

# -----------------------------------------------------------------------------
# 5. EGL/Rendering Diagnostics
# -----------------------------------------------------------------------------
echo -e "${YELLOW}5. EGL/Rendering Diagnostics${NC}"
echo "----------------------------"
echo "MUJOCO_GL: ${MUJOCO_GL:-not set}"
echo "DISPLAY: ${DISPLAY:-not set}"

echo ""
echo "DRI devices:"
ls -la /dev/dri/ 2>/dev/null || echo "  /dev/dri not available"
echo ""

echo "NVIDIA devices:"
ls -la /dev/nvidia* 2>/dev/null || echo "  No NVIDIA devices found"
echo ""

# Test MuJoCo
echo "MuJoCo test:"
python3 << 'PYTHON_EOF'
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
try:
    import mujoco
    print(f"  MuJoCo version: {mujoco.__version__}")
    print("  ✓ MuJoCo import successful")
except Exception as e:
    print(f"  ✗ MuJoCo import failed: {e}")
PYTHON_EOF
echo ""

# -----------------------------------------------------------------------------
# 6. Docker/Container Diagnostics
# -----------------------------------------------------------------------------
echo -e "${YELLOW}6. Container Diagnostics${NC}"
echo "------------------------"

# Check if running with --pid=host
if [ "$(cat /proc/1/cgroup 2>/dev/null | grep -c docker)" -eq 0 ]; then
    echo -e "${GREEN}✓ Appears to be running with --pid=host${NC}"
else
    HOST_PROCS=$(ps aux | grep -v grep | grep -c "dockerd\|containerd" 2>/dev/null || echo 0)
    if [ "$HOST_PROCS" -gt 0 ]; then
        echo -e "${GREEN}✓ Can see host processes (--pid=host working)${NC}"
    else
        echo -e "${YELLOW}⚠ May not have --pid=host - could affect performance${NC}"
    fi
fi

# Check ulimits
echo ""
echo "Resource limits:"
echo "  memlock: $(ulimit -l)"
echo "  rtprio: $(ulimit -r 2>/dev/null || echo 'N/A')"
echo "  nofile: $(ulimit -n)"
echo ""

# -----------------------------------------------------------------------------
# 7. Network Diagnostics
# -----------------------------------------------------------------------------
echo -e "${YELLOW}7. Network Diagnostics${NC}"
echo "----------------------"
echo "Localhost latency:"
ping -c 3 -q localhost 2>/dev/null | tail -1 || echo "  Unable to ping"
echo ""

# -----------------------------------------------------------------------------
# 8. ROS2 Topic Test (if ROS2 is running)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}8. ROS2 Topic Diagnostics${NC}"
echo "-------------------------"
echo "Active topics:"
ros2 topic list 2>/dev/null | head -20 || echo "  ROS2 not running or no topics"
echo ""

# Check specific topics if they exist
for topic in "/stretch/odom" "/stretch/cmd_vel" "/camera/color/image_raw"; do
    if ros2 topic list 2>/dev/null | grep -q "^${topic}$"; then
        echo "Checking $topic..."
        timeout 3 ros2 topic hz $topic 2>/dev/null | head -5 || echo "  Topic not publishing"
    fi
done
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "=========================================="
echo -e "${YELLOW}Summary & Recommendations${NC}"
echo "=========================================="

ISSUES=0

# Check critical items
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ GPU not accessible - check --gpus all flag${NC}"
    ((ISSUES++))
fi

if [ -z "$FASTRTPS_DEFAULT_PROFILES_FILE" ] || [ ! -f "$FASTRTPS_DEFAULT_PROFILES_FILE" ]; then
    echo -e "${YELLOW}⚠ No FastRTPS SHM profile - DDS may use UDP (slower)${NC}"
    ((ISSUES++))
fi

SHM_AVAILABLE=$(df /dev/shm 2>/dev/null | tail -1 | awk '{print $4}')
if [ -n "$SHM_AVAILABLE" ] && [ "$SHM_AVAILABLE" -lt 2097152 ] 2>/dev/null; then
    echo -e "${YELLOW}⚠ Low shared memory - increase --shm-size${NC}"
    ((ISSUES++))
fi

if [ "$ISSUES" -eq 0 ]; then
    echo -e "${GREEN}✓ No obvious issues detected${NC}"
    echo ""
    echo "If still slow, the issue may be:"
    echo "  1. ROS2 discovery overhead (try restarting ros2 daemon)"
    echo "  2. Python GIL contention in your nodes"
    echo "  3. Suboptimal node communication patterns"
fi

echo ""
echo "=========================================="
