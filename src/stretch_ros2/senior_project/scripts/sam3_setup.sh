#!/bin/bash
#
# SAM3 + ROS2 Integration Setup Script
# 
# This script sets up SAM3 in a separate conda environment while keeping
# ROS2 compatibility through a client-server architecture.
#
# Usage:
#   chmod +x setup_sam3.sh
#   ./setup_sam3.sh
#

set -e

echo "=============================================="
echo "SAM3 + ROS2 Integration Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for conda
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    exit 1
fi

# Step 1: Create SAM3 conda environment
echo -e "\n${YELLOW}Step 1: Creating SAM3 conda environment...${NC}"
if conda env list | grep -q "^sam3 "; then
    echo "Environment 'sam3' already exists. Skipping creation."
else
    conda create -n sam3 python=3.12 -y
fi

# Step 2: Install dependencies in sam3 environment
echo -e "\n${YELLOW}Step 2: Installing PyTorch and dependencies...${NC}"
eval "$(conda shell.bash hook)"
conda activate sam3

# Check CUDA version
CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "none")
if [ "$CUDA_VERSION" = "none" ]; then
    echo -e "${YELLOW}Warning: No NVIDIA GPU detected. Installing CPU version.${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "CUDA detected. Installing GPU version..."
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
fi

# Install server dependencies
pip install fastapi uvicorn requests pillow numpy opencv-python

# Step 3: Clone and install SAM3
echo -e "\n${YELLOW}Step 3: Installing SAM3...${NC}"
SAM3_DIR="$HOME/sam3"
if [ -d "$SAM3_DIR" ]; then
    echo "SAM3 directory exists. Updating..."
    cd "$SAM3_DIR"
    git pull || true
else
    git clone https://github.com/facebookresearch/sam3.git "$SAM3_DIR"
    cd "$SAM3_DIR"
fi

pip install -e .
pip install -e ".[notebooks]" || true  # Optional, may fail

# Step 4: Setup Hugging Face authentication
echo -e "\n${YELLOW}Step 4: Hugging Face Authentication${NC}"
echo "=============================================="
echo "SAM3 requires access approval from Meta."
echo ""
echo "1. Go to: https://huggingface.co/facebook/sam3"
echo "2. Click 'Request access' and wait for approval"
echo "3. Create an access token at: https://huggingface.co/settings/tokens"
echo ""
read -p "Have you been approved and have a token ready? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install huggingface_hub
    echo "Running: huggingface-cli login"
    huggingface-cli login
fi

# Step 5: Copy server and client files
echo -e "\n${YELLOW}Step 5: Setting up server and ROS node...${NC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy sam3_server.py
if [ -f "$SCRIPT_DIR/sam3_server.py" ]; then
    cp "$SCRIPT_DIR/sam3_server.py" "$SAM3_DIR/"
    echo "Copied sam3_server.py to $SAM3_DIR/"
fi

# Go back to original environment
conda deactivate

# Step 6: Install ROS2 dependencies
echo -e "\n${YELLOW}Step 6: Installing ROS2 node dependencies...${NC}"
pip install requests --break-system-packages || pip install requests

# Copy ROS node to your package
ROS_PKG_DIR="$HOME/ament_ws/src/senior_project/senior_project"
if [ -d "$ROS_PKG_DIR" ]; then
    if [ -f "$SCRIPT_DIR/sam3_ros_node.py" ]; then
        cp "$SCRIPT_DIR/sam3_ros_node.py" "$ROS_PKG_DIR/"
        echo "Copied sam3_ros_node.py to $ROS_PKG_DIR/"
    fi
fi

# Step 7: Create launch helpers
echo -e "\n${YELLOW}Step 7: Creating helper scripts...${NC}"

# Create start server script
cat > "$HOME/start_sam3_server.sh" << 'EOF'
#!/bin/bash
# Start SAM3 Server
eval "$(conda shell.bash hook)"
conda activate sam3
cd ~/sam3
python sam3_server.py
EOF
chmod +x "$HOME/start_sam3_server.sh"

# Create test script
cat > "$HOME/test_sam3.sh" << 'EOF'
#!/bin/bash
# Test SAM3 installation
eval "$(conda shell.bash hook)"
conda activate sam3

echo "Testing PyTorch..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "Testing SAM3 import..."
python -c "from sam3.model_builder import build_sam3_image_model; print('SAM3 import OK!')"

echo ""
echo "To download model weights, run:"
echo "  python -c \"from sam3.model_builder import build_sam3_image_model; m = build_sam3_image_model()\""
EOF
chmod +x "$HOME/test_sam3.sh"

echo ""
echo -e "${GREEN}=============================================="
echo "Setup Complete!"
echo "==============================================${NC}"
echo ""
echo "To use SAM3 with ROS2:"
echo ""
echo "  Terminal 1 (SAM3 Server):"
echo "    ~/start_sam3_server.sh"
echo ""
echo "  Terminal 2 (ROS2):"
echo "    source ~/ament_ws/install/setup.bash"
echo "    ros2 launch senior_project RL.launch.py"
echo ""
echo "  Terminal 3 (SAM3 ROS Node):"
echo "    source ~/ament_ws/install/setup.bash"
echo "    ros2 run senior_project sam3_ros_node --ros-args -p prompt:=\"chair\""
echo ""
echo "Test installation:"
echo "    ~/test_sam3.sh"
echo ""
echo "Change segmentation prompt dynamically:"
echo "    ros2 topic pub /sam3_segmentation_node/set_prompt std_msgs/String \"data: 'person'\""
echo ""
echo "View segmentation in RViz:"
echo "    Add Image display for /sam3_segmentation_node/visualization"