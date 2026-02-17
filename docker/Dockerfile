# =============================================================================
# Stretch 2 RL Training Docker Container
# ROS2 Humble + MuJoCo (headless) + PyTorch CUDA + RViz
# PERFORMANCE OPTIMIZED VERSION
# =============================================================================

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Detroit

# =============================================================================
# Base System Setup
# =============================================================================

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    locales \
    git \
    build-essential \
    cmake \
    pkg-config \
    python3-pip \
    python3-dev \
    python3-venv \
    sudo \
    vim \
    nano \
    htop \
    tmux \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# =============================================================================
# ROS2 Humble Installation
# =============================================================================

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    ros-humble-ros-base \
    ros-humble-ament-cmake \
    ros-humble-ament-cmake-python \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init || true
RUN rosdep update --rosdistro humble

# =============================================================================
# ROS2 Additional Packages + Cyclone DDS (PERFORMANCE FIX)
# =============================================================================

RUN apt-get update && apt-get install -y \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-image-transport-plugins \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf-transformations \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-urdf \
    ros-humble-rviz2 \
    ros-humble-nav2-bringup \
    ros-humble-navigation2 \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-controller-manager \
    ros-humble-sensor-msgs-py \
    ros-humble-rosbag2 \
    ros-humble-rosbridge-suite \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Graphics / Display Dependencies (for RViz)
# =============================================================================

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libosmesa6-dev \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    mesa-utils \
    x11-apps \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# MuJoCo Installation
# =============================================================================

ENV MUJOCO_GL=egl

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    libegl1-mesa-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Dependencies - CRITICAL: Order and versions matter!
# =============================================================================

# Step 1: Upgrade pip only
RUN pip3 install --upgrade pip wheel

# Step 3: Install PyTorch with CUDA
# ---- Fix distutils/apt SymPy conflict (required for pip torch install) ----
RUN apt-get update && apt-get remove -y --purge python3-sympy \
 && rm -rf /var/lib/apt/lists/*

# ---- PyTorch (clean install, CUDA 12.1 wheels) ----
RUN pip3 uninstall -y torch torchvision torchaudio "nvidia-*" || true \
 && pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio


RUN pip3 install filelock fsspec jinja2 networkx sympy typing-extensions

RUN pip3 install --no-cache-dir --ignore-installed --force-reinstall "blinker>=1.6"

RUN apt-get update && apt-get remove -y python3-scipy && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --force-reinstall "scipy>=1.11,<1.13"


# Step 4: Core ML/RL packages
RUN pip3 install \
    gymnasium \
    matplotlib \
    pandas \
    tensorboard \
    tqdm \
    pyyaml \
    opencv-python-headless \
    Pillow \
    flask \
    flask-socketio \
    termcolor

# Numba with compatible coverage
RUN pip3 install "coverage<7"

# Step 5: MuJoCo
RUN pip3 install 'mujoco==3.2.6'

# Step 6: ROS2 Python extras
RUN pip3 install \
    transforms3d \
    empy==3.3.4 \
    lark \
    catkin_pkg

# Step 7: Robosuite (required by stretch_mujoco)
RUN pip3 install --no-deps robosuite
RUN pip3 install --no-deps git+https://github.com/robocasa/robocasa.git || echo "Robocasa source install attempted"

# Step 9: CRITICAL - Pin setuptools LAST for ROS2/colcon compatibility
RUN pip3 install setuptools==58.2.0

    # =============================================================================
# Create Non-Root User
# =============================================================================

ARG USERNAME=stretch
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# =============================================================================
# Workspace Setup
# =============================================================================

ENV AMENT_WS=/home/$USERNAME/ament_ws
RUN mkdir -p $AMENT_WS/src && chown -R $USERNAME:$USERNAME /home/$USERNAME

# =============================================================================
# Clone stretch_ros2 Repository
# =============================================================================

USER $USERNAME
WORKDIR $AMENT_WS/src

RUN git clone https://github.com/hello-robot/stretch_mujoco.git

# PERFORMANCE PATCH: Reduce camera resolution for Docker (424x240 -> 212x120)
RUN sed -i 's/width=424/width=212/g; s/height=240/height=120/g' \
    $AMENT_WS/src/stretch_mujoco/stretch_mujoco/enums/stretch_cameras.py

RUN git clone https://github.com/hello-robot/stretch_urdf.git

ARG CACHEBUST=1
RUN echo "Cache busting: $CACHEBUST"
# Clone your repo and move src contents
RUN git clone https://github.com/Sean9574/senior_project_docker.git /tmp/senior_project && \
    cp -r /tmp/senior_project/src/* . && \
    rm -rf /tmp/senior_project

# =============================================================================
# Install stretch_mujoco Python package
# =============================================================================

WORKDIR $AMENT_WS/src/stretch_mujoco
RUN pip3 install --no-deps -e .

# =============================================================================
# Install Python dependencies for stretch packages
# =============================================================================
# =============================================================================
# Create directories for checkpoints and data
# =============================================================================

RUN mkdir -p /home/stretch/rl_checkpoints && chown stretch:stretch /home/stretch/rl_checkpoints
RUN mkdir -p /home/stretch/models && chown stretch:stretch /home/stretch/models
RUN mkdir -p /home/stretch/logs && chown stretch:stretch /home/stretch/logs


USER root

# SAM3 server dependencies
RUN pip3 install  uvicorn fastapi pydantic ftfy huggingface_hub iopath regex timm
RUN pip3 install --no-deps git+https://github.com/facebookresearch/sam3.git

# SAM3 runtime deps (since we install sam3 with --no-deps to avoid torch/CUDA conflicts)
RUN pip3 install --no-cache-dir \
    einops \
    fastapi \
    uvicorn \
    pillow \
    pydantic \
    decord \
    pycocotools \
    urchin \
    pyquaternion \
    requests

# Pre-download SAM3 model as stretch user (so cache is in right place)
# Pre-download SAM3 model weights only (no GPU needed)
ARG HF_TOKEN=""
ENV HF_HOME=/home/stretch/.cache/huggingface
RUN mkdir -p /home/stretch/.cache/huggingface && \
    chown -R stretch:stretch /home/stretch/.cache

USER stretch
RUN HUGGINGFACE_HUB_TOKEN=$HF_TOKEN python3 -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('/home/stretch/.cache/huggingface', exist_ok=True); \
path = hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt'); \
print(f'Downloaded SAM3 to: {path}')"

USER root


RUN pip3 install --no-deps \
    hello-robot-stretch-body \
    hello-robot-stretch-body-tools \
    hello-robot-stretch-factory \
    hello-robot-stretch-urdf || true

RUN pip3 install pyrealsense2 || true

# Re-pin setuptools after any installs
RUN pip3 install setuptools==58.2.0

RUN apt-get update && apt-get install -y \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

    # --- Intel RealSense SDK (librealsense) userspace (no DKMS inside Docker) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
 && mkdir -p /etc/apt/keyrings \
 && curl -sSfL https://librealsense.intel.com/Debian/librealsense.pgp \
    | gpg --dearmor -o /etc/apt/keyrings/librealsense.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/librealsense.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    librealsense2-dev \
    librealsense2-utils \
 && rm -rf /var/lib/apt/lists/*

 RUN pip3 uninstall -y numba llvmlite numpy || true && \
    pip3 install --no-cache-dir --force-reinstall \
      numpy==1.23.5 \
      llvmlite==0.42.0 \
      numba==0.59.1

RUN chown -R $USERNAME:$USERNAME $AMENT_WS
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME


USER $USERNAME

# =============================================================================
# Build ROS2 Workspace
# =============================================================================

WORKDIR $AMENT_WS

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y || true"


# Create missing README files
# Fix setup.py in stretch_urdf to handle missing README
RUN find $AMENT_WS/src -name "setup.py" | xargs -I {} sh -c \
    'sed -i "s/with open.*README.md.*as fh:/long_description = \"\"/g; s/long_description = fh.read()//g" {}'


# Build workspace with FastRTPS (default) so realsense compiles
# We switch to Cyclone DDS at RUNTIME in entrypoint.sh for better Docker performance
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Copy URDF file to expected location
RUN mkdir -p $AMENT_WS/install/stretch_description/share/stretch_description/urdf && \
    cp $AMENT_WS/src/stretch_urdf/stretch_urdf/RE2V0/stretch_description_RE2V0_eoa_wrist_dw3_tool_sg3.urdf \
       $AMENT_WS/install/stretch_description/share/stretch_description/urdf/stretch.urdf || true


# Copy custom MuJoCo scene (10x10m arena with obstacles)
RUN cp $AMENT_WS/src/stretch_ros2/senior_project/config/scene.xml \
       $AMENT_WS/src/stretch_mujoco/stretch_mujoco/models/scene.xml

# =============================================================================
# Environment Setup - PERFORMANCE OPTIMIZED
# =============================================================================

RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc
RUN echo "source $AMENT_WS/install/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc
RUN echo "export ROS_DOMAIN_ID=0" >> /home/$USERNAME/.bashrc
# USE CYCLONE DDS FOR BETTER DOCKER PERFORMANCE
RUN echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/$USERNAME/.bashrc
RUN echo "export MUJOCO_GL=egl" >> /home/$USERNAME/.bashrc
RUN echo "export PATH=/usr/local/cuda/bin:\$PATH" >> /home/$USERNAME/.bashrc
RUN echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> /home/$USERNAME/.bashrc
# CUDNN BENCHMARK MODE FOR FASTER REPEATED OPERATIONS
RUN echo "export CUDNN_BENCHMARK=1" >> /home/$USERNAME/.bashrc

# =============================================================================
# Create directories for checkpoints and data
# =============================================================================

RUN mkdir -p /home/stretch/rl_checkpoints && chown stretch:stretch /home/stretch/rl_checkpoints
RUN mkdir -p /home/stretch/models && chown stretch:stretch /home/stretch/models
RUN mkdir -p /home/stretch/logs && chown stretch:stretch /home/stretch/logs

# =============================================================================
# Entrypoint
# =============================================================================

WORKDIR /home/$USERNAME

COPY --chown=stretch:stretch docker/entrypoint.sh /home/stretch/entrypoint.sh
RUN chmod +x /home/stretch/entrypoint.sh

ENTRYPOINT ["/home/stretch/entrypoint.sh"]
CMD ["bash"]