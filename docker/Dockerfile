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
    termcolor \
    flask-cors \
    python-socketio


# Step 5: Robocasa (after torch to avoid conflicts)
RUN pip3 install --no-cache-dir robocasa || true


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
RUN pip3 install --no-deps robocasa || true
RUN pip3 install --no-deps git+https://github.com/robocasa/robocasa.git || true
RUN pip3 install --no-deps robosuite || true

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

# Pinned: the driver relies on StretchMujocoSimulator(start_translation=,
# start_rotation_quat=) for random spawns, and the sed patches below
RUN git clone https://github.com/hello-robot/stretch_mujoco.git && \
    git -C stretch_mujoco checkout c78d6a1cd79bb5ead1b679610299563c1a91124a

# ROBOCASA FIX: Replace robocasa_gen.py with a stub (we use custom scene.xml)
RUN echo '# Stub - robocasa kitchen generation disabled' > $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'LayoutType = None' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'StyleType = None' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'def get_styles(*args, **kwargs): return []' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'def layout_from_str(*args, **kwargs): return None' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'def layouts(*args, **kwargs): return []' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'def model_generation_wizard(*args, **kwargs): return None' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'def style_from_str(*args, **kwargs): return None' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo 'class RobocasaSceneBuilder:' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo '    def __init__(self, *args, **kwargs): pass' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py && \
    echo '    def generate(self, *args, **kwargs): return None' >> $AMENT_WS/src/stretch_mujoco/stretch_mujoco/robocasa_gen.py

# PERFORMANCE PATCH: Reduce camera resolution for Docker (424x240 -> 212x120)
RUN sed -i 's/width=424/width=212/g; s/height=240/height=120/g' \
    $AMENT_WS/src/stretch_mujoco/stretch_mujoco/enums/stretch_cameras.py

# VELOCITY PATCH: raise the wheel actuator limit from ±6 to ±30. The drivers
# send gear (3) x wheel speed, and a full-speed turn (0.3 m/s + 1.0 rad/s)
# needs about 27; ±6 would cap the robot far below cmd_vel
RUN sed -i 's/ctrlrange="-6 6"/ctrlrange="-30 30"/g' \
    $AMENT_WS/src/stretch_mujoco/stretch_mujoco/models/stretch.xml \
    $AMENT_WS/src/stretch_mujoco/stretch_mujoco/models/stretch_mj_3.3.0.xml

RUN git clone https://github.com/hello-robot/stretch_urdf.git

# NOTE: this repo's src/ is COPYed from the build context further down (after
# all pip/model layers), so the image always contains the code you build from
# and code edits don't invalidate the heavy layers.


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

# SAM3 turns the user's prompt into goals, so training needs it. It needs an
# approved Hugging Face token passed as a BuildKit secret (see README);
# --build-arg WITH_SAM3=0 builds a sim-only image without it.
ARG WITH_SAM3=1

# SAM3 server dependencies + runtime deps (sam3 itself --no-deps to avoid torch/CUDA conflicts)
RUN if [ "$WITH_SAM3" = "1" ]; then \
      pip3 install uvicorn fastapi pydantic ftfy huggingface_hub iopath regex timm && \
      pip3 install --no-deps git+https://github.com/facebookresearch/sam3.git && \
      pip3 install --no-cache-dir einops fastapi uvicorn pillow pydantic decord \
        pycocotools urchin pyquaternion requests ; \
    fi

# Pre-download SAM3 model weights as the stretch user (no GPU needed)
ENV HF_HOME=/home/stretch/.cache/huggingface
RUN mkdir -p /home/stretch/.cache/huggingface && \
    chown -R stretch:stretch /home/stretch/.cache

USER stretch
# Token is a BuildKit secret: available only during this RUN, never in image history
RUN --mount=type=secret,id=hf_token,uid=1000 \
    if [ "$WITH_SAM3" = "1" ]; then \
      if [ ! -s /run/secrets/hf_token ]; then \
        echo "ERROR: SAM3 needs a Hugging Face token: build with docker/docker_run.sh" \
             "(reads HF_TOKEN from ~/.hf.env) or --secret id=hf_token,env=HF_TOKEN;" \
             "or --build-arg WITH_SAM3=0 for a sim-only image." ; exit 1 ; \
      fi ; \
      HUGGINGFACE_HUB_TOKEN="$(cat /run/secrets/hf_token)" python3 -c "\
from huggingface_hub import hf_hub_download; \
path = hf_hub_download(repo_id='facebook/sam3', filename='sam3.pt'); \
print(f'Downloaded SAM3 to: {path}')" ; \
    fi

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

# --- Intel RealSense SDK (optional — not needed for MuJoCo sim) ---
# Wrapped in subshell so build continues if Intel repo is unreachable
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 lsb-release ca-certificates \
 && (mkdir -p /etc/apt/keyrings \
 && curl -sSfL --connect-timeout 10 --max-time 30 https://librealsense.intel.com/Debian/librealsense.pgp \
    | gpg --dearmor -o /etc/apt/keyrings/librealsense.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/librealsense.list \
 && apt-get update && apt-get install -y --no-install-recommends \
    librealsense2-dev \
    librealsense2-utils \
 || echo "WARNING: RealSense SDK install failed (Intel repo unreachable) — skipping, not needed for MuJoCo sim") \
 && rm -rf /var/lib/apt/lists/*

 RUN pip3 uninstall -y numba llvmlite numpy || true && \
    pip3 install --no-cache-dir --force-reinstall \
      numpy==1.23.5 \
      llvmlite==0.42.0 \
      numba==0.59.1

# This repo's workspace sources (filtered by .dockerignore)
COPY --chown=stretch:stretch src/ $AMENT_WS/src/

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


# Build workspace — skip realsense2_camera (not needed for MuJoCo sim,
# and librealsense2-dev may not be available if Intel repo was unreachable)
# We switch to Cyclone DDS at RUNTIME in entrypoint.sh for better Docker performance
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release \
    --packages-ignore realsense2_camera realsense2_camera_msgs realsense2_description"

# Copy URDF file to expected location
RUN mkdir -p $AMENT_WS/install/stretch_description/share/stretch_description/urdf && \
    cp $AMENT_WS/src/stretch_urdf/stretch_urdf/RE2V0/stretch_description_RE2V0_eoa_wrist_dw3_tool_sg3.urdf \
       $AMENT_WS/install/stretch_description/share/stretch_description/urdf/stretch.urdf


# Copy custom MuJoCo worlds next to stretch.xml (they include it relatively).
# scene.xml (10x10 m arena) replaces the default world every launch file uses;
# explore_arena.xml (30x30 m) is selectable with mujoco_xml:=<models>/explore_arena.xml
RUN cp $AMENT_WS/src/stretch_ros2/senior_project/config/scene.xml \
       $AMENT_WS/src/stretch_mujoco/stretch_mujoco/models/scene.xml && \
    cp $AMENT_WS/src/stretch_ros2/senior_project/config/can_label.png \
       $AMENT_WS/src/stretch_mujoco/stretch_mujoco/models/assets/can_label.png && \
    cp $AMENT_WS/src/stretch_ros2/senior_project/map/map.xml \
       $AMENT_WS/src/stretch_mujoco/stretch_mujoco/models/explore_arena.xml

# =============================================================================
# Environment Setup - PERFORMANCE OPTIMIZED
# =============================================================================

RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc
RUN echo "source $AMENT_WS/install/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc
# ROS_DOMAIN_ID comes from `docker run -e` (90 by default, away from the host's 0)
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