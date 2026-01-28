# syntax=docker/dockerfile:1
FROM ros:humble-ros-base-jammy

SHELL ["/bin/bash", "-lc"]
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ENV WS=/ws

# -----------------------------
# 1) System + ROS build tooling
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# rosdep setup
RUN rosdep init 2>/dev/null || true && rosdep update

# -----------------------------
# 2) Install uv and stretch_mujoco (as root)
# -----------------------------
  RUN python3 -m pip install --no-cache-dir uv && \
    uv pip install --system "hello-robot-stretch-mujoco[robocasa] @ git+https://github.com/hello-robot/stretch_mujoco.git" && \
    uv pip install --system "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git" --no-deps && \
    uv pip install --system "robocasa @ git+https://github.com/robocasa/robocasa.git" --no-deps
# -----------------------------
# 3) Create non-root user
# -----------------------------
ARG USERNAME=sean
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends sudo \
 && groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
 && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && rm -rf /var/lib/apt/lists/*

# -----------------------------
# 4) Workspace + copy sources
# -----------------------------
RUN mkdir -p ${WS}/src
WORKDIR ${WS}


# Copy workspace contents
COPY ./src ${WS}/src
COPY requirements.txt ${WS}/requirements.txt
COPY deps.repo[s] ${WS}/

# Give ownership to the non-root user
RUN chown -R ${USERNAME}:${USERNAME} ${WS}

# -----------------------------
# 5) Import source deps + pip deps as non-root
# -----------------------------
USER ${USERNAME}

# Import external source dependencies
RUN if [ -f "${WS}/deps.repos" ]; then \
      cd ${WS}/src && vcs import < ${WS}/deps.repos ; \
    else \
      echo "No deps.repos found; skipping vcs import."; \
    fi

# Install Python deps (excluding stretch_mujoco since it's already installed)
RUN if [ -f "${WS}/requirements.txt" ]; then \
      grep -v "stretch_mujoco" ${WS}/requirements.txt > /tmp/requirements_filtered.txt && \
      python3 -m pip install --no-cache-dir -r /tmp/requirements_filtered.txt ; \
    else \
      echo "No requirements.txt found; skipping pip install."; \
    fi

# -----------------------------
# 6) rosdep + build (root for apt installs)
# -----------------------------
USER root

RUN apt-get update && \
    source /opt/ros/${ROS_DISTRO}/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y --rosdistro ${ROS_DISTRO} && \
    rm -rf /var/lib/apt/lists/*

# Build the workspace
RUN source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install

# -----------------------------
# 7) Entrypoint that sources ROS + overlay
# -----------------------------
RUN printf '%s\n' \
'#!/bin/bash' \
'set -e' \
'source /opt/ros/'${ROS_DISTRO}'/setup.bash' \
'if [ -f /ws/install/setup.bash ]; then source /ws/install/setup.bash; fi' \
'exec "$@"' \
> /entrypoint.sh \
 && chmod +x /entrypoint.sh

# Default back to non-root for interactive use
USER ${USERNAME}
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]