# Senior Project — ROS 2 Humble Docker Workspace

This repository contains a full ROS 2 Humble workspace packaged in Docker.  
It builds all packages with `colcon` and runs the `senior_project` launch system (Stretch + MuJoCo + RL).

Supports:

- Headless simulation
- RViz GUI
- Detached/background mode
- Optional GPU + hardware passthrough

---

## Requirements

- Linux 22.04
- Docker installed
- X11 desktop (for RViz)
- (Optional) NVIDIA GPU + `nvidia-container-toolkit`

---

# Clone the Repo

```bash
git clone git@github.com:Sean9574/senior_project_docker.git
cd senior_project_docker

docker build -f docker/Dockerfile -t senior_project:humble . --progress=plain

docker images | grep senior_project

# **Allow Docker access to X server**
xhost +local:root

# **Run and enter the container**
docker run --rm -it --name senior_project_run --net=host senior_project:humble bash

source /opt/ros/humble/setup.bash
source /ws/install/setup.bash

ros2 launch senior_project RL.launch.py

