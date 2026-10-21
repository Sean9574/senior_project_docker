# Senior Project: TD3 Navigation for the Stretch Robot

A Stretch robot in MuJoCo learns to drive to goal points with TD3 (ROS 2 Humble, PyTorch).
Each episode places a random goal the lidar can see. Goals start 1–1.5 m away
and move out to 4 m as the success rate rises. A live dashboard shows rewards
and outcomes, and an evaluation mode measures a saved policy.

Everything runs inside Docker. The host's ROS version does not matter (a Jazzy
host is fine).

---

## Requirements

- Linux with an X11 desktop (only needed for RViz / the MuJoCo viewer)
- Docker 23+ (BuildKit is the default)
- NVIDIA GPU + `nvidia-container-toolkit`

## Quick start

```bash
git clone https://github.com/Sean9574/senior_project_docker.git
cd senior_project_docker
./docker/docker_run.sh          # builds the image, then opens a shell in the container
```

Variants of the last command:

| Command | What it does |
|---|---|
| `DISPLAY= ./docker/docker_run.sh` | headless (no X11) |
| `DETACH=1 ./docker/docker_run.sh` | keep the container running in the background; enter with `docker exec -it senior_project bash` |
| `CLEAN=1 ./docker/docker_run.sh` | rebuild without cache |
| `WITH_SAM3=1 ./docker/docker_run.sh` | also install SAM3 camera goals (see below) |

For RViz, first allow the container to use your display: `xhost +local:`.

## Train

Inside the container:

```bash
ros2 launch senior_project RL.launch.py                  # with RViz
ros2 launch senior_project RL.launch.py headless:=true   # no RViz / viewer
```

Early log lines have this shape (values vary):

```
[LIDAR] Self-mask: <n>/360 beams hit the robot body
[GOAL] (<x>, <y>) dist=<d>m (curriculum max 1.5m)
[EP 0001] GOAL <GOAL_REACHED|TIMEOUT> | Return <r> | Steps <n> | ... | GoalMax 1.5m
```

Training runs in three phases: 10,000 steps of a scripted expert, 5,000 random
steps, then the TD3 policy (200,000 steps total by default, `total_steps:=N`).
Stop with Ctrl+C; the run resumes from the last checkpoint next time.

Dashboard: http://127.0.0.1:5555 (the container shares the host network).

## Outputs

Everything persists in `~/rl_checkpoints` on the host (`/home/stretch/rl_checkpoints` in the container):

| File | Contents |
|---|---|
| `td3_safe_rl_agent.pt` | networks, optimizers, step counters, goal curriculum |
| `replay_buffer.npz` | replay buffer (saved every 20 episodes and on Ctrl+C) |
| `monitor/sim_<n>/episodes.csv`, `steps.csv` | per-episode / per-step logs from the dashboard (one folder per sim, numbered in discovery order) |
| `monitor/figures/` | graphs, generated when the dashboard stops |
| `eval_<timestamp>.csv` | evaluation results |

## Evaluate a trained policy

```bash
ros2 launch senior_project RL.launch.py eval_episodes:=50 headless:=true use_reward_monitor:=false
```

This runs 50 episodes with the frozen policy (no exploration noise, no learning)
on a fixed, seeded set of goals up to 4 m away. It writes one CSV row per
episode (success, time, path length, SPL, bumps, minimum clearance) and logs a
summary line: `[EVAL] success=<rate> SPL=<0-1> bumps/ep=<n> -> <csv path>`.
The training checkpoint is never modified.

## Compare against the scripted expert

```bash
ROS_DOMAIN_ID=100 ros2 launch senior_project expert_baseline.launch.py
```

The expert gets the same world, the same goals and the same rewards. Its
episodes are tagged `EXPERT_BASELINE`, kept out of the RL learning curves, and
compared against RL in Fig. 10.

## Parallel training

```bash
ros2 run senior_project parallel_runner --num_sims 4
```

Sims use ROS domains 90, 91, …; checkpoints go in `~/rl_checkpoints/sim_N`.
`federated_sync.py` averages the six networks every 300 s, and each learner
loads the merged weights at its next episode end. One dashboard (sim 0)
watches every sim.

## SAM3 camera goals (optional)

Driving to objects the camera finds ("go to the person") needs Meta's gated
SAM3 model:

1. Request access to `facebook/sam3` on Hugging Face.
2. Put `HF_TOKEN=hf_...` in `~/.hf.env`.
3. Build with `WITH_SAM3=1 ./docker/docker_run.sh`. The token is passed as a
   BuildKit secret and is not stored in the image.
4. Run `ros2 launch senior_project sam3_navigation.launch.py target:=person`.

## Other launch files

- `launch_sim.launch.py`: the simulator alone (RViz + MuJoCo viewer, no learner).
- `mujoco_xml:=/home/stretch/ament_ws/src/stretch_mujoco/stretch_mujoco/models/explore_arena.xml`:
  the larger 30 × 30 m arena instead of the default 10 × 10 m one.

## ROS domains

The container uses `ROS_DOMAIN_ID=90` (`DOMAIN=N ./docker/docker_run.sh` to
change it), parallel sims use 90–99, and the expert baseline uses 100. Host ROS
terminals normally use 0, so they never see the training traffic. Don't point
a host RViz (for example, Jazzy) at these domains: Humble and Jazzy are not
guaranteed to talk to each other. Use the container's RViz.

## Tests

The learning logic (lidar cleaning, goal sampling, curriculum, action limits,
dashboard episode accounting) has unit tests that don't need ROS:

```bash
python3 -m pytest src/stretch_ros2/senior_project/test
```

GitHub Actions runs them on Python 3.10 and 3.12.

## Running outside Docker

`requirements.txt` lists the Python packages. A native Jazzy build has not been
tested: `stretch_simulation` depends on Humble-era Hello Robot packages
(`stretch_core`, `laser_filters`, `rplidar_ros`).
