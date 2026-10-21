# Senior Project: Prompted Object Navigation for the Stretch Robot

A Stretch robot in MuJoCo learns, with TD3, to go to an object you name.
When training starts, the system asks **"What should the robot go to?"** and
SAM3 finds that object in the robot's camera image. Every episode the robot
starts at a random free spot, explores until it sees the object, and then
drives to it. A live dashboard shows rewards and outcomes, and an evaluation
mode measures a saved policy.

Everything runs inside Docker. The host's ROS version does not matter (a Jazzy
host is fine).

---

## Requirements

- Linux with an X11 desktop (only needed for RViz / the MuJoCo viewer)
- Docker 23+ (BuildKit is the default)
- NVIDIA GPU + `nvidia-container-toolkit` (the sim, SAM3 and training share it)
- Access to Meta's gated SAM3 model:
  1. Request access to `facebook/sam3` on Hugging Face.
  2. Put `HF_TOKEN=hf_...` in `~/.hf.env`.

## Quick start

```bash
git clone https://github.com/Sean9574/senior_project_docker.git
cd senior_project_docker
./docker/docker_run.sh          # builds the image (with SAM3), then opens a shell in the container
```

The token goes into the build as a BuildKit secret and is not stored in the image.

Variants of the last command:

| Command | What it does |
|---|---|
| `DISPLAY= ./docker/docker_run.sh` | headless (no X11) |
| `DETACH=1 ./docker/docker_run.sh` | keep the container running in the background; enter with `docker exec -it senior_project bash` |
| `CLEAN=1 ./docker/docker_run.sh` | rebuild without cache |
| `WITH_SAM3=0 ./docker/docker_run.sh` | sim-only image, no SAM3 and no token needed (no prompted goals) |

For RViz, first allow the container to use your display: `xhost +local:`.

## Train

Inside the container:

```bash
ros2 launch senior_project sam3_navigation.launch.py
```

```
What should the robot go to? (e.g. chair, potted plant, blue cylinder): chair
```

Pass `target:=chair` to skip the question (needed when there is no terminal to
ask in). The answer is kept for every episode, so training runs unattended. To
switch targets mid-run:

```bash
ros2 topic pub --once /sam3_goal_generator/set_target std_msgs/String "data: 'potted plant'"
```

The terminal turns into a live status panel (redrawn every half second):

```
 Stretch TD3  target "chair"  domain 90  up 01:08:42
 ------------------------------------------------------
 Phase    RL_POLICY   48,210 / 200,000  [##........] 24%
          ETA 3h48m   ~11 steps/s
 Episode  #412  GOAL-SEEK  12.4s  goal 1.8m  clr 0.62m
 Last 20  ++++++-+++++-+++-+++   success 85%  seen 95%
 Return   avg20 +5.8   best +10.5
 Health   SAM3 ok "chair" | step 0.11s | reset 3.2s
          ckpt saved 2m00s ago
 Alerts   14:17:18 WARN learner: replay save failed

 logs: /home/stretch/rl_checkpoints/logs   (Ctrl+C stops training)
```

- **Last 20:** `+` reached the object, `-` timed out. *seen* is how often SAM3 spotted it at all.
- **Health:** `SAM3 NOT CONFIRMED` means the server never accepted your prompt. If the
  learner stops reporting for 30 s, a `learner silent` line appears.
- **Alerts:** the latest warnings and errors from any node; repeats are counted (`x3`), not repeated.
- **Logs:** full node output goes to `~/rl_checkpoints/logs` on the host. Add
  `status_panel:=false` to get the old scrolling output instead. When output is
  redirected (`docker logs`, `nohup`), the panel prints one summary line every 30 s.

What happens each episode:

1. The robot spawns at a random pose at least 1 m from every obstacle (`[SPAWN] x=… y=… yaw=…`).
2. Until SAM3 sees the object there is no goal, and the robot is rewarded for exploring.
3. Once SAM3 sees it, the goal appears and the robot is rewarded for closing the distance.
4. The episode ends when the robot is within 0.45 m of the goal, or after 45 s.

Objects in the default 10 × 10 m world: blue cylinder on a table, chair,
bookshelf, potted plant, stool, trash can, coffee table, storage box.

Training runs in three phases: 10,000 steps of a scripted expert, 5,000 random
steps, then the TD3 policy (200,000 steps total by default, `total_steps:=N`).
Stop with Ctrl+C; the run resumes from the last checkpoint next time.

Dashboard: http://127.0.0.1:5555 (the container shares the host network).

## Outputs

Everything persists in `~/rl_checkpoints` on the host (`/home/stretch/rl_checkpoints` in the container):

| File | Contents |
|---|---|
| `td3_safe_rl_agent.pt` | networks, optimizers, step counters |
| `replay_buffer.npz` | replay buffer (saved every 20 episodes and on Ctrl+C) |
| `monitor/sim_<n>/episodes.csv`, `steps.csv` | per-episode / per-step logs from the dashboard (one folder per sim, numbered in discovery order) |
| `monitor/figures/` | graphs, generated when the dashboard stops |
| `eval_<timestamp>.csv` | evaluation results |

## Evaluate a trained policy

```bash
ros2 launch senior_project sam3_navigation.launch.py target:=chair eval_episodes:=50 headless:=true use_reward_monitor:=false
```

This runs 50 episodes with the frozen policy (no exploration noise, no
learning) from random spawns. It refuses to run if no trained checkpoint is
found. Each CSV row records:

- whether SAM3 ever saw the object (`goal_seen`, `time_to_seen_s`)
- success and episode time
- SPL: path efficiency, measured from the moment the object was first seen (0 for failures)
- bumps (counted at most once per 8 steps) and the minimum clearance

The summary line looks like `[EVAL] seen=<rate> success=<rate> SPL=<0-1> bumps/ep=<n> -> <csv path>`.

## Compare against the scripted expert

Start a SAM3 server if none is running, then launch the expert on its own domain:

```bash
python3 -m senior_project.sam3_server --port 8100 &
ROS_DOMAIN_ID=100 ros2 launch senior_project expert_baseline.launch.py
```

It asks for the same prompt and gets the same random spawns and rewards. Its
episodes are tagged `EXPERT_BASELINE`, kept out of the RL learning curves, and
compared against RL in Fig. 10.

## Parallel training

Start one shared SAM3 server, then the sims:

```bash
python3 -m senior_project.sam3_server --port 8100 &
ros2 run senior_project parallel_runner --num_sims 4      # asks what to go to
```

- **Shared target:** every sim chases the same target.
- **Domains and checkpoints:** sims use ROS domains 90, 91, …, and checkpoints go in `~/rl_checkpoints/sim_N`.
- **Weight sharing:** `federated_sync.py` averages the six networks every 300 s, and each learner loads the merged weights at its next episode end.
- **One dashboard:** sim 0's dashboard watches every sim.

## Other launch files

- `launch_sim.launch.py`: the simulator alone (RViz + MuJoCo viewer, fixed start, no learner).
- `random_spawn:=false` on the SAM3 launch: always start at the world's default pose.
- `mujoco_xml:=/home/stretch/ament_ws/src/stretch_mujoco/stretch_mujoco/models/explore_arena.xml`:
  the larger 30 × 30 m arena instead of the default 10 × 10 m one.

## ROS domains

The container uses `ROS_DOMAIN_ID=90` (`DOMAIN=N ./docker/docker_run.sh` to
change it), parallel sims use 90–99, and the expert baseline uses 100. Host ROS
terminals normally use 0, so they never see the training traffic. Don't point
a host RViz (for example, Jazzy) at these domains: Humble and Jazzy are not
guaranteed to talk to each other. Use the container's RViz.

## Tests

The learning logic has unit tests that don't need ROS: lidar cleaning, action
limits, rewards, spawn sampling, the prompt question and dashboard episode
accounting.

```bash
python3 -m pytest src/stretch_ros2/senior_project/test src/stretch_ros2/stretch_simulation/test
```

GitHub Actions runs them on Python 3.10 and 3.12.

## Running outside Docker

`requirements.txt` lists the Python packages. A native Jazzy build has not been
tested: `stretch_simulation` depends on Humble-era Hello Robot packages
(`stretch_core`, `laser_filters`, `rplidar_ros`).
