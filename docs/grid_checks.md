# Grid arena checks (run on the GPU server)

Run these inside the container (`./docker/docker_run.sh`, then a shell in it).
Each prints a few lines ending in `verdict`. Paste the output back.

## 1. Physics and lidar speed (no GPU needed)

```bash
python3 -m stretch_mujoco_driver.arena_bench --robots 8
```

Needs `verdict OK`: physics plus lidar use under 80% of one CPU core at a
4 ms physics step. On a 2024 laptop CPU it was about 100% (`TOO SLOW`), so
this decides whether 8 robots fit in one process on the server.

## 2. Camera renders

```bash
MUJOCO_GL=egl python3 -m stretch_mujoco_driver.arena_bench --robots 8 --cameras
```

The `camera` line shows the most frames per second each robot can get.
It should be well above 1 Hz (the per-robot SAM3 rate at 8 robots).

## 3. SAM3 throughput with 8 robots sharing one server

```bash
python3 -m senior_project.sam3_server --port 8100 &   # wait for "Uvicorn running"
python3 -m senior_project.sam3_bench --robots 8 --seconds 30
```

Needs `verdict OK`: each robot gets at least the planned 1 Hz.

## 4. First full launch

```bash
ros2 launch senior_project grid_navigation.launch.py target:=chair num_robots:=2
```

Watch the status panel for 2 minutes. Healthy looks like:
- both robot rows change episode numbers
- `SAM3 ok 2/2 "chair"`
- `step` about 0.10-0.15 s
- no `learner silent`

Then repeat with `num_robots:=8`. If anything goes wrong, the full logs are
in `~/rl_checkpoints/logs` on the host.
