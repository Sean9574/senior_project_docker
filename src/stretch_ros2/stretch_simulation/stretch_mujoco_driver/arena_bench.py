#!/usr/bin/env python3
"""
Grid arena speed check (no ROS): build the N-room world, then time physics,
lidar ray-casts and (optionally) camera renders. One result line per metric.

  python3 -m stretch_mujoco_driver.arena_bench --robots 8
  MUJOCO_GL=egl python3 -m stretch_mujoco_driver.arena_bench --robots 8 --cameras
"""

import argparse
import os
import time

import numpy as np

from stretch_mujoco_driver.arena_core import Arena
from stretch_mujoco_driver.arena_world import build_arena

CONTROL_HZ = 10.0
SCAN_HZ = 10.0


def _default_paths():
    from stretch_mujoco import utils
    robot = os.path.join(os.path.dirname(utils.default_scene_xml_path), "stretch.xml")
    try:
        from ament_index_python.packages import get_package_share_directory
        room = os.path.join(get_package_share_directory("senior_project"), "config", "scene.xml")
    except Exception:
        room = os.path.join(os.path.dirname(__file__), "..", "..", "senior_project", "config", "scene.xml")
    return robot, room


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=5.0, help="simulated seconds to time")
    parser.add_argument("--cameras", action="store_true", help="also time head-camera renders (needs GL)")
    parser.add_argument("--timestep", type=float, default=0.004, help="physics step (arena_sim default)")
    parser.add_argument("--robot-xml", default="")
    parser.add_argument("--room-xml", default="")
    args = parser.parse_args()
    robot_xml, room_xml = _default_paths() if not (args.robot_xml and args.room_xml) else (args.robot_xml, args.room_xml)

    t0 = time.perf_counter()
    world = build_arena(robot_xml, room_xml, args.robots)
    world.model.opt.timestep = args.timestep
    arena = Arena(world, rng=np.random.default_rng(0))
    for i in range(args.robots):
        arena.respawn(i)
        arena.set_cmd(i, 0.3, 0.3)
    print(f"build      {time.perf_counter() - t0:.2f} s for {args.robots} robots")

    steps = int(args.seconds / world.model.opt.timestep)
    t0 = time.perf_counter()
    arena.step(steps)
    wall = time.perf_counter() - t0
    print(f"physics    {wall / steps * 1000:.3f} ms/step   real-time factor {args.seconds / wall:.2f}x")

    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        for i in range(args.robots):
            arena.scan(i)
    per_scan = (time.perf_counter() - t0) / (reps * args.robots)
    print(f"lidar      {per_scan * 1000:.2f} ms/scan   {per_scan * SCAN_HZ * args.robots * 100:.0f}% of a core at {SCAN_HZ:.0f} Hz")

    if args.cameras:
        import mujoco
        r = mujoco.Renderer(world.model, width=212, height=120)
        t0 = time.perf_counter()
        for _ in range(reps):
            for room in world.rooms:
                r.update_scene(arena.data, camera=room.camera_rgb)
                r.render()
        per = (time.perf_counter() - t0) / (reps * args.robots)
        print(f"camera     {per * 1000:.2f} ms/render   max {1.0 / (per * args.robots):.1f} Hz per robot (RGB only)")

    budget = 1.0 / CONTROL_HZ
    load = wall / args.seconds + per_scan * SCAN_HZ * args.robots
    print(f"verdict    {'OK' if load < 0.8 else 'TOO SLOW'}: physics+lidar use {load * 100:.0f}% of one core "
          f"(control period {budget * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
