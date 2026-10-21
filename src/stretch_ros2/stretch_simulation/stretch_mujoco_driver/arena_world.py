"""
Build one MuJoCo world holding N walled rooms in a grid, one Stretch per room.

Room i is a copy of the room XML (config/scene.xml without its robot), with
walls raised so the head camera cannot see into the neighbouring rooms. Robot
i is the stretch_mujoco model attached with every name prefixed `r{i}_`; its
360 rangefinder sensors are removed (arena_core ray-casts from the same lidar
sites at the scan rate instead of every physics step).

Layout helpers are pure; build_arena() needs `mujoco` (imported lazily).
"""

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from stretch_mujoco_driver.spawn import Box, Circle

NUM_LIDAR_BEAMS = 360


def grid_shape(n: int) -> Tuple[int, int]:
    """(cols, rows) for n rooms: rows = floor(sqrt(n)), cols fill the rest."""
    rows = max(int(math.isqrt(n)), 1)
    cols = math.ceil(n / rows)
    return cols, rows


def room_origins(n: int, room_size: float = 10.0, gap: float = 2.0) -> List[Tuple[float, float]]:
    """World (x, y) of each room centre, filled row by row."""
    cols, _ = grid_shape(n)
    pitch = room_size + gap
    return [(float((i % cols) * pitch), float((i // cols) * pitch)) for i in range(n)]


@dataclass
class Room:
    index: int
    origin: Tuple[float, float]
    prefix: str
    base_body: int
    laser_body: int
    lidar_sites: List[int]
    camera_rgb: int
    camera_depth: int
    qpos_adr: int
    qvel_adr: int
    actuators: Dict[str, int]
    obstacles: list = field(default_factory=list)       # spawn.Box/Circle, room frame
    bounds: Tuple[float, float, float, float] = (-5.0, 5.0, -5.0, 5.0)  # room frame
    geoms: frozenset = frozenset()                      # world geoms of this room


@dataclass
class ArenaWorld:
    model: object
    rooms: List[Room]
    room_size: float
    gap: float


def _robot_spec(robot_xml: str):
    import mujoco
    text = open(robot_xml).read()
    # Keyframes don't survive a prefixed attach (MuJoCo warns and mangles them)
    text = re.sub(r"<keyframe>.*?</keyframe>", "", text, flags=re.S)
    spec = mujoco.MjSpec.from_string(text)
    assets = os.path.join(os.path.dirname(os.path.abspath(robot_xml)), "assets")
    spec.meshdir = assets
    spec.texturedir = assets
    for sensor in list(spec.sensors):
        if sensor.type == mujoco.mjtSensor.mjSENS_RANGEFINDER:
            sensor.delete()
    return spec


def _room_spec(room_xml: str, wall_height: float):
    import mujoco
    text = open(room_xml).read()
    text = re.sub(r'<include file="[^"]*stretch[^"]*\.xml"\s*/>', "", text)
    # One static carrier body so the whole room attaches (and moves) as a unit
    text = text.replace("<worldbody>", '<worldbody><body name="room">', 1)
    text = text.replace("</worldbody>", "</body></worldbody>", 1)
    spec = mujoco.MjSpec.from_string(text)
    for geom in spec.geoms:
        if geom.name.startswith("wall_"):
            geom.size = [geom.size[0], geom.size[1], wall_height / 2.0]
            geom.pos = [geom.pos[0], geom.pos[1], wall_height / 2.0]
    return spec


def _subtree(model, root: int) -> set:
    bodies = {root}
    for b in range(model.nbody):
        p = b
        while p > 0 and p not in bodies:
            p = model.body_parentid[p]
        if p in bodies:
            bodies.add(b)
    return bodies


def _room_footprints(model, data, geoms, origin):
    """Spawn obstacles (room frame) + room bounds from the floor plane."""
    import mujoco
    import numpy as np
    ox, oy = origin
    obstacles, bounds = [], None
    for g in sorted(geoms):
        gtype = model.geom_type[g]
        cx, cy = float(data.geom_xpos[g][0]) - ox, float(data.geom_xpos[g][1]) - oy
        size = model.geom_size[g]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            if size[0] > 0 and size[1] > 0:
                bounds = (cx - size[0], cx + size[0], cy - size[1], cy + size[1])
            continue
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            xmat = data.geom_xmat[g].reshape(3, 3)
            obstacles.append(Box(cx, cy, float(size[0]), float(size[1]),
                                 float(np.arctan2(xmat[1, 0], xmat[0, 0]))))
        elif gtype in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            obstacles.append(Circle(cx, cy, float(size[0])))
        else:
            obstacles.append(Circle(cx, cy, float(model.geom_rbound[g])))
    return obstacles, bounds


def build_arena(robot_xml: str, room_xml: str, n: int, room_size: float = 10.0,
                gap: float = 2.0, wall_height: float = 2.5) -> ArenaWorld:
    import mujoco
    robot = _robot_spec(robot_xml)
    room = _room_spec(room_xml, wall_height)

    arena = mujoco.MjSpec()
    arena.compiler.degree = robot.compiler.degree
    arena.meshdir = robot.meshdir
    arena.texturedir = robot.texturedir
    for key in ("timestep", "integrator", "impratio", "cone"):
        setattr(arena.option, key, getattr(robot.option, key))

    origins = room_origins(n, room_size, gap)
    for i, (ox, oy) in enumerate(origins):
        arena.worldbody.add_frame(pos=[ox, oy, 0.0]).attach_body(room.find_body("room"), f"room{i}_", "")
        arena.worldbody.add_frame(pos=[ox, oy, 0.0]).attach_body(robot.find_body("base_link"), f"r{i}_", "")
    model = arena.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    rooms = []
    for i, origin in enumerate(origins):
        p = f"r{i}_"
        base = bid(p + "base_link")
        joint = next(j for j in range(model.njnt) if model.jnt_bodyid[j] == base)
        room_bodies = _subtree(model, bid(f"room{i}_room"))
        robot_bodies = _subtree(model, base)
        room_geoms = frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] in room_bodies)
        robot_geoms = frozenset(g for g in range(model.ngeom) if model.geom_bodyid[g] in robot_bodies)
        obstacles, bounds = _room_footprints(model, data, room_geoms, origin)
        rooms.append(Room(
            index=i, origin=origin, prefix=p, base_body=base, laser_body=bid(p + "laser"),
            lidar_sites=[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{p}lidar{k:03d}")
                         for k in range(NUM_LIDAR_BEAMS)],
            camera_rgb=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, p + "d435i_camera_rgb"),
            camera_depth=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, p + "d435i_camera_depth"),
            qpos_adr=int(model.jnt_qposadr[joint]), qvel_adr=int(model.jnt_dofadr[joint]),
            actuators={model.actuator(a).name[len(p):]: a for a in range(model.nu)
                       if model.actuator(a).name.startswith(p)},
            obstacles=obstacles, bounds=bounds or (-room_size / 2, room_size / 2, -room_size / 2, room_size / 2),
            geoms=room_geoms | robot_geoms,
        ))
    return ArenaWorld(model=model, rooms=rooms, room_size=room_size, gap=gap)
