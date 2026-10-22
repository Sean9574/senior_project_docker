"""
Physics core of the grid arena (no ROS): one MjModel/MjData for N robots.

Per robot: wheel commands (same differential-drive conversion and limits as
the single-robot driver), a 360-beam lidar ray-cast from the model's own lidar
sites, room-frame odometry, and an in-place respawn that moves only that robot
and puts its arm and head back in the home pose.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

from stretch_mujoco_driver.spawn import sample_spawn_pose, yaw_to_quat

# stretch_mujoco config.robot_settings (c78d6a1)
WHEEL_RADIUS = 0.1016 / 2.0
WHEEL_SEPARATION = 0.3153
# Real Stretch base limits, same as stretch_mujoco_driver
MAX_LINEAR_MPS = 0.3
MAX_ANGULAR_RADPS = 1.0
# stretch.xml: <rangefinder ... cutoff="10.0"/>
LIDAR_CUTOFF_M = 10.0
# stretch.xml keyframe "home" (keyframes are stripped on attach, so set here)
HOME_CTRL = {"lift": 0.6, "arm": 0.1, "wrist_yaw": 0.0, "wrist_pitch": 0.0, "wrist_roll": 0.0,
             "gripper": 0.0, "head_pan": 0.0, "head_tilt": 0.0}


def clamp_cmd(v: float, w: float) -> Tuple[float, float]:
    return (max(-MAX_LINEAR_MPS, min(MAX_LINEAR_MPS, v)),
            max(-MAX_ANGULAR_RADPS, min(MAX_ANGULAR_RADPS, w)))


def wheel_speeds(v: float, w: float) -> Tuple[float, float]:
    """(left, right) wheel angular velocity, as stretch_mujoco.utils.diff_drive_inv_kinematics."""
    return ((v - w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS,
            (v + w * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS)


def steps_owed(sim_s: float, wall_elapsed: float, dt: float, max_steps: int) -> int:
    """Physics steps needed to bring simulated time up to wall time (pacing by
    deficit, so any timestep keeps 1.0x real time), capped per call so a
    stall doesn't turn into a long burst."""
    owed = int((wall_elapsed - sim_s) / dt + 1e-9)
    return max(0, min(owed, max_steps))


def _yaw(q) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Arena:
    def __init__(self, world, rng: Optional[np.random.Generator] = None,
                 clearance: float = 1.0, head_tilt: float = 0.0):
        import mujoco
        from stretch_mujoco_driver.arena_world import _subtree

        self._mj = mujoco
        self.world = world
        self.model = world.model
        self.data = mujoco.MjData(self.model)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.clearance = clearance
        self.cmd: List[Tuple[float, float]] = [(0.0, 0.0)] * len(world.rooms)
        self.spawn_fallback = False  # last respawn found no free spot

        # every velocity DOF of each robot, zeroed on respawn
        self._dofs = []
        for room in world.rooms:
            bodies = _subtree(self.model, room.base_body)
            dofs = []
            for j in range(self.model.njnt):
                if self.model.jnt_bodyid[j] in bodies:
                    n = {0: 6, 1: 3}.get(int(self.model.jnt_type[j]), 1)  # free, ball, slide/hinge
                    dofs.extend(range(self.model.jnt_dofadr[j], self.model.jnt_dofadr[j] + n))
            self._dofs.append(np.array(dofs, dtype=int))

        # home pose of each robot's arm and head (actuator targets and the
        # joint positions that meet them), restored on every respawn
        self._home_ctrl, self._home_qpos = [], []
        for room in world.rooms:
            ctrl, qpos = [], []
            for name, value in HOME_CTRL.items():
                if name in room.actuators:
                    a = room.actuators[name]
                    value = head_tilt if name == "head_tilt" else value
                    ctrl.append((a, value))
                    qpos.extend(self._target_qpos(a, value))
            self._home_ctrl.append(ctrl)
            self._home_qpos.append(qpos)
        for i in range(len(world.rooms)):
            self._go_home(i)
        mujoco.mj_forward(self.model, self.data)

    def _target_qpos(self, a: int, value: float) -> List[Tuple[int, float]]:
        """(qpos address, value) pairs that put actuator a's joint at its
        target; a fixed tendon (the telescoping arm) is split over its joints."""
        m, trn = self.model, int(self.model.actuator_trnid[a][0])
        if m.actuator_trntype[a] == self._mj.mjtTrn.mjTRN_JOINT:
            return [(int(m.jnt_qposadr[trn]), value)]
        if m.actuator_trntype[a] == self._mj.mjtTrn.mjTRN_TENDON:
            wraps = range(m.tendon_adr[trn], m.tendon_adr[trn] + m.tendon_num[trn])
            coef = {int(m.wrap_objid[w]): float(m.wrap_prm[w]) for w in wraps}
            norm = sum(c * c for c in coef.values())
            return [(int(m.jnt_qposadr[j]), value * c / norm) for j, c in coef.items()]
        return []

    def _go_home(self, i: int) -> None:
        for a, value in self._home_ctrl[i]:
            self.data.ctrl[a] = value
        for adr, value in self._home_qpos[i]:
            self.data.qpos[adr] = value

    # ------------------------------------------------------------ control
    def set_cmd(self, i: int, v: float, w: float) -> None:
        v, w = clamp_cmd(v, w)
        self.cmd[i] = (v, w)
        left, right = wheel_speeds(v, w)
        acts = self.world.rooms[i].actuators
        # Velocity actuators act on gear * joint velocity: scale by the gear,
        # or the robot drives at 1/gear of the commanded speed
        for name, speed in (("left_wheel_vel", left), ("right_wheel_vel", right)):
            a = acts[name]
            self.data.ctrl[a] = speed * self.model.actuator_gear[a][0]

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._mj.mj_step(self.model, self.data)

    # ------------------------------------------------------------ sensing
    def odom(self, i: int) -> Tuple[float, float, float, float, float]:
        """(x, y, yaw, v, w) of robot i in its room frame (room centre = origin)."""
        room = self.world.rooms[i]
        q = self.data.qpos[room.qpos_adr:room.qpos_adr + 7]
        qd = self.data.qvel[room.qvel_adr:room.qvel_adr + 6]
        yaw = _yaw(q[3:7])
        v = float(qd[0] * math.cos(yaw) + qd[1] * math.sin(yaw))  # free joint: linear vel in world frame
        return (float(q[0] - room.origin[0]), float(q[1] - room.origin[1]), yaw, v, float(qd[5]))

    def scan(self, i: int, return_geoms: bool = False):
        """360 ranges from robot i's lidar sites along each site's z axis
        (as the model's rangefinders do); miss = -1. Hits on anything outside
        robot i's room are reported as misses."""
        room = self.world.rooms[i]
        sites = room.lidar_sites
        pnt = self.data.site_xpos[sites[0]].copy()
        vec = self.data.site_xmat[sites].reshape(-1, 3, 3)[:, :, 2].ravel().copy()
        geomid = np.full(len(sites), -1, dtype=np.int32)
        dist = np.full(len(sites), -1.0)
        self._mj.mj_multiRay(self.model, self.data, pnt, vec, None, 1, room.laser_body,
                             geomid, dist, len(sites), LIDAR_CUTOFF_M)
        foreign = (geomid >= 0) & ~np.isin(geomid, list(room.geoms))
        dist[foreign] = -1.0
        # The model's rangefinders (cutoff 10) report far hits as exactly 10;
        # mj_multiRay's cutoff only prunes, so clamp to match
        dist[dist > LIDAR_CUTOFF_M] = LIDAR_CUTOFF_M
        geomid[foreign] = -1
        return (dist, geomid) if return_geoms else dist

    def camera_pose(self, i: int):
        """Pose of robot i's head colour camera as a ROS optical frame
        (x right, y down, z forward) in its room frame: (xyz, quat xyzw).
        Published as TF so the goal generator turns pixels into odom points
        exactly, whatever way the rendered image is rotated."""
        room = self.world.rooms[i]
        r_cam = self.data.cam_xmat[room.camera_rgb].reshape(3, 3)
        r_opt = r_cam @ np.diag([1.0, -1.0, -1.0])  # MuJoCo camera looks along -z, y up
        quat_wxyz = np.zeros(4)
        self._mj.mju_mat2Quat(quat_wxyz, r_opt.ravel())
        pos = self.data.cam_xpos[room.camera_rgb] - np.array([room.origin[0], room.origin[1], 0.0])
        w, x, y, z = quat_wxyz
        return pos.copy(), np.array([x, y, z, w])

    # ------------------------------------------------------------ episodes
    def respawn(self, i: int) -> Tuple[float, float, float]:
        """Move robot i to a random collision-free pose in its room (room
        frame), at rest. No other robot is touched. If no free spot is found
        it goes to the room centre and spawn_fallback is set."""
        room = self.world.rooms[i]
        pose = sample_spawn_pose(self.rng, room.obstacles, room.bounds, clearance=self.clearance)
        self.spawn_fallback = pose is None
        x, y, yaw = pose if pose is not None else (0.0, 0.0, 0.0)
        self.respawn_at(i, x, y, yaw)
        return x, y, yaw

    def respawn_at(self, i: int, x: float, y: float, yaw: float) -> None:
        """Put robot i at (x, y, yaw) in its room frame, at rest, arm and
        head in the home pose."""
        room = self.world.rooms[i]
        adr = room.qpos_adr
        self.data.qpos[adr:adr + 3] = [room.origin[0] + x, room.origin[1] + y, self.model.qpos0[adr + 2]]
        self.data.qpos[adr + 3:adr + 7] = yaw_to_quat(yaw)
        self.data.qvel[self._dofs[i]] = 0.0
        self._go_home(i)
        self.set_cmd(i, 0.0, 0.0)
        self._mj.mj_forward(self.model, self.data)
