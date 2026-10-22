"""Arena physics core: wheel commands, ray-cast lidar, room-frame odom, respawn.

Kinematics tests are pure. Model tests need `mujoco` + $STRETCH_MUJOCO_MODELS.
Run: STRETCH_MUJOCO_MODELS=<.../stretch_mujoco/models> python3 -m pytest src/stretch_ros2/stretch_simulation/test/test_arena_core.py
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stretch_mujoco_driver import arena_core  # noqa: E402

ROOM_XML = os.path.join(os.path.dirname(__file__), "..", "..", "senior_project", "config", "scene.xml")
MODELS = os.environ.get("STRETCH_MUJOCO_MODELS", "")
needs_mujoco = pytest.mark.skipif(
    not MODELS or not os.path.isdir(MODELS), reason="set STRETCH_MUJOCO_MODELS to run model tests")


def test_wheel_speeds_match_stretch_mujoco():
    # upstream: R = 0.1016/2, L = 0.3153; w_l = (v - w L/2)/R, w_r = (v + w L/2)/R
    wl, wr = arena_core.wheel_speeds(0.3, 1.0)
    assert wl == pytest.approx((0.3 - 0.3153 / 2) / 0.0508)
    assert wr == pytest.approx((0.3 + 0.3153 / 2) / 0.0508)


def test_commands_are_clamped_to_stretch_limits():
    assert arena_core.clamp_cmd(5.0, -5.0) == (0.3, -1.0)
    assert arena_core.clamp_cmd(-0.1, 0.2) == (-0.1, 0.2)


def _arena(n, **kw):
    pytest.importorskip("mujoco")
    from stretch_mujoco_driver import arena_world
    world = arena_world.build_arena(os.path.join(MODELS, "stretch.xml"), ROOM_XML, n, **kw)
    return arena_core.Arena(world, rng=np.random.default_rng(0))


@needs_mujoco
def test_driving_forward_moves_only_that_robot():
    a = _arena(2)
    a.set_cmd(0, 0.3, 0.0)
    a.step(500)  # 1 s at the model's 2 ms timestep
    x0, y0, yaw0, v0, _ = a.odom(0)
    x1, y1, _, _, _ = a.odom(1)
    forward = x0 * math.cos(yaw0) + y0 * math.sin(yaw0)
    # Same wheel conversion as stretch_mujoco / the single-robot driver; the
    # model's wheel gear (3) makes the real speed about a third of the command
    assert forward > 0.05
    assert math.hypot(x1, y1) < 0.02


@needs_mujoco
def test_positive_turn_rate_turns_counter_clockwise():
    a = _arena(1)
    a.set_cmd(0, 0.0, 1.0)
    a.step(250)
    assert a.odom(0)[2] > 0.1


@needs_mujoco
def test_odom_is_in_the_room_frame():
    a = _arena(4)
    for i in range(4):
        x, y, _, _, _ = a.odom(i)
        assert math.hypot(x, y) < 1e-6  # every robot starts at its own room centre


@needs_mujoco
def test_respawn_moves_only_that_robot_and_keeps_clearance():
    from stretch_mujoco_driver.spawn import distance_to_obstacle
    a = _arena(2)
    a.set_cmd(1, 0.3, 0.0)
    a.step(50)
    before = a.data.qpos.copy(), a.data.qvel.copy()
    r1 = a.world.rooms[1]
    for _ in range(20):
        x, y, yaw = a.respawn(0)
        assert min(distance_to_obstacle(x, y, o) for o in a.world.rooms[0].obstacles) >= 1.0
        ox, oy, oyaw, v, w = a.odom(0)
        assert (ox, oy) == pytest.approx((x, y), abs=1e-6) and v == pytest.approx(0.0, abs=1e-9)
    sl = slice(r1.qpos_adr, r1.qpos_adr + 7)
    assert np.array_equal(a.data.qpos[sl], before[0][sl])
    assert np.array_equal(a.data.qvel[r1.qvel_adr:r1.qvel_adr + 6], before[1][r1.qvel_adr:r1.qvel_adr + 6])


@needs_mujoco
def test_lidar_never_sees_another_room():
    a = _arena(2)
    for _ in range(30):
        a.respawn(0)
        a.respawn(1)
        for i in range(2):
            scan, hit_geoms = a.scan(i, return_geoms=True)
            assert scan.shape == (360,)
            assert all(g in a.world.rooms[i].geoms for g in hit_geoms if g >= 0)


@needs_mujoco
def test_ray_cast_scan_matches_the_models_own_rangefinders():
    import mujoco
    from stretch_mujoco_driver import arena_world
    world = arena_world.build_arena(os.path.join(MODELS, "stretch.xml"), ROOM_XML, 1, keep_rangefinders=True)
    a = arena_core.Arena(world, rng=np.random.default_rng(3))
    rf = [s for s in range(world.model.nsensor)
          if world.model.sensor_type[s] == mujoco.mjtSensor.mjSENS_RANGEFINDER]
    assert len(rf) == 360
    for _ in range(5):
        a.respawn(0)
        sensor = np.array([a.data.sensordata[world.model.sensor_adr[s]] for s in rf])
        ours = a.scan(0)
        both = (sensor > 0) & (ours > 0)
        assert both.sum() > 300
        assert np.max(np.abs(sensor[both] - ours[both])) < 0.01


@needs_mujoco
def test_respawn_at_places_the_robot_at_rest_in_room_frame():
    a = _arena(2)
    a.set_cmd(1, 0.3, 0.0)
    a.step(100)
    a.respawn_at(1, 1.5, -2.0, math.pi / 2)
    x, y, yaw, v, w = a.odom(1)
    assert (x, y, yaw) == pytest.approx((1.5, -2.0, math.pi / 2), abs=1e-6)
    assert v == pytest.approx(0.0, abs=1e-9) and w == pytest.approx(0.0, abs=1e-9)
    assert a.cmd[1] == (0.0, 0.0)


def _joint_qpos(a, name):
    import mujoco
    j = mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return a.data.qpos[a.model.jnt_qposadr[j]]


def _set_joint_qpos(a, name, value):
    import mujoco
    j = mujoco.mj_name2id(a.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    a.data.qpos[a.model.jnt_qposadr[j]] = value


@needs_mujoco
def test_respawn_puts_arm_and_head_back_in_the_home_pose():
    a = _arena(2)
    # last episode left robot 0's lift low, arm in and head turned
    for name, value in (("joint_lift", 0.1), ("joint_head_pan", 1.0), ("joint_head_tilt", -0.5),
                        ("joint_arm_l0", 0.0), ("joint_arm_l1", 0.0)):
        _set_joint_qpos(a, "r0_" + name, value)
    _set_joint_qpos(a, "r1_joint_lift", 0.2)
    a.respawn_at(0, 0.0, 0.0, 0.0)
    home = arena_core.HOME_CTRL
    assert _joint_qpos(a, "r0_joint_lift") == pytest.approx(home["lift"])
    assert _joint_qpos(a, "r0_joint_head_pan") == pytest.approx(home["head_pan"])
    assert _joint_qpos(a, "r0_joint_head_tilt") == pytest.approx(0.0)  # Arena head_tilt default
    arm = sum(_joint_qpos(a, f"r0_joint_arm_l{k}") for k in range(4))
    assert arm == pytest.approx(home["arm"])
    assert _joint_qpos(a, "r1_joint_lift") == pytest.approx(0.2)  # other robot untouched


@needs_mujoco
def test_respawn_reports_when_it_falls_back_to_the_room_centre():
    a = _arena(1)
    a.respawn(0)
    assert a.spawn_fallback is False
    a.clearance = 100.0  # no spot in a 10 m room is 100 m from everything
    assert a.respawn(0) == (0.0, 0.0, 0.0)
    assert a.spawn_fallback is True


def _quat_xyzw_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@needs_mujoco
def test_camera_optical_frame_looks_where_the_robot_faces():
    a = _arena(2)
    for yaw in (0.0, math.pi / 2):
        a.respawn_at(1, 1.0, 2.0, yaw)
        pos, quat = a.camera_pose(1)
        assert np.linalg.norm(quat) == pytest.approx(1.0)
        z_opt = _quat_xyzw_to_mat(quat)[:, 2]          # optical z = viewing direction
        assert z_opt @ np.array([math.cos(yaw), math.sin(yaw), 0.0]) > 0.7
        assert math.hypot(pos[0] - 1.0, pos[1] - 2.0) < 0.5   # on the robot, in the room frame
        assert 0.8 < pos[2] < 1.6                              # head height


@needs_mujoco
def test_camera_optical_frame_matches_mujoco_camera_axes():
    a = _arena(1)
    room = a.world.rooms[0]
    pos, quat = a.camera_pose(0)
    r_cam = a.data.cam_xmat[room.camera_rgb].reshape(3, 3)
    r_opt = _quat_xyzw_to_mat(quat)
    # ROS optical: x right, y down, z forward; MuJoCo camera: x right, y up, looks along -z
    assert np.allclose(r_opt[:, 0], r_cam[:, 0], atol=1e-6)
    assert np.allclose(r_opt[:, 1], -r_cam[:, 1], atol=1e-6)
    assert np.allclose(r_opt[:, 2], -r_cam[:, 2], atol=1e-6)


@needs_mujoco
def test_commanded_speed_is_the_real_speed():
    # The wheel actuators have gear 3: the command must be scaled by it, or the
    # robot drives at a third of the requested speed (0.089 m/s for 0.3 m/s)
    a = _arena(1)
    a.set_cmd(0, 0.3, 0.0)
    a.step(int(2.0 / a.model.opt.timestep))
    assert a.odom(0)[3] == pytest.approx(0.3, rel=0.15)


@needs_mujoco
def test_commanded_turn_rate_is_the_real_turn_rate():
    a = _arena(1)
    a.set_cmd(0, 0.0, 1.0)
    a.step(int(2.0 / a.model.opt.timestep))
    assert a.odom(0)[4] == pytest.approx(1.0, rel=0.15)



def test_paced_physics_keeps_up_with_wall_time_for_any_timestep():
    # 10 ms ticks with a 4 ms step: rounding a fixed substep count ran at 0.8x
    for dt in (0.002, 0.004, 0.003):
        sim = 0.0
        for k in range(1, 1001):
            n = arena_core.steps_owed(sim, wall_elapsed=k * 0.01, dt=dt, max_steps=50)
            sim += n * dt
        assert abs(sim - 10.0) <= dt


def test_paced_physics_never_bursts_more_than_the_cap():
    assert arena_core.steps_owed(0.0, wall_elapsed=5.0, dt=0.004, max_steps=25) == 25


@needs_mujoco
def test_far_beams_clamp_to_the_rangefinder_cutoff():
    a = _arena(1)
    a.respawn_at(0, -4.5, -4.5, math.pi / 4)   # corner: diagonal beams see > 10 m
    scan = a.scan(0)
    assert scan.max() == pytest.approx(10.0)
    assert (scan <= 10.0).all()
