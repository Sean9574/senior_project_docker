"""N-room arena world builder.

Layout tests are pure. Model tests need `mujoco` and the stretch_mujoco
models (directory in $STRETCH_MUJOCO_MODELS); they skip otherwise.
Run: STRETCH_MUJOCO_MODELS=<.../stretch_mujoco/models> python3 -m pytest src/stretch_ros2/stretch_simulation/test/test_arena_world.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stretch_mujoco_driver import arena_world  # noqa: E402

ROOM_XML = os.path.join(os.path.dirname(__file__), "..", "..", "senior_project", "config", "scene.xml")
MODELS = os.environ.get("STRETCH_MUJOCO_MODELS", "")


@pytest.mark.parametrize("n, shape", [(1, (1, 1)), (2, (2, 1)), (4, (2, 2)), (5, (3, 2)), (8, (4, 2)), (10, (4, 3))])
def test_grid_is_near_square(n, shape):
    assert arena_world.grid_shape(n) == shape


def test_room_origins_are_spaced_by_room_plus_gap():
    # 5 rooms -> 3 columns x 2 rows, filled row by row
    assert arena_world.room_origins(5, room_size=10.0, gap=2.0) == [
        (0.0, 0.0), (12.0, 0.0), (24.0, 0.0), (0.0, 12.0), (12.0, 12.0)]


needs_mujoco = pytest.mark.skipif(
    not MODELS or not os.path.isdir(MODELS), reason="set STRETCH_MUJOCO_MODELS to run model tests")


@pytest.fixture(scope="module")
def world8():
    pytest.importorskip("mujoco")
    return arena_world.build_arena(os.path.join(MODELS, "stretch.xml"), ROOM_XML, 8)


@needs_mujoco
def test_eight_prefixed_robots_without_rangefinders(world8):
    import mujoco
    m = world8.model
    for i in range(8):
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"r{i}_base_link") >= 0
    assert not any(m.sensor_type[s] == mujoco.mjtSensor.mjSENS_RANGEFINDER for s in range(m.nsensor))
    assert m.nkey == 0


@needs_mujoco
def test_each_robot_keeps_its_360_lidar_sites_and_cameras(world8):
    for room in world8.rooms:
        assert len(room.lidar_sites) == 360
        assert room.camera_rgb >= 0 and room.camera_depth >= 0
        assert room.prefix == f"r{room.index}_"


@needs_mujoco
def test_room_obstacles_are_room_local_and_walls_are_tall(world8):
    import mujoco
    for room in world8.rooms:
        assert room.obstacles, "room must have obstacles for spawning"
        for o in room.obstacles:
            assert abs(o.cx) <= 5.5 and abs(o.cy) <= 5.5
    m = world8.model
    wall = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "room0_wall_n")
    assert m.geom_size[wall][2] == pytest.approx(1.25)


@needs_mujoco
def test_robots_start_at_their_room_origins(world8):
    import mujoco
    d = mujoco.MjData(world8.model)
    mujoco.mj_forward(world8.model, d)
    for room in world8.rooms:
        x, y = d.qpos[room.qpos_adr:room.qpos_adr + 2]
        assert (x, y) == pytest.approx(room.origin, abs=1e-6)
