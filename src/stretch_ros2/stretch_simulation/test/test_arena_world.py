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
def test_target_is_a_real_size_soda_can_standing_on_the_table(world8):
    import mujoco
    m = world8.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert gid("room0_blue_target") < 0
    can, top = gid("room0_can_body"), gid("room0_table_top")
    assert can >= 0 and top >= 0
    # 12 oz can: 66 mm across, 122 mm tall including the lids (geom_aabb:
    # local centre + half sizes, the same for the mesh body and the lids)
    assert m.geom_aabb[can][3] == pytest.approx(0.033, abs=1e-3)
    lids = [gid("room0_can_top"), gid("room0_can_bottom")]
    tallest = max(d.geom_xpos[g][2] + m.geom_aabb[g][2] + m.geom_aabb[g][5] for g in [can] + lids)
    lowest = min(d.geom_xpos[g][2] + m.geom_aabb[g][2] - m.geom_aabb[g][5] for g in [can] + lids)
    assert tallest - lowest == pytest.approx(0.122, abs=2e-3)
    table_surface = d.geom_xpos[top][2] + m.geom_size[top][2]
    assert lowest == pytest.approx(table_surface, abs=1e-3)
    assert d.geom_xpos[can][:2] == pytest.approx(d.geom_xpos[top][:2], abs=1e-6)


@needs_mujoco
def test_can_wears_a_printed_label_so_sam3_sees_a_soda_can(world8):
    import mujoco
    m = world8.model
    can = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "room0_can_body")
    assert m.geom_type[can] == mujoco.mjtGeom.mjGEOM_MESH
    assert m.mesh_texcoordnum[m.geom_dataid[can]] > 0  # wrapped round the can
    tex = m.mat_texid[m.geom_matid[can]][mujoco.mjtTextureRole.mjTEXROLE_RGB]
    assert tex >= 0 and m.tex_type[tex] == mujoco.mjtTexture.mjTEXTURE_2D
    assert m.tex_width[tex] >= 512


@needs_mujoco
def test_robots_start_at_their_room_origins(world8):
    import mujoco
    d = mujoco.MjData(world8.model)
    mujoco.mj_forward(world8.model, d)
    for room in world8.rooms:
        x, y = d.qpos[room.qpos_adr:room.qpos_adr + 2]
        assert (x, y) == pytest.approx(room.origin, abs=1e-6)



@needs_mujoco
def test_one_light_and_one_colliding_floor_for_any_room_count(world8):
    import mujoco
    m = world8.model
    assert m.nlight == 1   # per-room lights would overexpose the SAM3 images
    planes = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    assert len(planes) == 8
    assert sum(1 for g in planes if m.geom_contype[g] or m.geom_conaffinity[g]) == 1


@needs_mujoco
def test_scene_headlight_is_kept(world8):
    assert list(world8.model.vis.headlight.diffuse) == pytest.approx([0.6, 0.6, 0.6])
    assert list(world8.model.vis.headlight.ambient) == pytest.approx([0.3, 0.3, 0.3])


@needs_mujoco
def test_robots_in_every_room_stand_on_the_floor(world8):
    import mujoco
    d = mujoco.MjData(world8.model)
    mujoco.mj_forward(world8.model, d)
    z0 = [d.qpos[r.qpos_adr + 2] for r in world8.rooms]
    for _ in range(250):
        mujoco.mj_step(world8.model, d)
    for room, z in zip(world8.rooms, z0):
        assert d.qpos[room.qpos_adr + 2] == pytest.approx(z, abs=0.02)
