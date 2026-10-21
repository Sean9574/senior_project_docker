"""federated_sync merges the learner's real checkpoint keys and hands them over
in a side file the learner loads (never by overwriting the learner's checkpoint).

Needs torch; skipped where torch is not installed.
Run: python3 -m pytest src/stretch_ros2/senior_project/test/test_federated_sync.py
"""

import json
import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from senior_project import federated_sync as fs  # noqa: E402

LEARNER_KEYS = ["actor", "actor_targ", "critic1", "critic2", "critic1_targ", "critic2_targ"]


def _write_sim(ckpt_dir, i, value, step=1000):
    sim_dir = ckpt_dir / f"sim_{i}"
    sim_dir.mkdir(parents=True)
    ckpt = {k: {"w": torch.full((2,), float(value))} for k in LEARNER_KEYS}
    ckpt["training_state"] = {"step": step}
    torch.save(ckpt, sim_dir / fs.CHECKPOINT_FILENAME)
    return sim_dir


def test_merge_averages_every_learner_network(tmp_path):
    _write_sim(tmp_path, 0, 1.0)
    _write_sim(tmp_path, 1, 3.0)
    assert fs.merge_and_redistribute(str(tmp_path), 2) is True
    for i in range(2):
        update = torch.load(tmp_path / f"sim_{i}" / fs.FEDERATED_UPDATE_FILENAME, weights_only=False)
        for k in LEARNER_KEYS:
            assert torch.allclose(update[k]["w"], torch.full((2,), 2.0)), k


def test_merge_leaves_learner_checkpoint_untouched(tmp_path):
    sim0 = _write_sim(tmp_path, 0, 1.0)
    _write_sim(tmp_path, 1, 3.0)
    fs.merge_and_redistribute(str(tmp_path), 2)
    own = torch.load(sim0 / fs.CHECKPOINT_FILENAME, weights_only=False)
    assert torch.allclose(own["actor"]["w"], torch.full((2,), 1.0))


def test_merge_writes_sync_status_json(tmp_path):
    _write_sim(tmp_path, 0, 1.0, step=500)
    _write_sim(tmp_path, 1, 3.0, step=700)
    fs.merge_and_redistribute(str(tmp_path), 2)
    status = json.loads((tmp_path / fs.SYNC_STATUS_FILENAME).read_text())
    assert status["steps_at_sync"] == [500, 700]


def test_merge_skipped_when_a_network_is_missing(tmp_path):
    _write_sim(tmp_path, 0, 1.0)
    sim1 = _write_sim(tmp_path, 1, 3.0)
    ckpt = torch.load(sim1 / fs.CHECKPOINT_FILENAME, weights_only=False)
    del ckpt["critic2"]
    torch.save(ckpt, sim1 / fs.CHECKPOINT_FILENAME)
    assert fs.merge_and_redistribute(str(tmp_path), 2) is False
    assert not (tmp_path / "sim_0" / fs.FEDERATED_UPDATE_FILENAME).exists()
