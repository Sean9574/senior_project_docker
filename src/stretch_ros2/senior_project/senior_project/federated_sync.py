#!/usr/bin/env python3
"""
Federated Weight Sync for Parallel TD3 Training

Runs alongside parallel_runner.py and periodically:
1. Waits for all sims to save checkpoints
2. Averages all sim weights together (federated averaging)
3. Pushes merged weights back to each sim's checkpoint dir

This means each sim benefits from ALL sims' exploration,
effectively multiplying the experience gathering rate by num_sims.

Merged weights go to sim_i/federated_update.pt (never over the learner's own
checkpoint, which it rewrites every episode). The learner loads that file at
its next episode end and deletes it. sync_status.json records each sync.

NOTE: This does NOT merge replay buffers — each sim keeps its own.
The speedup comes from shared weight updates, not shared experience.

Usage:
    # Run alongside parallel_runner.py
    python federated_sync.py --ckpt_dir ~/rl_checkpoints --num_sims 5 --sync_every 300

    # More aggressive syncing (every 2 min)
    python federated_sync.py --ckpt_dir ~/rl_checkpoints --num_sims 5 --sync_every 120

    # Weighted by training steps (sims that trained more get more influence)
    python federated_sync.py --ckpt_dir ~/rl_checkpoints --num_sims 5 --weighted
"""

import argparse
import copy
import json
import os
import shutil
import signal
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =========================================================================
# Config
# =========================================================================

CHECKPOINT_FILENAME = "td3_safe_rl_agent.pt"
REPLAY_FILENAME = "replay_buffer.npz"
FEDERATED_UPDATE_FILENAME = "federated_update.pt"  # read + deleted by learner_node
SYNC_STATUS_FILENAME = "sync_status.json"           # read by reward_monitor SyncMonitor

# Network keys TD3AgentCNN.save() writes (learner_node.py)
NETWORK_KEYS = ["actor", "actor_targ", "critic1", "critic2", "critic1_targ", "critic2_targ"]


def load_checkpoint(path: str) -> Optional[Dict]:
    """Safely load a checkpoint file."""
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        return ckpt
    except Exception as e:
        print(f"  [WARN] Failed to load {path}: {e}")
        return None


def get_training_step(ckpt: Dict) -> int:
    """Extract training step from checkpoint."""
    state = ckpt.get("training_state", {})
    return state.get("step", 0)


def average_state_dicts(
    state_dicts: List[OrderedDict],
    weights: Optional[List[float]] = None,
) -> OrderedDict:
    """Weighted average of state dicts."""
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    avg = OrderedDict()
    keys = state_dicts[0].keys()

    for key in keys:
        if not all(key in sd for sd in state_dicts):
            continue
        first = state_dicts[0][key]
        if not isinstance(first, torch.Tensor):
            avg[key] = first
            continue
        avg_tensor = torch.zeros_like(first, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            avg_tensor += w * sd[key].float()
        avg[key] = avg_tensor.to(first.dtype)

    return avg


def merge_and_redistribute(
    ckpt_dir: str,
    num_sims: int,
    weighted: bool = False,
    min_step_threshold: int = 100,
) -> bool:
    """
    Core sync operation:
    1. Load all sim checkpoints
    2. Average the network weights
    3. Write merged weights back to each sim dir

    Returns True if sync was performed, False if skipped.
    """
    sim_dirs = []
    checkpoints = []
    steps = []

    # --- Collect checkpoints from all sims ---
    for i in range(num_sims):
        sim_path = os.path.join(ckpt_dir, f"sim_{i}", CHECKPOINT_FILENAME)
        if not os.path.exists(sim_path):
            print(f"  [SKIP] sim_{i}: no checkpoint yet")
            return False
        ckpt = load_checkpoint(sim_path)
        if ckpt is None:
            print(f"  [SKIP] sim_{i}: checkpoint unreadable")
            return False
        step = get_training_step(ckpt)
        if step < min_step_threshold:
            print(f"  [SKIP] sim_{i}: only at step {step} (need {min_step_threshold})")
            return False
        sim_dirs.append(os.path.join(ckpt_dir, f"sim_{i}"))
        checkpoints.append(ckpt)
        steps.append(step)

    print(f"\n  All {num_sims} sims ready. Steps: {steps}")

    # --- Compute weights ---
    if weighted:
        total = sum(steps)
        weights = [s / total for s in steps]
        print(f"  Weighted merge: {[f'{w:.3f}' for w in weights]}")
    else:
        weights = None
        print(f"  Equal-weight merge")

    # --- Average each network (all six, or nothing: a partial merge would
    # pair an averaged actor with unaveraged critics) ---
    missing = [k for k in NETWORK_KEYS if not all(k in ckpt for ckpt in checkpoints)]
    if missing:
        print(f"  [SKIP] Networks missing from some sims: {missing}")
        return False
    merged_parts = {k: average_state_dicts([ckpt[k] for ckpt in checkpoints], weights)
                    for k in NETWORK_KEYS}
    print(f"  ✓ Merged: {', '.join(NETWORK_KEYS)} ({num_sims} sources)")

    sync_meta = {
        "time": datetime.now().isoformat(),
        "steps_at_sync": steps,
        "weighted": weighted,
    }

    # --- Hand merged weights to each sim (each keeps its own optimizer,
    # training_state and replay buffer) ---
    for i, sim_dir in enumerate(sim_dirs):
        out_path = os.path.join(sim_dir, FEDERATED_UPDATE_FILENAME)
        tmp_path = out_path + ".sync_tmp"
        try:
            torch.save({**merged_parts, "_last_sync": sync_meta}, tmp_path)
            os.replace(tmp_path, out_path)
            print(f"  ✓ Wrote merged weights to sim_{i}/{FEDERATED_UPDATE_FILENAME} (step {steps[i]})")
        except Exception as e:
            print(f"  ✗ Failed to write sim_{i}: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    with open(os.path.join(ckpt_dir, SYNC_STATUS_FILENAME), "w") as f:
        json.dump(sync_meta, f)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Federated weight sync for parallel TD3 training"
    )
    parser.add_argument(
        "--ckpt_dir", type=str,
        default=os.path.expanduser("~/rl_checkpoints"),
        help="Base checkpoint directory (contains sim_0/, sim_1/, ...)",
    )
    parser.add_argument(
        "--num_sims", "-n", type=int, required=True,
        help="Number of parallel sims to sync",
    )
    parser.add_argument(
        "--sync_every", type=int, default=300,
        help="Seconds between sync attempts (default: 300 = 5 min)",
    )
    parser.add_argument(
        "--weighted", action="store_true",
        help="Weight merge by training steps (sims with more steps get more influence)",
    )
    parser.add_argument(
        "--min_steps", type=int, default=500,
        help="Minimum steps before a sim participates in sync (default: 500)",
    )

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Federated Weight Sync")
    print(f"{'='*60}")
    print(f"  Checkpoint dir:  {args.ckpt_dir}")
    print(f"  Num sims:        {args.num_sims}")
    print(f"  Sync interval:   {args.sync_every}s")
    print(f"  Weighted:        {args.weighted}")
    print(f"  Min steps:       {args.min_steps}")
    print(f"{'='*60}\n")

    shutdown = False

    def handle_signal(sig, frame):
        nonlocal shutdown
        print("\n[Sync] Shutting down...")
        shutdown = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    sync_count = 0

    while not shutdown:
        print(f"\n[Sync] Attempt #{sync_count + 1} at {datetime.now().strftime('%H:%M:%S')}")

        success = merge_and_redistribute(
            ckpt_dir=args.ckpt_dir,
            num_sims=args.num_sims,
            weighted=args.weighted,
            min_step_threshold=args.min_steps,
        )

        if success:
            sync_count += 1
            print(f"[Sync] ✓ Sync #{sync_count} complete!")
        else:
            print(f"[Sync] Skipped (not all sims ready)")

        # Wait for next sync
        for _ in range(args.sync_every):
            if shutdown:
                break
            time.sleep(1)

    print(f"\n[Sync] Done. Performed {sync_count} syncs total.")


if __name__ == "__main__":
    main()