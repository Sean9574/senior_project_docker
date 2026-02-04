#!/usr/bin/env python3
"""
Checkpoint Merger for Parallel TD3 Training

Merges model checkpoints from multiple parallel simulations using
federated averaging (weight averaging).

Usage:
    # Basic merge (finds all sim_* directories)
    python merge_checkpoints.py --input_dir ~/ament_ws/src/stretch_ros2/senior_project/parallel_training --output merged.pt
    
    # Merge specific checkpoint files
    python merge_checkpoints.py --checkpoints sim_0/agent.pt sim_1/agent.pt --output merged.pt
    
    # Weighted merge (if some sims trained longer)
    python merge_checkpoints.py --input_dir ./parallel_training --output merged.pt --weighted

Supports:
    - Simple averaging: (w1 + w2 + ... + wN) / N
    - Weighted averaging: Based on training steps or reward
"""

import argparse
import copy
import os
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def find_checkpoints(input_dir: str, pattern: str = "*.pt") -> List[Path]:
    """
    Find all checkpoint files in sim_* subdirectories.
    
    Returns list of paths sorted by sim_id.
    """
    input_path = Path(input_dir).expanduser()
    checkpoints = []
    
    # Look for sim_* directories
    sim_dirs = sorted(input_path.glob("sim_*"))
    
    if sim_dirs:
        for sim_dir in sim_dirs:
            # Find .pt files in each sim directory
            pt_files = list(sim_dir.glob(pattern))
            if pt_files:
                # Take the most recent checkpoint
                latest = max(pt_files, key=lambda p: p.stat().st_mtime)
                checkpoints.append(latest)
                print(f"  Found: {latest}")
    else:
        # No sim_* dirs, look for .pt files directly
        pt_files = list(input_path.glob(pattern))
        checkpoints = sorted(pt_files)
        for ckpt in checkpoints:
            print(f"  Found: {ckpt}")
    
    return checkpoints


def load_checkpoint(path: Path) -> Dict:
    """Load a checkpoint file."""
    return torch.load(path, map_location="cpu")


def get_state_dict(checkpoint: Dict) -> OrderedDict:
    """
    Extract the model state dict from various checkpoint formats.
    
    Handles common formats:
    - Direct state dict
    - {'state_dict': ...}
    - {'model': ...}
    - {'actor': ..., 'critic': ...} (TD3/SAC style)
    """
    if isinstance(checkpoint, OrderedDict):
        return checkpoint
    
    # Common keys for model weights
    for key in ['state_dict', 'model', 'model_state_dict', 'network']:
        if key in checkpoint:
            return checkpoint[key]
    
    # TD3/SAC style with separate networks
    if 'actor' in checkpoint or 'critic' in checkpoint:
        return checkpoint  # Return full dict, handle specially
    
    # Assume it's already a state dict
    return checkpoint


def is_td3_checkpoint(checkpoint: Dict) -> bool:
    """Check if this is a TD3-style checkpoint with actor/critic."""
    return any(k in checkpoint for k in ['actor', 'actor_state_dict', 'policy'])


def average_state_dicts(
    state_dicts: List[OrderedDict],
    weights: Optional[List[float]] = None
) -> OrderedDict:
    """
    Average multiple state dicts together.
    
    Args:
        state_dicts: List of state dicts to average
        weights: Optional weights for weighted average (must sum to 1)
    
    Returns:
        Averaged state dict
    """
    if not state_dicts:
        raise ValueError("No state dicts to average")
    
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    
    # Start with a copy of the first state dict
    avg_state = OrderedDict()
    
    # Get all keys from first state dict
    keys = state_dicts[0].keys()
    
    for key in keys:
        # Check if all state dicts have this key
        if not all(key in sd for sd in state_dicts):
            print(f"  Warning: Key '{key}' not in all checkpoints, skipping")
            continue
        
        # Get the tensor from first state dict
        first_tensor = state_dicts[0][key]
        
        # Skip non-tensor entries
        if not isinstance(first_tensor, torch.Tensor):
            avg_state[key] = first_tensor
            continue
        
        # Weighted average of tensors
        avg_tensor = torch.zeros_like(first_tensor, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            avg_tensor += w * sd[key].float()
        
        # Convert back to original dtype
        avg_state[key] = avg_tensor.to(first_tensor.dtype)
    
    return avg_state


def merge_td3_checkpoints(
    checkpoints: List[Dict],
    weights: Optional[List[float]] = None
) -> Dict:
    """
    Merge TD3-style checkpoints with actor/critic networks.
    """
    merged = {}
    
    # Common TD3 checkpoint keys
    network_keys = [
        'actor', 'actor_state_dict', 'policy', 'policy_state_dict',
        'critic', 'critic_state_dict', 'q1', 'q2',
        'actor_target', 'critic_target', 'target_actor', 'target_critic',
        'actor_optimizer', 'critic_optimizer', 'optimizer',
    ]
    
    for key in network_keys:
        # Collect this key from all checkpoints that have it
        dicts_with_key = [(i, ckpt[key]) for i, ckpt in enumerate(checkpoints) if key in ckpt]
        
        if not dicts_with_key:
            continue
        
        indices, state_dicts = zip(*dicts_with_key)
        
        # Use corresponding weights
        if weights:
            key_weights = [weights[i] for i in indices]
        else:
            key_weights = None
        
        # Check if these are state dicts or optimizer states
        if isinstance(state_dicts[0], dict) and 'state' in state_dicts[0]:
            # Optimizer state - just take from first (not averaged)
            merged[key] = state_dicts[0]
            print(f"  {key}: copied from first checkpoint (optimizer state)")
        else:
            # Model weights - average them
            merged[key] = average_state_dicts(list(state_dicts), key_weights)
            print(f"  {key}: averaged from {len(state_dicts)} checkpoints")
    
    # Copy non-network metadata from first checkpoint
    for key in checkpoints[0]:
        if key not in merged and key not in network_keys:
            merged[key] = checkpoints[0][key]
    
    return merged


def merge_checkpoints(
    checkpoint_paths: List[Path],
    output_path: Path,
    weights: Optional[List[float]] = None,
) -> Dict:
    """
    Main merge function.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        output_path: Where to save merged checkpoint
        weights: Optional weights for weighted average
    
    Returns:
        Merged checkpoint dict
    """
    print(f"\nMerging {len(checkpoint_paths)} checkpoints...")
    
    # Load all checkpoints
    checkpoints = []
    for path in checkpoint_paths:
        print(f"  Loading: {path}")
        ckpt = load_checkpoint(path)
        checkpoints.append(ckpt)
    
    # Determine checkpoint type and merge
    if is_td3_checkpoint(checkpoints[0]):
        print("\n  Detected TD3-style checkpoint")
        merged = merge_td3_checkpoints(checkpoints, weights)
    else:
        print("\n  Detected standard checkpoint")
        state_dicts = [get_state_dict(ckpt) for ckpt in checkpoints]
        merged = {
            'state_dict': average_state_dicts(state_dicts, weights),
            'merged_from': [str(p) for p in checkpoint_paths],
            'merge_time': datetime.now().isoformat(),
            'num_sources': len(checkpoints),
        }
    
    # Add merge metadata
    merged['_merge_info'] = {
        'sources': [str(p) for p in checkpoint_paths],
        'weights': weights,
        'merge_time': datetime.now().isoformat(),
    }
    
    # Save merged checkpoint
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(merged, output_path)
    print(f"\n  Saved merged checkpoint to: {output_path}")
    
    return merged


def compute_weights_from_steps(checkpoint_paths: List[Path]) -> List[float]:
    """
    Compute weights based on training steps in each checkpoint.
    
    Checkpoints with more training steps get higher weight.
    """
    steps = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu")
        
        # Try to find step count
        step = 0
        for key in ['step', 'steps', 'total_steps', 'global_step', 'iteration']:
            if key in ckpt:
                step = ckpt[key]
                break
        
        steps.append(max(step, 1))  # Avoid zero weight
    
    total = sum(steps)
    weights = [s / total for s in steps]
    
    print(f"\n  Computed weights from training steps:")
    for path, w, s in zip(checkpoint_paths, weights, steps):
        print(f"    {path.name}: {s} steps -> weight={w:.3f}")
    
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Merge TD3 checkpoints from parallel training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-find and merge all sim_* checkpoints
    python merge_checkpoints.py -i ~/rl_checkpoints -o ~/rl_checkpoints/merged.pt
    
    # Merge specific files
    python merge_checkpoints.py -c ckpt1.pt ckpt2.pt ckpt3.pt -o merged.pt
    
    # Weighted merge based on training steps
    python merge_checkpoints.py -i ~/rl_checkpoints -o merged.pt --weighted
    
    # Custom weights
    python merge_checkpoints.py -c a.pt b.pt c.pt -o merged.pt -w 0.5 0.3 0.2
        """
    )
    
    parser.add_argument(
        "--input_dir", "-i", type=str,
        help="Directory containing sim_* subdirectories with checkpoints"
    )
    parser.add_argument(
        "--checkpoints", "-c", nargs="+", type=str,
        help="Explicit list of checkpoint files to merge"
    )
    parser.add_argument(
        "--output", "-o", type=str, required=True,
        help="Output path for merged checkpoint"
    )
    parser.add_argument(
        "--pattern", "-p", type=str, default="*.pt",
        help="Glob pattern for finding checkpoints (default: *.pt)"
    )
    parser.add_argument(
        "--weights", "-w", nargs="+", type=float,
        help="Manual weights for each checkpoint (must match number of checkpoints)"
    )
    parser.add_argument(
        "--weighted", action="store_true",
        help="Compute weights based on training steps in each checkpoint"
    )
    
    args = parser.parse_args()
    
    # Find checkpoints
    if args.checkpoints:
        checkpoint_paths = [Path(p).expanduser() for p in args.checkpoints]
    elif args.input_dir:
        print(f"Scanning {args.input_dir} for checkpoints...")
        checkpoint_paths = find_checkpoints(args.input_dir, args.pattern)
    else:
        parser.error("Must specify either --input_dir or --checkpoints")
    
    if not checkpoint_paths:
        print("Error: No checkpoints found!")
        return 1
    
    # Determine weights
    weights = None
    if args.weights:
        if len(args.weights) != len(checkpoint_paths):
            parser.error(f"Number of weights ({len(args.weights)}) must match "
                        f"number of checkpoints ({len(checkpoint_paths)})")
        weights = args.weights
    elif args.weighted:
        weights = compute_weights_from_steps(checkpoint_paths)
    
    # Merge
    merge_checkpoints(checkpoint_paths, Path(args.output), weights)
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
