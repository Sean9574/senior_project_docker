#!/usr/bin/env python3
"""
Parallel Simulation Runner for SAM3 Navigation

Spawns multiple simulations in parallel, each with its own:
  - ROS_DOMAIN_ID (sim_base_domain + sim_id)
  - Server ports (with offsets)
  - Checkpoint directory

Supports two SAM3 server modes:
  - dedicated: Each sim gets its own SAM3 server (ports 8100, 8101, 8102, ...)
  - shared: All sims share one SAM3 server (port 8100)

Usage:
    # Run 4 parallel sims with dedicated SAM3 servers (recommended for multi-GPU)
    python parallel_runner.py --num_sims 4 --target "cup" --sam3_mode dedicated

    # Run 4 parallel sims with shared SAM3 server (lower memory usage)
    python parallel_runner.py --num_sims 4 --target "cup" --sam3_mode shared

    # Run and merge on completion
    python parallel_runner.py --num_sims 4 --target "cup" --merge_on_exit
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional


def _str2bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")


class ParallelRunner:
    def __init__(
        self,
        num_sims: int,
        target: str = "person",
        total_steps: int = 200000,
        rollout_steps: int = 2048,
        base_ckpt_dir: str = "~/rl_checkpoints",
        launch_file: str = "sam3_navigation.launch.py",
        package: str = "senior_project",
        extra_args: Optional[Dict[str, str]] = None,
        merge_on_exit: bool = False,
        stagger_delay: float = 10.0,
        headless: bool = True,
        sam3_domain: int = 0,
        sim_base_domain: int = 10,
        sam3_mode: str = "dedicated",
        sam3_base_port: int = 8100,
        sync_every: int = 300,
    ):
        self.num_sims = num_sims
        self.target = target
        self.total_steps = total_steps
        self.rollout_steps = rollout_steps
        self.base_ckpt_dir = os.path.expanduser(base_ckpt_dir)
        self.launch_file = launch_file
        self.package = package
        self.extra_args = extra_args or {}
        self.merge_on_exit = merge_on_exit
        self.stagger_delay = stagger_delay
        self.headless = bool(headless)
        self.sam3_domain = sam3_domain
        self.sim_base_domain = sim_base_domain
        self.sam3_mode = sam3_mode
        self.sam3_base_port = sam3_base_port
        self.sync_every = sync_every

        self.processes: List[subprocess.Popen] = []
        self.sync_process: Optional[subprocess.Popen] = None
        self.start_time: Optional[datetime] = None
        self._shutdown_requested = False

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        if self._shutdown_requested:
            print("\n[Runner] Force killing all processes...")
            self._force_kill()
            sys.exit(1)

        print("\n[Runner] Shutdown requested, stopping simulations...")
        self._shutdown_requested = True
        self.stop_all()

    def _force_kill(self):
        """Force kill all processes."""
        for proc in self.processes:
            try:
                proc.kill()
            except Exception:
                pass

    def _get_sim_domain(self, sim_id: int) -> int:
        """Get the ROS_DOMAIN_ID for a specific simulation."""
        return self.sim_base_domain + sim_id

    def _build_launch_command(self, sim_id: int) -> List[str]:
        """Build the ros2 launch command for a simulation."""
        ckpt_dir = os.path.join(self.base_ckpt_dir, f"sim_{sim_id}")

        # Determine SAM3 port based on mode
        if self.sam3_mode == "dedicated":
            # Each sim gets its own SAM3 server
            sam3_port = self.sam3_base_port + sim_id
        else:
            # All sims share one SAM3 server
            sam3_port = self.sam3_base_port
        
        # Mono depth port (offset from SAM3 port)
        mono_port = sam3_port + 50  # e.g., 8150 for sim 0 if sam3 is 8100

        cmd = [
            "ros2", "launch", self.package, self.launch_file,
            f"sim_id:={sim_id}",
            f"target:={self.target}",
            f"total_steps:={self.total_steps}",
            f"rollout_steps:={self.rollout_steps}",
            f"ckpt_dir:={ckpt_dir}",
            f"headless:={str(self.headless).lower()}",
            # Tell child simulations NOT to start their own SAM3 server
            "start_sam3_server:=false",
            # Point to the correct SAM3 server port
            f"sam3_server_port:={sam3_port}",
            f"sam3_server_host:=localhost",
            # Also disable local mono depth server (unless explicitly requested)
            "start_mono_depth_server:=false",
            # Pass the SAM3 domain for reference (though HTTP is domain-agnostic)
            f"sam3_domain:={self.sam3_domain}",
        ]

        # Add any extra arguments (but never allow extra to override critical settings)
        forbidden_keys = {"headless", "sam3_domain", "start_sam3_server", 
                          "sam3_server_port", "start_mono_depth_server", "sim_id"}
        for key, value in self.extra_args.items():
            key_lower = key.strip().lower()
            if key_lower in forbidden_keys:
                print(f"[Runner] WARNING: Ignoring extra arg '{key}' (reserved)")
                continue
            cmd.append(f"{key}:={value}")

        return cmd

    def start_simulation(self, sim_id: int) -> subprocess.Popen:
        """Start a single simulation instance."""
        cmd = self._build_launch_command(sim_id)

        # Set environment with unique ROS_DOMAIN_ID for this sim
        sim_domain = self._get_sim_domain(sim_id)
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(sim_domain)

        # Create log directory
        log_dir = os.path.join(self.base_ckpt_dir, f"sim_{sim_id}", "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Open log files
        stdout_log = None
        stderr_log = None

        # Determine SAM3 port for logging
        if self.sam3_mode == "dedicated":
            sam3_port = self.sam3_base_port + sim_id
            port_info = f"SAM3_PORT={sam3_port} [dedicated]"
        else:
            sam3_port = self.sam3_base_port
            port_info = f"SAM3_PORT={sam3_port} [shared]"

        print(
            f"[Runner] Starting sim {sim_id} "
            f"(DOMAIN_ID={sim_domain}, SAM3_DOMAIN={self.sam3_domain}, "
            f"{port_info}, headless={self.headless})"
        )

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_log,
            stderr=stderr_log,
            preexec_fn=os.setsid,  # Create new process group for clean shutdown
        )

        # Store log file handles for cleanup
        proc._log_files = (stdout_log, stderr_log)

        return proc

    def start_all(self):
        """Start all simulations with staggered delays."""
        self.start_time = datetime.now()

        print(f"\n{'='*60}")
        print(f"  Parallel SAM3 Training - {self.num_sims} Simulations")
        print(f"{'='*60}")
        print(f"  Target: {self.target}")
        print(f"  Total Steps: {self.total_steps}")
        print(f"  Rollout Steps: {self.rollout_steps}")
        print(f"  Headless: {self.headless}")
        print(f"  Checkpoint Dir: {self.base_ckpt_dir}")
        print(f"  Merge on Exit: {self.merge_on_exit}")
        if self.sam3_mode == "dedicated":
            print(f"  SAM3 Server Mode: DEDICATED (ports {self.sam3_base_port}-{self.sam3_base_port + self.num_sims - 1})")
        else:
            print(f"  SAM3 Server Mode: SHARED (port {self.sam3_base_port})")
        print(f"  Sim Domains: {self.sim_base_domain} - {self.sim_base_domain + self.num_sims - 1}")
        print(f"  NOTE: All sims use SHARED SAM3 server (not starting their own)")
        if self.sync_every > 0 and self.num_sims > 1:
            print(f"  Federated Sync: every {self.sync_every}s")
        else:
            print(f"  Federated Sync: disabled")
        print(f"{'='*60}\n")

        for sim_id in range(self.num_sims):
            if self._shutdown_requested:
                break

            proc = self.start_simulation(sim_id)
            self.processes.append(proc)

            # Stagger launches to avoid resource contention
            if sim_id < self.num_sims - 1:
                print(f"[Runner] Waiting {self.stagger_delay}s before next launch.")
                time.sleep(self.stagger_delay)

        print(f"\n[Runner] All {len(self.processes)} simulations started!")
        print(f"[Runner] Logs: {self.base_ckpt_dir}/sim_*/logs/")

        # Start federated weight sync if multiple sims
        if self.num_sims > 1 and self.sync_every > 0:
            sync_script = os.path.join(os.path.dirname(__file__), "federated_sync.py")
            if os.path.exists(sync_script):
                sync_cmd = [
                    sys.executable, sync_script,
                    "--num_sims", str(self.num_sims),
                    "--ckpt_dir", self.base_ckpt_dir,
                    "--sync_every", str(self.sync_every),
                    "--weighted",
                ]
                self.sync_process = subprocess.Popen(sync_cmd, preexec_fn=os.setsid)
                print(f"[Runner] Federated sync started (every {self.sync_every}s, pid={self.sync_process.pid})")
            else:
                print(f"[Runner] WARNING: federated_sync.py not found at {sync_script}, skipping sync")

        print("[Runner] Press Ctrl+C to stop all simulations\n")

    def stop_all(self):
        """Stop all simulations gracefully."""
        print("[Runner] Stopping all simulations.")

        # Stop federated sync first
        if self.sync_process and self.sync_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.sync_process.pid), signal.SIGTERM)
                self.sync_process.wait(timeout=5)
                print("[Runner] Federated sync stopped.")
            except Exception:
                try:
                    os.killpg(os.getpgid(self.sync_process.pid), signal.SIGKILL)
                except Exception:
                    pass

        for i, proc in enumerate(self.processes):
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                print(f"[Runner] Sent SIGTERM to sim {i} (pid={proc.pid})")
            except Exception as e:
                print(f"[Runner] Could not stop sim {i}: {e}")

        # Wait a bit for graceful shutdown
        time.sleep(3)

        # Force kill any remaining
        for i, proc in enumerate(self.processes):
            try:
                if proc.poll() is None:  # Still running
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    print(f"[Runner] Force killed sim {i}")
            except Exception:
                pass

        # Close log files
        for proc in self.processes:
            try:
                if hasattr(proc, "_log_files"):
                    for f in proc._log_files:
                        try:
                            f.close()
                        except Exception:
                            pass
            except Exception:
                pass

        if self.merge_on_exit:
            self._merge_checkpoints()

        self._print_summary()

    def _merge_checkpoints(self):
        """Run merge_checkpoints.py to combine trained agents."""
        merge_script = os.path.join(os.path.dirname(__file__), "merge_checkpoints.py")
        output_path = os.path.join(self.base_ckpt_dir, "merged.pt")

        if not os.path.exists(merge_script):
            print(f"[Runner] Warning: Merge script not found at {merge_script}")
            return

        try:
            result = subprocess.run(
                [
                    "python3", merge_script,
                    "--input_dir", self.base_ckpt_dir,
                    "--output", output_path,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print(f"[Runner] Merged checkpoint saved to: {output_path}")
            else:
                print(f"[Runner] Merge failed: {result.stderr}")
        except Exception as e:
            print(f"[Runner] Error running merge: {e}")

    def wait(self):
        """Wait for all simulations to complete."""
        print("[Runner] Monitoring simulations...")

        while not self._shutdown_requested:
            alive = 0
            for i, proc in enumerate(self.processes):
                ret = proc.poll()
                if ret is None:
                    alive += 1
                elif ret != 0:
                    print(f"[Runner] WARNING: Sim {i} exited with code {ret}")

            if alive == 0:
                print("[Runner] All simulations completed!")
                break

            elapsed = (datetime.now() - self.start_time).total_seconds()
            if int(elapsed) % 60 == 0:
                print(
                    f"[Runner] Status: {alive}/{self.num_sims} sims running, "
                    f"elapsed: {elapsed/60:.1f}min"
                )

            time.sleep(5)

        self.stop_all()

    def _print_summary(self):
        """Print training summary."""
        if not self.start_time:
            return
        elapsed = (datetime.now() - self.start_time).total_seconds()

        print(f"\n{'='*60}")
        print("  Training Summary")
        print(f"{'='*60}")
        print(f"  Total Time: {elapsed/60:.1f} minutes")
        print(f"  Simulations: {self.num_sims}")
        print(f"  Headless: {self.headless}")
        if self.sam3_mode == "dedicated":
            print(f"  SAM3 Server Mode: DEDICATED")
        else:
            print(f"  SAM3 Server Mode: SHARED")
        print(f"  Sim Domains: {self.sim_base_domain} - {self.sim_base_domain + self.num_sims - 1}")
        print(f"  Checkpoints: {self.base_ckpt_dir}/sim_*/")
        if self.merge_on_exit:
            print(f"  Merged Model: {self.base_ckpt_dir}/merged.pt")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel SAM3 navigation training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--num_sims", "-n", type=int, default=4,
        help="Number of parallel simulations (default: 4)"
    )
    parser.add_argument(
        "--target", "-t", type=str, default="person",
        help="Target object to navigate to (default: person)"
    )
    parser.add_argument(
        "--total_steps", type=int, default=200000,
        help="Total training steps per simulation (default: 200000)"
    )
    parser.add_argument(
        "--rollout_steps", type=int, default=2048,
        help="Rollout steps per update (default: 2048)"
    )
    parser.add_argument(
        "--ckpt_dir", type=str,
        default=os.path.expanduser("~/rl_checkpoints"),
        help="Base checkpoint directory"
    )
    parser.add_argument(
        "--merge_on_exit", "-m", action="store_true",
        help="Automatically merge checkpoints when training ends"
    )
    parser.add_argument(
        "--sync_every", type=int, default=300,
        help="Seconds between federated weight syncs (0 to disable, default: 300)"
    )
    parser.add_argument(
        "--stagger_delay", type=float, default=10.0,
        help="Delay between launching simulations in seconds (default: 10.0)"
    )
    parser.add_argument(
        "--launch_file", type=str, default="sam3_navigation.launch.py",
        help="Launch file to use for each simulation"
    )
    parser.add_argument(
        "--package", type=str, default="senior_project",
        help="ROS package containing launch file"
    )
    parser.add_argument(
        "--headless", type=str, default="true",
        help="Headless mode: true/false (propagated into each sim launch)"
    )

    # Domain configuration
    parser.add_argument(
        "--sam3_domain", type=int, default=0,
        help="ROS_DOMAIN_ID where the SAM3 server(s) are running (default: 0)"
    )
    parser.add_argument(
        "--sim_base_domain", type=int, default=10,
        help="Base ROS_DOMAIN_ID for simulations; sim N uses base + N (default: 10)"
    )
    
    # SAM3 server configuration
    parser.add_argument(
        "--sam3_mode", type=str, default="dedicated",
        choices=["dedicated", "shared"],
        help="SAM3 server mode: 'dedicated' (one per sim) or 'shared' (single server)"
    )
    parser.add_argument(
        "--sam3_base_port", type=int, default=8100,
        help="Base port for SAM3 servers (sim N uses base_port + N in dedicated mode)"
    )

    # Collect any extra launch arguments
    parser.add_argument(
        "--extra", "-e", nargs="*", default=[],
        help="Extra launch arguments as key=value pairs"
    )

    args = parser.parse_args()

    # Parse extra arguments
    extra_args: Dict[str, str] = {}
    for item in args.extra:
        if "=" in item:
            key, value = item.split("=", 1)
            extra_args[key] = value

    runner = ParallelRunner(
        num_sims=args.num_sims,
        target=args.target,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        base_ckpt_dir=args.ckpt_dir,
        launch_file=args.launch_file,
        package=args.package,
        extra_args=extra_args,
        merge_on_exit=args.merge_on_exit,
        stagger_delay=args.stagger_delay,
        headless=_str2bool(args.headless),
        sam3_domain=args.sam3_domain,
        sim_base_domain=args.sim_base_domain,
        sam3_mode=args.sam3_mode,
        sam3_base_port=args.sam3_base_port,
        sync_every=args.sync_every,
    )

    runner.start_all()
    runner.wait()


if __name__ == "__main__":
    main()