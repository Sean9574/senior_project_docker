#!/usr/bin/env python3
"""
Multi-Domain RL Training Monitor + Camera Streaming

Updated for v5 (never-terminate) learner:
- Episode outcomes: GOAL_REACHED / TIMEOUT only (collisions never terminate)
- Bumps tracked as mid-episode events, not outcomes
- Reward components: discovery, revisit, proximity, progress, step, collision, goal, timeout
- Proximity threshold line on min-distance chart
"""

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import rclpy
from cv_bridge import CvBridge
from flask import Flask, Response, jsonify, render_template_string, request
from flask_cors import CORS
from flask_socketio import SocketIO
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32
from std_msgs.msg import String as StringMsg

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = 5555
MAX_HISTORY_LENGTH = 1000
UPDATE_RATE_HZ = 5
DEFAULT_SCAN_START = 0
DEFAULT_SCAN_END = 50
DEFAULT_SCAN_INTERVAL = 15.0

DEFAULT_CAMERA_FPS = 15.0
DEFAULT_CAMERA_JPEG_QUALITY = 85
DEFAULT_CAMERA_MAX_WIDTH = 800

# =============================================================================
# Experiment CSV Logging
# =============================================================================

STEP_CSV_COLUMNS = [
    "wall_clock", "sim_id", "episode", "step", "total_step", "mode", "reward",
    "r_discovery", "r_revisit", "r_proximity", "r_collision", "r_progress",
    "r_goal", "r_step", "r_timeout",
    "min_distance", "nav_zone", "velocity_v", "velocity_w",
    "robot_x", "robot_y", "goal_dist", "cells_discovered", "frontier_count",
]

EPISODE_CSV_COLUMNS = [
    "wall_clock", "sim_id", "episode", "total_steps_at_end", "episode_steps",
    "episode_return", "outcome", "goals_reached", "cells_discovered",
    "avg_min_distance", "avg_velocity", "bumps", "training_phase",
]

SYNC_CSV_COLUMNS = [
    "wall_clock", "sync_number", "sim_steps", "weighted",
]

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DomainStats:
    domain_id: int
    namespace: str
    display_name: str
    rewards: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    components: Dict[str, deque] = field(default_factory=dict)
    episode_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    current_episode_return: float = 0.0
    current_episode_steps: int = 0
    last_episode_num: int = 0
    episode_count: int = 0
    total_steps: int = 0
    last_update: float = 0.0
    goals_reached: int = 0
    frontiers: int = 0
    cells_discovered: int = 0

    # Episode outcome tracking (v5: only goal or timeout end episodes)
    collisions: int = 0       # Legacy — kept for backward compat but always 0 in v5
    successes: int = 0
    timeouts: int = 0
    bumps: int = 0            # Mid-episode collision bumps (never terminate)
    _last_success: bool = False

    # Min distance tracking
    min_distances: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    nav_zones: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))

    # Velocity tracking
    velocities: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))

    # Goal generator status
    goal_status: Dict = field(default_factory=dict)
    goal_status_time: float = 0.0

    # Expert baseline tracking
    expert_avg_return: float = 0.0
    training_phase: str = "UNKNOWN"  # EXPERT_DEMO / RANDOM_EXPLORE / RL_POLICY
    expert_episode_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    rl_episode_returns: deque = field(default_factory=lambda: deque(maxlen=100))

    # --- CSV Experiment Logging ---
    _csv_steps_writer: Optional[object] = field(default=None, repr=False)
    _csv_steps_file: Optional[object] = field(default=None, repr=False)
    _csv_episodes_writer: Optional[object] = field(default=None, repr=False)
    _csv_episodes_file: Optional[object] = field(default=None, repr=False)
    _csv_start_time: float = 0.0
    _csv_episode_min_dists: list = field(default_factory=list)
    _csv_episode_velocities: list = field(default_factory=list)
    _csv_episode_bumps: int = 0
    _csv_episode_cells: int = 0
    _csv_episode_goals: int = 0
    _csv_steps_logged: int = 0
    _csv_episodes_logged: int = 0

    def init_csv_logging(self, experiment_dir: str, sim_id: int, start_time: float):
        """Initialize CSV writers for this domain. Call once after creation."""
        self._csv_start_time = start_time
        sim_dir = os.path.join(experiment_dir, f"sim_{sim_id}")
        os.makedirs(sim_dir, exist_ok=True)

        self._csv_steps_file = open(os.path.join(sim_dir, "steps.csv"), "w", newline="")
        self._csv_steps_writer = csv.DictWriter(self._csv_steps_file, fieldnames=STEP_CSV_COLUMNS)
        self._csv_steps_writer.writeheader()

        self._csv_episodes_file = open(os.path.join(sim_dir, "episodes.csv"), "w", newline="")
        self._csv_episodes_writer = csv.DictWriter(self._csv_episodes_file, fieldnames=EPISODE_CSV_COLUMNS)
        self._csv_episodes_writer.writeheader()

    def close_csv(self):
        """Flush final episode and close CSV files."""
        if self._csv_episodes_writer and self.current_episode_steps > 0:
            self._flush_csv_episode(time.time() - self._csv_start_time, success=False)
        if self._csv_steps_file:
            self._csv_steps_file.close()
        if self._csv_episodes_file:
            self._csv_episodes_file.close()

    def add_reward(self, reward: float, timestamp: float):
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        self.current_episode_return += reward
        self.current_episode_steps += 1
        self.total_steps += 1
        self.last_update = time.time()

    def add_breakdown(self, breakdown: Dict):
        timestamp = time.time()

        # Reward components
        reward_terms = breakdown.get("reward_terms", {})
        for key, value in reward_terms.items():
            if key not in self.components:
                self.components[key] = deque(maxlen=MAX_HISTORY_LENGTH)
            self.components[key].append((timestamp, value))

        # Min distance (top-level field)
        min_dist = breakdown.get("min_distance", 0.0)
        if min_dist > 0:
            self.min_distances.append((timestamp, min_dist))

        # Nav zone (top-level field)
        zone = breakdown.get("nav_zone", "UNKNOWN")
        self.nav_zones.append((timestamp, zone))

        # Velocity
        vel = breakdown.get("velocity", {})
        v = vel.get("v", 0.0)
        self.velocities.append((timestamp, v))

        # Goals
        goals = breakdown.get("goals_reached", 0)
        if goals > self.goals_reached:
            self.goals_reached = goals

        # Explore stats
        explore_stats = breakdown.get("explore_stats", {})
        self.cells_discovered = explore_stats.get("total_discovered", self.cells_discovered)
        self.frontiers = explore_stats.get("frontier_count", self.frontiers)

        # Episode boundary detection — track outcomes
        # v5: collisions never terminate. Only goal/timeout end episodes.
        collision = breakdown.get("collision", False)
        success = breakdown.get("success", False)

        # Count mid-episode bumps
        if collision:
            self.bumps += 1

        episode_num = breakdown.get("episode", 0)
        if episode_num > self.last_episode_num and self.last_episode_num > 0:
            if self.current_episode_steps > 0:
                self.episode_returns.append(self.current_episode_return)
                self.episode_count += 1
                # Record how the PREVIOUS episode ended (only goal or timeout in v5)
                if self._last_success:
                    self.successes += 1
                else:
                    self.timeouts += 1
                # Flush CSV episode row
                if self._csv_episodes_writer:
                    now = time.time() - self._csv_start_time
                    self._flush_csv_episode(now, success=self._last_success)
            self.current_episode_return = 0.0
            self.current_episode_steps = 0
            self.goals_reached = 0
        self.last_episode_num = episode_num

        # Remember flags for next episode boundary
        self._last_success = success

        # Training phase and expert baseline
        self.training_phase = breakdown.get("training_phase", self.training_phase)
        self.expert_avg_return = breakdown.get("expert_avg_return", self.expert_avg_return)

        # --- CSV Logging ---
        if self._csv_steps_writer:
            now = time.time() - self._csv_start_time
            vel = breakdown.get("velocity", {})
            state = breakdown.get("state", {})
            explore_stats = breakdown.get("explore_stats", {})
            reward_terms = breakdown.get("reward_terms", {})

            self._csv_steps_writer.writerow({
                "wall_clock": f"{now:.2f}",
                "sim_id": self.domain_id,
                "episode": episode_num,
                "step": breakdown.get("step", 0),
                "total_step": self.total_steps,
                "mode": breakdown.get("mode", ""),
                "reward": f"{breakdown.get('reward', 0.0):.4f}",
                "r_discovery": f"{reward_terms.get('discovery', 0.0):.4f}",
                "r_revisit": f"{reward_terms.get('revisit', 0.0):.4f}",
                "r_proximity": f"{reward_terms.get('proximity', 0.0):.4f}",
                "r_collision": f"{reward_terms.get('collision', 0.0):.4f}",
                "r_progress": f"{reward_terms.get('progress', 0.0):.4f}",
                "r_goal": f"{reward_terms.get('goal', 0.0):.4f}",
                "r_step": f"{reward_terms.get('step', 0.0):.4f}",
                "r_timeout": f"{reward_terms.get('timeout', 0.0):.4f}",
                "min_distance": f"{min_dist:.4f}",
                "nav_zone": zone,
                "velocity_v": f"{vel.get('v', 0.0):.4f}",
                "velocity_w": f"{vel.get('w', 0.0):.4f}",
                "robot_x": f"{state.get('x', 0.0):.3f}",
                "robot_y": f"{state.get('y', 0.0):.3f}",
                "goal_dist": f"{state.get('goal_dist', 0.0):.3f}",
                "cells_discovered": explore_stats.get("total_discovered", 0),
                "frontier_count": explore_stats.get("frontier_count", 0),
            })
            self._csv_steps_logged += 1
            if self._csv_steps_logged % 500 == 0:
                self._csv_steps_file.flush()

            # Track per-episode accumulators for CSV
            if min_dist > 0:
                self._csv_episode_min_dists.append(min_dist)
            self._csv_episode_velocities.append(abs(vel.get("v", 0.0)))
            if collision:
                self._csv_episode_bumps += 1
            self._csv_episode_cells = explore_stats.get("total_discovered", 0)
            self._csv_episode_goals = breakdown.get("goals_reached", 0)

    def _flush_csv_episode(self, wall_clock: float, success: bool = False):
        """Write one episode summary row to CSV."""
        if not self._csv_episodes_writer:
            return
        avg_min_d = (
            sum(self._csv_episode_min_dists) / len(self._csv_episode_min_dists)
            if self._csv_episode_min_dists else 0.0
        )
        avg_vel = (
            sum(self._csv_episode_velocities) / len(self._csv_episode_velocities)
            if self._csv_episode_velocities else 0.0
        )
        self._csv_episodes_writer.writerow({
            "wall_clock": f"{wall_clock:.2f}",
            "sim_id": self.domain_id,
            "episode": self.last_episode_num,
            "total_steps_at_end": self.total_steps,
            "episode_steps": self.current_episode_steps,
            "episode_return": f"{self.current_episode_return:.2f}",
            "outcome": "goal_reached" if success else "timeout",
            "goals_reached": self._csv_episode_goals,
            "cells_discovered": self._csv_episode_cells,
            "avg_min_distance": f"{avg_min_d:.4f}",
            "avg_velocity": f"{avg_vel:.4f}",
            "bumps": self._csv_episode_bumps,
            "training_phase": self.training_phase,
        })
        self._csv_episodes_file.flush()
        self._csv_episodes_logged += 1
        # Reset accumulators
        self._csv_episode_min_dists = []
        self._csv_episode_velocities = []
        self._csv_episode_bumps = 0

    def update_goal_status(self, status: Dict):
        self.goal_status = status
        self.goal_status_time = time.time()

    def get_summary(self) -> Dict:
        recent_rewards = list(self.rewards)[-100:] if self.rewards else [0]
        recent_min_dists = [x[1] for x in list(self.min_distances)[-100:]] if self.min_distances else [0]
        goal_status_fresh = (time.time() - self.goal_status_time) < 2.0
        total_outcomes = self.successes + self.timeouts

        return {
            "domain_id": self.domain_id,
            "namespace": self.namespace,
            "display_name": self.display_name,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "goals_reached": self.goals_reached,
            "cells_discovered": self.cells_discovered,
            "frontiers": self.frontiers,
            "current_episode_return": self.current_episode_return,
            "current_episode_steps": self.current_episode_steps,
            "avg_reward_100": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0,
            "avg_episode_return": sum(self.episode_returns) / len(self.episode_returns) if self.episode_returns else 0,
            "avg_min_distance": sum(recent_min_dists) / len(recent_min_dists) if recent_min_dists else 0,
            "min_min_distance": min(recent_min_dists) if recent_min_dists else 0,
            "bumps": self.bumps,
            "collisions": self.collisions,
            "successes": self.successes,
            "timeouts": self.timeouts,
            "collision_rate": 0,  # v5: collisions don't end episodes
            "success_rate": self.successes / total_outcomes if total_outcomes > 0 else 0,
            "active": time.time() - self.last_update < 5.0,
            "goal_status": self.goal_status if goal_status_fresh else {},
            "training_phase": self.training_phase,
            "expert_avg_return": self.expert_avg_return,
        }

    def get_plot_data(self, max_points: int = 300) -> Dict:
        step = max(1, len(self.rewards) // max_points)

        rewards = list(self.rewards)[::step]
        timestamps = list(self.timestamps)[::step]
        t0 = timestamps[0] if timestamps else 0
        rel_times = [(t - t0) for t in timestamps]

        episode_returns = list(self.episode_returns)
        episode_indices = list(range(1, len(episode_returns) + 1))

        components_data = {}
        for key, values in self.components.items():
            vals = list(values)[::step]
            if vals:
                t0_comp = vals[0][0]
                components_data[key] = {
                    "times": [(v[0] - t0_comp) for v in vals],
                    "values": [v[1] for v in vals]
                }

        # Min distance plot
        md_list = list(self.min_distances)[::step]
        if md_list:
            t0_md = md_list[0][0]
            md_times = [(v[0] - t0_md) for v in md_list]
            md_values = [v[1] for v in md_list]
        else:
            md_times, md_values = [], []

        # Velocity plot
        vel_list = list(self.velocities)[::step]
        if vel_list:
            t0_vel = vel_list[0][0]
            vel_times = [(v[0] - t0_vel) for v in vel_list]
            vel_values = [v[1] for v in vel_list]
        else:
            vel_times, vel_values = [], []

        # Zone counts (last 500 steps)
        zone_counts = {"FREE": 0, "AWARE": 0, "CAUTION": 0, "DANGER": 0, "EMERGENCY": 0, "CRITICAL": 0}
        for _, zone in list(self.nav_zones)[-500:]:
            if zone in zone_counts:
                zone_counts[zone] += 1

        # Episode outcomes
        outcomes = {
            "collisions": self.collisions,
            "successes": self.successes,
            "timeouts": self.timeouts,
        }

        return {
            "domain_id": self.domain_id,
            "display_name": self.display_name,
            "rewards": {"times": rel_times, "values": rewards},
            "episode_returns": {"episodes": episode_indices, "values": episode_returns},
            "components": components_data,
            "min_distance": {"times": md_times, "values": md_values},
            "velocity": {"times": vel_times, "values": vel_values},
            "nav_zones": zone_counts,
            "outcomes": outcomes,
        }


# =============================================================================
# ROS2 Collectors
# =============================================================================

class DomainCollector:
    def __init__(self, domain_id: int, stats_store: Dict[str, DomainStats], namespaces: List[str] = None):
        self.domain_id = domain_id
        self.stats = stats_store
        self.namespaces = namespaces or [""]
        self.running = False

        self.context = Context()
        rclpy.init(context=self.context, domain_id=domain_id)
        self.node = rclpy.create_node(f"monitor_{domain_id}", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)

        self._subs = []
        self._setup_subscriptions()

    def _setup_subscriptions(self):
        for ns in self.namespaces:
            self._subscribe_namespace(ns)

    def _subscribe_namespace(self, namespace: str):
        if namespace:
            reward_topic = f"/{namespace}/reward"
            breakdown_topic = f"/{namespace}/reward_breakdown"
            display_name = f"D{self.domain_id}:{namespace}"
        else:
            reward_topic = "/reward"
            breakdown_topic = "/reward_breakdown"
            display_name = f"Domain {self.domain_id}"

        stats_key = f"{self.domain_id}:{namespace}"

        if stats_key not in self.stats:
            self.stats[stats_key] = DomainStats(
                domain_id=self.domain_id,
                namespace=namespace,
                display_name=display_name
            )

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=10)

        self._subs.append(self.node.create_subscription(Float32, reward_topic, lambda msg, k=stats_key: self._reward_cb(msg, k), qos))
        self._subs.append(self.node.create_subscription(StringMsg, breakdown_topic, lambda msg, k=stats_key: self._breakdown_cb(msg, k), qos))

        goal_status_topics = [
            "/sam3_goal_generator/status",
            f"/{namespace}/sam3_goal_generator/status" if namespace else None,
        ]
        for topic in goal_status_topics:
            if topic:
                try:
                    self._subs.append(self.node.create_subscription(StringMsg, topic, lambda msg, k=stats_key: self._goal_status_cb(msg, k), qos))
                except:
                    pass

    def _reward_cb(self, msg: Float32, key: str):
        if key in self.stats:
            self.stats[key].add_reward(msg.data, time.time())

    def _breakdown_cb(self, msg: StringMsg, key: str):
        try:
            if key in self.stats:
                self.stats[key].add_breakdown(json.loads(msg.data))
        except:
            pass

    def _goal_status_cb(self, msg: StringMsg, key: str):
        try:
            if key in self.stats:
                self.stats[key].update_goal_status(json.loads(msg.data))
        except:
            pass

    def discover_namespaces(self) -> List[str]:
        discovered = []
        for topic, _ in self.node.get_topic_names_and_types():
            if topic.endswith("/reward"):
                ns = topic.rsplit("/reward", 1)[0].lstrip("/")
                if ns not in self.namespaces:
                    discovered.append(ns)
                    self._subscribe_namespace(ns)
                    self.namespaces.append(ns)
        return discovered

    def spin(self):
        self.running = True
        while self.running:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except:
                pass

    def stop(self):
        self.running = False
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            rclpy.shutdown(context=self.context)
        except:
            pass


class CameraCollector:
    def __init__(self, domain_id: int, fps: float = DEFAULT_CAMERA_FPS, quality: int = DEFAULT_CAMERA_JPEG_QUALITY, max_width: int = DEFAULT_CAMERA_MAX_WIDTH):
        self.domain_id = domain_id
        self.fps = max(1.0, fps)
        self.quality = int(max(10, min(95, quality)))
        self.max_width = int(max_width)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ts: float = 0.0

        self._context = Context()
        rclpy.init(context=self._context, domain_id=domain_id)
        self._node = rclpy.create_node(f"cam_{domain_id}", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)

        self._sub = None
        self._sub_topic: Optional[str] = None
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._running:
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except:
                pass

    def ensure_subscription(self, topic: str) -> bool:
        topic = (topic or "").strip()
        if not topic:
            return False
        if not topic.startswith("/"):
            topic = "/" + topic
        if self._sub is not None and self._sub_topic == topic:
            return True
        try:
            if self._sub is not None:
                try:
                    self._node.destroy_subscription(self._sub)
                except:
                    pass
                self._sub = None
            self._sub = self._node.create_subscription(RosImage, topic, self._image_cb, qos_profile_sensor_data)
            self._sub_topic = topic
            return True
        except:
            self._sub_topic = None
            self._sub = None
            return False

    def _image_cb(self, msg: RosImage):
        now = time.time()
        with self._lock:
            if (now - self._latest_ts) < (1.0 / self.fps):
                return
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.max_width and cv_img is not None:
                h, w = cv_img.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / float(w)
                    cv_img = cv2.resize(cv_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_ts = now
        except:
            pass

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def stop(self):
        self._running = False
        try:
            if self._sub:
                self._node.destroy_subscription(self._sub)
            self._executor.shutdown()
            self._node.destroy_node()
            rclpy.shutdown(context=self._context)
        except:
            pass


class MultiDomainManager:
    def __init__(self, stats_store: Dict[str, DomainStats]):
        self.stats = stats_store
        self.collectors: Dict[int, DomainCollector] = {}
        self.threads: Dict[int, threading.Thread] = {}
        self.cameras: Dict[int, CameraCollector] = {}
        self.domain_image_topics: Dict[int, List[str]] = {}
        self.lock = threading.Lock()

    def add_domain(self, domain_id: int, namespaces: List[str] = None) -> bool:
        with self.lock:
            if domain_id in self.collectors:
                return False
            try:
                collector = DomainCollector(domain_id, self.stats, namespaces)
                self.collectors[domain_id] = collector
                thread = threading.Thread(target=collector.spin, daemon=True)
                thread.start()
                self.threads[domain_id] = thread
                if domain_id not in self.cameras:
                    self.cameras[domain_id] = CameraCollector(domain_id)
                return True
            except:
                return False

    def _scan_domain_topics(self, did: int) -> Tuple[int, bool, List[str]]:
        ctx = Context()
        image_topics: List[str] = []
        try:
            rclpy.init(context=ctx, domain_id=did)
            node = rclpy.create_node(f"scan_{did}", context=ctx)
            time.sleep(0.3)
            topics = node.get_topic_names_and_types()
            has_reward = any(t[0].endswith("/reward") for t in topics)
            for tname, ttypes in topics:
                if "sensor_msgs/msg/Image" in ttypes:
                    image_topics.append(tname)
            node.destroy_node()
            rclpy.shutdown(context=ctx)
            return did, has_reward, sorted(image_topics)
        except:
            try:
                rclpy.shutdown(context=ctx)
            except:
                pass
            return did, False, []

    def discover_domains(self, start: int, end: int) -> List[int]:
        discovered = []
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(self._scan_domain_topics, range(start, end + 1)))
        for did, has_reward, image_topics in results:
            if image_topics:
                self.domain_image_topics[did] = image_topics
            elif did not in self.domain_image_topics:
                self.domain_image_topics[did] = []
            if has_reward and did not in self.collectors:
                if self.add_domain(did):
                    discovered.append(did)
        return discovered

    def discover_namespaces_all(self):
        for c in self.collectors.values():
            try:
                c.discover_namespaces()
            except:
                pass

    def get_image_topics(self, domain_id: int) -> List[str]:
        return self.domain_image_topics.get(domain_id, [])

    def get_camera(self, domain_id: int) -> Optional[CameraCollector]:
        return self.cameras.get(domain_id)

    def stop_all(self):
        for c in self.collectors.values():
            c.stop()
        for cam in self.cameras.values():
            cam.stop()


# =============================================================================
# Dashboard HTML
# =============================================================================

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RL Training Monitor</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f1117; --bg-raised: #161920; --bg-card: #1c1f2b;
      --bg-card-hover: #222636; --border: #2a2e3d; --border-light: #353a4d;
      --text: #e2e8f0; --text-dim: #94a3b8; --text-muted: #64748b;
      --accent: #6366f1; --accent-glow: rgba(99,102,241,0.15);
      --green: #22c55e; --green-dim: rgba(34,197,94,0.15);
      --red: #ef4444; --red-dim: rgba(239,68,68,0.12);
      --amber: #f59e0b; --amber-dim: rgba(245,158,11,0.12);
      --cyan: #06b6d4; --purple: #a855f7; --orange: #f97316;
      --pink: #ec4899; --lime: #84cc16; --teal: #14b8a6;
      --radius: 12px; --radius-sm: 8px;
      --shadow: 0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2);
      --shadow-lg: 0 4px 16px rgba(0,0,0,0.4);
      --mono: 'JetBrains Mono', 'Fira Code', monospace;
      --sans: 'DM Sans', -apple-system, sans-serif;
    }
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family:var(--sans); background:var(--bg); color:var(--text); line-height:1.5; min-height:100vh; }

    /* ── Header ── */
    .header {
      background:var(--bg-raised); border-bottom:1px solid var(--border);
      padding:14px 28px; display:flex; justify-content:space-between; align-items:center;
      position:sticky; top:0; z-index:100; backdrop-filter:blur(12px);
    }
    .header-left { display:flex; align-items:center; gap:14px; }
    .header h1 { font-family:var(--mono); font-size:15px; font-weight:700; letter-spacing:-0.3px; }
    .header h1 span { color:var(--accent); }
    .version-tag {
      font-family:var(--mono); font-size:10px; font-weight:600; letter-spacing:0.5px;
      background:var(--accent-glow); color:var(--accent); padding:3px 10px; border-radius:20px;
      border:1px solid rgba(99,102,241,0.25);
    }
    .phase-badge {
      font-family:var(--mono); font-size:11px; font-weight:600; padding:5px 14px;
      border-radius:20px; letter-spacing:0.3px; transition:all 0.3s;
    }
    .phase-badge.expert { background:var(--amber-dim); color:var(--amber); border:1px solid rgba(245,158,11,0.3); }
    .phase-badge.random { background:rgba(168,85,247,0.12); color:var(--purple); border:1px solid rgba(168,85,247,0.3); }
    .phase-badge.rl { background:var(--green-dim); color:var(--green); border:1px solid rgba(34,197,94,0.3); }
    .phase-badge.unknown { background:rgba(100,116,139,0.12); color:var(--text-muted); border:1px solid var(--border); }
    .conn-badge {
      font-family:var(--mono); font-size:10px; padding:4px 12px; border-radius:20px;
      background:var(--bg-card); border:1px solid var(--border); color:var(--text-muted);
    }
    .conn-badge.live { color:var(--green); border-color:rgba(34,197,94,0.3); background:var(--green-dim); }

    /* ── Layout ── */
    .container { padding:20px 28px; max-width:1600px; margin:0 auto; }

    /* ── Tabs ── */
    .tabs { display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }
    .tab {
      font-family:var(--mono); font-size:12px; font-weight:500; padding:8px 16px;
      background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius-sm);
      cursor:pointer; color:var(--text-muted); transition:all 0.2s; display:flex; align-items:center; gap:8px;
    }
    .tab:hover { border-color:var(--accent); color:var(--text-dim); }
    .tab.active { background:var(--accent); border-color:var(--accent); color:#fff; box-shadow:0 0 20px var(--accent-glow); }
    .tab .dot { width:7px; height:7px; border-radius:50%; background:var(--text-muted); flex-shrink:0; }
    .tab.live .dot { background:var(--green); box-shadow:0 0 6px var(--green); animation:pulse 2s infinite; }
    .tab.active .dot { background:rgba(255,255,255,0.5); box-shadow:none; }
    .tab.active.live .dot { background:#fff; box-shadow:0 0 6px #fff; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

    /* ── Cards ── */
    .card {
      background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius);
      padding:20px; box-shadow:var(--shadow); margin-bottom:16px; transition:border-color 0.2s;
    }
    .card:hover { border-color:var(--border-light); }
    .card h3 {
      font-family:var(--mono); font-size:11px; font-weight:600; color:var(--text-muted);
      text-transform:uppercase; letter-spacing:1px; margin-bottom:16px;
      display:flex; align-items:center; gap:8px;
    }
    .card h3::before { content:''; width:3px; height:14px; background:var(--accent); border-radius:2px; }

    /* ── Stats Row ── */
    .stats-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px; margin-bottom:20px; }
    .stat {
      background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius-sm);
      padding:14px 16px; transition:border-color 0.2s;
    }
    .stat:hover { border-color:var(--border-light); }
    .stat .label { font-family:var(--mono); font-size:10px; font-weight:600; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }
    .stat .value { font-family:var(--mono); font-size:20px; font-weight:700; margin-top:4px; line-height:1.2; }
    .stat .sub { font-family:var(--mono); font-size:10px; color:var(--text-muted); margin-top:3px; }
    .green { color:var(--green)!important; }
    .red { color:var(--red)!important; }
    .blue { color:var(--accent)!important; }
    .amber { color:var(--amber)!important; }
    .purple { color:var(--purple)!important; }
    .cyan { color:var(--cyan)!important; }

    /* ── Grid Layouts ── */
    .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
    .grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }
    .grid-2-1 { display:grid; grid-template-columns:2fr 1fr; gap:16px; }
    .grid-1-2 { display:grid; grid-template-columns:1fr 2fr; gap:16px; }
    @media(max-width:1200px) { .grid-3 { grid-template-columns:1fr 1fr; } }
    @media(max-width:900px) { .grid-2,.grid-3,.grid-2-1,.grid-1-2 { grid-template-columns:1fr; } }
    .chart { height:260px; }
    .chart-tall { height:320px; }

    /* ── Camera ── */
    .camera-container { position:relative; background:#000; border-radius:var(--radius-sm); overflow:hidden; }
    .camera-img { width:100%; height:280px; object-fit:contain; display:block; }

    /* ── Outcome Bar ── */
    .outcome-bar { display:flex; height:6px; border-radius:3px; overflow:hidden; margin-bottom:16px; background:var(--border); }
    .outcome-bar .seg { transition:width 0.5s ease; }
    .outcome-bar .seg-success { background:var(--green); }
    .outcome-bar .seg-timeout { background:var(--amber); }

    /* ── Goal Status ── */
    .goal-status {
      margin-top:12px; padding:12px 16px; background:var(--bg-raised);
      border:1px solid var(--border); border-radius:var(--radius-sm); font-size:12px;
    }
    .goal-status.found { border-color:rgba(34,197,94,0.3); }
    .goal-status.searching { border-color:rgba(245,158,11,0.3); }
    .goal-status .status-row { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid var(--border); }
    .goal-status .status-row:last-child { border-bottom:none; }
    .goal-status .status-label { font-family:var(--mono); color:var(--text-muted); font-size:11px; }
    .goal-status .status-value { font-family:var(--mono); font-weight:600; font-size:11px; }
    .goal-status .status-value.found { color:var(--green); }
    .goal-status .status-value.searching { color:var(--amber); }

    /* ── Expert Comparison Box ── */
    .expert-compare {
      background:linear-gradient(135deg, rgba(99,102,241,0.06) 0%, rgba(168,85,247,0.06) 100%);
      border:1px solid rgba(99,102,241,0.2); border-radius:var(--radius);
      padding:16px 20px; margin-bottom:16px; display:flex; align-items:center; justify-content:space-between;
      gap:24px;
    }
    .expert-compare .ec-item { text-align:center; }
    .expert-compare .ec-label { font-family:var(--mono); font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }
    .expert-compare .ec-value { font-family:var(--mono); font-size:22px; font-weight:700; margin-top:2px; }
    .expert-compare .ec-vs { font-family:var(--mono); font-size:12px; color:var(--text-muted); font-weight:600; }
    .expert-compare .ec-verdict {
      font-family:var(--mono); font-size:11px; font-weight:700; padding:4px 12px;
      border-radius:20px; letter-spacing:0.3px;
    }
    .ec-verdict.winning { background:var(--green-dim); color:var(--green); }
    .ec-verdict.losing { background:var(--amber-dim); color:var(--amber); }

    /* ── Reward Legend ── */
    .reward-legend { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
    .reward-legend .item { display:flex; align-items:center; gap:5px; font-family:var(--mono); font-size:10px; color:var(--text-dim); }
    .reward-legend .swatch { width:10px; height:3px; border-radius:2px; }

    /* ── Overview Summary ── */
    .summary-box {
      background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius);
      padding:24px; margin-bottom:20px;
    }
    .summary-box h3 { font-family:var(--mono); font-size:12px; font-weight:600; color:var(--accent); text-transform:uppercase; letter-spacing:1px; margin-bottom:20px; }
    .summary-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:20px; }
    .summary-stat { text-align:center; }
    .summary-stat .label { font-family:var(--mono); font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.5px; }
    .summary-stat .value { font-family:var(--mono); font-size:22px; font-weight:700; margin-top:4px; }

    .empty { text-align:center; padding:80px; color:var(--text-muted); font-family:var(--mono); font-size:13px; }
  </style>
</head>
<body>
  <div class="header">
    <div class="header-left">
      <h1><span>//</span> RL Training Monitor</h1>
      <span class="version-tag">v5 — NEVER TERMINATE</span>
      <span class="phase-badge unknown" id="phaseBadge">—</span>
    </div>
    <span class="conn-badge" id="badge">Connecting...</span>
  </div>
  <div class="container">
    <div class="tabs" id="tabs"></div>
    <div id="content"><div class="empty">Scanning for ROS2 domains...</div></div>
  </div>

<script>
const socket = io();
let domains = {};
let selected = 'all';

const C = {
  accent:'#6366f1', green:'#22c55e', red:'#ef4444', amber:'#f59e0b',
  purple:'#a855f7', cyan:'#06b6d4', orange:'#f97316', pink:'#ec4899',
  lime:'#84cc16', teal:'#14b8a6', gray:'#64748b',
  greenDim:'rgba(34,197,94,0.5)', redDim:'rgba(239,68,68,0.5)',
};

const compColors = {
  discovery:C.purple, revisit:C.pink,
  proximity:C.orange, collision:C.red,
  progress:C.green, goal:C.amber,
  step:C.gray, timeout:'#92400e',
};

const plotBg = 'rgba(0,0,0,0)';
const gridColor = 'rgba(255,255,255,0.04)';
const zeroLine = 'rgba(255,255,255,0.08)';
const layoutBase = {
  paper_bgcolor:plotBg, plot_bgcolor:plotBg,
  font:{family:'JetBrains Mono, monospace', color:'#94a3b8', size:10},
  margin:{l:50, r:16, t:28, b:36},
  xaxis:{gridcolor:gridColor, zerolinecolor:zeroLine, tickfont:{size:9}},
  yaxis:{gridcolor:gridColor, zerolinecolor:zeroLine, tickfont:{size:9}, rangemode:'tozero'},
  showlegend:false,
};
const cfg = {displayModeBar:false, responsive:true};

socket.on('connect', () => { const b=document.getElementById('badge'); b.textContent='Connected'; b.className='conn-badge live'; });
socket.on('disconnect', () => { const b=document.getElementById('badge'); b.textContent='Disconnected'; b.className='conn-badge'; });

socket.on('update', data => {
  domains = data.domains || {};
  const n = Object.keys(domains).length;
  const b = document.getElementById('badge');
  b.textContent = n + ' Domain' + (n!==1?'s':'');
  b.className = 'conn-badge live';

  // Update phase badge from first active domain
  const firstActive = Object.values(domains).find(d => d.summary.active);
  if (firstActive) updatePhaseBadge(firstActive.summary.training_phase);

  renderTabs();
  if (selected === 'all') updateAllView();
  else if (domains[selected]) updateDomainView(selected);
});

function updatePhaseBadge(phase) {
  const el = document.getElementById('phaseBadge');
  if (!el) return;
  const map = {
    'EXPERT_DEMO': ['EXPERT DEMO', 'expert'],
    'RANDOM_EXPLORE': ['RANDOM EXPLORE', 'random'],
    'RL_POLICY': ['RL POLICY', 'rl'],
  };
  const [text, cls] = map[phase] || [phase||'—', 'unknown'];
  el.textContent = text;
  el.className = 'phase-badge ' + cls;
}

function renderTabs() {
  const keys = Object.keys(domains);
  let h = '<div class="tab '+(selected==='all'?'active':'')+'" onclick="sel(\'all\')">Overview</div>';
  keys.forEach(k => {
    const s = domains[k].summary;
    h += '<div class="tab '+(selected===k?'active':'')+' '+(s.active?'live':'')+'" onclick="sel(\''+k+'\')">' +
         '<div class="dot"></div>'+s.display_name+'</div>';
  });
  document.getElementById('tabs').innerHTML = h;
}

function sel(k) { selected = k; render(); }
function render() { renderTabs(); if (selected==='all') renderAllView(); else renderDomainView(selected); }

// ═══════════ ALL VIEW ═══════════

function renderAllView() {
  const vals = Object.values(domains);
  if (!vals.length) { document.getElementById('content').innerHTML = '<div class="empty">Scanning for active ROS2 domains...</div>'; return; }
  const totSteps = vals.reduce((a,d)=>a+d.summary.total_steps,0);
  const totEp = vals.reduce((a,d)=>a+d.summary.episode_count,0);
  const totBumps = vals.reduce((a,d)=>a+d.summary.bumps,0);
  const totSuc = vals.reduce((a,d)=>a+d.summary.successes,0);
  const totTo = vals.reduce((a,d)=>a+d.summary.timeouts,0);
  const active = vals.filter(d=>d.summary.active).length;
  const rets = vals.filter(d=>d.summary.episode_count>0).map(d=>d.summary.avg_episode_return);
  const avgRet = rets.length ? rets.reduce((a,b)=>a+b,0)/rets.length : 0;
  const expertAvg = vals[0]?.summary?.expert_avg_return || 0;

  let h = '<div class="summary-box"><h3>Cross-Domain Summary</h3><div class="summary-grid">' +
    ss('Total Steps', totSteps.toLocaleString()) +
    ss('Episodes', totEp) +
    ss('Bumps', totBumps, C.orange) +
    ss('Goals', totSuc, C.green) +
    ss('Timeouts', totTo, C.amber) +
    ss('Expert Avg', expertAvg.toFixed(1), C.purple) +
    ss('Avg Return', avgRet.toFixed(1), avgRet>=0?C.green:C.red) +
    ss('Active', active+'/'+vals.length, C.green) +
  '</div></div>';
  h += '<div class="card"><h3>Episode Returns — All Domains</h3><div id="allChart" class="chart-tall"></div></div>';
  document.getElementById('content').innerHTML = h;
  drawAllChart();
}

function updateAllView() {
  if (!document.getElementById('allChart')) { renderAllView(); return; }
  // Update summary stats in place
  const vals = Object.values(domains);
  drawAllChart();
}

function drawAllChart() {
  const traces = [];
  const cls = [C.accent, C.green, C.purple, C.cyan, C.orange, C.pink];
  let ci = 0;
  Object.values(domains).forEach(d => {
    const p = d.plot_data;
    if (p.episode_returns.values.length) {
      traces.push({x:p.episode_returns.episodes, y:p.episode_returns.values, type:'scatter', mode:'lines',
        name:p.display_name, line:{color:cls[ci++%cls.length], width:2}});
    }
  });
  // Add expert baseline
  const expertAvg = Object.values(domains)[0]?.summary?.expert_avg_return;
  if (expertAvg && traces.length) {
    const maxEp = Math.max(...traces.map(t => Math.max(...t.x)));
    traces.push({x:[1, maxEp], y:[expertAvg, expertAvg], type:'scatter', mode:'lines',
      name:'Expert Baseline', line:{color:C.amber, width:2, dash:'dot'}});
  }
  if (traces.length) {
    Plotly.react('allChart', traces, {
      ...layoutBase, showlegend:true, legend:{bgcolor:'rgba(0,0,0,0)', font:{size:10, color:'#94a3b8'}},
      xaxis:{...layoutBase.xaxis, title:{text:'Episode', font:{size:10}}},
      yaxis:{...layoutBase.yaxis, title:{text:'Return', font:{size:10}}, rangemode:'tozero'}
    }, cfg);
  }
}

function ss(label, value, color) {
  return '<div class="summary-stat"><div class="label">'+label+'</div><div class="value" style="'+(color?'color:'+color:'')+'">' + value + '</div></div>';
}

// ═══════════ DOMAIN VIEW ═══════════

function renderDomainView(k) {
  const d = domains[k]; if (!d) return;
  const s = d.summary;
  const defaultTopic = '/sam3_goal_generator/visualization';
  const total = s.successes + s.timeouts;
  const sucPct = total>0 ? (s.successes/total*100).toFixed(0) : 0;
  const toPct  = total>0 ? (s.timeouts/total*100).toFixed(0) : 0;

  // Expert vs RL comparison bar
  let expertCompare = '';
  if (s.expert_avg_return !== 0 || s.training_phase === 'RL_POLICY') {
    const rlAvg = s.avg_episode_return;
    const expAvg = s.expert_avg_return;
    const winning = rlAvg > expAvg;
    expertCompare = '<div class="expert-compare">' +
      '<div class="ec-item"><div class="ec-label">Expert Baseline</div><div class="ec-value" style="color:'+C.amber+'">'+expAvg.toFixed(1)+'</div></div>' +
      '<div class="ec-vs">vs</div>' +
      '<div class="ec-item"><div class="ec-label">RL Average</div><div class="ec-value" style="color:'+(winning?C.green:C.red)+'">'+rlAvg.toFixed(1)+'</div></div>' +
      '<div class="ec-verdict '+(winning?'winning':'losing')+'">'+(winning?'RL SURPASSED':'EXPERT LEADS')+'</div>' +
    '</div>';
  }

  let h = expertCompare;

  // Stats row
  h += '<div class="stats-row">' +
    statBox('Steps', s.total_steps.toLocaleString(), '', '') +
    statBox('Episodes', s.episode_count, '', '') +
    statBox('Bumps', s.bumps, 'amber', 'no termination') +
    statBox('Goals', s.successes, 'green', sucPct+'% of eps') +
    statBox('Timeouts', s.timeouts, 'amber', toPct+'% of eps') +
    statBox('Cells', s.cells_discovered, 'cyan', s.frontiers+' frontiers') +
    statBox('Cur Return', s.current_episode_return.toFixed(1), s.current_episode_return>=0?'green':'red', 'step '+s.current_episode_steps) +
    statBox('Avg Return', s.avg_episode_return.toFixed(1), s.avg_episode_return>=0?'green':'red', 'last '+Math.min(s.episode_count,100)+' eps') +
    statBox('Avg Min Dist', s.avg_min_distance.toFixed(2)+'m', s.avg_min_distance>0.4?'green':'amber', 'closest '+s.min_min_distance.toFixed(2)+'m') +
  '</div>';

  // Outcome bar
  if (total > 0) {
    h += '<div class="outcome-bar">';
    if (s.successes>0) h += '<div class="seg seg-success" style="width:'+sucPct+'%"></div>';
    if (s.timeouts>0) h += '<div class="seg seg-timeout" style="width:'+toPct+'%"></div>';
    h += '</div>';
  }

  // Row 1: Camera + Episode Returns (hero chart)
  h += '<div class="grid-2">' +
    '<div class="card"><h3>Camera Feed</h3>' +
    '<div class="camera-container"><img class="camera-img" id="camImg" src="/stream/'+s.domain_id+'?topic='+encodeURIComponent(defaultTopic)+'"/></div>' +
    '<div class="goal-status" id="goalStatus"></div></div>' +
    '<div class="card"><h3>Episode Returns vs Expert</h3><div id="epChart" class="chart-tall"></div></div>' +
  '</div>';

  // Row 2: Reward timeseries + Components
  h += '<div class="grid-2">' +
    '<div class="card"><h3>Real-Time Reward</h3><div id="rewChart" class="chart"></div></div>' +
    '<div class="card"><h3>Reward Components</h3><div id="compChart" class="chart"></div>' +
    '<div class="reward-legend" id="compLegend"></div></div>' +
  '</div>';

  // Row 3: Min distance + Velocity
  h += '<div class="grid-2">' +
    '<div class="card"><h3>Min Distance to Obstacle</h3><div id="minDistChart" class="chart"></div></div>' +
    '<div class="card"><h3>Velocity Output</h3><div id="velChart" class="chart"></div></div>' +
  '</div>';

  // Row 4: Outcomes + Zone Distribution
  h += '<div class="grid-2">' +
    '<div class="card"><h3>Episode Outcomes</h3><div id="outcomeChart" class="chart"></div></div>' +
    '<div class="card"><h3>Nav Zone Distribution</h3><div id="zoneChart" class="chart"></div></div>' +
  '</div>';

  document.getElementById('content').innerHTML = h;
  updateDomainView(k);
}

function statBox(label, value, colorClass, sub) {
  return '<div class="stat"><div class="label">'+label+'</div><div class="value '+colorClass+'">'+value+'</div>'+(sub?'<div class="sub">'+sub+'</div>':'')+'</div>';
}

function updateDomainView(k) {
  const d = domains[k]; if (!d) return;
  const s = d.summary, p = d.plot_data;

  updateGoalStatus(s.goal_status);

  // Reward chart — fixed at zero
  if (p.rewards.values.length) {
    Plotly.react('rewChart', [{x:p.rewards.times, y:p.rewards.values, type:'scatter', mode:'lines',
      line:{color:C.accent, width:1.5}, fill:'tozeroy', fillcolor:'rgba(99,102,241,0.08)'}],
    {...layoutBase, xaxis:{...layoutBase.xaxis, title:{text:'Time (s)', font:{size:9}}},
     yaxis:{...layoutBase.yaxis, rangemode:'tozero'}}, cfg);
  }

  // Episode returns + rolling avg + expert baseline
  if (p.episode_returns.values.length) {
    const ra = [];
    for (let i=0; i<p.episode_returns.values.length; i++) {
      const w = p.episode_returns.values.slice(Math.max(0,i-9), i+1);
      ra.push(w.reduce((a,b)=>a+b,0)/w.length);
    }
    const traces = [
      {x:p.episode_returns.episodes, y:p.episode_returns.values, type:'scatter', mode:'lines',
        name:'Return', line:{color:C.accent, width:1.5}, opacity:0.5},
      {x:p.episode_returns.episodes, y:ra, type:'scatter', mode:'lines',
        name:'10-ep Avg', line:{color:C.green, width:2.5}},
    ];
    // Expert baseline line
    const expertAvg = s.expert_avg_return;
    if (expertAvg !== 0) {
      const maxEp = Math.max(...p.episode_returns.episodes);
      traces.push({x:[1, maxEp], y:[expertAvg, expertAvg], type:'scatter', mode:'lines',
        name:'Expert Baseline ('+expertAvg.toFixed(1)+')', line:{color:C.amber, width:2, dash:'dot'}});
    }
    // Zero line
    const maxEp2 = Math.max(...p.episode_returns.episodes);
    traces.push({x:[1, maxEp2], y:[0, 0], type:'scatter', mode:'lines',
      name:'Zero', line:{color:'rgba(255,255,255,0.15)', width:1}, showlegend:false});

    Plotly.react('epChart', traces, {
      ...layoutBase, showlegend:true,
      legend:{bgcolor:'rgba(0,0,0,0)', font:{size:9, color:'#94a3b8'}, x:0.02, y:0.98},
      xaxis:{...layoutBase.xaxis, title:{text:'Episode', font:{size:9}}},
      yaxis:{...layoutBase.yaxis, title:{text:'Return', font:{size:9}}, rangemode:'tozero'}
    }, cfg);
  }

  // Min distance with threshold lines
  if (p.min_distance.values.length) {
    const t0 = p.min_distance.times[0];
    const tEnd = p.min_distance.times[p.min_distance.times.length-1];
    Plotly.react('minDistChart', [
      {x:p.min_distance.times, y:p.min_distance.values, type:'scatter', mode:'lines',
        name:'Min Dist', line:{color:C.orange, width:1.5}, fill:'tozeroy', fillcolor:'rgba(249,115,22,0.06)'},
      {x:[t0,tEnd], y:[0.65,0.65], type:'scatter', mode:'lines', name:'Proximity',
        line:{color:C.amber, width:1, dash:'dot'}, hoverinfo:'name'},
      {x:[t0,tEnd], y:[0.30,0.30], type:'scatter', mode:'lines', name:'Collision',
        line:{color:C.red, width:1, dash:'dash'}, hoverinfo:'name'}
    ], {...layoutBase, showlegend:true, legend:{bgcolor:'rgba(0,0,0,0)', font:{size:9, color:'#94a3b8'}},
      xaxis:{...layoutBase.xaxis, title:{text:'Time (s)', font:{size:9}}},
      yaxis:{...layoutBase.yaxis, title:{text:'meters', font:{size:9}}, rangemode:'tozero'}}, cfg);
  }

  // Outcomes donut
  const oc = p.outcomes;
  const ovals=[], olabels=[], ocolors=[];
  if (oc.successes>0) { ovals.push(oc.successes); olabels.push('Goal ('+oc.successes+')'); ocolors.push(C.green); }
  if (oc.timeouts>0)  { ovals.push(oc.timeouts);  olabels.push('Timeout ('+oc.timeouts+')'); ocolors.push(C.amber); }
  if (ovals.length) {
    Plotly.react('outcomeChart', [{values:ovals, labels:olabels, type:'pie',
      marker:{colors:ocolors}, textinfo:'label+percent', textfont:{size:10, color:'#e2e8f0'},
      hole:0.55, sort:false}],
    {...layoutBase}, cfg);
  }

  // Reward components
  const traces = [], legendItems = [];
  const compOrder = ['discovery','revisit','proximity','progress','goal','step','collision','timeout'];
  compOrder.forEach(n => {
    const cd = p.components[n];
    if (cd && cd.values.length) {
      const color = compColors[n] || C.gray;
      traces.push({x:cd.times, y:cd.values, type:'scatter', mode:'lines', name:n, line:{color, width:1.5}});
      legendItems.push({name:n, color});
    }
  });
  Object.keys(p.components).forEach(n => {
    if (!compOrder.includes(n)) {
      const cd = p.components[n];
      if (cd && cd.values.length) {
        traces.push({x:cd.times, y:cd.values, type:'scatter', mode:'lines', name:n, line:{color:C.gray, width:1}});
        legendItems.push({name:n, color:C.gray});
      }
    }
  });
  if (traces.length) {
    Plotly.react('compChart', traces, {...layoutBase, showlegend:false,
      xaxis:{...layoutBase.xaxis, title:{text:'Time (s)', font:{size:9}}},
      yaxis:{...layoutBase.yaxis, rangemode:'tozero'}}, cfg);
    const el = document.getElementById('compLegend');
    if (el) el.innerHTML = legendItems.map(i =>
      '<div class="item"><div class="swatch" style="background:'+i.color+'"></div>'+i.name+'</div>'
    ).join('');
  }

  // Velocity — fixed zero
  if (p.velocity.values.length) {
    Plotly.react('velChart', [{x:p.velocity.times, y:p.velocity.values, type:'scatter', mode:'lines',
      line:{color:C.cyan, width:1.5}, fill:'tozeroy', fillcolor:'rgba(6,182,212,0.06)'}],
    {...layoutBase, xaxis:{...layoutBase.xaxis, title:{text:'Time (s)', font:{size:9}}},
     yaxis:{...layoutBase.yaxis, title:{text:'m/s', font:{size:9}}, rangemode:'tozero'}}, cfg);
  }

  // Nav zones donut
  const zc = {FREE:C.green, AWARE:C.lime, CAUTION:C.amber, DANGER:C.orange, EMERGENCY:C.red, CRITICAL:'#991b1b'};
  const zones = Object.entries(p.nav_zones).filter(z=>z[1]>0);
  if (zones.length) {
    Plotly.react('zoneChart', [{
      values:zones.map(z=>z[1]), labels:zones.map(z=>z[0]), type:'pie',
      marker:{colors:zones.map(z=>zc[z[0]]||C.gray)}, textinfo:'label+percent',
      textfont:{size:10, color:'#e2e8f0'}, hole:0.55, sort:false
    }], {...layoutBase}, cfg);
  }
}

function updateGoalStatus(gs) {
  const el = document.getElementById('goalStatus');
  if (!el) return;
  if (!gs || Object.keys(gs).length===0) {
    el.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:6px;font-family:var(--mono);font-size:11px;">No goal generator data</div>';
    el.className = 'goal-status'; return;
  }
  const found = gs.found===true;
  el.className = 'goal-status '+(found?'found':'searching');
  let html = row('Target', gs.target||'\u2014');
  html += row('Status', found?'\u2713 Found':'\u25CB Searching...', found?'found':'searching');
  if (found) {
    html += row('Distance', gs.distance?gs.distance.toFixed(2)+'m':'\u2014');
    html += row('Confidence', gs.confidence?(gs.confidence*100).toFixed(0)+'%':'\u2014');
    if (gs.depth_method) html += row('Depth', gs.depth_method);
    if (gs.goal_position) html += row('Goal', '('+gs.goal_position[0].toFixed(2)+', '+gs.goal_position[1].toFixed(2)+')');
  } else if (gs.message) {
    html += row('Info', gs.message);
  }
  if (gs.locked_goal) html += row('Locked', '('+gs.locked_goal[0].toFixed(2)+', '+gs.locked_goal[1].toFixed(2)+')', 'found');
  el.innerHTML = html;
}

function row(label, value, cls) {
  return '<div class="status-row"><span class="status-label">'+label+'</span><span class="status-value'+(cls?' '+cls:'')+'">'+value+'</span></div>';
}
</script>
</body>
</html>
"""


# =============================================================================
# Sync Event Monitor (watches checkpoints for weight averaging events)
# =============================================================================

class SyncMonitor:
    """Watches checkpoint dirs for _last_sync metadata changes."""

    def __init__(self, ckpt_dir: str, num_sims: int, experiment_dir: str, start_time: float):
        self.ckpt_dir = ckpt_dir
        self.num_sims = num_sims
        self.start_time = start_time
        self.last_sync_time_str = None
        self.sync_count = 0

        self.sync_file = open(os.path.join(experiment_dir, "sync_events.csv"), "w", newline="")
        self.sync_writer = csv.DictWriter(self.sync_file, fieldnames=SYNC_CSV_COLUMNS)
        self.sync_writer.writeheader()

    def check(self):
        """Check if a new sync happened by reading checkpoint metadata."""
        try:
            import torch
        except ImportError:
            return
        ckpt_path = os.path.join(self.ckpt_dir, "sim_0", "td3_safe_rl_agent.pt")
        if not os.path.exists(ckpt_path):
            return
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            sync_meta = ckpt.get("_last_sync")
            if sync_meta is None:
                return
            sync_time_str = sync_meta.get("time", "")
            if sync_time_str == self.last_sync_time_str:
                return
            self.last_sync_time_str = sync_time_str
            self.sync_count += 1
            now = time.time() - self.start_time
            self.sync_writer.writerow({
                "wall_clock": f"{now:.2f}",
                "sync_number": self.sync_count,
                "sim_steps": json.dumps(sync_meta.get("steps_at_sync", [])),
                "weighted": sync_meta.get("weighted", False),
            })
            self.sync_file.flush()
            print(f"  [SYNC] #{self.sync_count} detected at {now:.0f}s")
        except Exception:
            pass  # Checkpoint might be mid-write

    def close(self):
        self.sync_file.close()


# =============================================================================
# Graph Generation (reads CSVs, produces matplotlib figures)
# =============================================================================

def generate_experiment_graphs(experiment_dirs: List[str], output_dir: str, window: int = 20):
    """
    Generate publication-quality graphs comparing experiment runs.
    Each experiment_dir should contain sim_*/episodes.csv, sim_*/steps.csv, etc.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np
        import pandas as pd
    except ImportError as e:
        print(f"[GRAPHS] Missing dependency: {e}. Install matplotlib, pandas, numpy.")
        return []

    COLORS = ["#2563eb", "#dc2626", "#16a34a", "#ea580c", "#9333ea", "#0891b2", "#ca8a04", "#db2777"]
    SIM_COLORS = ["#60a5fa", "#f87171", "#4ade80", "#fb923c", "#c084fc", "#22d3ee", "#facc15", "#f472b6"]

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "axes.spines.top": False, "axes.spines.right": False,
        "grid.alpha": 0.3, "grid.linewidth": 0.5, "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 11,
        "legend.fontsize": 9, "legend.framealpha": 0.9,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # Load experiments
    experiments = []
    for exp_dir in experiment_dirs:
        exp_dir = os.path.expanduser(exp_dir)
        config = {}
        cfg_path = os.path.join(exp_dir, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                config = json.load(f)

        label = config.get("label", os.path.basename(exp_dir))
        sim_episodes = {}
        sim_steps = {}
        for sim_dir in sorted(Path(exp_dir).glob("sim_*")):
            sid = int(sim_dir.name.split("_")[1])
            ep_path = sim_dir / "episodes.csv"
            if ep_path.exists():
                df = pd.read_csv(ep_path)
                if len(df) > 0:
                    sim_episodes[sid] = df
            st_path = sim_dir / "steps.csv"
            if st_path.exists():
                df = pd.read_csv(st_path)
                if len(df) > 0:
                    sim_steps[sid] = df

        sync_events = None
        sync_path = os.path.join(exp_dir, "sync_events.csv")
        if os.path.exists(sync_path):
            df = pd.read_csv(sync_path)
            if len(df) > 0:
                sync_events = df

        num_sims = config.get("num_sims", max(len(sim_episodes), 1))
        experiments.append({
            "label": label, "num_sims": num_sims,
            "sim_episodes": sim_episodes, "sim_steps": sim_steps,
            "sync_events": sync_events,
        })

    if not experiments:
        print("[GRAPHS] No experiment data found.")
        return []

    def rolling(values, w):
        return pd.Series(values).rolling(window=w, min_periods=1).mean().values

    def aggregate(exp):
        frames = list(exp["sim_episodes"].values())
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True).sort_values("wall_clock").reset_index(drop=True)
        return df

    def sim_label(exp):
        n = exp["num_sims"]
        return f"{exp['label']} ({n} sim{'s' if n > 1 else ''})"

    # --- Fig 1: Convergence vs wall-clock ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(experiments):
        c = COLORS[i % len(COLORS)]
        eps = aggregate(exp)
        if eps.empty:
            continue
        t = eps["wall_clock"].values / 60.0
        r = eps["episode_return"].values
        ax.plot(t, r, color=c, alpha=0.12, linewidth=0.5)
        ax.plot(t, rolling(r, window), color=c, linewidth=2.2, label=sim_label(exp))
    ax.set_xlabel("Wall-Clock Time (minutes)")
    ax.set_ylabel("Episode Return")
    ax.set_title("Convergence: Episode Return vs Wall-Clock Time")
    ax.legend(loc="lower right")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    p = os.path.join(output_dir, "01_convergence_wallclock.png")
    fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 2: Convergence vs total steps ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(experiments):
        c = COLORS[i % len(COLORS)]
        eps = aggregate(exp)
        if eps.empty:
            continue
        s = eps["total_steps_at_end"].values
        r = eps["episode_return"].values
        ax.plot(s, r, color=c, alpha=0.12, linewidth=0.5)
        ax.plot(s, rolling(r, window), color=c, linewidth=2.2, label=sim_label(exp))
    ax.set_xlabel("Total Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title("Sample Efficiency: Episode Return vs Total Steps")
    ax.legend(loc="lower right")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    p = os.path.join(output_dir, "02_convergence_steps.png")
    fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 3: Goal success rate ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(experiments):
        c = COLORS[i % len(COLORS)]
        eps = aggregate(exp)
        if eps.empty:
            continue
        t = eps["wall_clock"].values / 60.0
        succ = (eps["outcome"] == "goal_reached").astype(float).values
        ax.plot(t, rolling(succ, 50) * 100, color=c, linewidth=2.2, label=sim_label(exp))
    ax.set_xlabel("Wall-Clock Time (minutes)")
    ax.set_ylabel("Goal Success Rate (%)")
    ax.set_title("Goal Success Rate (50-episode rolling window)")
    ax.legend(loc="lower right")
    ax.set_ylim(-2, 102)
    p = os.path.join(output_dir, "03_success_rate.png")
    fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 4: Exploration coverage ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, exp in enumerate(experiments):
        c = COLORS[i % len(COLORS)]
        for j, (sid, df) in enumerate(exp["sim_episodes"].items()):
            if df.empty:
                continue
            label = sim_label(exp) if j == 0 else None
            alpha = 0.8 if exp["num_sims"] == 1 else 0.5
            ax.plot(df["wall_clock"].values / 60.0, df["cells_discovered"].values,
                    color=c, alpha=alpha, linewidth=1.5, label=label)
    ax.set_xlabel("Wall-Clock Time (minutes)")
    ax.set_ylabel("Cells Discovered")
    ax.set_title("Exploration Coverage Over Time (per sim)")
    ax.legend(loc="lower right")
    p = os.path.join(output_dir, "04_exploration_coverage.png")
    fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 5: Reward breakdown per experiment ---
    comp_defs = [
        ("r_discovery", "Discovery", "#8b5cf6"), ("r_progress", "Progress", "#22c55e"),
        ("r_step", "Step Cost", "#64748b"), ("r_revisit", "Revisit", "#ec4899"),
        ("r_proximity", "Proximity", "#f97316"), ("r_collision", "Collision", "#ef4444"),
        ("r_goal", "Goal", "#eab308"), ("r_timeout", "Timeout", "#92400e"),
    ]
    for exp in experiments:
        frames = list(exp["sim_steps"].values())
        if not frames:
            continue
        all_st = pd.concat(frames, ignore_index=True).sort_values("wall_clock")
        fig, ax = plt.subplots(figsize=(12, 5))
        t = all_st["wall_clock"].values / 60.0
        for col, clabel, color in comp_defs:
            if col in all_st.columns:
                ax.plot(t, rolling(all_st[col].values, 500), color=color, linewidth=1.5,
                        label=clabel, alpha=0.85)
        ax.set_xlabel("Wall-Clock Time (minutes)")
        ax.set_ylabel("Reward Component (rolling avg)")
        ax.set_title(f"Reward Breakdown — {exp['label']}")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        safe = exp["label"].replace(" ", "_").replace("/", "_")[:40]
        p = os.path.join(output_dir, f"05_reward_breakdown_{safe}.png")
        fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 6: Sync event impact ---
    for exp in experiments:
        if exp["sync_events"] is None or exp["sync_events"].empty:
            continue
        eps = aggregate(exp)
        if eps.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        t = eps["wall_clock"].values / 60.0
        r = eps["episode_return"].values
        ax.plot(t, r, color="#94a3b8", alpha=0.15, linewidth=0.5)
        ax.plot(t, rolling(r, 20), color=COLORS[0], linewidth=2.0, label="Return (smoothed)")
        for j, st in enumerate(exp["sync_events"]["wall_clock"].values):
            ax.axvline(x=st / 60.0, color="#ef4444", linewidth=1.0, linestyle="--", alpha=0.6,
                       label="Weight Sync" if j == 0 else None)
        ax.set_xlabel("Wall-Clock Time (minutes)")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"Sync Event Impact — {exp['label']}")
        ax.legend(loc="lower right")
        safe = exp["label"].replace(" ", "_").replace("/", "_")[:40]
        p = os.path.join(output_dir, f"06_sync_impact_{safe}.png")
        fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 7: Time to threshold ---
    all_returns = []
    for exp in experiments:
        eps = aggregate(exp)
        if not eps.empty:
            all_returns.extend(eps["episode_return"].values)
    if all_returns:
        max_ret = np.percentile(all_returns, 90)
        thresholds = [max_ret * 0.25, max_ret * 0.5, max_ret * 0.75] if max_ret > 0 else [-100, -50, 0]
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(thresholds))
        bw = 0.8 / max(len(experiments), 1)
        for i, exp in enumerate(experiments):
            eps = aggregate(exp)
            if eps.empty:
                continue
            t = eps["wall_clock"].values / 60.0
            r = eps["episode_return"].values
            sm = rolling(r, window)
            reach = []
            for th in thresholds:
                hit = np.where(sm >= th)[0]
                reach.append(t[hit[0]] if len(hit) > 0 else (t[-1] if len(t) > 0 else 0))
            off = (i - len(experiments) / 2 + 0.5) * bw
            bars = ax.bar(x + off, reach, bw, color=COLORS[i % len(COLORS)],
                          label=sim_label(exp), alpha=0.85, edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, reach):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xlabel("Return Threshold")
        ax.set_ylabel("Time to Reach (minutes)")
        ax.set_title("Time to Convergence Threshold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.0f}" for t in thresholds])
        ax.legend()
        p = os.path.join(output_dir, "07_time_to_threshold.png")
        fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 8: Per-sim returns ---
    for exp in experiments:
        if exp["num_sims"] <= 1:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        for sid, df in exp["sim_episodes"].items():
            if df.empty:
                continue
            t = df["wall_clock"].values / 60.0
            r = df["episode_return"].values
            c = SIM_COLORS[sid % len(SIM_COLORS)]
            ax.plot(t, r, color=c, alpha=0.1, linewidth=0.5)
            ax.plot(t, rolling(r, 15), color=c, linewidth=1.8, label=f"Sim {sid}")
        ax.set_xlabel("Wall-Clock Time (minutes)")
        ax.set_ylabel("Episode Return")
        ax.set_title(f"Per-Sim Learning Curves — {exp['label']}")
        ax.legend(loc="lower right", ncol=2)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
        safe = exp["label"].replace(" ", "_").replace("/", "_")[:40]
        p = os.path.join(output_dir, f"08_per_sim_returns_{safe}.png")
        fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 9: Safety metrics ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, exp in enumerate(experiments):
        c = COLORS[i % len(COLORS)]
        eps = aggregate(exp)
        if eps.empty:
            continue
        lb = sim_label(exp)
        t = eps["wall_clock"].values / 60.0
        if "avg_min_distance" in eps.columns:
            ax1.plot(t, rolling(eps["avg_min_distance"].values, 50), color=c, linewidth=2.0, label=lb)
        if "bumps" in eps.columns:
            ax2.plot(t, rolling(eps["bumps"].values.astype(float), 50), color=c, linewidth=2.0, label=lb)
    ax1.set_xlabel("Wall-Clock Time (min)"); ax1.set_ylabel("Avg Min Distance (m)")
    ax1.set_title("Obstacle Awareness"); ax1.legend(loc="lower right", fontsize=8)
    ax2.set_xlabel("Wall-Clock Time (min)"); ax2.set_ylabel("Bumps per Episode")
    ax2.set_title("Collision Frequency"); ax2.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    p = os.path.join(output_dir, "09_safety_metrics.png")
    fig.savefig(p); plt.close(fig); saved_files.append(p)

    # --- Fig 10: Expert State Machine vs RL Policy convergence ---
    for exp in experiments:
        eps = aggregate(exp)
        if eps.empty or "training_phase" not in eps.columns:
            continue

        expert_eps = eps[eps["training_phase"] == "EXPERT_DEMO"]
        rl_eps = eps[eps["training_phase"] == "RL_POLICY"]

        if expert_eps.empty and rl_eps.empty:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 10a: Episode return over time — expert vs RL with baseline
        ax = axes[0, 0]
        if not expert_eps.empty:
            t_exp = expert_eps["wall_clock"].values / 60.0
            r_exp = expert_eps["episode_return"].values
            ax.plot(t_exp, r_exp, color="#f59e0b", alpha=0.2, linewidth=0.5)
            ax.plot(t_exp, rolling(r_exp, min(15, len(r_exp))), color="#f59e0b",
                    linewidth=2.5, label="Expert State Machine")
            expert_mean = r_exp.mean()
            # Extend baseline across full time range
            all_t = eps["wall_clock"].values / 60.0
            ax.axhline(y=expert_mean, color="#f59e0b", linewidth=1.5, linestyle="--",
                        alpha=0.7, label=f"Expert Avg ({expert_mean:.0f})")
        if not rl_eps.empty:
            t_rl = rl_eps["wall_clock"].values / 60.0
            r_rl = rl_eps["episode_return"].values
            ax.plot(t_rl, r_rl, color="#2563eb", alpha=0.15, linewidth=0.5)
            ax.plot(t_rl, rolling(r_rl, min(15, len(r_rl))), color="#2563eb",
                    linewidth=2.5, label="RL Policy")
        # Mark random exploration phase
        random_eps = eps[eps["training_phase"] == "RANDOM_EXPLORE"]
        if not random_eps.empty:
            t_rand = random_eps["wall_clock"].values / 60.0
            r_rand = random_eps["episode_return"].values
            ax.plot(t_rand, rolling(r_rand, min(10, len(r_rand))), color="#94a3b8",
                    linewidth=1.5, linestyle=":", label="Random Explore", alpha=0.7)
        ax.set_xlabel("Wall-Clock Time (minutes)")
        ax.set_ylabel("Episode Return")
        ax.set_title("Expert vs RL: Episode Return")
        ax.legend(loc="lower right", fontsize=8)
        ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--", alpha=0.3)

        # 10b: Goal success rate — expert vs RL
        ax = axes[0, 1]
        if not expert_eps.empty:
            succ_exp = (expert_eps["outcome"] == "goal_reached").astype(float).values
            t_exp = expert_eps["wall_clock"].values / 60.0
            w_exp = min(20, len(succ_exp))
            ax.plot(t_exp, rolling(succ_exp, w_exp) * 100, color="#f59e0b",
                    linewidth=2.5, label="Expert")
        if not rl_eps.empty:
            succ_rl = (rl_eps["outcome"] == "goal_reached").astype(float).values
            t_rl = rl_eps["wall_clock"].values / 60.0
            w_rl = min(20, len(succ_rl))
            ax.plot(t_rl, rolling(succ_rl, w_rl) * 100, color="#2563eb",
                    linewidth=2.5, label="RL Policy")
        ax.set_xlabel("Wall-Clock Time (minutes)")
        ax.set_ylabel("Goal Success Rate (%)")
        ax.set_title("Expert vs RL: Success Rate")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_ylim(-2, 102)

        # 10c: Bumps per episode — expert vs RL (safety comparison)
        ax = axes[1, 0]
        if not expert_eps.empty and "bumps" in expert_eps.columns:
            t_exp = expert_eps["wall_clock"].values / 60.0
            b_exp = expert_eps["bumps"].values.astype(float)
            ax.plot(t_exp, rolling(b_exp, min(15, len(b_exp))), color="#f59e0b",
                    linewidth=2.5, label="Expert")
        if not rl_eps.empty and "bumps" in rl_eps.columns:
            t_rl = rl_eps["wall_clock"].values / 60.0
            b_rl = rl_eps["bumps"].values.astype(float)
            ax.plot(t_rl, rolling(b_rl, min(15, len(b_rl))), color="#2563eb",
                    linewidth=2.5, label="RL Policy")
        ax.set_xlabel("Wall-Clock Time (minutes)")
        ax.set_ylabel("Bumps per Episode")
        ax.set_title("Expert vs RL: Collision Safety")
        ax.legend(loc="upper right", fontsize=8)

        # 10d: Summary bar chart — side by side averages
        ax = axes[1, 1]
        metrics = []
        expert_vals = []
        rl_vals = []
        if not expert_eps.empty and not rl_eps.empty:
            metrics.append("Avg Return")
            expert_vals.append(expert_eps["episode_return"].mean())
            rl_vals.append(rl_eps["episode_return"].mean())

            metrics.append("Success %")
            expert_vals.append((expert_eps["outcome"] == "goal_reached").mean() * 100)
            rl_vals.append((rl_eps["outcome"] == "goal_reached").mean() * 100)

            if "bumps" in expert_eps.columns:
                metrics.append("Avg Bumps")
                expert_vals.append(expert_eps["bumps"].astype(float).mean())
                rl_vals.append(rl_eps["bumps"].astype(float).mean())

            if "avg_min_distance" in expert_eps.columns:
                metrics.append("Avg Min Dist")
                expert_vals.append(expert_eps["avg_min_distance"].mean())
                rl_vals.append(rl_eps["avg_min_distance"].mean())

        if metrics:
            x = np.arange(len(metrics))
            bw = 0.35
            ax.bar(x - bw/2, expert_vals, bw, color="#f59e0b", label="Expert", alpha=0.85)
            ax.bar(x + bw/2, rl_vals, bw, color="#2563eb", label="RL Policy", alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics, fontsize=9)
            ax.legend(fontsize=8)
            # Add value labels
            for xi, (ev, rv) in enumerate(zip(expert_vals, rl_vals)):
                ax.text(xi - bw/2, ev + 0.5, f"{ev:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#b45309")
                ax.text(xi + bw/2, rv + 0.5, f"{rv:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold", color="#1d4ed8")
        ax.set_title("Expert vs RL: Summary Comparison")

        fig.suptitle(f"Expert State Machine vs RL Policy — {exp['label']}", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()
        safe = exp["label"].replace(" ", "_").replace("/", "_")[:40]
        p = os.path.join(output_dir, f"10_expert_vs_rl_{safe}.png")
        fig.savefig(p); plt.close(fig); saved_files.append(p)

    print(f"[GRAPHS] Generated {len(saved_files)} figures in {output_dir}")
    return saved_files


# =============================================================================
# Flask App
# =============================================================================

def create_app(stats: Dict[str, DomainStats], mgr: MultiDomainManager, experiment_dir: str = ""):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "rl_monitor"
    CORS(app)
    sio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/cameras")
    def api_cameras():
        try:
            domain_id = int(request.args.get("domain_id", "0"))
        except:
            domain_id = 0
        return jsonify({"domain_id": domain_id, "topics": mgr.get_image_topics(domain_id)})

    @app.route("/api/generate_graphs", methods=["POST"])
    def api_generate_graphs():
        """Generate graphs from current experiment data. POST with optional JSON body:
        {"dirs": ["path1", "path2"], "output": "figures/", "window": 20}
        If no dirs given, uses the current experiment_dir."""
        data = request.get_json(silent=True) or {}
        dirs = data.get("dirs", [experiment_dir] if experiment_dir else [])
        out = data.get("output", os.path.join(experiment_dir, "figures") if experiment_dir else "./figures")
        w = data.get("window", 20)
        if not dirs:
            return jsonify({"error": "No experiment directories configured"}), 400
        try:
            files = generate_experiment_graphs(dirs, out, window=w)
            return jsonify({"success": True, "figures": files, "output_dir": out})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/experiment_status")
    def api_experiment_status():
        """Return current experiment logging status."""
        csv_stats = {}
        for key, ds in stats.items():
            csv_stats[key] = {
                "steps_logged": ds._csv_steps_logged,
                "episodes_logged": ds._csv_episodes_logged,
            }
        return jsonify({
            "experiment_dir": experiment_dir,
            "logging_active": bool(experiment_dir),
            "domains": csv_stats,
        })

    def mjpeg_generator(domain_id: int, topic: str):
        cam = mgr.get_camera(domain_id)
        if cam is None:
            return
        cam.ensure_subscription(topic)
        boundary = b"--frame\r\n"
        while True:
            frame = cam.latest_jpeg()
            if frame:
                yield boundary + b"Content-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.03)

    @app.route("/stream/<int:domain_id>")
    def stream(domain_id: int):
        topic = request.args.get("topic", "").strip()
        if not topic:
            return "Missing ?topic=", 400
        if not topic.startswith("/"):
            topic = "/" + topic
        return Response(mjpeg_generator(domain_id, topic), mimetype="multipart/x-mixed-replace; boundary=frame")

    def updater():
        while True:
            time.sleep(1.0 / UPDATE_RATE_HZ)
            data = {"domains": {k: {"summary": v.get_summary(), "plot_data": v.get_plot_data()} for k, v in stats.items()}}
            sio.emit("update", data)

    return app, sio, updater


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RL Training Monitor with experiment data logging and graph generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Live monitor only (original behavior)
    python reward_monitor.py

    # Monitor + log experiment data
    python reward_monitor.py --experiment ~/experiments/baseline_1sim --label "1 Sim Baseline"

    # Monitor + log + track weight sync events
    python reward_monitor.py --experiment ~/experiments/parallel_4sim \\
        --label "4 Sims, sync 5min" --sync-dir ~/rl_checkpoints

    # Generate graphs from saved experiments (no live monitor)
    python reward_monitor.py --generate-graphs \\
        --graph-dirs ~/experiments/baseline ~/experiments/parallel_4sim \\
        --graph-output ~/figures
        """,
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--scan-start", type=int, default=DEFAULT_SCAN_START)
    parser.add_argument("--scan-end", type=int, default=DEFAULT_SCAN_END)
    parser.add_argument("--scan-interval", type=float, default=DEFAULT_SCAN_INTERVAL)
    parser.add_argument("--namespaces", nargs="*", default=["", "stretch"])

    # Experiment logging
    parser.add_argument("--experiment", type=str, default="",
                        help="Directory to save experiment CSVs (enables logging)")
    parser.add_argument("--label", type=str, default="",
                        help="Human-readable experiment label")
    parser.add_argument("--sync-dir", type=str, default="",
                        help="Checkpoint directory to monitor for sync events")
    parser.add_argument("--sync-check-interval", type=float, default=30.0,
                        help="Seconds between sync event checks (default: 30)")

    # Graph generation (offline mode)
    parser.add_argument("--generate-graphs", action="store_true",
                        help="Generate graphs and exit (no live monitor)")
    parser.add_argument("--graph-dirs", type=str, nargs="+", default=[],
                        help="Experiment directories to compare in graphs")
    parser.add_argument("--graph-output", type=str, default="./figures",
                        help="Output directory for generated figures")
    parser.add_argument("--graph-window", type=int, default=20,
                        help="Rolling average window for graphs (default: 20)")

    args = parser.parse_args()

    # --- Offline graph generation mode ---
    if args.generate_graphs:
        if not args.graph_dirs:
            print("Error: --graph-dirs required with --generate-graphs")
            sys.exit(1)
        generate_experiment_graphs(args.graph_dirs, args.graph_output, window=args.graph_window)
        sys.exit(0)

    # --- Experiment directory setup ---
    experiment_dir = ""
    start_time = time.time()
    sync_monitor = None

    if args.experiment:
        experiment_dir = os.path.expanduser(args.experiment)
        os.makedirs(experiment_dir, exist_ok=True)
        config = {
            "label": args.label or os.path.basename(experiment_dir),
            "start_time": datetime.now().isoformat(),
            "scan_start": args.scan_start,
            "scan_end": args.scan_end,
            "sync_dir": args.sync_dir,
        }
        with open(os.path.join(experiment_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        if args.sync_dir:
            sync_dir = os.path.expanduser(args.sync_dir)
            # Count sim dirs to determine num_sims
            num_sims = len(list(Path(sync_dir).glob("sim_*"))) or 1
            sync_monitor = SyncMonitor(sync_dir, num_sims, experiment_dir, start_time)

    print(f"""
    RL Training Monitor (Reward v5)
    ================================
    http://localhost:{args.port}
    Scanning domains {args.scan_start}-{args.scan_end}
    {"Experiment: " + experiment_dir if experiment_dir else "Experiment logging: OFF"}
    {"Label: " + args.label if args.label else ""}
    {"Sync monitor: " + args.sync_dir if args.sync_dir else ""}

    REWARD SYSTEM:
      Terminal:  goal=+2000, timeout=-50 (collision NEVER terminates)
      Goal:     ratchet progress (only new-best distance pays)
      Explore:  discovery (new cells only)
      Revisit:  penalty for returning to explored areas
      Obstacle: proximity cost + bump penalty (-5, no termination)
      Collision threshold: 0.30m (LIDAR min distance)
    """)

    stats: Dict[str, DomainStats] = {}
    mgr = MultiDomainManager(stats)
    mgr.discover_domains(args.scan_start, args.scan_end)

    # Initialize CSV logging for any domains already discovered
    if experiment_dir:
        _sim_counter = [0]  # mutable counter for closure
        def _init_csv_for_new_domains():
            for key, ds in stats.items():
                if ds._csv_steps_writer is None:
                    ds.init_csv_logging(experiment_dir, _sim_counter[0], start_time)
                    _sim_counter[0] += 1
                    print(f"  [CSV] Logging sim_{_sim_counter[0]-1} (domain {ds.domain_id}:{ds.namespace})")
        _init_csv_for_new_domains()

    app, sio, updater = create_app(stats, mgr, experiment_dir=experiment_dir)
    threading.Thread(target=updater, daemon=True).start()

    def scan_loop():
        while True:
            time.sleep(args.scan_interval)
            mgr.discover_domains(args.scan_start, args.scan_end)
            mgr.discover_namespaces_all()
            # Initialize CSV for newly discovered domains
            if experiment_dir:
                _init_csv_for_new_domains()

    threading.Thread(target=scan_loop, daemon=True).start()

    # Sync monitor thread
    if sync_monitor:
        def sync_loop():
            while True:
                time.sleep(args.sync_check_interval)
                sync_monitor.check()
        threading.Thread(target=sync_loop, daemon=True).start()

    def shutdown(*_):
        # Close CSV files
        for ds in stats.values():
            ds.close_csv()
        if sync_monitor:
            sync_monitor.close()
        # Auto-generate graphs on shutdown if experiment logging was active
        if experiment_dir:
            print(f"\n[SHUTDOWN] Generating graphs from {experiment_dir}...")
            try:
                fig_dir = os.path.join(experiment_dir, "figures")
                generate_experiment_graphs([experiment_dir], fig_dir)
                print(f"[SHUTDOWN] Figures saved to {fig_dir}/")
            except Exception as e:
                print(f"[SHUTDOWN] Graph generation failed: {e}")

            # Print summary
            total_steps = sum(ds._csv_steps_logged for ds in stats.values())
            total_eps = sum(ds._csv_episodes_logged for ds in stats.values())
            elapsed = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"  Experiment Complete")
            print(f"{'='*60}")
            print(f"  Duration:  {elapsed/60:.1f} minutes")
            print(f"  Steps:     {total_steps}")
            print(f"  Episodes:  {total_eps}")
            print(f"  Data:      {experiment_dir}")
            if sync_monitor:
                print(f"  Syncs:     {sync_monitor.sync_count}")
            print(f"{'='*60}")
            print(f"\n  To compare experiments:")
            print(f"  python reward_monitor.py --generate-graphs \\")
            print(f"    --graph-dirs {experiment_dir} <other_experiment_dir> \\")
            print(f"    --graph-output ./figures\n")

        mgr.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    try:
        sio.run(app, host="0.0.0.0", port=args.port, debug=False, use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)
    except:
        shutdown()


if __name__ == "__main__":
    main()