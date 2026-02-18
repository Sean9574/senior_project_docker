#!/usr/bin/env python3
"""
Multi-Domain RL Training Monitor + Camera Streaming

Features:
- Auto-discovers ROS2 domains with reward topics
- Streams camera/image topics via MJPEG
- Smooth, non-jittery chart updates
- Clean, soft UI design
"""

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
UPDATE_RATE_HZ = 5  # Slower updates = smoother charts
DEFAULT_SCAN_START = 0
DEFAULT_SCAN_END = 50
DEFAULT_SCAN_INTERVAL = 15.0

DEFAULT_CAMERA_FPS = 15.0
DEFAULT_CAMERA_JPEG_QUALITY = 85
DEFAULT_CAMERA_MAX_WIDTH = 800

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
    safety_zones: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    intervention_rates: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    total_steps: int = 0
    last_update: float = 0.0
    goals_reached: int = 0
    frontiers: int = 0
    cells_discovered: int = 0
    
    # Goal generator status
    goal_status: Dict = field(default_factory=dict)
    goal_status_time: float = 0.0

    def add_reward(self, reward: float, timestamp: float):
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        self.current_episode_return += reward
        self.current_episode_steps += 1
        self.total_steps += 1
        self.last_update = time.time()

    def add_breakdown(self, breakdown: Dict):
        timestamp = time.time()

        reward_terms = breakdown.get("reward_terms", {})
        for key, value in reward_terms.items():
            if key not in self.components:
                self.components[key] = deque(maxlen=MAX_HISTORY_LENGTH)
            self.components[key].append((timestamp, value))

        safety = breakdown.get("safety", {})
        zone = safety.get("zone", "UNKNOWN")
        intervention = safety.get("intervention", 0.0)
        self.safety_zones.append((timestamp, zone))
        self.intervention_rates.append((timestamp, intervention))

        goals = breakdown.get("goals_reached", 0)
        if goals > self.goals_reached:
            self.goals_reached = goals

        explore_stats = breakdown.get("explore_stats", {})
        self.cells_discovered = explore_stats.get("total_discovered", self.cells_discovered)
        self.frontiers = explore_stats.get("frontier_count", self.frontiers)

        episode_num = breakdown.get("episode", 0)
        if episode_num > self.last_episode_num and self.last_episode_num > 0:
            if self.current_episode_steps > 0:
                self.episode_returns.append(self.current_episode_return)
                self.episode_count += 1
            self.current_episode_return = 0.0
            self.current_episode_steps = 0
            self.goals_reached = 0
        self.last_episode_num = episode_num

    def update_goal_status(self, status: Dict):
        """Update goal generator status."""
        self.goal_status = status
        self.goal_status_time = time.time()

    def get_summary(self) -> Dict:
        recent_rewards = list(self.rewards)[-100:] if self.rewards else [0]
        recent_interventions = [x[1] for x in list(self.intervention_rates)[-100:]] if self.intervention_rates else [0]
        
        # Goal status freshness (stale after 2 seconds)
        goal_status_fresh = (time.time() - self.goal_status_time) < 2.0

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
            "avg_intervention": sum(recent_interventions) / len(recent_interventions) if recent_interventions else 0,
            "active": time.time() - self.last_update < 5.0,
            "goal_status": self.goal_status if goal_status_fresh else {},
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

        zone_counts = {"FREE": 0, "AWARE": 0, "CAUTION": 0, "DANGER": 0, "EMERGENCY": 0}
        for _, zone in list(self.safety_zones)[-500:]:
            if zone in zone_counts:
                zone_counts[zone] += 1

        interventions = list(self.intervention_rates)[::step]
        if interventions:
            t0_int = interventions[0][0]
            intervention_times = [(v[0] - t0_int) for v in interventions]
            intervention_values = [v[1] for v in interventions]
        else:
            intervention_times, intervention_values = [], []

        return {
            "domain_id": self.domain_id,
            "display_name": self.display_name,
            "rewards": {"times": rel_times, "values": rewards},
            "episode_returns": {"episodes": episode_indices, "values": episode_returns},
            "components": components_data,
            "safety_zones": zone_counts,
            "interventions": {"times": intervention_times, "values": intervention_values}
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
        
        # Subscribe to goal generator status topic (try multiple patterns)
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
# Dashboard HTML - Clean Soft Design with Smooth Charts
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RL Training Monitor</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
  <style>
    :root {
      --bg: #f4f7fa;
      --card: #ffffff;
      --border: #e8ecf1;
      --text: #2d3748;
      --text-light: #718096;
      --text-muted: #a0aec0;
      --blue: #667eea;
      --green: #48bb78;
      --red: #fc8181;
      --yellow: #f6e05e;
      --purple: #9f7aea;
      --radius: 16px;
      --shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }

    .header {
      background: var(--card);
      padding: 16px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-shadow: var(--shadow);
    }

    .header h1 {
      font-size: 18px;
      font-weight: 700;
      color: var(--text);
    }

    .badge {
      background: linear-gradient(135deg, #eef2ff 0%, #f5f3ff 100%);
      border: 1px solid #c7d2fe;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      color: var(--blue);
    }

    .container {
      padding: 20px 24px;
      max-width: 1400px;
      margin: 0 auto;
    }

    .tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
      flex-wrap: wrap;
    }

    .tab {
      padding: 10px 18px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-light);
      transition: all 0.2s;
      display: flex;
      align-items: center;
      gap: 8px;
      box-shadow: var(--shadow);
    }

    .tab:hover {
      border-color: var(--blue);
      color: var(--blue);
    }

    .tab.active {
      background: linear-gradient(135deg, var(--blue) 0%, #7c3aed 100%);
      border-color: transparent;
      color: white;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }

    .tab .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--text-muted);
    }

    .tab.live .dot {
      background: var(--green);
      animation: pulse 2s infinite;
    }

    .tab.active .dot {
      background: rgba(255,255,255,0.6);
    }

    .tab.active.live .dot {
      background: white;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px;
      box-shadow: var(--shadow);
      margin-bottom: 16px;
    }

    .card h3 {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-light);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 16px;
    }

    .stats-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-bottom: 20px;
    }

    .stat {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: var(--shadow);
    }

    .stat .label {
      font-size: 11px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.3px;
    }

    .stat .value {
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }

    .stat .value.green { color: var(--green); }
    .stat .value.red { color: var(--red); }
    .stat .value.blue { color: var(--blue); }
    .stat .value.purple { color: var(--purple); }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }

    @media (max-width: 1000px) {
      .grid { grid-template-columns: 1fr; }
    }

    .chart { height: 240px; }

    .camera-container {
      position: relative;
      background: #1a202c;
      border-radius: 12px;
      overflow: hidden;
    }

    .camera-img {
      width: 100%;
      height: 300px;
      object-fit: contain;
      display: block;
    }

    .camera-select {
      padding: 8px 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 12px;
      background: var(--card);
      color: var(--text);
      min-width: 200px;
    }

    .zones {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 12px;
    }

    .zone {
      padding: 4px 12px;
      border-radius: 16px;
      font-size: 11px;
      font-weight: 600;
    }

    .zone.FREE { background: #c6f6d5; color: #276749; }
    .zone.AWARE { background: #fefcbf; color: #975a16; }
    .zone.CAUTION { background: #fed7aa; color: #c05621; }
    .zone.DANGER { background: #fed7d7; color: #c53030; }
    .zone.EMERGENCY { background: #feb2b2; color: #9b2c2c; }

    .empty {
      text-align: center;
      padding: 60px;
      color: var(--text-muted);
    }

    .summary-box {
      background: linear-gradient(135deg, #eef2ff 0%, #faf5ff 100%);
      border: 1px solid #ddd6fe;
      border-radius: var(--radius);
      padding: 20px;
      margin-bottom: 20px;
    }

    .summary-box h3 {
      color: var(--blue);
      margin-bottom: 16px;
    }

    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 16px;
    }

    .summary-stat {
      text-align: center;
    }

    .summary-stat .label {
      font-size: 10px;
      color: var(--text-muted);
      text-transform: uppercase;
    }

    .summary-stat .value {
      font-size: 20px;
      font-weight: 700;
      margin-top: 4px;
    }

    .goal-status {
      margin-top: 12px;
      padding: 12px 16px;
      background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%);
      border: 1px solid #bbf7d0;
      border-radius: 12px;
      font-size: 13px;
    }

    .goal-status.searching {
      background: linear-gradient(135deg, #fefce8 0%, #fef9c3 100%);
      border-color: #fde047;
    }

    .goal-status .status-row {
      display: flex;
      justify-content: space-between;
      padding: 4px 0;
      border-bottom: 1px solid rgba(0,0,0,0.05);
    }

    .goal-status .status-row:last-child {
      border-bottom: none;
    }

    .goal-status .status-label {
      color: var(--text-muted);
      font-weight: 500;
    }

    .goal-status .status-value {
      font-weight: 600;
      color: var(--text);
    }

    .goal-status .status-value.found {
      color: var(--green);
    }

    .goal-status .status-value.searching {
      color: #ca8a04;
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>RL Training Monitor</h1>
    <span class="badge" id="badge">Scanning...</span>
  </div>

  <div class="container">
    <div class="tabs" id="tabs"></div>
    <div id="content"><div class="empty">Scanning for ROS2 domains...</div></div>
  </div>

  <script>
    const socket = io();
    let domains = {};
    let selected = 'all';
    let chartsInitialized = {};

    const colors = {
      blue: '#667eea', green: '#48bb78', red: '#fc8181',
      yellow: '#ecc94b', purple: '#9f7aea', cyan: '#4fd1c5'
    };

    const layoutBase = {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: { color: '#718096', size: 10 },
      margin: { l: 40, r: 16, t: 24, b: 32 },
      xaxis: { gridcolor: '#edf2f7', zerolinecolor: '#e2e8f0' },
      yaxis: { gridcolor: '#edf2f7', zerolinecolor: '#e2e8f0' },
      showlegend: false
    };

    const config = { displayModeBar: false, responsive: true };

    socket.on('connect', () => document.getElementById('badge').textContent = 'Connected');
    socket.on('disconnect', () => document.getElementById('badge').textContent = 'Disconnected');

    socket.on('update', data => {
      domains = data.domains || {};
      const n = Object.keys(domains).length;
      document.getElementById('badge').textContent = n + ' Domain' + (n !== 1 ? 's' : '');
      
      // Only rebuild tabs, not full content
      renderTabs();
      
      // Update charts without rebuilding HTML
      if (selected === 'all') {
        updateAllCharts();
      } else if (domains[selected]) {
        updateDomainCharts(selected);
        updateGoalStatus(domains[selected].summary.goal_status);
      }
    });

    function render() {
      renderTabs();
      if (selected === 'all') renderAllView();
      else renderDomainView(selected);
    }

    function renderTabs() {
      const keys = Object.keys(domains);
      let h = '<div class="tab ' + (selected === 'all' ? 'active' : '') + '" onclick="sel(\\'all\\')">Overview</div>';
      keys.forEach(k => {
        const s = domains[k].summary;
        h += '<div class="tab ' + (selected === k ? 'active' : '') + ' ' + (s.active ? 'live' : '') + '" onclick="sel(\\'' + k + '\\')">' +
             '<div class="dot"></div>' + s.display_name + '</div>';
      });
      document.getElementById('tabs').innerHTML = h;
    }

    function sel(k) {
      selected = k;
      chartsInitialized = {};
      render();  // Full render only on tab switch
    }
    
    function updateAllCharts() {
      if (!document.getElementById('allChart')) return;
      const vals = Object.values(domains);
      if (!vals.length) return;
      
      const traces = [];
      const cls = [colors.blue, colors.green, colors.purple, colors.red, colors.cyan, colors.yellow];
      let ci = 0;
      Object.values(domains).forEach(d => {
        const p = d.plot_data;
        if (p.episode_returns.values.length) {
          traces.push({
            x: p.episode_returns.episodes,
            y: p.episode_returns.values,
            type: 'scatter',
            mode: 'lines',
            name: p.display_name,
            line: { color: cls[ci++ % cls.length], width: 2 }
          });
        }
      });
      if (traces.length) {
        Plotly.react('allChart', traces, {
          ...layoutBase,
          showlegend: true,
          legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
          xaxis: { ...layoutBase.xaxis, title: 'Episode' },
          yaxis: { ...layoutBase.yaxis, title: 'Return' }
        }, config);
      }
      
      // Update summary stats
      const totSteps = vals.reduce((a, d) => a + d.summary.total_steps, 0);
      const totEp = vals.reduce((a, d) => a + d.summary.episode_count, 0);
      const active = vals.filter(d => d.summary.active).length;
      document.querySelectorAll('.summary-stat .value').forEach((el, i) => {
        if (i === 0) el.textContent = totSteps.toLocaleString();
        if (i === 1) el.textContent = totEp;
        if (i === 4) el.textContent = active + '/' + vals.length;
      });
    }
    
    function updateDomainCharts(k) {
      const d = domains[k];
      if (!d) return;
      
      const s = d.summary;
      const p = d.plot_data;
      
      // Update stat values without rebuilding HTML
      const statEls = document.querySelectorAll('.stat .value');
      if (statEls.length >= 7) {
        statEls[0].textContent = s.total_steps.toLocaleString();
        statEls[1].textContent = s.episode_count;
        statEls[2].textContent = s.goals_reached;
        statEls[3].textContent = s.cells_discovered;
        statEls[4].textContent = s.current_episode_return.toFixed(1);
        statEls[5].textContent = s.avg_episode_return.toFixed(1);
        statEls[6].textContent = (s.avg_intervention * 100).toFixed(0) + '%';
      }
      
      // Update charts
      drawRewardChart(p);
      drawEpisodeChart(p);
      drawInterventionChart(p);
      drawZoneChart(p);
      drawComponentsChart(p);
    }

    function renderAllView() {
      const vals = Object.values(domains);
      if (!vals.length) {
        document.getElementById('content').innerHTML = '<div class="empty">No domains found yet...</div>';
        return;
      }

      const totSteps = vals.reduce((a, d) => a + d.summary.total_steps, 0);
      const totEp = vals.reduce((a, d) => a + d.summary.episode_count, 0);
      const totGoals = vals.reduce((a, d) => a + d.summary.goals_reached, 0);
      const active = vals.filter(d => d.summary.active).length;
      const rets = vals.filter(d => d.summary.episode_count > 0).map(d => d.summary.avg_episode_return);
      const avgRet = rets.length ? rets.reduce((a, b) => a + b, 0) / rets.length : 0;

      let h = '<div class="summary-box"><h3>Cross-Domain Summary</h3><div class="summary-grid">' +
        '<div class="summary-stat"><div class="label">Total Steps</div><div class="value">' + totSteps.toLocaleString() + '</div></div>' +
        '<div class="summary-stat"><div class="label">Episodes</div><div class="value">' + totEp + '</div></div>' +
        '<div class="summary-stat"><div class="label">Goals</div><div class="value">' + totGoals + '</div></div>' +
        '<div class="summary-stat"><div class="label">Avg Return</div><div class="value" style="color:' + (avgRet >= 0 ? colors.green : colors.red) + '">' + avgRet.toFixed(1) + '</div></div>' +
        '<div class="summary-stat"><div class="label">Active</div><div class="value" style="color:' + colors.green + '">' + active + '/' + vals.length + '</div></div>' +
      '</div></div>';

      h += '<div class="card"><h3>Episode Returns (All Domains)</h3><div id="allChart" class="chart" style="height:280px"></div></div>';

      document.getElementById('content').innerHTML = h;

      // Draw comparison chart
      const traces = [];
      const cls = [colors.blue, colors.green, colors.purple, colors.red, colors.cyan, colors.yellow];
      let ci = 0;
      Object.values(domains).forEach(d => {
        const p = d.plot_data;
        if (p.episode_returns.values.length) {
          traces.push({
            x: p.episode_returns.episodes,
            y: p.episode_returns.values,
            type: 'scatter',
            mode: 'lines',
            name: p.display_name,
            line: { color: cls[ci++ % cls.length], width: 2 }
          });
        }
      });
      if (traces.length) {
        Plotly.newPlot('allChart', traces, {
          ...layoutBase,
          showlegend: true,
          legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } },
          xaxis: { ...layoutBase.xaxis, title: 'Episode' },
          yaxis: { ...layoutBase.yaxis, title: 'Return' }
        }, config);
      }
    }

    async function renderDomainView(k) {
      const d = domains[k];
      if (!d) return;

      const s = d.summary;
      const p = d.plot_data;

      // Fetch camera topics
      const resp = await fetch('/api/cameras?domain_id=' + s.domain_id);
      const camData = await resp.json();
      const topics = camData.topics || [];
      const defaultTopic = topics.includes('/sam3_goal_generator/visualization')
        ? '/sam3_goal_generator/visualization' : (topics[0] || '');

      let options = topics.length
        ? topics.map(t => '<option value="' + t + '"' + (t === defaultTopic ? ' selected' : '') + '>' + t + '</option>').join('')
        : '<option value="">No image topics</option>';

      let h = '<div class="stats-row">' +
        '<div class="stat"><div class="label">Steps</div><div class="value">' + s.total_steps.toLocaleString() + '</div></div>' +
        '<div class="stat"><div class="label">Episodes</div><div class="value">' + s.episode_count + '</div></div>' +
        '<div class="stat"><div class="label">Goals</div><div class="value green">' + s.goals_reached + '</div></div>' +
        '<div class="stat"><div class="label">Cells</div><div class="value blue">' + s.cells_discovered + '</div></div>' +
        '<div class="stat"><div class="label">Current Return</div><div class="value ' + (s.current_episode_return >= 0 ? 'green' : 'red') + '">' + s.current_episode_return.toFixed(1) + '</div></div>' +
        '<div class="stat"><div class="label">Avg Return</div><div class="value ' + (s.avg_episode_return >= 0 ? 'green' : 'red') + '">' + s.avg_episode_return.toFixed(1) + '</div></div>' +
        '<div class="stat"><div class="label">Intervention</div><div class="value ' + (s.avg_intervention < 0.3 ? 'green' : '') + '">' + (s.avg_intervention * 100).toFixed(0) + '%</div></div>' +
      '</div>';

      h += '<div class="grid">' +
        '<div class="card"><h3>Camera Feed</h3>' +
        '<div style="margin-bottom:12px"><select class="camera-select" id="camSelect">' + options + '</select></div>' +
        '<div class="camera-container"><img class="camera-img" id="camImg" src="' +
        (defaultTopic ? '/stream/' + s.domain_id + '?topic=' + encodeURIComponent(defaultTopic) : '') + '"/></div>' +
        '<div class="goal-status" id="goalStatus"></div></div>' +
        '<div class="card"><h3>Real-Time Reward</h3><div id="rewChart" class="chart"></div></div>' +
      '</div>';

      h += '<div class="grid">' +
        '<div class="card"><h3>Episode Returns</h3><div id="epChart" class="chart"></div></div>' +
        '<div class="card"><h3>Intervention Rate</h3><div id="intChart" class="chart"></div></div>' +
      '</div>';

      h += '<div class="grid">' +
        '<div class="card"><h3>Safety Zones</h3><div id="zoneChart" class="chart"></div>' +
        '<div class="zones">' + Object.entries(p.safety_zones).filter(z => z[1] > 0).map(z => '<span class="zone ' + z[0] + '">' + z[0] + ': ' + z[1] + '</span>').join('') + '</div></div>' +
        '<div class="card"><h3>Reward Components</h3><div id="compChart" class="chart"></div></div>' +
      '</div>';

      document.getElementById('content').innerHTML = h;

      // Camera select handler
      const camSelect = document.getElementById('camSelect');
      const camImg = document.getElementById('camImg');
      camSelect.onchange = () => {
        const t = camSelect.value;
        camImg.src = t ? '/stream/' + s.domain_id + '?topic=' + encodeURIComponent(t) + '&_t=' + Date.now() : '';
      };

      // Update goal status panel
      updateGoalStatus(s.goal_status);

      // Draw charts using Plotly.react for smooth updates
      drawRewardChart(p);
      drawEpisodeChart(p);
      drawInterventionChart(p);
      drawZoneChart(p);
      drawComponentsChart(p);
    }

    function updateGoalStatus(gs) {
      const el = document.getElementById('goalStatus');
      if (!el) return;

      if (!gs || Object.keys(gs).length === 0) {
        el.innerHTML = '<div style="color:var(--text-muted);text-align:center;padding:8px;">No goal generator data</div>';
        el.className = 'goal-status';
        return;
      }

      const found = gs.found === true;
      el.className = 'goal-status' + (found ? '' : ' searching');

      let html = '<div class="status-row"><span class="status-label">Target</span><span class="status-value">' + (gs.target || '—') + '</span></div>';
      html += '<div class="status-row"><span class="status-label">Status</span><span class="status-value ' + (found ? 'found' : 'searching') + '">' + (found ? '✓ Found' : '○ Searching...') + '</span></div>';

      if (found) {
        html += '<div class="status-row"><span class="status-label">Distance</span><span class="status-value">' + (gs.distance ? gs.distance.toFixed(2) + 'm' : '—') + '</span></div>';
        html += '<div class="status-row"><span class="status-label">Confidence</span><span class="status-value">' + (gs.confidence ? (gs.confidence * 100).toFixed(0) + '%' : '—') + '</span></div>';
        html += '<div class="status-row"><span class="status-label">Depth Method</span><span class="status-value">' + (gs.depth_method || '—') + '</span></div>';
        if (gs.goal_position) {
          html += '<div class="status-row"><span class="status-label">Goal Position</span><span class="status-value">(' + gs.goal_position[0].toFixed(2) + ', ' + gs.goal_position[1].toFixed(2) + ')</span></div>';
        }
      } else if (gs.message) {
        html += '<div class="status-row"><span class="status-label">Info</span><span class="status-value" style="font-size:11px">' + gs.message + '</span></div>';
      }

      if (gs.available_methods) {
        html += '<div class="status-row"><span class="status-label">Available</span><span class="status-value" style="font-size:11px">' + gs.available_methods.join(', ') + '</span></div>';
      }

      el.innerHTML = html;
    }

    function drawRewardChart(p) {
      if (!p.rewards.values.length) return;
      Plotly.react('rewChart', [{
        x: p.rewards.times, y: p.rewards.values,
        type: 'scatter', mode: 'lines',
        line: { color: colors.blue, width: 2 },
        fill: 'tozeroy', fillcolor: 'rgba(102,126,234,0.1)'
      }], { ...layoutBase, xaxis: { ...layoutBase.xaxis, title: 'Time (s)' } }, config);
    }

    function drawEpisodeChart(p) {
      if (!p.episode_returns.values.length) return;
      const ra = [];
      for (let i = 0; i < p.episode_returns.values.length; i++) {
        const w = p.episode_returns.values.slice(Math.max(0, i - 9), i + 1);
        ra.push(w.reduce((a, b) => a + b, 0) / w.length);
      }
      Plotly.react('epChart', [
        { x: p.episode_returns.episodes, y: p.episode_returns.values, type: 'scatter', mode: 'lines+markers', name: 'Return', line: { color: colors.green, width: 2 }, marker: { size: 4 } },
        { x: p.episode_returns.episodes, y: ra, type: 'scatter', mode: 'lines', name: 'Avg', line: { color: colors.yellow, width: 2.5 } }
      ], { ...layoutBase, showlegend: true, legend: { bgcolor: 'rgba(0,0,0,0)', x: 0.02, y: 0.98 }, xaxis: { ...layoutBase.xaxis, title: 'Episode' } }, config);
    }

    function drawInterventionChart(p) {
      if (!p.interventions.values.length) return;
      Plotly.react('intChart', [{
        x: p.interventions.times, y: p.interventions.values.map(v => v * 100),
        type: 'scatter', mode: 'lines',
        line: { color: colors.red, width: 2 },
        fill: 'tozeroy', fillcolor: 'rgba(252,129,129,0.1)'
      }], { ...layoutBase, xaxis: { ...layoutBase.xaxis, title: 'Time (s)' }, yaxis: { ...layoutBase.yaxis, title: '%', range: [0, 100] } }, config);
    }

    function drawZoneChart(p) {
      const zc = { FREE: colors.green, AWARE: '#9ae6b4', CAUTION: colors.yellow, DANGER: '#feb2b2', EMERGENCY: colors.red };
      const zones = Object.entries(p.safety_zones).filter(z => z[1] > 0);
      if (!zones.length) return;
      Plotly.react('zoneChart', [{
        values: zones.map(z => z[1]),
        labels: zones.map(z => z[0]),
        type: 'pie',
        marker: { colors: zones.map(z => zc[z[0]] || '#a0aec0') },
        textinfo: 'label+percent',
        textfont: { size: 10 },
        hole: 0.5
      }], { ...layoutBase }, config);
    }

    function drawComponentsChart(p) {
      const cc = { discovery: colors.purple, proximity: colors.red, step: '#a0aec0', novelty: colors.cyan, rnd: '#4fd1c5', goal: colors.yellow };
      const traces = [];
      Object.entries(p.components).forEach(([n, d]) => {
        if (d.values.length) traces.push({ x: d.times, y: d.values, type: 'scatter', mode: 'lines', name: n, line: { color: cc[n] || '#a0aec0', width: 1.5 } });
      });
      if (traces.length) {
        Plotly.react('compChart', traces, { ...layoutBase, showlegend: true, legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 9 } }, xaxis: { ...layoutBase.xaxis, title: 'Time (s)' } }, config);
      }
    }
  </script>
</body>
</html>
"""


# =============================================================================
# Flask App
# =============================================================================

def create_app(stats: Dict[str, DomainStats], mgr: MultiDomainManager):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--scan-start", type=int, default=DEFAULT_SCAN_START)
    parser.add_argument("--scan-end", type=int, default=DEFAULT_SCAN_END)
    parser.add_argument("--scan-interval", type=float, default=DEFAULT_SCAN_INTERVAL)
    parser.add_argument("--namespaces", nargs="*", default=["", "stretch"])
    args = parser.parse_args()

    print(f"""
    RL Training Monitor
    ===================
    http://localhost:{args.port}
    Scanning domains {args.scan_start}-{args.scan_end}
    """)

    stats: Dict[str, DomainStats] = {}
    mgr = MultiDomainManager(stats)

    mgr.discover_domains(args.scan_start, args.scan_end)

    app, sio, updater = create_app(stats, mgr)
    threading.Thread(target=updater, daemon=True).start()

    def scan_loop():
        while True:
            time.sleep(args.scan_interval)
            mgr.discover_domains(args.scan_start, args.scan_end)
            mgr.discover_namespaces_all()

    threading.Thread(target=scan_loop, daemon=True).start()

    def shutdown(*_):
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