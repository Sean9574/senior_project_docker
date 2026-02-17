#!/usr/bin/env python3
"""
Multi-Domain RL Training Monitor + Per-Domain Camera Streaming (MJPEG)

- Automatically discovers ROS2 domains that have reward topics (as before).
- ALSO discovers Image topics (sensor_msgs/msg/Image) in each domain.
- Exposes a per-domain MJPEG endpoint for any image topic:
    /stream/<domain_id>?topic=/sam3_goal_generator/visualization

Open dashboard:
  http://localhost:5555

Direct stream test (example):
  http://localhost:5555/stream/7?topic=/sam3_goal_generator/visualization

Docker reminder:
  docker run ... -p 5555:5555 ...
"""

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data

from std_msgs.msg import Float32, String as StringMsg
from sensor_msgs.msg import Image as RosImage

from flask import Flask, render_template_string, Response, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

import cv2
from cv_bridge import CvBridge

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = 5555
MAX_HISTORY_LENGTH = 1000
UPDATE_RATE_HZ = 10
DEFAULT_SCAN_START = 0
DEFAULT_SCAN_END = 50
DEFAULT_SCAN_INTERVAL = 15.0

# Camera defaults
DEFAULT_CAMERA_FPS = 10.0
DEFAULT_CAMERA_JPEG_QUALITY = 80
DEFAULT_CAMERA_MAX_WIDTH = 960  # resize for performance; set 0 to disable

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DomainStats:
    """Statistics for a single robot domain."""
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

    def add_reward(self, reward: float, timestamp: float):
        self.rewards.append(reward)
        self.timestamps.append(timestamp)
        self.current_episode_return += reward
        self.current_episode_steps += 1
        self.total_steps += 1
        self.last_update = time.time()

    def add_breakdown(self, breakdown: Dict):
        timestamp = time.time()

        # Track reward components
        reward_terms = breakdown.get("reward_terms", {})
        for key, value in reward_terms.items():
            if key not in self.components:
                self.components[key] = deque(maxlen=MAX_HISTORY_LENGTH)
            self.components[key].append((timestamp, value))

        # Track safety
        safety = breakdown.get("safety", {})
        zone = safety.get("zone", "UNKNOWN")
        intervention = safety.get("intervention", 0.0)
        self.safety_zones.append((timestamp, zone))
        self.intervention_rates.append((timestamp, intervention))

        # Track goals from breakdown
        goals = breakdown.get("goals_reached", 0)
        if goals > self.goals_reached:
            self.goals_reached = goals

        # Track exploration stats
        explore_stats = breakdown.get("explore_stats", {})
        self.cells_discovered = explore_stats.get("total_discovered", self.cells_discovered)
        self.frontiers = explore_stats.get("frontier_count", self.frontiers)

        # Detect episode changes by watching episode number
        episode_num = breakdown.get("episode", 0)
        if episode_num > self.last_episode_num and self.last_episode_num > 0:
            if self.current_episode_steps > 0:
                self.episode_returns.append(self.current_episode_return)
                self.episode_count += 1
            self.current_episode_return = 0.0
            self.current_episode_steps = 0
            self.goals_reached = 0
        self.last_episode_num = episode_num

    def get_summary(self) -> Dict:
        recent_rewards = list(self.rewards)[-100:] if self.rewards else [0]
        recent_interventions = [x[1] for x in list(self.intervention_rates)[-100:]] if self.intervention_rates else [0]

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
        }

    def get_plot_data(self, max_points: int = 500) -> Dict:
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
# Per-Domain ROS2 Subscriber (Rewards)
# =============================================================================

class DomainCollector:
    def __init__(self, domain_id: int, stats_store: Dict[str, DomainStats],
                 namespaces: List[str] = None):
        self.domain_id = domain_id
        self.stats = stats_store
        self.namespaces = namespaces or [""]
        self.running = False

        self.context = Context()
        rclpy.init(context=self.context, domain_id=domain_id)

        self.node = rclpy.create_node(
            f"reward_monitor_{domain_id}",
            context=self.context
        )

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

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self._subs.append(self.node.create_subscription(
            Float32, reward_topic,
            lambda msg, k=stats_key: self._reward_cb(msg, k), qos
        ))

        self._subs.append(self.node.create_subscription(
            StringMsg, breakdown_topic,
            lambda msg, k=stats_key: self._breakdown_cb(msg, k), qos
        ))

        print(f"[Domain {self.domain_id}] Subscribed: {reward_topic}")

    def _reward_cb(self, msg: Float32, key: str):
        if key in self.stats:
            self.stats[key].add_reward(msg.data, time.time())

    def _breakdown_cb(self, msg: StringMsg, key: str):
        try:
            data = json.loads(msg.data)
            if key in self.stats:
                self.stats[key].add_breakdown(data)
        except Exception:
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
            except Exception:
                pass

    def stop(self):
        self.running = False
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            rclpy.shutdown(context=self.context)
        except Exception:
            pass


# =============================================================================
# Per-Domain ROS2 Camera Collector (Image -> JPEG)
# =============================================================================

class CameraCollector:
    """
    One CameraCollector per ROS2 domain.

    - Keeps ONE active image subscription at a time per domain (chosen by browser request).
    - Stores latest JPEG bytes for MJPEG streaming.
    """
    def __init__(self, domain_id: int,
                 fps: float = DEFAULT_CAMERA_FPS,
                 jpeg_quality: int = DEFAULT_CAMERA_JPEG_QUALITY,
                 max_width: int = DEFAULT_CAMERA_MAX_WIDTH):
        self.domain_id = domain_id
        self.fps = max(1.0, float(fps))
        self.jpeg_quality = int(max(10, min(95, jpeg_quality)))
        self.max_width = int(max_width)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_ts: float = 0.0

        self._context = Context()
        rclpy.init(context=self._context, domain_id=domain_id)
        self._node = rclpy.create_node(f"camera_monitor_{domain_id}", context=self._context)

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
            except Exception:
                pass

    def ensure_subscription(self, topic: str) -> bool:
        topic = (topic or "").strip()
        if not topic:
            return False
        if not topic.startswith("/"):
            topic = "/" + topic

        # Already subscribed
        if self._sub is not None and self._sub_topic == topic:
            return True

        try:
            # Replace old subscription
            if self._sub is not None:
                try:
                    self._node.destroy_subscription(self._sub)
                except Exception:
                    pass
                self._sub = None

            self._sub = self._node.create_subscription(
                RosImage,
                topic,
                self._image_cb,
                qos_profile_sensor_data
            )
            self._sub_topic = topic
            print(f"[Camera D{self.domain_id}] Subscribed: {topic}")
            return True
        except Exception as e:
            print(f"[Camera D{self.domain_id}] Failed to subscribe {topic}: {e}")
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
                    cv_img = cv2.resize(
                        cv_img,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA
                    )

            ok, buf = cv2.imencode(
                ".jpg",
                cv_img,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                return

            jpeg_bytes = buf.tobytes()
            with self._lock:
                self._latest_jpeg = jpeg_bytes
                self._latest_ts = now

        except Exception:
            return

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def latest_age_sec(self) -> float:
        with self._lock:
            if self._latest_ts == 0:
                return 1e9
            return time.time() - self._latest_ts

    def stop(self):
        self._running = False
        try:
            if self._sub is not None:
                try:
                    self._node.destroy_subscription(self._sub)
                except Exception:
                    pass
            self._executor.shutdown()
            self._node.destroy_node()
            rclpy.shutdown(context=self._context)
        except Exception:
            pass


# =============================================================================
# Multi-Domain Manager (Rewards + Cameras + Scanning)
# =============================================================================

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

                # Create camera collector too (lazy subscribe when requested)
                if domain_id not in self.cameras:
                    self.cameras[domain_id] = CameraCollector(domain_id)

                print(f"[Manager] Added domain {domain_id}")
                return True
            except Exception as e:
                print(f"[Manager] Failed to add domain {domain_id}: {e}")
                return False

    def _scan_domain_topics(self, did: int) -> Tuple[int, bool, List[str]]:
        """
        Returns: (did, has_reward, image_topics)
        """
        ctx = Context()
        image_topics: List[str] = []
        try:
            rclpy.init(context=ctx, domain_id=did)
            node = rclpy.create_node(f"scan_{did}", context=ctx)

            # tiny delay to allow discovery
            time.sleep(0.3)
            topics = node.get_topic_names_and_types()

            has_reward = any(t[0].endswith("/reward") for t in topics)

            # Detect image topics by type name
            # types look like: ['sensor_msgs/msg/Image']
            for tname, ttypes in topics:
                if "sensor_msgs/msg/Image" in ttypes:
                    image_topics.append(tname)

            node.destroy_node()
            rclpy.shutdown(context=ctx)
            return did, has_reward, sorted(image_topics)
        except Exception:
            try:
                rclpy.shutdown(context=ctx)
            except Exception:
                pass
            return did, False, []

    def discover_domains(self, start: int, end: int) -> List[int]:
        discovered = []

        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(self._scan_domain_topics, range(start, end + 1)))

        for did, has_reward, image_topics in results:
            if image_topics:
                self.domain_image_topics[did] = image_topics
            else:
                # keep old list if already known; don't wipe it
                if did not in self.domain_image_topics:
                    self.domain_image_topics[did] = []

            if has_reward and did not in self.collectors:
                if self.add_domain(did):
                    discovered.append(did)

        if discovered:
            print(f"[Manager] Found reward domains: {discovered}")
        return discovered

    def discover_namespaces_all(self):
        for c in self.collectors.values():
            try:
                c.discover_namespaces()
            except Exception:
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
# Dashboard HTML (adds a camera panel)
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
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#f8fafc; margin:0; }
    .header { background:white; padding:16px 20px; border-bottom:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; }
    .container { padding:18px 20px; max-width:1500px; margin:0 auto; }
    .tabs { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px; }
    .tab { padding:10px 14px; background:white; border:1px solid #e2e8f0; border-radius:12px; cursor:pointer; }
    .tab.active { background:#6366f1; border-color:#6366f1; color:white; }
    .card { background:white; border:1px solid #e2e8f0; border-radius:16px; padding:16px; margin-bottom:16px; }
    .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
    @media(max-width:1000px){ .grid2{ grid-template-columns: 1fr; } }
    .row { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    select, input { padding:8px 10px; border-radius:10px; border:1px solid #cbd5e1; }
    .cam {
      width:100%;
      max-height:480px;
      object-fit:contain;
      border-radius:14px;
      border:1px solid #e2e8f0;
      background:#0b1220;
    }
    .muted { color:#64748b; font-size:13px; }
  </style>
</head>
<body>
  <div class="header">
    <div style="font-weight:700">RL Training Monitor</div>
    <div class="muted" id="badge">Connecting...</div>
  </div>

  <div class="container">
    <div class="tabs" id="tabs"></div>
    <div id="content"><div class="muted">Scanning for domains...</div></div>
  </div>

  <script>
    const socket = io();
    let domains = {};
    let selected = 'all';

    socket.on('update', data => {
      domains = data.domains || {};
      render();
    });

    function render() {
      const keys = Object.keys(domains);
      document.getElementById('badge').textContent = keys.length + ' domain(s)';
      renderTabs(keys);

      if (selected === 'all') {
        document.getElementById('content').innerHTML = '<div class="muted">Select a domain to view charts + camera.</div>';
      } else {
        renderSingle(selected);
      }
    }

    function renderTabs(keys) {
      let h = '<div class="tab ' + (selected==='all'?'active':'') + '" onclick="selectTab(\\'all\\')">All</div>';
      keys.forEach(k => {
        const s = domains[k].summary;
        h += '<div class="tab ' + (selected===k?'active':'') + '" onclick="selectTab(\\'' + k + '\\')">' + s.display_name + '</div>';
      });
      document.getElementById('tabs').innerHTML = h;
    }

    function selectTab(k){ selected=k; render(); }

    async function renderSingle(k) {
      const d = domains[k];
      if (!d) return;

      const s = d.summary;

      // Fetch camera topics for this domain_id
      const resp = await fetch('/api/cameras?domain_id=' + s.domain_id);
      const camData = await resp.json();
      const topics = (camData.topics || []);

      const defaultTopic = topics.includes('/sam3_goal_generator/visualization')
        ? '/sam3_goal_generator/visualization'
        : (topics[0] || '');

      const camSelectId = 'cam_topic_select';
      const camImgId = 'cam_img';

      let options = '';
      if (!topics.length) {
        options = '<option value="">(no Image topics found)</option>';
      } else {
        topics.forEach(t => {
          const sel = (t === defaultTopic) ? 'selected' : '';
          options += '<option value="' + t + '" ' + sel + '>' + t + '</option>';
        });
      }

      const streamUrl = defaultTopic ? ('/stream/' + s.domain_id + '?topic=' + encodeURIComponent(defaultTopic)) : '';

      let h = '';
      h += '<div class="card">';
      h += '<div style="font-size:18px;font-weight:700">' + s.display_name + '</div>';
      h += '<div class="muted">Domain ' + s.domain_id + ' | Steps ' + s.total_steps.toLocaleString() + ' | Episodes ' + s.episode_count + '</div>';
      h += '</div>';

      h += '<div class="grid2">';
      h += '  <div class="card">';
      h += '    <div style="font-weight:700; margin-bottom:10px">Live Camera</div>';
      h += '    <div class="row" style="margin-bottom:10px">';
      h += '      <div class="muted">Topic:</div>';
      h += '      <select id="' + camSelectId + '">' + options + '</select>';
      h += '      <span class="muted" id="cam_status"></span>';
      h += '    </div>';
      h += '    <img class="cam" id="' + camImgId + '" src="' + streamUrl + '" />';
      h += '    <div class="muted" style="margin-top:8px">Stream: /stream/' + s.domain_id + '?topic=...</div>';
      h += '  </div>';

      h += '  <div class="card">';
      h += '    <div style="font-weight:700; margin-bottom:10px">Reward (last 100)</div>';
      h += '    <div id="rewChart" style="height:320px"></div>';
      h += '  </div>';
      h += '</div>';

      document.getElementById('content').innerHTML = h;

      // Hook topic change -> update img src
      const sel = document.getElementById(camSelectId);
      const img = document.getElementById(camImgId);
      const status = document.getElementById('cam_status');

      function updateStream() {
        const topic = sel.value;
        if (!topic) {
          img.removeAttribute('src');
          status.textContent = ' (no topic)';
          return;
        }
        img.src = '/stream/' + s.domain_id + '?topic=' + encodeURIComponent(topic) + '&_t=' + Date.now();
        status.textContent = '';
      }
      sel.addEventListener('change', updateStream);

      // Plot reward mini chart
      const p = d.plot_data;
      if (p && p.rewards && p.rewards.values && p.rewards.values.length) {
        const xs = p.rewards.times.slice(-100);
        const ys = p.rewards.values.slice(-100);
        Plotly.newPlot('rewChart', [{
          x: xs, y: ys, type:'scatter', mode:'lines', fill:'tozeroy'
        }], {
          paper_bgcolor:'rgba(0,0,0,0)',
          plot_bgcolor:'rgba(0,0,0,0)',
          margin:{l:45,r:10,t:10,b:35},
          xaxis:{gridcolor:'#e2e8f0'},
          yaxis:{gridcolor:'#e2e8f0'},
          showlegend:false
        }, {displayModeBar:false, responsive:true});
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
        except Exception:
            domain_id = 0
        topics = mgr.get_image_topics(domain_id)
        return jsonify({"domain_id": domain_id, "topics": topics})

    def mjpeg_generator(domain_id: int, topic: str):
        cam = mgr.get_camera(domain_id)
        if cam is None:
            # domain camera not initialized (shouldn't happen if domain is added)
            while True:
                time.sleep(0.2)
                yield b""

        # ensure subscription to requested topic
        cam.ensure_subscription(topic)

        boundary = b"--frame\r\n"
        content_type = b"Content-Type: image/jpeg\r\n"
        while True:
            # if frames are stale, still keep connection alive (blank wait)
            frame = cam.latest_jpeg()
            if frame is None:
                time.sleep(0.05)
                continue

            yield boundary + content_type + b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n"
            # throttle a bit; camera collector already throttles, but this avoids hot loop
            time.sleep(0.001)

    @app.route("/stream/<int:domain_id>")
    def stream(domain_id: int):
        topic = request.args.get("topic", "").strip()
        if not topic:
            return "Missing ?topic=/your/image/topic", 400
        if not topic.startswith("/"):
            topic = "/" + topic

        return Response(
            mjpeg_generator(domain_id, topic),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def updater():
        while True:
            time.sleep(1.0 / UPDATE_RATE_HZ)
            data = {
                "domains": {
                    k: {
                        "summary": v.get_summary(),
                        "plot_data": v.get_plot_data(),
                    } for k, v in stats.items()
                }
            }
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
+--------------------------------------------------------------+
|                   RL Training Monitor                        |
+--------------------------------------------------------------+
|  Browser:  http://localhost:{args.port}
|  Stream:   http://localhost:{args.port}/stream/<domain>?topic=/sam3_goal_generator/visualization
|  Scanning: domains {args.scan_start}-{args.scan_end} every {args.scan_interval}s
+--------------------------------------------------------------+
    """)

    stats: Dict[str, DomainStats] = {}
    mgr = MultiDomainManager(stats)

    print("[Scan] Initial scan...")
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
        print("\n[Monitor] Shutting down...")
        mgr.stop_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        sio.run(
            app,
            host="0.0.0.0",
            port=args.port,
            debug=False,
            use_reloader=False,
            log_output=False,
            allow_unsafe_werkzeug=True,
        )
    except Exception as e:
        print(f"[Error] {e}")
        shutdown()


if __name__ == "__main__":
    main()
