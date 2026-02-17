#!/usr/bin/env python3
"""
Multi-Domain Real-Time Reward Visualization Server

Automatically discovers and monitors ALL active ROS2 domains with reward topics.
Displays individual domain metrics and cross-domain averages.

Features:
- Auto-discovery of all active ROS2 domains
- Real-time graphs per domain
- Cross-domain average metrics
- Professional, clean UI design
"""

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.node import Node
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, String as StringMsg

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = 5555
MAX_HISTORY_LENGTH = 1000
UPDATE_RATE_HZ = 10
DISCOVERY_TIMEOUT = 2.0

# Default scan range
DEFAULT_SCAN_START = 0
DEFAULT_SCAN_END = 50
DEFAULT_SCAN_INTERVAL = 15.0

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
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=100))
    current_episode_return: float = 0.0
    current_episode_steps: int = 0
    episode_count: int = 0
    safety_zones: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    intervention_rates: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    total_steps: int = 0
    last_update: float = 0.0
    active: bool = True
    goals_reached: int = 0
    
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
        
        # Track goals
        if breakdown.get("success", False):
            self.goals_reached += 1
        
        # Check for episode end
        if breakdown.get("collision", False) or breakdown.get("mission_complete", False):
            self.end_episode()
    
    def end_episode(self):
        if self.current_episode_steps > 0:
            self.episode_returns.append(self.current_episode_return)
            self.episode_lengths.append(self.current_episode_steps)
            self.episode_count += 1
            self.current_episode_return = 0.0
            self.current_episode_steps = 0
    
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
            "current_episode_return": self.current_episode_return,
            "current_episode_steps": self.current_episode_steps,
            "avg_reward_100": sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0,
            "avg_episode_return": sum(self.episode_returns) / len(self.episode_returns) if self.episode_returns else 0,
            "avg_intervention": sum(recent_interventions) / len(recent_interventions) if recent_interventions else 0,
            "active": time.time() - self.last_update < 5.0,
            "last_update": self.last_update,
        }
    
    def get_plot_data(self, max_points: int = 500) -> Dict:
        step = max(1, len(self.rewards) // max_points)
        
        rewards = list(self.rewards)[::step]
        timestamps = list(self.timestamps)[::step]
        
        if timestamps:
            t0 = timestamps[0]
            rel_times = [(t - t0) for t in timestamps]
        else:
            rel_times = []
            t0 = 0
        
        episode_returns = list(self.episode_returns)
        episode_indices = list(range(len(episode_returns)))
        
        components_data = {}
        for key, values in self.components.items():
            vals = list(values)[::step]
            t0_comp = vals[0][0] if vals else t0
            components_data[key] = {
                "times": [(v[0] - t0_comp) for v in vals],
                "values": [v[1] for v in vals]
            }
        
        zone_counts = {"FREE": 0, "AWARE": 0, "CAUTION": 0, "DANGER": 0, "EMERGENCY": 0}
        for _, zone in list(self.safety_zones)[-500:]:
            if zone in zone_counts:
                zone_counts[zone] += 1
        
        interventions = list(self.intervention_rates)[::step]
        t0_int = interventions[0][0] if interventions else t0
        intervention_times = [(v[0] - t0_int) for v in interventions]
        intervention_values = [v[1] for v in interventions]
        
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
# Per-Domain ROS2 Subscriber
# =============================================================================

class DomainCollector:
    """Collects reward data from a single ROS2 domain."""
    
    def __init__(self, domain_id: int, stats_store: Dict[str, DomainStats], 
                 namespaces: List[str] = None):
        self.domain_id = domain_id
        self.stats = stats_store
        self.namespaces = namespaces or [""]
        self.running = False
        
        self.context = Context()
        rclpy.init(context=self.context, domain_id=domain_id)
        
        self.node = rclpy.create_node(
            f"reward_viz_domain_{domain_id}",
            context=self.context,
            automatically_declare_parameters_from_overrides=True
        )
        
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        
        self._topic_subs = []
        self._setup_subscriptions()
        
        self.node.get_logger().info(f"[Domain {domain_id}] Collector initialized")
    
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
        
        reward_sub = self.node.create_subscription(
            Float32, reward_topic,
            lambda msg, k=stats_key: self._reward_callback(msg, k),
            qos
        )
        self._topic_subs.append(reward_sub)
        
        breakdown_sub = self.node.create_subscription(
            StringMsg, breakdown_topic,
            lambda msg, k=stats_key: self._breakdown_callback(msg, k),
            qos
        )
        self._topic_subs.append(breakdown_sub)
        
        self.node.get_logger().info(f"[Domain {self.domain_id}] Subscribed: {reward_topic}")
    
    def _reward_callback(self, msg: Float32, stats_key: str):
        if stats_key in self.stats:
            self.stats[stats_key].add_reward(msg.data, time.time())
    
    def _breakdown_callback(self, msg: StringMsg, stats_key: str):
        try:
            breakdown = json.loads(msg.data)
            if stats_key in self.stats:
                self.stats[stats_key].add_breakdown(breakdown)
        except json.JSONDecodeError:
            pass
    
    def discover_namespaces(self) -> List[str]:
        discovered = []
        topic_names_and_types = self.node.get_topic_names_and_types()
        
        for topic_name, _ in topic_names_and_types:
            if topic_name.endswith("/reward"):
                parts = topic_name.rsplit("/reward", 1)[0]
                ns = parts.lstrip("/") if parts else ""
                
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
            except Exception as e:
                self.node.get_logger().warn(f"[Domain {self.domain_id}] Spin error: {e}")
    
    def stop(self):
        self.running = False
        try:
            self.executor.shutdown()
            self.node.destroy_node()
            rclpy.shutdown(context=self.context)
        except Exception:
            pass


# =============================================================================
# Multi-Domain Manager
# =============================================================================

class MultiDomainManager:
    """Manages collectors for multiple ROS2 domains."""
    
    def __init__(self, stats_store: Dict[str, DomainStats]):
        self.stats = stats_store
        self.collectors: Dict[int, DomainCollector] = {}
        self.threads: Dict[int, threading.Thread] = {}
        self.lock = threading.Lock()
    
    def add_domain(self, domain_id: int, namespaces: List[str] = None) -> bool:
        with self.lock:
            if domain_id in self.collectors:
                return False
            
            try:
                collector = DomainCollector(domain_id, self.stats, namespaces)
                self.collectors[domain_id] = collector
                
                thread = threading.Thread(
                    target=collector.spin,
                    daemon=True,
                    name=f"domain_{domain_id}_spin"
                )
                thread.start()
                self.threads[domain_id] = thread
                
                print(f"[Manager] Added domain {domain_id}")
                return True
            except Exception as e:
                print(f"[Manager] Failed to add domain {domain_id}: {e}")
                return False
    
    def discover_domains(self, scan_start: int, scan_end: int) -> List[int]:
        discovered = []
        
        print(f"[Manager] Scanning domains {scan_start}-{scan_end}...")
        
        def check_domain(domain_id: int) -> Optional[int]:
            if domain_id in self.collectors:
                return None
            
            try:
                ctx = Context()
                rclpy.init(context=ctx, domain_id=domain_id)
                node = rclpy.create_node(f"scanner_{domain_id}", context=ctx)
                
                time.sleep(0.3)
                
                topics = node.get_topic_names_and_types()
                has_reward = any(t[0].endswith("/reward") for t in topics)
                
                node.destroy_node()
                rclpy.shutdown(context=ctx)
                
                if has_reward:
                    return domain_id
                return None
            except Exception:
                return None
        
        scan_range = range(scan_start, scan_end + 1)
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(check_domain, scan_range))
        
        for domain_id in results:
            if domain_id is not None and domain_id not in self.collectors:
                if self.add_domain(domain_id):
                    discovered.append(domain_id)
        
        if discovered:
            print(f"[Manager] Discovered {len(discovered)} domains: {discovered}")
        
        return discovered
    
    def discover_namespaces_all(self):
        for domain_id, collector in self.collectors.items():
            try:
                new_ns = collector.discover_namespaces()
                if new_ns:
                    print(f"[Manager] Domain {domain_id} new namespaces: {new_ns}")
            except Exception as e:
                print(f"[Manager] Discovery error in domain {domain_id}: {e}")
    
    def stop_all(self):
        for collector in self.collectors.values():
            collector.stop()


# =============================================================================
# Flask Web Server - Professional Clean Design
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
            --bg-primary: #1a1d23;
            --bg-secondary: #22262e;
            --bg-tertiary: #2a2f38;
            --bg-hover: #323842;
            --border-color: #3a3f4a;
            --text-primary: #e4e6ea;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-yellow: #f59e0b;
            --accent-red: #ef4444;
            --accent-purple: #8b5cf6;
            --accent-cyan: #06b6d4;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
        }
        
        .header {
            background: var(--bg-secondary);
            padding: 18px 28px;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
            letter-spacing: -0.3px;
        }
        
        .header-info {
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .domain-count {
            background: var(--bg-tertiary);
            padding: 6px 14px;
            border-radius: var(--radius-md);
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
            border: 1px solid var(--border-color);
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent-red);
            transition: all 0.3s ease;
        }
        
        .status-dot.connected {
            background: var(--accent-green);
            box-shadow: 0 0 8px rgba(16, 185, 129, 0.5);
        }
        
        .container {
            padding: 24px;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .domain-tabs {
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            flex-wrap: wrap;
        }
        
        .domain-tab {
            padding: 10px 18px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
        }
        
        .domain-tab:hover {
            background: var(--bg-hover);
            border-color: var(--accent-blue);
        }
        
        .domain-tab.active {
            background: var(--accent-blue);
            border-color: var(--accent-blue);
            color: white;
        }
        
        .domain-tab .indicator {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--text-muted);
        }
        
        .domain-tab.active-domain .indicator {
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }
        
        .domain-tab.active .indicator {
            background: rgba(255,255,255,0.6);
        }
        
        .domain-tab.active.active-domain .indicator {
            background: white;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.4; }
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }
        
        .stat-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 18px 20px;
            transition: all 0.2s ease;
        }
        
        .stat-card:hover {
            border-color: var(--accent-blue);
            transform: translateY(-1px);
        }
        
        .stat-card .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
            font-weight: 500;
        }
        
        .stat-card .value {
            font-size: 26px;
            font-weight: 600;
            letter-spacing: -0.5px;
        }
        
        .stat-card .value.positive { color: var(--accent-green); }
        .stat-card .value.negative { color: var(--accent-red); }
        .stat-card .value.neutral { color: var(--accent-yellow); }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 18px;
            margin-bottom: 18px;
        }
        
        @media (max-width: 1100px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
        
        .chart-container {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 20px;
        }
        
        .chart-container h3 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }
        
        .chart { height: 280px; }
        .chart.tall { height: 360px; }
        
        .safety-zones {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 12px;
        }
        
        .zone-badge {
            padding: 4px 12px;
            border-radius: var(--radius-sm);
            font-size: 11px;
            font-weight: 600;
        }
        
        .zone-badge.FREE { background: rgba(16, 185, 129, 0.15); color: #34d399; }
        .zone-badge.AWARE { background: rgba(132, 204, 22, 0.15); color: #a3e635; }
        .zone-badge.CAUTION { background: rgba(245, 158, 11, 0.15); color: #fbbf24; }
        .zone-badge.DANGER { background: rgba(249, 115, 22, 0.15); color: #fb923c; }
        .zone-badge.EMERGENCY { background: rgba(239, 68, 68, 0.15); color: #f87171; }
        
        .domain-list { margin-top: 20px; }
        
        .domain-row {
            background: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: var(--radius-lg);
            padding: 16px 20px;
            margin-bottom: 10px;
            display: grid;
            grid-template-columns: 180px 1fr repeat(4, 90px);
            align-items: center;
            gap: 16px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .domain-row:hover {
            background: var(--bg-hover);
            border-color: var(--accent-blue);
        }
        
        .domain-row .domain-name {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .domain-row .domain-name .title {
            font-weight: 600;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .domain-row .domain-name .subtitle {
            font-size: 11px;
            color: var(--text-muted);
        }
        
        .domain-row .mini-chart { height: 45px; }
        
        .domain-row .stat {
            text-align: center;
        }
        
        .domain-row .stat .label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .domain-row .stat .value {
            font-size: 16px;
            font-weight: 600;
            margin-top: 2px;
        }
        
        .no-data {
            text-align: center;
            padding: 80px 20px;
            color: var(--text-muted);
        }
        
        .no-data h2 {
            font-size: 18px;
            font-weight: 500;
            margin-bottom: 8px;
            color: var(--text-secondary);
        }
        
        .no-data p {
            font-size: 14px;
        }
        
        .section-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--text-primary);
        }
        
        .average-section {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: var(--radius-lg);
            padding: 20px;
            margin-bottom: 24px;
        }
        
        .average-section h3 {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 16px;
            color: var(--accent-blue);
        }
        
        .average-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 16px;
        }
        
        .average-stat {
            text-align: center;
        }
        
        .average-stat .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .average-stat .value {
            font-size: 22px;
            font-weight: 600;
            color: var(--text-primary);
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RL Training Monitor</h1>
        <div class="header-info">
            <span class="domain-count" id="domainCount">Scanning...</span>
            <div class="connection-status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="domain-tabs" id="domainTabs">
            <div class="domain-tab active" onclick="selectDomain('all')">All Domains</div>
        </div>
        <div id="content">
            <div class="no-data">
                <h2>Scanning for active domains...</h2>
                <p>Looking for ROS2 domains with reward topics</p>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let domains = {};
        let selectedDomain = 'all';
        
        const plotlyLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#9ca3af', size: 11 },
            margin: { l: 45, r: 15, t: 25, b: 35 },
            xaxis: { gridcolor: '#3a3f4a', zerolinecolor: '#3a3f4a', tickfont: { size: 10 } },
            yaxis: { gridcolor: '#3a3f4a', zerolinecolor: '#3a3f4a', tickfont: { size: 10 } },
            showlegend: true,
            legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#9ca3af', size: 10 } }
        };
        const plotlyConfig = { displayModeBar: false, responsive: true };
        
        socket.on('connect', () => {
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Connected';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('statusDot').classList.remove('connected');
            document.getElementById('statusText').textContent = 'Disconnected';
        });
        
        socket.on('update', (data) => {
            domains = data.domains;
            updateUI();
        });
        
        function updateUI() {
            const count = Object.keys(domains).length;
            document.getElementById('domainCount').textContent = count + ' Domain' + (count !== 1 ? 's' : '');
            updateTabs();
            updateContent();
        }
        
        function updateTabs() {
            const tabsContainer = document.getElementById('domainTabs');
            const domainKeys = Object.keys(domains);
            
            let html = '<div class="domain-tab ' + (selectedDomain === 'all' ? 'active' : '') + '" onclick="selectDomain(\\'all\\')">All Domains (' + domainKeys.length + ')</div>';
            
            domainKeys.forEach(key => {
                const d = domains[key];
                const stats = d.summary;
                const isActive = selectedDomain === key;
                const isLive = stats.active;
                html += '<div class="domain-tab ' + (isActive ? 'active' : '') + ' ' + (isLive ? 'active-domain' : '') + '" onclick="selectDomain(\\'' + key + '\\')">' +
                    '<div class="indicator"></div>' +
                    '<span>' + stats.display_name + '</span>' +
                '</div>';
            });
            
            tabsContainer.innerHTML = html;
        }
        
        function selectDomain(key) {
            selectedDomain = key;
            updateTabs();
            updateContent();
        }
        
        function updateContent() {
            const container = document.getElementById('content');
            if (Object.keys(domains).length === 0) {
                container.innerHTML = '<div class="no-data"><h2>Scanning for active domains...</h2><p>Looking for ROS2 domains with reward topics</p></div>';
                return;
            }
            if (selectedDomain === 'all') {
                renderAllDomainsView(container);
            } else {
                renderSingleDomainView(container, selectedDomain);
            }
        }
        
        function renderAllDomainsView(container) {
            const vals = Object.values(domains);
            
            // Calculate averages across all domains
            const totalSteps = vals.reduce((s, d) => s + d.summary.total_steps, 0);
            const totalEpisodes = vals.reduce((s, d) => s + d.summary.episode_count, 0);
            const totalGoals = vals.reduce((s, d) => s + d.summary.goals_reached, 0);
            const activeCount = vals.filter(d => d.summary.active).length;
            
            const avgReturns = vals.filter(d => d.summary.episode_count > 0).map(d => d.summary.avg_episode_return);
            const avgReturn = avgReturns.length > 0 ? avgReturns.reduce((a, b) => a + b, 0) / avgReturns.length : 0;
            
            const avgInterventions = vals.filter(d => d.summary.total_steps > 0).map(d => d.summary.avg_intervention);
            const avgIntervention = avgInterventions.length > 0 ? avgInterventions.reduce((a, b) => a + b, 0) / avgInterventions.length : 0;
            
            let html = '<div class="average-section">' +
                '<h3>Cross-Domain Averages</h3>' +
                '<div class="average-stats">' +
                    '<div class="average-stat"><div class="label">Total Steps</div><div class="value">' + totalSteps.toLocaleString() + '</div></div>' +
                    '<div class="average-stat"><div class="label">Total Episodes</div><div class="value">' + totalEpisodes + '</div></div>' +
                    '<div class="average-stat"><div class="label">Total Goals</div><div class="value">' + totalGoals + '</div></div>' +
                    '<div class="average-stat"><div class="label">Avg Return</div><div class="value" style="color:' + (avgReturn >= 0 ? '#10b981' : '#ef4444') + '">' + avgReturn.toFixed(1) + '</div></div>' +
                    '<div class="average-stat"><div class="label">Avg Intervention</div><div class="value">' + (avgIntervention * 100).toFixed(1) + '%</div></div>' +
                    '<div class="average-stat"><div class="label">Active</div><div class="value" style="color:#10b981">' + activeCount + ' / ' + vals.length + '</div></div>' +
                '</div>' +
            '</div>';
            
            html += '<div class="chart-container" style="margin-bottom:20px"><h3>Episode Returns - All Domains</h3><div id="comparisonChart" class="chart tall"></div></div>';
            
            html += '<div class="section-title">Individual Domains</div><div class="domain-list">';
            
            Object.entries(domains).forEach(([key, data]) => {
                const s = data.summary;
                const safeKey = key.replace(/[^a-zA-Z0-9]/g, '_');
                html += '<div class="domain-row" onclick="selectDomain(\\'' + key + '\\')">' +
                    '<div class="domain-name">' +
                        '<div class="title"><div style="width:6px;height:6px;border-radius:50%;background:' + (s.active ? '#10b981' : '#6b7280') + '"></div>' + s.display_name + '</div>' +
                        '<div class="subtitle">Domain ' + s.domain_id + '</div>' +
                    '</div>' +
                    '<div class="mini-chart" id="mini_' + safeKey + '"></div>' +
                    '<div class="stat"><div class="label">Steps</div><div class="value">' + s.total_steps.toLocaleString() + '</div></div>' +
                    '<div class="stat"><div class="label">Episodes</div><div class="value">' + s.episode_count + '</div></div>' +
                    '<div class="stat"><div class="label">Avg Return</div><div class="value" style="color:' + (s.avg_episode_return >= 0 ? '#10b981' : '#ef4444') + '">' + s.avg_episode_return.toFixed(1) + '</div></div>' +
                    '<div class="stat"><div class="label">Goals</div><div class="value" style="color:#8b5cf6">' + s.goals_reached + '</div></div>' +
                '</div>';
            });
            
            html += '</div>';
            container.innerHTML = html;
            
            // Draw comparison chart
            const traces = [];
            const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16'];
            let ci = 0;
            
            Object.entries(domains).forEach(([key, data]) => {
                const pd = data.plot_data;
                if (pd.episode_returns.values.length > 0) {
                    traces.push({
                        x: pd.episode_returns.episodes,
                        y: pd.episode_returns.values,
                        type: 'scatter',
                        mode: 'lines',
                        name: pd.display_name,
                        line: { color: colors[ci++ % colors.length], width: 2 }
                    });
                }
            });
            
            if (traces.length > 0) {
                Plotly.newPlot('comparisonChart', traces, {
                    ...plotlyLayout,
                    xaxis: { ...plotlyLayout.xaxis, title: 'Episode' },
                    yaxis: { ...plotlyLayout.yaxis, title: 'Return' }
                }, plotlyConfig);
            }
            
            // Mini charts
            Object.entries(domains).forEach(([key, data]) => {
                const safeKey = key.replace(/[^a-zA-Z0-9]/g, '_');
                const el = document.getElementById('mini_' + safeKey);
                if (el && data.plot_data.rewards.values.length > 0) {
                    const recentRewards = data.plot_data.rewards.values.slice(-100);
                    const recentTimes = data.plot_data.rewards.times.slice(-100);
                    Plotly.newPlot('mini_' + safeKey, [{
                        x: recentTimes,
                        y: recentRewards,
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: '#3b82f6', width: 1.5 },
                        fill: 'tozeroy',
                        fillcolor: 'rgba(59, 130, 246, 0.15)'
                    }], {
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        margin: { l: 0, r: 0, t: 0, b: 0 },
                        xaxis: { visible: false },
                        yaxis: { visible: false },
                        showlegend: false
                    }, plotlyConfig);
                }
            });
        }
        
        function renderSingleDomainView(container, key) {
            const data = domains[key];
            if (!data) return;
            
            const s = data.summary;
            const pd = data.plot_data;
            
            let html = '<div style="margin-bottom:24px">' +
                '<h2 style="font-size:20px;font-weight:600;margin-bottom:4px">' + s.display_name + '</h2>' +
                '<p style="color:#6b7280;font-size:13px">Domain ' + s.domain_id + '</p>' +
            '</div>';
            
            html += '<div class="stats-grid">' +
                '<div class="stat-card"><div class="label">Total Steps</div><div class="value">' + s.total_steps.toLocaleString() + '</div></div>' +
                '<div class="stat-card"><div class="label">Episodes</div><div class="value">' + s.episode_count + '</div></div>' +
                '<div class="stat-card"><div class="label">Goals Reached</div><div class="value positive">' + s.goals_reached + '</div></div>' +
                '<div class="stat-card"><div class="label">Current Return</div><div class="value ' + (s.current_episode_return >= 0 ? 'positive' : 'negative') + '">' + s.current_episode_return.toFixed(1) + '</div></div>' +
                '<div class="stat-card"><div class="label">Avg Return</div><div class="value ' + (s.avg_episode_return >= 0 ? 'positive' : 'negative') + '">' + s.avg_episode_return.toFixed(1) + '</div></div>' +
                '<div class="stat-card"><div class="label">Avg Intervention</div><div class="value ' + (s.avg_intervention < 0.2 ? 'positive' : 'neutral') + '">' + (s.avg_intervention * 100).toFixed(1) + '%</div></div>' +
            '</div>';
            
            html += '<div class="charts-grid">' +
                '<div class="chart-container"><h3>Real-Time Reward</h3><div id="rewardChart" class="chart"></div></div>' +
                '<div class="chart-container"><h3>Episode Returns</h3><div id="episodeChart" class="chart"></div></div>' +
                '<div class="chart-container"><h3>Safety Intervention Rate</h3><div id="interventionChart" class="chart"></div></div>' +
                '<div class="chart-container"><h3>Safety Zone Distribution</h3><div id="zoneChart" class="chart"></div>' +
                    '<div class="safety-zones">' + Object.entries(pd.safety_zones).map(function(z) { return '<span class="zone-badge ' + z[0] + '">' + z[0] + ': ' + z[1] + '</span>'; }).join('') + '</div>' +
                '</div>' +
            '</div>';
            
            html += '<div class="chart-container"><h3>Reward Components</h3><div id="componentsChart" class="chart tall"></div></div>';
            
            container.innerHTML = html;
            
            // Reward chart
            if (pd.rewards.values.length > 0) {
                Plotly.newPlot('rewardChart', [{
                    x: pd.rewards.times,
                    y: pd.rewards.values,
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#3b82f6', width: 1.5 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(59, 130, 246, 0.15)'
                }], {
                    ...plotlyLayout,
                    xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' },
                    yaxis: { ...plotlyLayout.yaxis, title: 'Reward' }
                }, plotlyConfig);
            }
            
            // Episode returns chart
            if (pd.episode_returns.values.length > 0) {
                const windowSize = 10;
                const rollingAvg = [];
                for (let i = 0; i < pd.episode_returns.values.length; i++) {
                    const window = pd.episode_returns.values.slice(Math.max(0, i - windowSize + 1), i + 1);
                    rollingAvg.push(window.reduce((a, b) => a + b, 0) / window.length);
                }
                Plotly.newPlot('episodeChart', [
                    {
                        x: pd.episode_returns.episodes,
                        y: pd.episode_returns.values,
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: 'Return',
                        line: { color: '#10b981', width: 2 },
                        marker: { size: 5 }
                    },
                    {
                        x: pd.episode_returns.episodes,
                        y: rollingAvg,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Rolling Avg',
                        line: { color: '#f59e0b', width: 2.5 }
                    }
                ], {
                    ...plotlyLayout,
                    xaxis: { ...plotlyLayout.xaxis, title: 'Episode' },
                    yaxis: { ...plotlyLayout.yaxis, title: 'Return' }
                }, plotlyConfig);
            }
            
            // Intervention chart
            if (pd.interventions.values.length > 0) {
                Plotly.newPlot('interventionChart', [{
                    x: pd.interventions.times,
                    y: pd.interventions.values.map(v => v * 100),
                    type: 'scatter',
                    mode: 'lines',
                    line: { color: '#ef4444', width: 1.5 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(239, 68, 68, 0.15)'
                }], {
                    ...plotlyLayout,
                    xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' },
                    yaxis: { ...plotlyLayout.yaxis, title: '%', range: [0, 100] }
                }, plotlyConfig);
            }
            
            // Zone chart
            const zoneColors = {
                'FREE': '#10b981',
                'AWARE': '#84cc16',
                'CAUTION': '#f59e0b',
                'DANGER': '#f97316',
                'EMERGENCY': '#ef4444'
            };
            const zones = Object.entries(pd.safety_zones).filter(z => z[1] > 0);
            if (zones.length > 0) {
                Plotly.newPlot('zoneChart', [{
                    values: zones.map(z => z[1]),
                    labels: zones.map(z => z[0]),
                    type: 'pie',
                    marker: { colors: zones.map(z => zoneColors[z[0]] || '#6b7280') },
                    textinfo: 'label+percent',
                    textfont: { color: '#fff', size: 11 },
                    hole: 0.45
                }], {
                    ...plotlyLayout,
                    showlegend: false
                }, plotlyConfig);
            }
            
            // Components chart
            const componentColors = {
                'progress': '#10b981',
                'discovery': '#8b5cf6',
                'proximity': '#ef4444',
                'intervention_penalty': '#dc2626',
                'step': '#6b7280',
                'novelty': '#ec4899',
                'rnd': '#06b6d4',
                'goal': '#f59e0b',
                'alignment': '#3b82f6',
                'mission_complete': '#22d3ee'
            };
            const compTraces = [];
            Object.entries(pd.components).forEach(([name, d]) => {
                if (d.values.length > 0) {
                    compTraces.push({
                        x: d.times,
                        y: d.values,
                        type: 'scatter',
                        mode: 'lines',
                        name: name,
                        line: { color: componentColors[name] || '#6b7280', width: 1.5 }
                    });
                }
            });
            if (compTraces.length > 0) {
                Plotly.newPlot('componentsChart', compTraces, {
                    ...plotlyLayout,
                    xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' },
                    yaxis: { ...plotlyLayout.yaxis, title: 'Value' }
                }, plotlyConfig);
            }
        }
    </script>
</body>
</html>
"""


def create_app(stats_store: Dict[str, DomainStats]):
    """Create Flask app with SocketIO."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'multi_domain_viz'
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/api/domains')
    def get_domains():
        result = {}
        for key, stats in stats_store.items():
            result[key] = {"summary": stats.get_summary(), "plot_data": stats.get_plot_data()}
        return jsonify(result)
    
    def background_update():
        while True:
            time.sleep(1.0 / UPDATE_RATE_HZ)
            data = {"domains": {}}
            for key, stats in stats_store.items():
                data["domains"][key] = {"summary": stats.get_summary(), "plot_data": stats.get_plot_data()}
            socketio.emit('update', data)
    
    return app, socketio, background_update


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-domain RL reward visualization")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Web server port (default: {DEFAULT_PORT})")
    parser.add_argument("--domains", nargs="*", type=int, default=None,
                        help="Specific domain IDs to monitor")
    parser.add_argument("--scan-start", type=int, default=DEFAULT_SCAN_START,
                        help=f"Start of scan range (default: {DEFAULT_SCAN_START})")
    parser.add_argument("--scan-end", type=int, default=DEFAULT_SCAN_END,
                        help=f"End of scan range (default: {DEFAULT_SCAN_END})")
    parser.add_argument("--scan-interval", type=float, default=DEFAULT_SCAN_INTERVAL,
                        help=f"Seconds between scans (default: {DEFAULT_SCAN_INTERVAL})")
    parser.add_argument("--namespaces", nargs="*", default=["", "stretch"],
                        help="Namespaces to monitor")
    args = parser.parse_args()
    
    print("""
+--------------------------------------------------------------+
|                   RL Training Monitor                        |
+--------------------------------------------------------------+
|  Open in browser:  http://localhost:{}                    |
|  Scan range: {}-{}                                           |
|  Auto-scanning for active domains...                         |
+--------------------------------------------------------------+
    """.format(args.port, args.scan_start, args.scan_end))
    
    # Shared stats store
    stats_store: Dict[str, DomainStats] = {}
    
    # Create domain manager
    manager = MultiDomainManager(stats_store)
    
    # Add specific domains if provided
    if args.domains:
        for domain_id in args.domains:
            manager.add_domain(domain_id, args.namespaces.copy())
    
    # Initial scan
    print(f"[Scan] Initial scan of domains {args.scan_start}-{args.scan_end}...")
    manager.discover_domains(args.scan_start, args.scan_end)
    
    # Create Flask app
    app, socketio, background_update = create_app(stats_store)
    
    # Start background update thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    # Continuous scanning thread
    def scan_loop():
        while True:
            time.sleep(args.scan_interval)
            try:
                manager.discover_domains(args.scan_start, args.scan_end)
                manager.discover_namespaces_all()
            except Exception as e:
                print(f"[Scan] Error: {e}")
    
    scan_thread = threading.Thread(target=scan_loop, daemon=True)
    scan_thread.start()
    
    # Shutdown handler
    def shutdown(signum=None, frame=None):
        print("\n[Monitor] Shutting down...")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Run Flask server
    try:
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False,
                     use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"[Monitor] Server error: {e}")
        shutdown()


if __name__ == "__main__":
    main()