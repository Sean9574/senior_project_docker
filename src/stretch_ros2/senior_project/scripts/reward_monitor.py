#!/usr/bin/env python3
"""
Multi-Domain RL Training Monitor

Automatically discovers and monitors all active ROS2 domains with reward topics.
Professional, clean interface with soft colors and rounded design.
"""

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, String as StringMsg

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = 5555
MAX_HISTORY_LENGTH = 1000
UPDATE_RATE_HZ = 10
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
            # Episode changed - save the previous episode's return
            if self.current_episode_steps > 0:
                self.episode_returns.append(self.current_episode_return)
                self.episode_count += 1
            self.current_episode_return = 0.0
            self.current_episode_steps = 0
            self.goals_reached = 0  # Reset goals for new episode
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
# Per-Domain ROS2 Subscriber
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


# =============================================================================
# Multi-Domain Manager
# =============================================================================

class MultiDomainManager:
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
                thread = threading.Thread(target=collector.spin, daemon=True)
                thread.start()
                self.threads[domain_id] = thread
                print(f"[Manager] Added domain {domain_id}")
                return True
            except Exception as e:
                print(f"[Manager] Failed to add domain {domain_id}: {e}")
                return False
    
    def discover_domains(self, start: int, end: int) -> List[int]:
        discovered = []
        
        def check(did):
            if did in self.collectors:
                return None
            try:
                ctx = Context()
                rclpy.init(context=ctx, domain_id=did)
                node = rclpy.create_node(f"scan_{did}", context=ctx)
                time.sleep(0.3)
                topics = node.get_topic_names_and_types()
                has_reward = any(t[0].endswith("/reward") for t in topics)
                node.destroy_node()
                rclpy.shutdown(context=ctx)
                return did if has_reward else None
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=10) as ex:
            results = list(ex.map(check, range(start, end + 1)))
        
        for did in results:
            if did is not None and did not in self.collectors:
                if self.add_domain(did):
                    discovered.append(did)
        
        if discovered:
            print(f"[Manager] Found domains: {discovered}")
        return discovered
    
    def discover_namespaces_all(self):
        for c in self.collectors.values():
            try:
                c.discover_namespaces()
            except:
                pass
    
    def stop_all(self):
        for c in self.collectors.values():
            c.stop()


# =============================================================================
# Dashboard HTML - Soft, Professional Design
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
            --bg-base: #f8fafc;
            --bg-card: #ffffff;
            --bg-card-hover: #f1f5f9;
            --border: #e2e8f0;
            --border-hover: #cbd5e1;
            --text-primary: #1e293b;
            --text-secondary: #475569;
            --text-muted: #94a3b8;
            --accent-blue: #6366f1;
            --accent-green: #22c55e;
            --accent-yellow: #eab308;
            --accent-red: #ef4444;
            --accent-purple: #a855f7;
            --accent-cyan: #06b6d4;
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.06);
            --radius: 16px;
            --radius-sm: 10px;
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: var(--bg-base);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .header {
            background: var(--bg-card);
            padding: 20px 32px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: var(--shadow-sm);
        }
        
        .header h1 {
            font-size: 22px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        
        .domain-badge {
            background: linear-gradient(135deg, #f0f4ff 0%, #e8f0fe 100%);
            border: 1px solid #c7d2fe;
            padding: 8px 16px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 600;
            color: var(--accent-blue);
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent-red);
            transition: all 0.3s;
        }
        
        .status-dot.connected {
            background: var(--accent-green);
            box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.2);
        }
        
        .container {
            padding: 28px 32px;
            max-width: 1500px;
            margin: 0 auto;
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 28px;
            flex-wrap: wrap;
        }
        
        .tab {
            padding: 12px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-sm);
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
            font-weight: 500;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: var(--shadow-sm);
        }
        
        .tab:hover {
            border-color: var(--accent-blue);
            color: var(--accent-blue);
            transform: translateY(-1px);
        }
        
        .tab.active {
            background: linear-gradient(135deg, var(--accent-blue) 0%, #818cf8 100%);
            border-color: transparent;
            color: white;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        .tab .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--text-muted);
        }
        
        .tab.live .dot {
            background: var(--accent-green);
            animation: pulse 2s infinite;
        }
        
        .tab.active .dot {
            background: rgba(255,255,255,0.7);
        }
        
        .tab.active.live .dot {
            background: white;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(0.9); }
        }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 24px;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s;
        }
        
        .card:hover {
            box-shadow: var(--shadow-md);
        }
        
        .card h3 {
            font-size: 14px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 16px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }
        
        .stat {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px 24px;
            box-shadow: var(--shadow-sm);
            transition: all 0.2s;
        }
        
        .stat:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .stat .label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .stat .value {
            font-size: 28px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.5px;
        }
        
        .stat .value.green { color: var(--accent-green); }
        .stat .value.red { color: var(--accent-red); }
        .stat .value.yellow { color: var(--accent-yellow); }
        .stat .value.blue { color: var(--accent-blue); }
        .stat .value.purple { color: var(--accent-purple); }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1000px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
        
        .chart { height: 260px; }
        .chart.tall { height: 320px; }
        
        .zones {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 16px;
        }
        
        .zone {
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .zone.FREE { background: #dcfce7; color: #166534; }
        .zone.AWARE { background: #fef9c3; color: #854d0e; }
        .zone.CAUTION { background: #fed7aa; color: #9a3412; }
        .zone.DANGER { background: #fecaca; color: #991b1b; }
        .zone.EMERGENCY { background: #fca5a5; color: #7f1d1d; }
        
        .domain-list {
            margin-top: 28px;
        }
        
        .domain-item {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 20px 24px;
            margin-bottom: 12px;
            display: grid;
            grid-template-columns: 200px 1fr repeat(4, 100px);
            align-items: center;
            gap: 20px;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: var(--shadow-sm);
        }
        
        .domain-item:hover {
            border-color: var(--accent-blue);
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
        }
        
        .domain-item .name {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }
        
        .domain-item .name .title {
            font-weight: 600;
            font-size: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .domain-item .name .sub {
            font-size: 12px;
            color: var(--text-muted);
        }
        
        .domain-item .mini { height: 50px; }
        
        .domain-item .col {
            text-align: center;
        }
        
        .domain-item .col .label {
            font-size: 10px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .domain-item .col .value {
            font-size: 18px;
            font-weight: 700;
            margin-top: 4px;
        }
        
        .empty {
            text-align: center;
            padding: 80px 20px;
            color: var(--text-muted);
        }
        
        .empty h2 {
            font-size: 20px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
        }
        
        .summary-card {
            background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 100%);
            border: 1px solid #e0e7ff;
            border-radius: var(--radius);
            padding: 24px;
            margin-bottom: 28px;
        }
        
        .summary-card h3 {
            font-size: 14px;
            font-weight: 700;
            color: var(--accent-blue);
            margin-bottom: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .summary-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 20px;
        }
        
        .summary-stat {
            text-align: center;
        }
        
        .summary-stat .label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }
        
        .summary-stat .value {
            font-size: 24px;
            font-weight: 700;
            color: var(--text-primary);
            margin-top: 6px;
        }
        
        .section-title {
            font-size: 16px;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 16px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>RL Training Monitor</h1>
        <div class="header-right">
            <span class="domain-badge" id="domainBadge">Scanning...</span>
            <div class="status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="tabs" id="tabs"></div>
        <div id="content">
            <div class="empty">
                <h2>Scanning for domains...</h2>
                <p>Looking for active ROS2 domains with reward topics</p>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        let domains = {};
        let selected = 'all';
        
        const colors = {
            blue: '#6366f1', green: '#22c55e', yellow: '#eab308',
            red: '#ef4444', purple: '#a855f7', cyan: '#06b6d4',
            pink: '#ec4899', orange: '#f97316'
        };
        
        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#64748b', size: 11 },
            margin: { l: 50, r: 20, t: 30, b: 40 },
            xaxis: { gridcolor: '#e2e8f0', zerolinecolor: '#e2e8f0' },
            yaxis: { gridcolor: '#e2e8f0', zerolinecolor: '#e2e8f0' },
            showlegend: true,
            legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 10 } }
        };
        const config = { displayModeBar: false, responsive: true };
        
        socket.on('connect', () => {
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Connected';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('statusDot').classList.remove('connected');
            document.getElementById('statusText').textContent = 'Disconnected';
        });
        
        socket.on('update', data => { domains = data.domains; render(); });
        
        function render() {
            const keys = Object.keys(domains);
            document.getElementById('domainBadge').textContent = keys.length + ' Domain' + (keys.length !== 1 ? 's' : '');
            renderTabs(keys);
            if (selected === 'all') renderAll();
            else renderSingle(selected);
        }
        
        function renderTabs(keys) {
            let h = '<div class="tab ' + (selected === 'all' ? 'active' : '') + '" onclick="select(\\'all\\')">All Domains (' + keys.length + ')</div>';
            keys.forEach(k => {
                const s = domains[k].summary;
                h += '<div class="tab ' + (selected === k ? 'active' : '') + ' ' + (s.active ? 'live' : '') + '" onclick="select(\\'' + k + '\\')">' +
                    '<div class="dot"></div>' + s.display_name + '</div>';
            });
            document.getElementById('tabs').innerHTML = h;
        }
        
        function select(k) { selected = k; render(); }
        
        function renderAll() {
            const vals = Object.values(domains);
            if (!vals.length) {
                document.getElementById('content').innerHTML = '<div class="empty"><h2>No domains found</h2><p>Waiting for domains...</p></div>';
                return;
            }
            
            const totSteps = vals.reduce((a, d) => a + d.summary.total_steps, 0);
            const totEp = vals.reduce((a, d) => a + d.summary.episode_count, 0);
            const totGoals = vals.reduce((a, d) => a + d.summary.goals_reached, 0);
            const active = vals.filter(d => d.summary.active).length;
            const returns = vals.filter(d => d.summary.episode_count > 0).map(d => d.summary.avg_episode_return);
            const avgRet = returns.length ? returns.reduce((a,b) => a+b, 0) / returns.length : 0;
            const ints = vals.filter(d => d.summary.total_steps > 0).map(d => d.summary.avg_intervention);
            const avgInt = ints.length ? ints.reduce((a,b) => a+b, 0) / ints.length : 0;
            
            let h = '<div class="summary-card"><h3>Cross-Domain Summary</h3><div class="summary-stats">' +
                '<div class="summary-stat"><div class="label">Total Steps</div><div class="value">' + totSteps.toLocaleString() + '</div></div>' +
                '<div class="summary-stat"><div class="label">Episodes</div><div class="value">' + totEp + '</div></div>' +
                '<div class="summary-stat"><div class="label">Goals</div><div class="value">' + totGoals + '</div></div>' +
                '<div class="summary-stat"><div class="label">Avg Return</div><div class="value" style="color:' + (avgRet >= 0 ? colors.green : colors.red) + '">' + avgRet.toFixed(1) + '</div></div>' +
                '<div class="summary-stat"><div class="label">Avg Intervention</div><div class="value">' + (avgInt * 100).toFixed(1) + '%</div></div>' +
                '<div class="summary-stat"><div class="label">Active</div><div class="value" style="color:' + colors.green + '">' + active + '/' + vals.length + '</div></div>' +
            '</div></div>';
            
            h += '<div class="card" style="margin-bottom:28px"><h3>Episode Returns</h3><div id="cmpChart" class="chart tall"></div></div>';
            
            h += '<div class="section-title">Individual Domains</div><div class="domain-list">';
            Object.entries(domains).forEach(([k, d]) => {
                const s = d.summary;
                const sk = k.replace(/[^a-zA-Z0-9]/g, '_');
                h += '<div class="domain-item" onclick="select(\\'' + k + '\\')">' +
                    '<div class="name"><div class="title"><div style="width:8px;height:8px;border-radius:50%;background:' + (s.active ? colors.green : '#cbd5e1') + '"></div>' + s.display_name + '</div>' +
                    '<div class="sub">Domain ' + s.domain_id + ' | Cells: ' + s.cells_discovered + '</div></div>' +
                    '<div class="mini" id="m_' + sk + '"></div>' +
                    '<div class="col"><div class="label">Steps</div><div class="value">' + s.total_steps.toLocaleString() + '</div></div>' +
                    '<div class="col"><div class="label">Episodes</div><div class="value">' + s.episode_count + '</div></div>' +
                    '<div class="col"><div class="label">Avg Return</div><div class="value" style="color:' + (s.avg_episode_return >= 0 ? colors.green : colors.red) + '">' + s.avg_episode_return.toFixed(1) + '</div></div>' +
                    '<div class="col"><div class="label">Goals</div><div class="value" style="color:' + colors.purple + '">' + s.goals_reached + '</div></div>' +
                '</div>';
            });
            h += '</div>';
            
            document.getElementById('content').innerHTML = h;
            
            // Comparison chart
            const traces = [];
            const cls = [colors.blue, colors.green, colors.yellow, colors.red, colors.purple, colors.cyan, colors.pink, colors.orange];
            let ci = 0;
            Object.values(domains).forEach(d => {
                const p = d.plot_data;
                if (p.episode_returns.values.length) {
                    traces.push({ x: p.episode_returns.episodes, y: p.episode_returns.values, type: 'scatter', mode: 'lines', name: p.display_name, line: { color: cls[ci++ % cls.length], width: 2.5 } });
                }
            });
            if (traces.length) Plotly.newPlot('cmpChart', traces, { ...layout, xaxis: { ...layout.xaxis, title: 'Episode' }, yaxis: { ...layout.yaxis, title: 'Return' } }, config);
            
            // Mini charts
            Object.entries(domains).forEach(([k, d]) => {
                const sk = k.replace(/[^a-zA-Z0-9]/g, '_');
                const el = document.getElementById('m_' + sk);
                if (el && d.plot_data.rewards.values.length) {
                    Plotly.newPlot('m_' + sk, [{ x: d.plot_data.rewards.times.slice(-100), y: d.plot_data.rewards.values.slice(-100), type: 'scatter', mode: 'lines', line: { color: colors.blue, width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(99,102,241,0.1)' }],
                        { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: { l: 0, r: 0, t: 0, b: 0 }, xaxis: { visible: false }, yaxis: { visible: false }, showlegend: false }, config);
                }
            });
        }
        
        function renderSingle(k) {
            const d = domains[k];
            if (!d) return;
            const s = d.summary, p = d.plot_data;
            
            let h = '<div style="margin-bottom:24px"><h2 style="font-size:24px;font-weight:700">' + s.display_name + '</h2>' +
                '<p style="color:#64748b;margin-top:4px">Domain ' + s.domain_id + ' | Cells discovered: ' + s.cells_discovered + ' | Frontiers: ' + s.frontiers + '</p></div>';
            
            h += '<div class="stats-row">' +
                '<div class="stat"><div class="label">Steps</div><div class="value">' + s.total_steps.toLocaleString() + '</div></div>' +
                '<div class="stat"><div class="label">Episodes</div><div class="value">' + s.episode_count + '</div></div>' +
                '<div class="stat"><div class="label">Goals</div><div class="value green">' + s.goals_reached + '</div></div>' +
                '<div class="stat"><div class="label">Current Return</div><div class="value ' + (s.current_episode_return >= 0 ? 'green' : 'red') + '">' + s.current_episode_return.toFixed(1) + '</div></div>' +
                '<div class="stat"><div class="label">Avg Return</div><div class="value ' + (s.avg_episode_return >= 0 ? 'green' : 'red') + '">' + s.avg_episode_return.toFixed(1) + '</div></div>' +
                '<div class="stat"><div class="label">Intervention</div><div class="value ' + (s.avg_intervention < 0.3 ? 'green' : 'yellow') + '">' + (s.avg_intervention * 100).toFixed(1) + '%</div></div>' +
            '</div>';
            
            h += '<div class="charts-grid">' +
                '<div class="card"><h3>Real-Time Reward</h3><div id="rewChart" class="chart"></div></div>' +
                '<div class="card"><h3>Episode Returns</h3><div id="epChart" class="chart"></div></div>' +
                '<div class="card"><h3>Intervention Rate</h3><div id="intChart" class="chart"></div></div>' +
                '<div class="card"><h3>Safety Zones</h3><div id="zoneChart" class="chart"></div>' +
                '<div class="zones">' + Object.entries(p.safety_zones).filter(z => z[1] > 0).map(z => '<span class="zone ' + z[0] + '">' + z[0] + ': ' + z[1] + '</span>').join('') + '</div></div>' +
            '</div>';
            
            h += '<div class="card"><h3>Reward Components</h3><div id="compChart" class="chart tall"></div></div>';
            
            document.getElementById('content').innerHTML = h;
            
            // Reward chart
            if (p.rewards.values.length) {
                Plotly.newPlot('rewChart', [{ x: p.rewards.times, y: p.rewards.values, type: 'scatter', mode: 'lines', line: { color: colors.blue, width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(99,102,241,0.1)' }],
                    { ...layout, xaxis: { ...layout.xaxis, title: 'Time (s)' }, yaxis: { ...layout.yaxis, title: 'Reward' } }, config);
            }
            
            // Episode chart
            if (p.episode_returns.values.length) {
                const ra = [];
                for (let i = 0; i < p.episode_returns.values.length; i++) {
                    const w = p.episode_returns.values.slice(Math.max(0, i - 9), i + 1);
                    ra.push(w.reduce((a, b) => a + b, 0) / w.length);
                }
                Plotly.newPlot('epChart', [
                    { x: p.episode_returns.episodes, y: p.episode_returns.values, type: 'scatter', mode: 'lines+markers', name: 'Return', line: { color: colors.green, width: 2 }, marker: { size: 5 } },
                    { x: p.episode_returns.episodes, y: ra, type: 'scatter', mode: 'lines', name: 'Rolling Avg', line: { color: colors.yellow, width: 3 } }
                ], { ...layout, xaxis: { ...layout.xaxis, title: 'Episode' }, yaxis: { ...layout.yaxis, title: 'Return' } }, config);
            }
            
            // Intervention chart
            if (p.interventions.values.length) {
                Plotly.newPlot('intChart', [{ x: p.interventions.times, y: p.interventions.values.map(v => v * 100), type: 'scatter', mode: 'lines', line: { color: colors.red, width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(239,68,68,0.1)' }],
                    { ...layout, xaxis: { ...layout.xaxis, title: 'Time (s)' }, yaxis: { ...layout.yaxis, title: '%', range: [0, 100] } }, config);
            }
            
            // Zone chart
            const zc = { FREE: colors.green, AWARE: '#84cc16', CAUTION: colors.yellow, DANGER: colors.orange, EMERGENCY: colors.red };
            const zones = Object.entries(p.safety_zones).filter(z => z[1] > 0);
            if (zones.length) {
                Plotly.newPlot('zoneChart', [{ values: zones.map(z => z[1]), labels: zones.map(z => z[0]), type: 'pie', marker: { colors: zones.map(z => zc[z[0]] || '#94a3b8') }, textinfo: 'label+percent', textfont: { size: 11 }, hole: 0.5 }],
                    { ...layout, showlegend: false }, config);
            }
            
            // Components chart
            const cc = { progress: colors.green, discovery: colors.purple, proximity: colors.red, intervention_penalty: '#dc2626', step: '#94a3b8', novelty: colors.pink, rnd: colors.cyan, goal: colors.yellow, alignment: colors.blue };
            const traces = [];
            Object.entries(p.components).forEach(([n, d]) => {
                if (d.values.length) traces.push({ x: d.times, y: d.values, type: 'scatter', mode: 'lines', name: n, line: { color: cc[n] || '#94a3b8', width: 1.5 } });
            });
            if (traces.length) Plotly.newPlot('compChart', traces, { ...layout, xaxis: { ...layout.xaxis, title: 'Time (s)' }, yaxis: { ...layout.yaxis, title: 'Value' } }, config);
        }
    </script>
</body>
</html>
"""


def create_app(stats: Dict[str, DomainStats]):
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'rl_monitor'
    CORS(app)
    sio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)
    
    def updater():
        while True:
            time.sleep(1.0 / UPDATE_RATE_HZ)
            data = {"domains": {k: {"summary": v.get_summary(), "plot_data": v.get_plot_data()} for k, v in stats.items()}}
            sio.emit('update', data)
    
    return app, sio, updater


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
|  Browser:  http://localhost:{args.port}                           |
|  Scanning: domains {args.scan_start}-{args.scan_end} every {args.scan_interval}s                   |
+--------------------------------------------------------------+
    """)
    
    stats: Dict[str, DomainStats] = {}
    mgr = MultiDomainManager(stats)
    
    print(f"[Scan] Initial scan...")
    mgr.discover_domains(args.scan_start, args.scan_end)
    
    app, sio, updater = create_app(stats)
    
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
        sio.run(app, host='0.0.0.0', port=args.port, debug=False, use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"[Error] {e}")
        shutdown()


if __name__ == "__main__":
    main()