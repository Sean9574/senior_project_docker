#!/usr/bin/env python3
"""
Multi-Domain Real-Time Reward Visualization Server

Collects reward data from MULTIPLE ROS2 domains simultaneously and serves
a real-time dashboard via localhost web interface.

KEY FEATURE: Monitors multiple ROS_DOMAIN_IDs at once!

Each ROS2 domain is isolated, so we spawn a separate rclpy context for each
domain we want to monitor.
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
DISCOVERY_TIMEOUT = 2.0  # Seconds to wait for topics in each domain

# Default scan range - covers typical simulation setups
DEFAULT_SCAN_START = 0
DEFAULT_SCAN_END = 30

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
        
        # Check for episode end
        if breakdown.get("collision", False) or breakdown.get("success", False):
            self.end_episode()
    
    def end_episode(self):
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
        
        episode_returns = list(self.episode_returns)
        episode_indices = list(range(len(episode_returns)))
        
        components_data = {}
        for key, values in self.components.items():
            vals = list(values)[::step]
            t0_comp = vals[0][0] if vals else (t0 if timestamps else 0)
            components_data[key] = {
                "times": [(v[0] - t0_comp) for v in vals],
                "values": [v[1] for v in vals]
            }
        
        zone_counts = {"FREE": 0, "AWARE": 0, "CAUTION": 0, "DANGER": 0, "EMERGENCY": 0, "CRITICAL": 0}
        for _, zone in list(self.safety_zones)[-500:]:
            if zone in zone_counts:
                zone_counts[zone] += 1
        
        interventions = list(self.intervention_rates)[::step]
        t0_int = interventions[0][0] if interventions else (t0 if timestamps else 0)
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
    """
    Collects reward data from a single ROS2 domain.
    Each domain gets its own rclpy context and executor.
    """
    
    def __init__(self, domain_id: int, stats_store: Dict[str, DomainStats], 
                 namespaces: List[str] = None):
        self.domain_id = domain_id
        self.stats = stats_store
        self.namespaces = namespaces or [""]
        self.running = False
        
        # Create isolated ROS2 context for this domain
        self.context = Context()
        rclpy.init(context=self.context, domain_id=domain_id)
        
        # Create node in this context
        self.node = rclpy.create_node(
            f"reward_viz_domain_{domain_id}",
            context=self.context,
            automatically_declare_parameters_from_overrides=True
        )
        
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        
        self._subscriptions = []
        self._setup_subscriptions()
        
        self.node.get_logger().info(f"[Domain {domain_id}] Collector initialized")
    
    def _setup_subscriptions(self):
        """Subscribe to reward topics for all namespaces."""
        for ns in self.namespaces:
            self._subscribe_namespace(ns)
    
    def _subscribe_namespace(self, namespace: str):
        """Subscribe to reward topics for a specific namespace."""
        # Build topic names
        if namespace:
            reward_topic = f"/{namespace}/reward"
            breakdown_topic = f"/{namespace}/reward_breakdown"
            display_name = f"D{self.domain_id}:{namespace}"
        else:
            reward_topic = "/reward"
            breakdown_topic = "/reward_breakdown"
            display_name = f"Domain {self.domain_id}"
        
        # Create unique key for this domain+namespace combo
        stats_key = f"{self.domain_id}:{namespace}"
        
        if stats_key not in self.stats:
            self.stats[stats_key] = DomainStats(
                domain_id=self.domain_id,
                namespace=namespace,
                display_name=display_name
            )
        
        # QoS profile
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Subscribe to reward
        reward_sub = self.node.create_subscription(
            Float32, reward_topic,
            lambda msg, k=stats_key: self._reward_callback(msg, k),
            qos
        )
        self._subscriptions.append(reward_sub)
        
        # Subscribe to breakdown
        breakdown_sub = self.node.create_subscription(
            StringMsg, breakdown_topic,
            lambda msg, k=stats_key: self._breakdown_callback(msg, k),
            qos
        )
        self._subscriptions.append(breakdown_sub)
        
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
        """Discover namespaces with reward topics in this domain."""
        discovered = []
        
        topic_names_and_types = self.node.get_topic_names_and_types()
        
        for topic_name, _ in topic_names_and_types:
            if topic_name.endswith("/reward"):
                # Extract namespace
                parts = topic_name.rsplit("/reward", 1)[0]
                if parts:
                    ns = parts.lstrip("/")
                else:
                    ns = ""
                
                if ns not in self.namespaces:
                    discovered.append(ns)
                    self._subscribe_namespace(ns)
                    self.namespaces.append(ns)
        
        return discovered
    
    def spin(self):
        """Spin the executor (call from thread)."""
        self.running = True
        while self.running:
            try:
                self.executor.spin_once(timeout_sec=0.1)
            except Exception as e:
                self.node.get_logger().warn(f"[Domain {self.domain_id}] Spin error: {e}")
    
    def stop(self):
        """Stop the collector."""
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
        """Add a domain to monitor."""
        with self.lock:
            if domain_id in self.collectors:
                return False
            
            try:
                collector = DomainCollector(domain_id, self.stats, namespaces)
                self.collectors[domain_id] = collector
                
                # Start spin thread
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
    
    def discover_domains(self, scan_start: int = DEFAULT_SCAN_START, 
                         scan_end: int = DEFAULT_SCAN_END) -> List[int]:
        """Scan for active domains with reward topics."""
        discovered = []
        
        print(f"[Manager] Scanning domains {scan_start}-{scan_end}...")
        
        def check_domain(domain_id: int) -> Optional[int]:
            """Check if a domain has reward topics."""
            # Skip if already monitoring
            if domain_id in self.collectors:
                return None
                
            try:
                ctx = Context()
                rclpy.init(context=ctx, domain_id=domain_id)
                node = rclpy.create_node(f"scanner_{domain_id}", context=ctx)
                
                # Wait a bit for discovery
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
        
        # Scan in parallel
        scan_range = range(scan_start, scan_end + 1)
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(check_domain, scan_range))
        
        for domain_id in results:
            if domain_id is not None and domain_id not in self.collectors:
                if self.add_domain(domain_id):
                    discovered.append(domain_id)
        
        print(f"[Manager] Discovered {len(discovered)} domains: {discovered}")
        return discovered
    
    def discover_namespaces_all(self):
        """Discover new namespaces in all monitored domains."""
        for domain_id, collector in self.collectors.items():
            try:
                new_ns = collector.discover_namespaces()
                if new_ns:
                    print(f"[Manager] Domain {domain_id} new namespaces: {new_ns}")
            except Exception as e:
                print(f"[Manager] Discovery error in domain {domain_id}: {e}")
    
    def stop_all(self):
        """Stop all collectors."""
        for collector in self.collectors.values():
            collector.stop()


# =============================================================================
# Flask Web Server (same HTML as before, but with domain ID display)
# =============================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Reward Visualizer - Multi-Domain</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419; color: #e7e9ea; min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1f2e 0%, #2d1f3d 100%);
            padding: 20px 30px; border-bottom: 1px solid #2f3336;
            display: flex; justify-content: space-between; align-items: center;
        }
        .header h1 {
            font-size: 24px; font-weight: 600;
            background: linear-gradient(90deg, #7c3aed, #2dd4bf);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .header-info { display: flex; align-items: center; gap: 20px; }
        .domain-badge {
            background: #7c3aed; padding: 5px 12px; border-radius: 15px;
            font-size: 12px; font-weight: 600;
        }
        .connection-status { display: flex; align-items: center; gap: 8px; font-size: 14px; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #ef4444; }
        .status-dot.connected { background: #22c55e; box-shadow: 0 0 10px #22c55e; }
        .container { padding: 20px; max-width: 1800px; margin: 0 auto; }
        .domain-tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .domain-tab {
            padding: 10px 20px; background: #1e2732; border: 1px solid #2f3336;
            border-radius: 8px; cursor: pointer; transition: all 0.2s;
            display: flex; align-items: center; gap: 8px;
        }
        .domain-tab:hover { background: #2a3544; }
        .domain-tab.active { background: #7c3aed; border-color: #7c3aed; }
        .domain-tab .indicator { width: 8px; height: 8px; border-radius: 50%; background: #6b7280; }
        .domain-tab.active-domain .indicator { background: #22c55e; animation: pulse 2s infinite; }
        .domain-tab .domain-id { font-size: 10px; background: rgba(0,0,0,0.3); padding: 2px 6px; border-radius: 4px; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }
        .stat-card { background: #1e2732; border: 1px solid #2f3336; border-radius: 12px; padding: 20px; }
        .stat-card .label { font-size: 12px; color: #71767b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
        .stat-card .value { font-size: 28px; font-weight: 700; }
        .stat-card .value.positive { color: #22c55e; }
        .stat-card .value.negative { color: #ef4444; }
        .stat-card .value.neutral { color: #f59e0b; }
        .charts-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 20px; }
        @media (max-width: 1200px) { .charts-grid { grid-template-columns: 1fr; } }
        .chart-container { background: #1e2732; border: 1px solid #2f3336; border-radius: 12px; padding: 20px; }
        .chart-container h3 { font-size: 16px; margin-bottom: 15px; color: #e7e9ea; }
        .chart { height: 300px; }
        .chart.tall { height: 400px; }
        .safety-zones { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
        .zone-badge { padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .zone-badge.FREE { background: #166534; color: #86efac; }
        .zone-badge.AWARE { background: #365314; color: #bef264; }
        .zone-badge.CAUTION { background: #713f12; color: #fde047; }
        .zone-badge.DANGER { background: #7c2d12; color: #fdba74; }
        .zone-badge.EMERGENCY { background: #7f1d1d; color: #fca5a5; }
        .zone-badge.CRITICAL { background: #450a0a; color: #fecaca; }
        .all-domains-view { margin-top: 20px; }
        .domain-row {
            background: #1e2732; border: 1px solid #2f3336; border-radius: 12px;
            padding: 15px 20px; margin-bottom: 10px;
            display: grid; grid-template-columns: 200px 1fr repeat(4, 100px);
            align-items: center; gap: 20px; cursor: pointer;
        }
        .domain-row:hover { background: #2a3544; }
        .domain-row .domain-name { font-weight: 600; display: flex; align-items: center; gap: 10px; flex-direction: column; align-items: flex-start; }
        .domain-row .domain-name .ns { font-size: 11px; color: #71767b; }
        .domain-row .mini-chart { height: 50px; }
        .domain-row .stat { text-align: center; }
        .domain-row .stat .label { font-size: 10px; color: #71767b; }
        .domain-row .stat .value { font-size: 18px; font-weight: 600; }
        .no-data { text-align: center; padding: 60px; color: #71767b; }
        .no-data h2 { margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RL Reward Visualizer - Multi-Domain</h1>
        <div class="header-info">
            <span class="domain-badge" id="domainCount">0 Domains</span>
            <div class="connection-status">
                <div class="status-dot" id="statusDot"></div>
                <span id="statusText">Connecting...</span>
            </div>
        </div>
    </div>
    <div class="container">
        <div class="domain-tabs" id="domainTabs">
            <div class="domain-tab active" onclick="selectDomain('all')"><span>All Domains</span></div>
        </div>
        <div id="content">
            <div class="no-data">
                <h2>Waiting for data...</h2>
                <p>Scanning ROS2 domains for reward topics...</p>
            </div>
        </div>
    </div>
    <script>
        const socket = io();
        let domains = {};
        let selectedDomain = 'all';
        
        const plotlyLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e7e9ea', size: 12 },
            margin: { l: 50, r: 20, t: 30, b: 40 },
            xaxis: { gridcolor: '#2f3336', zerolinecolor: '#2f3336' },
            yaxis: { gridcolor: '#2f3336', zerolinecolor: '#2f3336' },
            showlegend: true, legend: { bgcolor: 'rgba(0,0,0,0)', font: { color: '#e7e9ea' } }
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
        socket.on('update', (data) => { domains = data.domains; updateUI(); });
        
        function updateUI() {
            const count = Object.keys(domains).length;
            document.getElementById('domainCount').textContent = `${count} Domain${count !== 1 ? 's' : ''}`;
            updateTabs();
            updateContent();
        }
        
        function updateTabs() {
            const tabsContainer = document.getElementById('domainTabs');
            const domainKeys = Object.keys(domains);
            
            let html = `<div class="domain-tab ${selectedDomain === 'all' ? 'active' : ''}" onclick="selectDomain('all')"><span>All (${domainKeys.length})</span></div>`;
            
            domainKeys.forEach(key => {
                const d = domains[key];
                const stats = d.summary;
                html += `<div class="domain-tab ${selectedDomain === key ? 'active' : ''} ${stats.active ? 'active-domain' : ''}" onclick="selectDomain('${key}')">
                    <div class="indicator"></div>
                    <span>${stats.display_name}</span>
                    <span class="domain-id">D${stats.domain_id}</span>
                </div>`;
            });
            
            tabsContainer.innerHTML = html;
        }
        
        function selectDomain(key) { selectedDomain = key; updateTabs(); updateContent(); }
        
        function updateContent() {
            const container = document.getElementById('content');
            if (Object.keys(domains).length === 0) {
                container.innerHTML = '<div class="no-data"><h2>Waiting for data...</h2><p>Scanning ROS2 domains for reward topics...</p></div>';
                return;
            }
            if (selectedDomain === 'all') renderAllDomainsView(container);
            else renderSingleDomainView(container, selectedDomain);
        }
        
        function renderAllDomainsView(container) {
            const vals = Object.values(domains);
            const totalSteps = vals.reduce((s, d) => s + d.summary.total_steps, 0);
            const totalEpisodes = vals.reduce((s, d) => s + d.summary.episode_count, 0);
            const avgReturn = vals.reduce((s, d) => s + d.summary.avg_episode_return, 0) / Math.max(vals.length, 1);
            const activeCount = vals.filter(d => d.summary.active).length;
            
            let html = `<div class="all-domains-view">
                <div class="stats-grid">
                    <div class="stat-card"><div class="label">Total Steps</div><div class="value">${totalSteps.toLocaleString()}</div></div>
                    <div class="stat-card"><div class="label">Total Episodes</div><div class="value">${totalEpisodes}</div></div>
                    <div class="stat-card"><div class="label">Avg Episode Return</div><div class="value ${avgReturn >= 0 ? 'positive' : 'negative'}">${avgReturn.toFixed(1)}</div></div>
                    <div class="stat-card"><div class="label">Active / Total</div><div class="value positive">${activeCount} / ${vals.length}</div></div>
                </div>
                <div class="chart-container"><h3>Episode Returns Comparison</h3><div id="comparisonChart" class="chart tall"></div></div>`;
            
            Object.entries(domains).forEach(([key, data]) => {
                const s = data.summary;
                html += `<div class="domain-row" onclick="selectDomain('${key}')">
                    <div class="domain-name">
                        <div><div class="indicator" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${s.active ? '#22c55e' : '#6b7280'};margin-right:8px"></div>${s.display_name}</div>
                        <div class="ns">Domain ${s.domain_id} | ${s.namespace || '/'}</div>
                    </div>
                    <div class="mini-chart" id="mini_${key.replace(/[^a-zA-Z0-9]/g, '_')}"></div>
                    <div class="stat"><div class="label">Steps</div><div class="value">${s.total_steps.toLocaleString()}</div></div>
                    <div class="stat"><div class="label">Episodes</div><div class="value">${s.episode_count}</div></div>
                    <div class="stat"><div class="label">Avg Return</div><div class="value" style="color:${s.avg_episode_return >= 0 ? '#22c55e' : '#ef4444'}">${s.avg_episode_return.toFixed(1)}</div></div>
                    <div class="stat"><div class="label">Intervention</div><div class="value" style="color:${s.avg_intervention < 0.2 ? '#22c55e' : '#f59e0b'}">${(s.avg_intervention * 100).toFixed(0)}%</div></div>
                </div>`;
            });
            html += '</div>';
            container.innerHTML = html;
            
            // Comparison chart
            const traces = [], colors = ['#7c3aed', '#2dd4bf', '#f59e0b', '#ef4444', '#22c55e', '#3b82f6', '#ec4899', '#8b5cf6'];
            let ci = 0;
            Object.entries(domains).forEach(([key, data]) => {
                const pd = data.plot_data;
                if (pd.episode_returns.values.length > 0) {
                    traces.push({ x: pd.episode_returns.episodes, y: pd.episode_returns.values, type: 'scatter', mode: 'lines', name: pd.display_name, line: { color: colors[ci++ % colors.length], width: 2 } });
                }
            });
            if (traces.length > 0) {
                Plotly.newPlot('comparisonChart', traces, { ...plotlyLayout, xaxis: { ...plotlyLayout.xaxis, title: 'Episode' }, yaxis: { ...plotlyLayout.yaxis, title: 'Return' } }, plotlyConfig);
            }
            
            // Mini charts
            Object.entries(domains).forEach(([key, data]) => {
                const id = `mini_${key.replace(/[^a-zA-Z0-9]/g, '_')}`;
                const el = document.getElementById(id);
                if (el && data.plot_data.rewards.values.length > 0) {
                    Plotly.newPlot(id, [{ x: data.plot_data.rewards.times.slice(-100), y: data.plot_data.rewards.values.slice(-100), type: 'scatter', mode: 'lines', line: { color: '#7c3aed', width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(124, 58, 237, 0.2)' }], { paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', margin: { l: 0, r: 0, t: 0, b: 0 }, xaxis: { visible: false }, yaxis: { visible: false }, showlegend: false }, plotlyConfig);
                }
            });
        }
        
        function renderSingleDomainView(container, key) {
            const data = domains[key];
            if (!data) return;
            const s = data.summary, pd = data.plot_data;
            
            let html = `
                <h2 style="margin-bottom:20px">${s.display_name} <span style="color:#71767b;font-size:14px">Domain ${s.domain_id} | ${s.namespace || '/'}</span></h2>
                <div class="stats-grid">
                    <div class="stat-card"><div class="label">Total Steps</div><div class="value">${s.total_steps.toLocaleString()}</div></div>
                    <div class="stat-card"><div class="label">Episodes</div><div class="value">${s.episode_count}</div></div>
                    <div class="stat-card"><div class="label">Current Return</div><div class="value ${s.current_episode_return >= 0 ? 'positive' : 'negative'}">${s.current_episode_return.toFixed(1)}</div></div>
                    <div class="stat-card"><div class="label">Avg Return</div><div class="value ${s.avg_episode_return >= 0 ? 'positive' : 'negative'}">${s.avg_episode_return.toFixed(1)}</div></div>
                    <div class="stat-card"><div class="label">Avg Reward (100)</div><div class="value ${s.avg_reward_100 >= 0 ? 'positive' : 'negative'}">${s.avg_reward_100.toFixed(3)}</div></div>
                    <div class="stat-card"><div class="label">Intervention</div><div class="value ${s.avg_intervention < 0.2 ? 'positive' : 'neutral'}">${(s.avg_intervention * 100).toFixed(1)}%</div></div>
                </div>
                <div class="charts-grid">
                    <div class="chart-container"><h3>Real-Time Reward</h3><div id="rewardChart" class="chart"></div></div>
                    <div class="chart-container"><h3>Episode Returns</h3><div id="episodeChart" class="chart"></div></div>
                    <div class="chart-container"><h3>Intervention Rate</h3><div id="interventionChart" class="chart"></div></div>
                    <div class="chart-container"><h3>Safety Zones</h3><div id="zoneChart" class="chart"></div>
                        <div class="safety-zones">${Object.entries(pd.safety_zones).map(([z,c]) => `<span class="zone-badge ${z}">${z}: ${c}</span>`).join('')}</div>
                    </div>
                </div>
                <div class="chart-container"><h3>Reward Components</h3><div id="componentsChart" class="chart tall"></div></div>`;
            container.innerHTML = html;
            
            // Charts
            if (pd.rewards.values.length > 0) {
                Plotly.newPlot('rewardChart', [{ x: pd.rewards.times, y: pd.rewards.values, type: 'scatter', mode: 'lines', line: { color: '#7c3aed', width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(124, 58, 237, 0.2)' }], { ...plotlyLayout, xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' }, yaxis: { ...plotlyLayout.yaxis, title: 'Reward' } }, plotlyConfig);
            }
            
            if (pd.episode_returns.values.length > 0) {
                const ws = 10, ra = [];
                for (let i = 0; i < pd.episode_returns.values.length; i++) {
                    const w = pd.episode_returns.values.slice(Math.max(0, i - ws + 1), i + 1);
                    ra.push(w.reduce((a, b) => a + b, 0) / w.length);
                }
                Plotly.newPlot('episodeChart', [
                    { x: pd.episode_returns.episodes, y: pd.episode_returns.values, type: 'scatter', mode: 'lines+markers', name: 'Return', line: { color: '#2dd4bf', width: 2 }, marker: { size: 6 } },
                    { x: pd.episode_returns.episodes, y: ra, type: 'scatter', mode: 'lines', name: 'Avg', line: { color: '#f59e0b', width: 3 } }
                ], { ...plotlyLayout, xaxis: { ...plotlyLayout.xaxis, title: 'Episode' }, yaxis: { ...plotlyLayout.yaxis, title: 'Return' } }, plotlyConfig);
            }
            
            if (pd.interventions.values.length > 0) {
                Plotly.newPlot('interventionChart', [{ x: pd.interventions.times, y: pd.interventions.values.map(v => v * 100), type: 'scatter', mode: 'lines', line: { color: '#ef4444', width: 1.5 }, fill: 'tozeroy', fillcolor: 'rgba(239, 68, 68, 0.2)' }], { ...plotlyLayout, xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' }, yaxis: { ...plotlyLayout.yaxis, title: '%', range: [0, 100] } }, plotlyConfig);
            }
            
            const zc = { 'FREE': '#22c55e', 'AWARE': '#84cc16', 'CAUTION': '#eab308', 'DANGER': '#f97316', 'EMERGENCY': '#ef4444', 'CRITICAL': '#b91c1c' };
            const zones = Object.entries(pd.safety_zones).filter(z => z[1] > 0);
            if (zones.length > 0) {
                Plotly.newPlot('zoneChart', [{ values: zones.map(z => z[1]), labels: zones.map(z => z[0]), type: 'pie', marker: { colors: zones.map(z => zc[z[0]] || '#6b7280') }, textinfo: 'label+percent', textfont: { color: '#fff' }, hole: 0.4 }], { ...plotlyLayout, showlegend: false }, plotlyConfig);
            }
            
            const cc = { 'progress': '#22c55e', 'discovery': '#8b5cf6', 'proximity': '#ef4444', 'intervention_penalty': '#dc2626', 'step': '#6b7280', 'novelty': '#ec4899', 'rnd': '#06b6d4' };
            const compTraces = [];
            Object.entries(pd.components).forEach(([n, d]) => {
                if (d.values.length > 0) compTraces.push({ x: d.times, y: d.values, type: 'scatter', mode: 'lines', name: n, line: { color: cc[n] || '#6b7280', width: 1.5 } });
            });
            if (compTraces.length > 0) {
                Plotly.newPlot('componentsChart', compTraces, { ...plotlyLayout, xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' }, yaxis: { ...plotlyLayout.yaxis, title: 'Value' } }, plotlyConfig);
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
                        help="Specific ROS2 domain IDs to monitor (e.g., --domains 10 11 12)")
    parser.add_argument("--domain-range", nargs=2, type=int, metavar=("START", "END"),
                        help="Range of domain IDs to monitor (e.g., --domain-range 10 15)")
    parser.add_argument("--namespaces", nargs="*", default=["", "stretch"],
                        help="Namespaces to monitor in each domain")
    parser.add_argument("--scan", action="store_true",
                        help="Continuously scan for new domains")
    parser.add_argument("--scan-range", nargs=2, type=int, metavar=("START", "END"),
                        default=[0, 30],
                        help="Range to scan for domains (default: 0 30)")
    parser.add_argument("--scan-interval", type=float, default=10.0,
                        help="Seconds between scans (default: 10)")
    args = parser.parse_args()
    
    # Determine which domains to monitor initially
    if args.domain_range:
        domain_ids = list(range(args.domain_range[0], args.domain_range[1] + 1))
    elif args.domains:
        domain_ids = args.domains
    else:
        domain_ids = []  # Will scan to find domains
    
    scan_start, scan_end = args.scan_range
    
    print(f"""
+==============================================================+
|        RL Reward Visualizer - MULTI-DOMAIN                   |
+==============================================================+
|  Open in browser:  http://localhost:{args.port:<5}                 |
|  Initial domains: {str(domain_ids) if domain_ids else '(scanning...)':<40}|
|  Namespaces: {str(args.namespaces):<47}|
|  Continuous scan: {str(args.scan):<40}|
|  Scan range: {scan_start}-{scan_end:<46}|
+==============================================================+
    """)
    
    # Shared stats store
    stats_store: Dict[str, DomainStats] = {}
    
    # Create domain manager
    manager = MultiDomainManager(stats_store)
    
    # Add initial domains if specified
    for domain_id in domain_ids:
        manager.add_domain(domain_id, args.namespaces.copy())
    
    # If no initial domains or scanning enabled, do initial scan
    if not domain_ids or args.scan:
        print(f"[Manager] Initial scan of domains {scan_start}-{scan_end}...")
        manager.discover_domains(scan_start, scan_end)
    
    # Create Flask app
    app, socketio, background_update = create_app(stats_store)
    
    # Start background update thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    # Continuous scanning thread
    if args.scan:
        def scan_loop():
            while True:
                time.sleep(args.scan_interval)
                try:
                    new_domains = manager.discover_domains(scan_start, scan_end)
                    if new_domains:
                        print(f"[Manager] Found new domains: {new_domains}")
                    manager.discover_namespaces_all()
                except Exception as e:
                    print(f"[Scan] Error: {e}")
        
        scan_thread = threading.Thread(target=scan_loop, daemon=True)
        scan_thread.start()
    
    # Shutdown handler
    def shutdown(signum=None, frame=None):
        print("\n[VIZ] Shutting down...")
        manager.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Run Flask server
    try:
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False,
                     use_reloader=False, log_output=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"[VIZ] Server error: {e}")
        shutdown()


if __name__ == "__main__":
    main()