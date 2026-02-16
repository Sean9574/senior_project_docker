#!/usr/bin/env python3
"""
Real-Time Reward Visualization Server

Collects reward data from multiple robot domains via ROS topics and serves
a real-time dashboard via localhost web interface.

Features:
- Auto-discovers robot domains or accepts manual configuration
- Real-time WebSocket updates to browser
- Tracks reward components breakdown
- Shows safety intervention rates
- Supports multiple concurrent domains

Usage:
    python3 reward_visualizer.py --domains stretch1 stretch2 stretch3
    python3 reward_visualizer.py --auto-discover
    
Then open http://localhost:5555 in your browser
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
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Set

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, String as StringMsg

from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_PORT = 5555
MAX_HISTORY_LENGTH = 1000  # Points to keep per domain
UPDATE_RATE_HZ = 10  # How often to push updates to browser
DISCOVERY_INTERVAL = 5.0  # Seconds between auto-discovery scans

# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class DomainStats:
    """Statistics for a single robot domain."""
    domain: str
    rewards: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    
    # Reward breakdown components
    components: Dict[str, deque] = field(default_factory=dict)
    
    # Episode tracking
    episode_returns: deque = field(default_factory=lambda: deque(maxlen=100))
    episode_lengths: deque = field(default_factory=lambda: deque(maxlen=100))
    current_episode_return: float = 0.0
    current_episode_steps: int = 0
    episode_count: int = 0
    
    # Safety stats
    safety_zones: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    intervention_rates: deque = field(default_factory=lambda: deque(maxlen=MAX_HISTORY_LENGTH))
    
    # Running stats
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
        """Add reward breakdown data."""
        timestamp = time.time()
        
        # Extract reward terms
        reward_terms = breakdown.get("reward_terms", {})
        for key, value in reward_terms.items():
            if key not in self.components:
                self.components[key] = deque(maxlen=MAX_HISTORY_LENGTH)
            self.components[key].append((timestamp, value))
        
        # Extract safety info
        safety = breakdown.get("safety", {})
        zone = safety.get("zone", "UNKNOWN")
        intervention = safety.get("intervention", 0.0)
        
        self.safety_zones.append((timestamp, zone))
        self.intervention_rates.append((timestamp, intervention))
        
        # Check for episode end
        if breakdown.get("collision", False) or breakdown.get("success", False):
            self.end_episode()
    
    def end_episode(self):
        """Record episode completion."""
        self.episode_returns.append(self.current_episode_return)
        self.episode_lengths.append(self.current_episode_steps)
        self.episode_count += 1
        self.current_episode_return = 0.0
        self.current_episode_steps = 0
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        recent_rewards = list(self.rewards)[-100:] if self.rewards else [0]
        recent_interventions = [x[1] for x in list(self.intervention_rates)[-100:]] if self.intervention_rates else [0]
        
        return {
            "domain": self.domain,
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
        """Get data formatted for plotting."""
        # Downsample if needed
        step = max(1, len(self.rewards) // max_points)
        
        rewards = list(self.rewards)[::step]
        timestamps = list(self.timestamps)[::step]
        
        # Convert timestamps to relative seconds
        if timestamps:
            t0 = timestamps[0]
            rel_times = [(t - t0) for t in timestamps]
        else:
            rel_times = []
        
        # Episode returns
        episode_returns = list(self.episode_returns)
        episode_indices = list(range(len(episode_returns)))
        
        # Component data
        components_data = {}
        for key, values in self.components.items():
            vals = list(values)[::step]
            components_data[key] = {
                "times": [(v[0] - t0) if timestamps else 0 for v in vals],
                "values": [v[1] for v in vals]
            }
        
        # Safety zone distribution
        zone_counts = {"FREE": 0, "AWARE": 0, "CAUTION": 0, "DANGER": 0, "EMERGENCY": 0, "CRITICAL": 0}
        for _, zone in list(self.safety_zones)[-500:]:
            if zone in zone_counts:
                zone_counts[zone] += 1
        
        # Intervention over time
        interventions = list(self.intervention_rates)[::step]
        intervention_times = [(v[0] - t0) if timestamps else 0 for v in interventions]
        intervention_values = [v[1] for v in interventions]
        
        return {
            "domain": self.domain,
            "rewards": {
                "times": rel_times,
                "values": rewards
            },
            "episode_returns": {
                "episodes": episode_indices,
                "values": episode_returns
            },
            "components": components_data,
            "safety_zones": zone_counts,
            "interventions": {
                "times": intervention_times,
                "values": intervention_values
            }
        }


# =============================================================================
# ROS Subscriber Node
# =============================================================================

class RewardCollectorNode(Node):
    """ROS node that subscribes to reward topics from multiple domains."""
    
    def __init__(self, domains: List[str], stats_store: Dict[str, DomainStats]):
        super().__init__("reward_visualizer")
        
        self.stats = stats_store
        self.subscriptions = {}
        self.discovered_domains: Set[str] = set()
        
        # Subscribe to specified domains
        for domain in domains:
            self._subscribe_domain(domain)
        
        self.get_logger().info(f"[VIZ] Collecting rewards from {len(domains)} domains")
    
    def _subscribe_domain(self, domain: str):
        """Subscribe to reward topics for a domain."""
        if domain in self.subscriptions:
            return
        
        # Initialize stats for this domain
        if domain not in self.stats:
            self.stats[domain] = DomainStats(domain=domain)
        
        # Determine topic names
        if domain:
            reward_topic = f"/{domain}/reward"
            breakdown_topic = f"/{domain}/reward_breakdown"
        else:
            reward_topic = "/reward"
            breakdown_topic = "/reward_breakdown"
        
        # Create subscriptions
        reward_sub = self.create_subscription(
            Float32, reward_topic,
            lambda msg, d=domain: self._reward_callback(msg, d),
            10
        )
        
        breakdown_sub = self.create_subscription(
            StringMsg, breakdown_topic,
            lambda msg, d=domain: self._breakdown_callback(msg, d),
            10
        )
        
        self.subscriptions[domain] = {
            "reward": reward_sub,
            "breakdown": breakdown_sub
        }
        
        self.discovered_domains.add(domain)
        self.get_logger().info(f"[VIZ] Subscribed to domain: {domain or 'default'}")
    
    def _reward_callback(self, msg: Float32, domain: str):
        """Handle incoming reward message."""
        if domain in self.stats:
            self.stats[domain].add_reward(msg.data, time.time())
    
    def _breakdown_callback(self, msg: StringMsg, domain: str):
        """Handle incoming reward breakdown message."""
        try:
            breakdown = json.loads(msg.data)
            if domain in self.stats:
                self.stats[domain].add_breakdown(breakdown)
        except json.JSONDecodeError:
            pass
    
    def discover_domains(self) -> List[str]:
        """Discover available domains by scanning topics."""
        # Get all topics
        topic_names_and_types = self.get_topic_names_and_types()
        
        new_domains = []
        for topic_name, _ in topic_names_and_types:
            if topic_name.endswith("/reward"):
                # Extract domain name
                parts = topic_name.split("/")
                if len(parts) >= 3:
                    domain = parts[1]
                    if domain not in self.discovered_domains:
                        new_domains.append(domain)
                        self._subscribe_domain(domain)
                elif topic_name == "/reward":
                    if "" not in self.discovered_domains:
                        new_domains.append("")
                        self._subscribe_domain("")
        
        return new_domains


# =============================================================================
# Flask Web Server
# =============================================================================

# HTML Template for the dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RL Reward Visualizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0f1419;
            color: #e7e9ea;
            min-height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1f2e 0%, #2d1f3d 100%);
            padding: 20px 30px;
            border-bottom: 1px solid #2f3336;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .header h1 {
            font-size: 24px;
            font-weight: 600;
            background: linear-gradient(90deg, #7c3aed, #2dd4bf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ef4444;
        }
        
        .status-dot.connected {
            background: #22c55e;
            box-shadow: 0 0 10px #22c55e;
        }
        
        .container {
            padding: 20px;
            max-width: 1800px;
            margin: 0 auto;
        }
        
        .domain-tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .domain-tab {
            padding: 10px 20px;
            background: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .domain-tab:hover {
            background: #2a3544;
        }
        
        .domain-tab.active {
            background: #7c3aed;
            border-color: #7c3aed;
        }
        
        .domain-tab .indicator {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #6b7280;
        }
        
        .domain-tab.active-domain .indicator {
            background: #22c55e;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .stat-card {
            background: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 12px;
            padding: 20px;
        }
        
        .stat-card .label {
            font-size: 12px;
            color: #71767b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }
        
        .stat-card .value {
            font-size: 28px;
            font-weight: 700;
        }
        
        .stat-card .value.positive { color: #22c55e; }
        .stat-card .value.negative { color: #ef4444; }
        .stat-card .value.neutral { color: #f59e0b; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }
        
        @media (max-width: 1200px) {
            .charts-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .chart-container {
            background: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 12px;
            padding: 20px;
        }
        
        .chart-container h3 {
            font-size: 16px;
            margin-bottom: 15px;
            color: #e7e9ea;
        }
        
        .chart {
            height: 300px;
        }
        
        .chart.tall {
            height: 400px;
        }
        
        .safety-zones {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 10px;
        }
        
        .zone-badge {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .zone-badge.FREE { background: #166534; color: #86efac; }
        .zone-badge.AWARE { background: #365314; color: #bef264; }
        .zone-badge.CAUTION { background: #713f12; color: #fde047; }
        .zone-badge.DANGER { background: #7c2d12; color: #fdba74; }
        .zone-badge.EMERGENCY { background: #7f1d1d; color: #fca5a5; }
        .zone-badge.CRITICAL { background: #450a0a; color: #fecaca; }
        
        .all-domains-view {
            margin-top: 20px;
        }
        
        .domain-row {
            background: #1e2732;
            border: 1px solid #2f3336;
            border-radius: 12px;
            padding: 15px 20px;
            margin-bottom: 10px;
            display: grid;
            grid-template-columns: 150px 1fr repeat(4, 100px);
            align-items: center;
            gap: 20px;
        }
        
        .domain-row .domain-name {
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .domain-row .mini-chart {
            height: 50px;
        }
        
        .domain-row .stat {
            text-align: center;
        }
        
        .domain-row .stat .label {
            font-size: 10px;
            color: #71767b;
        }
        
        .domain-row .stat .value {
            font-size: 18px;
            font-weight: 600;
        }
        
        .no-data {
            text-align: center;
            padding: 60px;
            color: #71767b;
        }
        
        .no-data h2 {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 RL Reward Visualizer</h1>
        <div class="connection-status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Connecting...</span>
        </div>
    </div>
    
    <div class="container">
        <div class="domain-tabs" id="domainTabs">
            <div class="domain-tab active" data-domain="all">
                <span>📊 All Domains</span>
            </div>
        </div>
        
        <div id="content">
            <div class="no-data">
                <h2>Waiting for data...</h2>
                <p>Make sure your RL training is running and publishing to /reward topics</p>
            </div>
        </div>
    </div>
    
    <script>
        // Connect to WebSocket
        const socket = io();
        
        let domains = {};
        let selectedDomain = 'all';
        let charts = {};
        
        // Plotly dark theme
        const plotlyLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#e7e9ea', size: 12 },
            margin: { l: 50, r: 20, t: 30, b: 40 },
            xaxis: {
                gridcolor: '#2f3336',
                zerolinecolor: '#2f3336',
            },
            yaxis: {
                gridcolor: '#2f3336',
                zerolinecolor: '#2f3336',
            },
            showlegend: true,
            legend: {
                bgcolor: 'rgba(0,0,0,0)',
                font: { color: '#e7e9ea' }
            }
        };
        
        const plotlyConfig = {
            displayModeBar: false,
            responsive: true
        };
        
        // Connection status
        socket.on('connect', () => {
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Connected';
        });
        
        socket.on('disconnect', () => {
            document.getElementById('statusDot').classList.remove('connected');
            document.getElementById('statusText').textContent = 'Disconnected';
        });
        
        // Handle data updates
        socket.on('update', (data) => {
            domains = data.domains;
            updateTabs();
            updateContent();
        });
        
        function updateTabs() {
            const tabsContainer = document.getElementById('domainTabs');
            const domainNames = Object.keys(domains);
            
            // Keep "All Domains" tab, update domain tabs
            let html = `
                <div class="domain-tab ${selectedDomain === 'all' ? 'active' : ''}" 
                     data-domain="all" onclick="selectDomain('all')">
                    <span>📊 All Domains (${domainNames.length})</span>
                </div>
            `;
            
            domainNames.forEach(domain => {
                const stats = domains[domain].summary;
                const isActive = stats.active;
                const displayName = domain || 'default';
                
                html += `
                    <div class="domain-tab ${selectedDomain === domain ? 'active' : ''} ${isActive ? 'active-domain' : ''}"
                         data-domain="${domain}" onclick="selectDomain('${domain}')">
                        <div class="indicator"></div>
                        <span>${displayName}</span>
                    </div>
                `;
            });
            
            tabsContainer.innerHTML = html;
        }
        
        function selectDomain(domain) {
            selectedDomain = domain;
            updateTabs();
            updateContent();
        }
        
        function updateContent() {
            const container = document.getElementById('content');
            
            if (Object.keys(domains).length === 0) {
                container.innerHTML = `
                    <div class="no-data">
                        <h2>Waiting for data...</h2>
                        <p>Make sure your RL training is running and publishing to /reward topics</p>
                    </div>
                `;
                return;
            }
            
            if (selectedDomain === 'all') {
                renderAllDomainsView(container);
            } else {
                renderSingleDomainView(container, selectedDomain);
            }
        }
        
        function renderAllDomainsView(container) {
            let html = '<div class="all-domains-view">';
            
            // Summary stats
            const totalSteps = Object.values(domains).reduce((sum, d) => sum + d.summary.total_steps, 0);
            const totalEpisodes = Object.values(domains).reduce((sum, d) => sum + d.summary.episode_count, 0);
            const avgReturn = Object.values(domains).reduce((sum, d) => sum + d.summary.avg_episode_return, 0) / Object.keys(domains).length;
            
            html += `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Total Steps (All Domains)</div>
                        <div class="value">${totalSteps.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Total Episodes</div>
                        <div class="value">${totalEpisodes}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Avg Episode Return</div>
                        <div class="value ${avgReturn >= 0 ? 'positive' : 'negative'}">${avgReturn.toFixed(1)}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Active Domains</div>
                        <div class="value positive">${Object.values(domains).filter(d => d.summary.active).length}</div>
                    </div>
                </div>
            `;
            
            // Comparison chart
            html += `
                <div class="chart-container">
                    <h3>Episode Returns Comparison</h3>
                    <div id="comparisonChart" class="chart tall"></div>
                </div>
            `;
            
            // Domain rows
            Object.entries(domains).forEach(([domain, data]) => {
                const stats = data.summary;
                const displayName = domain || 'default';
                
                html += `
                    <div class="domain-row" onclick="selectDomain('${domain}')">
                        <div class="domain-name">
                            <div class="indicator" style="background: ${stats.active ? '#22c55e' : '#6b7280'}"></div>
                            ${displayName}
                        </div>
                        <div class="mini-chart" id="miniChart_${domain.replace(/[^a-zA-Z0-9]/g, '_')}"></div>
                        <div class="stat">
                            <div class="label">Steps</div>
                            <div class="value">${stats.total_steps.toLocaleString()}</div>
                        </div>
                        <div class="stat">
                            <div class="label">Episodes</div>
                            <div class="value">${stats.episode_count}</div>
                        </div>
                        <div class="stat">
                            <div class="label">Avg Return</div>
                            <div class="value" style="color: ${stats.avg_episode_return >= 0 ? '#22c55e' : '#ef4444'}">
                                ${stats.avg_episode_return.toFixed(1)}
                            </div>
                        </div>
                        <div class="stat">
                            <div class="label">Intervention</div>
                            <div class="value" style="color: ${stats.avg_intervention < 0.2 ? '#22c55e' : '#f59e0b'}">
                                ${(stats.avg_intervention * 100).toFixed(0)}%
                            </div>
                        </div>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
            
            // Render comparison chart
            renderComparisonChart();
            
            // Render mini charts
            Object.entries(domains).forEach(([domain, data]) => {
                renderMiniChart(domain, data.plot_data);
            });
        }
        
        function renderComparisonChart() {
            const traces = [];
            const colors = ['#7c3aed', '#2dd4bf', '#f59e0b', '#ef4444', '#22c55e', '#3b82f6'];
            
            let colorIndex = 0;
            Object.entries(domains).forEach(([domain, data]) => {
                const plotData = data.plot_data;
                if (plotData.episode_returns.values.length > 0) {
                    traces.push({
                        x: plotData.episode_returns.episodes,
                        y: plotData.episode_returns.values,
                        type: 'scatter',
                        mode: 'lines',
                        name: domain || 'default',
                        line: { color: colors[colorIndex % colors.length], width: 2 }
                    });
                }
                colorIndex++;
            });
            
            const layout = {
                ...plotlyLayout,
                xaxis: { ...plotlyLayout.xaxis, title: 'Episode' },
                yaxis: { ...plotlyLayout.yaxis, title: 'Return' },
            };
            
            Plotly.newPlot('comparisonChart', traces, layout, plotlyConfig);
        }
        
        function renderMiniChart(domain, plotData) {
            const elementId = `miniChart_${domain.replace(/[^a-zA-Z0-9]/g, '_')}`;
            const element = document.getElementById(elementId);
            if (!element) return;
            
            const trace = {
                x: plotData.rewards.times.slice(-100),
                y: plotData.rewards.values.slice(-100),
                type: 'scatter',
                mode: 'lines',
                line: { color: '#7c3aed', width: 1.5 },
                fill: 'tozeroy',
                fillcolor: 'rgba(124, 58, 237, 0.2)'
            };
            
            const layout = {
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                margin: { l: 0, r: 0, t: 0, b: 0 },
                xaxis: { visible: false },
                yaxis: { visible: false },
                showlegend: false
            };
            
            Plotly.newPlot(elementId, [trace], layout, plotlyConfig);
        }
        
        function renderSingleDomainView(container, domain) {
            const data = domains[domain];
            if (!data) return;
            
            const stats = data.summary;
            const plotData = data.plot_data;
            const displayName = domain || 'default';
            
            let html = `
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="label">Total Steps</div>
                        <div class="value">${stats.total_steps.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Episodes</div>
                        <div class="value">${stats.episode_count}</div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Current Episode Return</div>
                        <div class="value ${stats.current_episode_return >= 0 ? 'positive' : 'negative'}">
                            ${stats.current_episode_return.toFixed(1)}
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Avg Episode Return</div>
                        <div class="value ${stats.avg_episode_return >= 0 ? 'positive' : 'negative'}">
                            ${stats.avg_episode_return.toFixed(1)}
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Avg Reward (Last 100)</div>
                        <div class="value ${stats.avg_reward_100 >= 0 ? 'positive' : 'negative'}">
                            ${stats.avg_reward_100.toFixed(3)}
                        </div>
                    </div>
                    <div class="stat-card">
                        <div class="label">Safety Intervention</div>
                        <div class="value ${stats.avg_intervention < 0.2 ? 'positive' : 'neutral'}">
                            ${(stats.avg_intervention * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
                
                <div class="charts-grid">
                    <div class="chart-container">
                        <h3>📈 Real-Time Reward</h3>
                        <div id="rewardChart" class="chart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>🏆 Episode Returns</h3>
                        <div id="episodeChart" class="chart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>🛡️ Safety Intervention Rate</h3>
                        <div id="interventionChart" class="chart"></div>
                    </div>
                    
                    <div class="chart-container">
                        <h3>🎯 Safety Zone Distribution</h3>
                        <div id="zoneChart" class="chart"></div>
                        <div class="safety-zones">
                            ${Object.entries(plotData.safety_zones).map(([zone, count]) => 
                                `<span class="zone-badge ${zone}">${zone}: ${count}</span>`
                            ).join('')}
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <h3>📊 Reward Components Breakdown</h3>
                    <div id="componentsChart" class="chart tall"></div>
                </div>
            `;
            
            container.innerHTML = html;
            
            // Render charts
            renderRewardChart(plotData);
            renderEpisodeChart(plotData);
            renderInterventionChart(plotData);
            renderZoneChart(plotData);
            renderComponentsChart(plotData);
        }
        
        function renderRewardChart(plotData) {
            const trace = {
                x: plotData.rewards.times,
                y: plotData.rewards.values,
                type: 'scatter',
                mode: 'lines',
                name: 'Reward',
                line: { color: '#7c3aed', width: 1.5 },
                fill: 'tozeroy',
                fillcolor: 'rgba(124, 58, 237, 0.2)'
            };
            
            const layout = {
                ...plotlyLayout,
                xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' },
                yaxis: { ...plotlyLayout.yaxis, title: 'Reward' },
            };
            
            Plotly.newPlot('rewardChart', [trace], layout, plotlyConfig);
        }
        
        function renderEpisodeChart(plotData) {
            const trace = {
                x: plotData.episode_returns.episodes,
                y: plotData.episode_returns.values,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Episode Return',
                line: { color: '#2dd4bf', width: 2 },
                marker: { size: 6 }
            };
            
            // Add rolling average
            const windowSize = 10;
            const rollingAvg = [];
            for (let i = 0; i < plotData.episode_returns.values.length; i++) {
                const start = Math.max(0, i - windowSize + 1);
                const window = plotData.episode_returns.values.slice(start, i + 1);
                rollingAvg.push(window.reduce((a, b) => a + b, 0) / window.length);
            }
            
            const avgTrace = {
                x: plotData.episode_returns.episodes,
                y: rollingAvg,
                type: 'scatter',
                mode: 'lines',
                name: 'Rolling Avg (10)',
                line: { color: '#f59e0b', width: 3 }
            };
            
            const layout = {
                ...plotlyLayout,
                xaxis: { ...plotlyLayout.xaxis, title: 'Episode' },
                yaxis: { ...plotlyLayout.yaxis, title: 'Return' },
            };
            
            Plotly.newPlot('episodeChart', [trace, avgTrace], layout, plotlyConfig);
        }
        
        function renderInterventionChart(plotData) {
            const trace = {
                x: plotData.interventions.times,
                y: plotData.interventions.values.map(v => v * 100),
                type: 'scatter',
                mode: 'lines',
                name: 'Intervention %',
                line: { color: '#ef4444', width: 1.5 },
                fill: 'tozeroy',
                fillcolor: 'rgba(239, 68, 68, 0.2)'
            };
            
            const layout = {
                ...plotlyLayout,
                xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' },
                yaxis: { ...plotlyLayout.yaxis, title: 'Intervention %', range: [0, 100] },
            };
            
            Plotly.newPlot('interventionChart', [trace], layout, plotlyConfig);
        }
        
        function renderZoneChart(plotData) {
            const zones = Object.entries(plotData.safety_zones);
            const colors = {
                'FREE': '#22c55e',
                'AWARE': '#84cc16',
                'CAUTION': '#eab308',
                'DANGER': '#f97316',
                'EMERGENCY': '#ef4444',
                'CRITICAL': '#b91c1c'
            };
            
            const trace = {
                values: zones.map(z => z[1]),
                labels: zones.map(z => z[0]),
                type: 'pie',
                marker: {
                    colors: zones.map(z => colors[z[0]] || '#6b7280')
                },
                textinfo: 'label+percent',
                textfont: { color: '#fff' },
                hole: 0.4
            };
            
            const layout = {
                ...plotlyLayout,
                showlegend: false,
            };
            
            Plotly.newPlot('zoneChart', [trace], layout, plotlyConfig);
        }
        
        function renderComponentsChart(plotData) {
            const traces = [];
            const componentColors = {
                'progress': '#22c55e',
                'alignment': '#3b82f6',
                'discovery': '#8b5cf6',
                'novelty': '#ec4899',
                'frontier': '#f59e0b',
                'proximity': '#ef4444',
                'intervention_penalty': '#dc2626',
                'rnd': '#06b6d4',
                'step': '#6b7280',
                'collision': '#7f1d1d',
                'goal': '#15803d',
            };
            
            Object.entries(plotData.components).forEach(([name, data]) => {
                if (data.values.length > 0) {
                    traces.push({
                        x: data.times,
                        y: data.values,
                        type: 'scatter',
                        mode: 'lines',
                        name: name,
                        line: { 
                            color: componentColors[name] || '#6b7280',
                            width: 1.5
                        }
                    });
                }
            });
            
            const layout = {
                ...plotlyLayout,
                xaxis: { ...plotlyLayout.xaxis, title: 'Time (s)' },
                yaxis: { ...plotlyLayout.yaxis, title: 'Reward Component' },
            };
            
            Plotly.newPlot('componentsChart', traces, layout, plotlyConfig);
        }
    </script>
</body>
</html>
"""


def create_app(stats_store: Dict[str, DomainStats]) -> tuple:
    """Create Flask app with SocketIO."""
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'reward_viz_secret'
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    @app.route('/')
    def index():
        return render_template_string(DASHBOARD_HTML)
    
    @app.route('/api/domains')
    def get_domains():
        """Get list of domains and their stats."""
        result = {}
        for domain, stats in stats_store.items():
            result[domain] = {
                "summary": stats.get_summary(),
                "plot_data": stats.get_plot_data()
            }
        return jsonify(result)
    
    def background_update():
        """Background thread that pushes updates to connected clients."""
        while True:
            time.sleep(1.0 / UPDATE_RATE_HZ)
            
            data = {"domains": {}}
            for domain, stats in stats_store.items():
                data["domains"][domain] = {
                    "summary": stats.get_summary(),
                    "plot_data": stats.get_plot_data()
                }
            
            socketio.emit('update', data)
    
    return app, socketio, background_update


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Real-time RL reward visualization")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, 
                        help=f"Web server port (default: {DEFAULT_PORT})")
    parser.add_argument("--domains", nargs="*", default=[""],
                        help="Domain namespaces to monitor (default: '' for /reward)")
    parser.add_argument("--auto-discover", action="store_true",
                        help="Auto-discover domains from ROS topics")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           🤖 RL Reward Visualizer Server                     ║
╠══════════════════════════════════════════════════════════════╣
║  Open in browser:  http://localhost:{args.port:<5}                 ║
║  Domains: {str(args.domains):<49}║
║  Auto-discover: {str(args.auto_discover):<42}║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize ROS
    rclpy.init()
    
    # Shared stats store
    stats_store: Dict[str, DomainStats] = {}
    
    # Create ROS node
    ros_node = RewardCollectorNode(args.domains, stats_store)
    
    # Create ROS executor
    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(ros_node)
    
    # Create Flask app
    app, socketio, background_update = create_app(stats_store)
    
    # Start ROS spinning in background thread
    def ros_spin():
        try:
            ros_executor.spin()
        except Exception as e:
            print(f"[ROS] Error: {e}")
    
    ros_thread = threading.Thread(target=ros_spin, daemon=True)
    ros_thread.start()
    
    # Start background update thread
    update_thread = threading.Thread(target=background_update, daemon=True)
    update_thread.start()
    
    # Auto-discovery thread
    if args.auto_discover:
        def discovery_loop():
            while True:
                time.sleep(DISCOVERY_INTERVAL)
                try:
                    new_domains = ros_node.discover_domains()
                    if new_domains:
                        print(f"[VIZ] Discovered new domains: {new_domains}")
                except Exception as e:
                    print(f"[VIZ] Discovery error: {e}")
        
        discovery_thread = threading.Thread(target=discovery_loop, daemon=True)
        discovery_thread.start()
    
    # Shutdown handler
    def shutdown(signum=None, frame=None):
        print("\n[VIZ] Shutting down...")
        try:
            ros_executor.shutdown()
            ros_node.destroy_node()
            rclpy.shutdown()
        except:
            pass
        sys.exit(0)
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    # Run Flask server
    try:
        socketio.run(app, host='0.0.0.0', port=args.port, debug=False, 
                     use_reloader=False, log_output=False)
    except Exception as e:
        print(f"[VIZ] Server error: {e}")
        shutdown()


if __name__ == "__main__":
    main()