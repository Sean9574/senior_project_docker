#!/usr/bin/env python3
"""
RL Training Monitor - Episode-Level Analytics
Fixed: Match learner_node reward_breakdown schema and improve analytics UI.
"""
import json
import threading

import rclpy
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from rclpy.node import Node
from std_msgs.msg import String

app = Flask(__name__)
app.config["SECRET_KEY"] = "reward_monitor_secret"
socketio = SocketIO(app, cors_allowed_origins="*")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RL Training Analytics</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        :root {
            --bg-primary: #0f1419;
            --bg-secondary: #1a1f2e;
            --bg-tertiary: #242d3d;
            --border: #2d3748;
            --text-primary: #e2e8f0;
            --text-secondary: #a0aec0;
            --text-muted: #718096;
            --accent: #3b82f6;
            --positive: #10b981;
            --negative: #ef4444;
            --warning: #f59e0b;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            font-size: 13px;
        }

        .header {
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            padding: 12px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-title {
            font-size: 15px;
            font-weight: 600;
        }

        .status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--negative);
        }

        .status-dot.connected {
            background: var(--positive);
            box-shadow: 0 0 8px var(--positive);
        }

        .main {
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 1px;
            background: var(--border);
            height: calc(100vh - 49px);
        }

        .sidebar {
            background: var(--bg-primary);
            overflow-y: auto;
            padding: 16px;
        }

        .content {
            background: var(--bg-primary);
            overflow-y: auto;
            padding: 16px;
        }

        .card {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
        }

        .card-title {
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 12px;
        }

        .big-stat {
            text-align: center;
            padding: 16px;
        }

        .big-stat-value {
            font-size: 36px;
            font-weight: 700;
            font-family: 'SF Mono', monospace;
        }

        .big-stat-label {
            font-size: 11px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 4px;
        }

        .big-stat-value.neutral { color: var(--accent); }

        .stats-row {
            display: flex;
            gap: 12px;
            margin-bottom: 16px;
        }

        .stat-box {
            flex: 1;
            background: var(--bg-tertiary);
            border-radius: 6px;
            padding: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 20px;
            font-weight: 600;
            font-family: 'SF Mono', monospace;
        }

        .stat-label {
            font-size: 9px;
            color: var(--text-muted);
            text-transform: uppercase;
            margin-top: 2px;
        }

        .learning-indicator {
            padding: 16px;
            border-radius: 8px;
            text-align: center;
            margin-bottom: 16px;
        }

        .learning-indicator.learning {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--positive);
        }

        .learning-indicator.struggling {
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid var(--warning);
        }

        .learning-indicator.not-learning {
            background: rgba(239, 68, 68, 0.15);
            border: 1px solid var(--negative);
        }

        .learning-status {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 4px;
        }

        .learning-detail {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .progress-item {
            margin-bottom: 12px;
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 11px;
        }

        .progress-label {
            color: var(--text-secondary);
        }

        .progress-value {
            font-family: 'SF Mono', monospace;
        }

        .progress-bar {
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.3s;
        }

        .progress-fill.positive { background: var(--positive); }
        .progress-fill.negative { background: var(--negative); }
        .progress-fill.neutral { background: var(--accent); }

        .chart-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }

        .chart-card {
            background: var(--bg-secondary);
            border-radius: 8px;
            padding: 16px;
        }

        .chart-card.full-width {
            grid-column: 1 / -1;
        }

        .chart-container {
            height: 200px;
            position: relative;
        }

        .reward-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 6px 0;
            border-bottom: 1px solid var(--bg-tertiary);
        }

        .reward-name {
            font-size: 11px;
            color: var(--text-secondary);
        }

        .reward-value {
            font-family: 'SF Mono', monospace;
            font-size: 12px;
            font-weight: 600;
        }

        .reward-value.positive { color: var(--positive); }
        .reward-value.negative { color: var(--negative); }

        .episode-log {
            max-height: 250px;
            overflow-y: auto;
            font-family: 'SF Mono', monospace;
            font-size: 11px;
        }

        .episode-entry {
            display: grid;
            grid-template-columns: 50px 60px 80px 1fr;
            gap: 8px;
            padding: 6px 8px;
            border-bottom: 1px solid var(--bg-tertiary);
        }

        .episode-entry.success { background: rgba(16, 185, 129, 0.1); }
        .episode-entry.collision { background: rgba(239, 68, 68, 0.1); }
        .episode-entry.timeout { background: rgba(245, 158, 11, 0.1); }

        .angle-gauge {
            text-align: center;
            padding: 20px;
        }

        .angle-value {
            font-size: 48px;
            font-weight: 700;
            font-family: 'SF Mono', monospace;
        }

        .angle-label {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">RL Training Analytics</div>
        <div class="status">
            <div class="status-dot" id="statusDot"></div>
            <span id="statusText">Disconnected</span>
        </div>
    </div>

    <div class="main">
        <div class="sidebar">
            <div class="card">
                <div class="big-stat">
                    <div class="big-stat-value neutral" id="currentEpisode">0</div>
                    <div class="big-stat-label">Episode</div>
                </div>
                <div style="text-align: center; color: var(--text-muted); font-size: 12px;">
                    Step <span id="currentStep">0</span> / 300
                </div>
            </div>

            <div class="card">
                <div class="card-title">Goal Angle (Training Target)</div>
                <div class="angle-gauge">
                    <div class="angle-value" id="goalAngle">--</div>
                    <div class="angle-label">degrees from goal</div>
                </div>
            </div>

            <div class="learning-indicator not-learning" id="learningIndicator">
                <div class="learning-status" id="learningStatus">Waiting for data</div>
                <div class="learning-detail" id="learningDetail">Need at least 10 episodes</div>
            </div>

            <div class="card">
                <div class="card-title">Session Totals</div>
                <div class="stats-row">
                    <div class="stat-box">
                        <div class="stat-value" style="color: var(--positive);" id="totalSuccess">0</div>
                        <div class="stat-label">Success</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" style="color: var(--negative);" id="totalCollision">0</div>
                        <div class="stat-label">Collision</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" style="color: var(--warning);" id="totalTimeout">0</div>
                        <div class="stat-label">Timeout</div>
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-header">
                        <span class="progress-label">Avg Episode Length</span>
                        <span class="progress-value" id="avgLength">--</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill neutral" id="lengthBar" style="width: 0%;"></div>
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-header">
                        <span class="progress-label">Success Rate</span>
                        <span class="progress-value" id="successRate">--</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill positive" id="successBar" style="width: 0%;"></div>
                    </div>
                </div>

                <div class="progress-item">
                    <div class="progress-header">
                        <span class="progress-label">Collision Rate</span>
                        <span class="progress-value" id="collisionRate">--</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill negative" id="collisionBar" style="width: 0%;"></div>
                    </div>
                </div>
            </div>

            <div class="card">
                <div class="card-title">Live Rewards</div>
                <div id="rewardList">
                    <div class="reward-item"><span class="reward-name">Progress</span><span class="reward-value" id="rProgress">--</span></div>
                    <div class="reward-item"><span class="reward-name">Movement</span><span class="reward-value" id="rMovement">--</span></div>
                    <div class="reward-item"><span class="reward-name">Alignment</span><span class="reward-value" id="rAlign">--</span></div>
                    <div class="reward-item"><span class="reward-name">Exploration</span><span class="reward-value" id="rExplore">--</span></div>
                    <div class="reward-item"><span class="reward-name">Slowness</span><span class="reward-value" id="rSlow">--</span></div>
                    <div class="reward-item"><span class="reward-name">Obstacle</span><span class="reward-value" id="rObstacle">--</span></div>
                    <div class="reward-item"><span class="reward-name">Goal</span><span class="reward-value" id="rGoal">--</span></div>
                    <div class="reward-item"><span class="reward-name">Fail</span><span class="reward-value" id="rFail">--</span></div>
                    <div class="reward-item" style="border-top: 2px solid var(--accent); margin-top: 8px; padding-top: 8px;">
                        <span class="reward-name" style="font-weight: 600;">TOTAL</span>
                        <span class="reward-value" id="rTotal">--</span>
                    </div>
                </div>
            </div>
        </div>

        <div class="content">
            <div class="chart-grid">
                <div class="chart-card full-width">
                    <div class="card-title">Episode Returns (Is it improving?)</div>
                    <div class="chart-container"><canvas id="returnChart"></canvas></div>
                </div>

                <div class="chart-card">
                    <div class="card-title">Episode Length (Surviving longer?)</div>
                    <div class="chart-container"><canvas id="lengthChart"></canvas></div>
                </div>

                <div class="chart-card">
                    <div class="card-title">Avg Angle per Episode (Getting aligned?)</div>
                    <div class="chart-container"><canvas id="angleChart"></canvas></div>
                </div>
            </div>

            <div class="chart-card" style="margin-top: 16px;">
                <div class="card-title">Episode Analytics</div>
                <div class="stats-row">
                    <div class="stat-box">
                        <div class="stat-value" id="rollingReturn10">--</div>
                        <div class="stat-label">Rolling Return (10)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="rollingLength10">--</div>
                        <div class="stat-label">Rolling Length (10)</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="rollingAngle10">--</div>
                        <div class="stat-label">Rolling Angle (10)</div>
                    </div>
                </div>
                <div class="stats-row">
                    <div class="stat-box">
                        <div class="stat-value" id="bestReturn">--</div>
                        <div class="stat-label">Best Return</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="bestAngle">--</div>
                        <div class="stat-label">Best Avg Angle</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value" id="episodesCount">0</div>
                        <div class="stat-label">Episodes Logged</div>
                    </div>
                </div>
            </div>

            <div class="chart-card" style="margin-top: 16px;">
                <div class="card-title">Episode History (most recent first)</div>
                <div class="episode-log" id="episodeLog">
                    <div class="episode-entry" style="font-weight: 600; background: var(--bg-tertiary);">
                        <span>EP</span><span>STEPS</span><span>RETURN</span><span>AVG ANGLE</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();

        let episodes = [];
        let monitorEpisodeCount = 0;
        let currentEp = { steps: 0, totalReward: 0, lastStep: -1, angles: [], lastDataTime: 0 };
        let rollingReturns = [];
        let rollingLengths = [];
        let rollingAngles = [];

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: '#1a1f2e' }, ticks: { color: '#4a5568', font: { size: 9 } } },
                y: { grid: { color: '#1a1f2e' }, ticks: { color: '#4a5568', font: { size: 9 } } }
            }
        };

        const returnChart = new Chart(document.getElementById('returnChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Episode Return',
                        data: [],
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        borderWidth: 1.5,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 3
                    },
                    {
                        label: 'Rolling Avg (10)',
                        data: [],
                        borderColor: '#10b981',
                        borderWidth: 2,
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0
                    }
                ]
            },
            options: {
                ...chartOptions,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: { color: '#718096', font: { size: 10 }, boxWidth: 12 }
                    }
                }
            }
        });

        const lengthChart = new Chart(document.getElementById('lengthChart'), {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: '#3b82f680',
                    borderColor: '#3b82f6',
                    borderWidth: 1
                }]
            },
            options: chartOptions
        });

        const angleChart = new Chart(document.getElementById('angleChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 1.5,
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3
                }]
            },
            options: chartOptions
        });

        function updateLearningIndicator() {
            const indicator = document.getElementById('learningIndicator');
            const status = document.getElementById('learningStatus');
            const detail = document.getElementById('learningDetail');

            if (episodes.length < 10) {
                indicator.className = 'learning-indicator not-learning';
                status.textContent = 'Collecting data';
                detail.textContent = `${episodes.length}/10 episodes`;
                return;
            }

            const recent = episodes.slice(-10);
            const recentAvg = recent.reduce((a, b) => a + b.return, 0) / 10;
            const recentAngle = recent.reduce((a, b) => a + (b.avgAngle || 90), 0) / 10;

            if (episodes.length >= 20) {
                const previous = episodes.slice(-20, -10);
                const prevAvg = previous.reduce((a, b) => a + b.return, 0) / 10;
                const prevAngle = previous.reduce((a, b) => a + (b.avgAngle || 90), 0) / 10;

                const returnDiff = recentAvg - prevAvg;
                const angleDiff = prevAngle - recentAngle;

                if (returnDiff > 50 || angleDiff > 5) {
                    indicator.className = 'learning-indicator learning';
                    status.textContent = 'Learning';
                    detail.textContent = `Return Δ: ${returnDiff > 0 ? '+' : ''}${returnDiff.toFixed(0)} | Angle Δ: ${angleDiff > 0 ? '-' : '+'}${Math.abs(angleDiff).toFixed(1)}°`;
                } else if (returnDiff > -50 && angleDiff > -5) {
                    indicator.className = 'learning-indicator struggling';
                    status.textContent = 'Plateau';
                    detail.textContent = `Avg return: ${recentAvg.toFixed(0)} | Avg angle: ${recentAngle.toFixed(1)}°`;
                } else {
                    indicator.className = 'learning-indicator not-learning';
                    status.textContent = 'Performance declining';
                    detail.textContent = 'Returns and alignment are trending down';
                }
            } else {
                indicator.className = 'learning-indicator struggling';
                status.textContent = 'Early training';
                detail.textContent = `Avg return: ${recentAvg.toFixed(0)} | Avg angle: ${recentAngle.toFixed(1)}°`;
            }
        }

        function addEpisode(ep) {
            episodes.push(ep);

            const maxPoints = 50;
            if (returnChart.data.labels.length >= maxPoints) {
                returnChart.data.labels.shift();
                returnChart.data.datasets[0].data.shift();
                returnChart.data.datasets[1].data.shift();
                lengthChart.data.labels.shift();
                lengthChart.data.datasets[0].data.shift();
                angleChart.data.labels.shift();
                angleChart.data.datasets[0].data.shift();
            }

            returnChart.data.labels.push(ep.episode);
            returnChart.data.datasets[0].data.push(ep.return);

            rollingReturns.push(ep.return);
            if (rollingReturns.length > 10) rollingReturns.shift();
            const avgReturn10 = rollingReturns.reduce((a, b) => a + b, 0) / rollingReturns.length;
            returnChart.data.datasets[1].data.push(avgReturn10);

            lengthChart.data.labels.push(ep.episode);
            lengthChart.data.datasets[0].data.push(ep.steps);
            rollingLengths.push(ep.steps);
            if (rollingLengths.length > 10) rollingLengths.shift();
            const avgLength10 = rollingLengths.reduce((a, b) => a + b, 0) / rollingLengths.length;

            const angleForEp = ep.avgAngle || 0;
            angleChart.data.labels.push(ep.episode);
            angleChart.data.datasets[0].data.push(angleForEp);
            rollingAngles.push(angleForEp);
            if (rollingAngles.length > 10) rollingAngles.shift();
            const avgAngle10 = rollingAngles.reduce((a, b) => a + b, 0) / rollingAngles.length;

            returnChart.update('none');
            lengthChart.update('none');
            angleChart.update('none');

            const successes = episodes.filter(e => e.outcome === 'success').length;
            const collisions = episodes.filter(e => e.outcome === 'collision').length;
            const timeouts = episodes.filter(e => e.outcome === 'timeout').length;
            const totalEpisodes = episodes.length;

            document.getElementById('totalSuccess').textContent = successes;
            document.getElementById('totalCollision').textContent = collisions;
            document.getElementById('totalTimeout').textContent = timeouts;

            const avgLen = episodes.reduce((a, b) => a + b.steps, 0) / totalEpisodes;
            document.getElementById('avgLength').textContent = `${avgLen.toFixed(0)} steps`;
            document.getElementById('lengthBar').style.width = `${Math.min(avgLen / 3, 100)}%`;

            const successRate = (successes / totalEpisodes) * 100;
            const collisionRate = (collisions / totalEpisodes) * 100;

            document.getElementById('successRate').textContent =
                isFinite(successRate) ? `${successRate.toFixed(1)}%` : '--';
            document.getElementById('collisionRate').textContent =
                isFinite(collisionRate) ? `${collisionRate.toFixed(1)}%` : '--';

            document.getElementById('successBar').style.width =
                `${Math.min(successRate, 100)}%`;
            document.getElementById('collisionBar').style.width =
                `${Math.min(collisionRate, 100)}%`;

            document.getElementById('rollingReturn10').textContent = avgReturn10.toFixed(1);
            document.getElementById('rollingLength10').textContent = avgLength10.toFixed(1);
            document.getElementById('rollingAngle10').textContent = avgAngle10.toFixed(1) + '°';
            document.getElementById('episodesCount').textContent = totalEpisodes;

            const bestReturnEp = episodes.reduce((best, e) => e.return > best.return ? e : best, episodes[0]);
            const bestAngleEp = episodes.reduce((best, e) =>
                (e.avgAngle || 999) < (best.avgAngle || 999) ? e : best, episodes[0]);

            document.getElementById('bestReturn').textContent = bestReturnEp.return.toFixed(1);
            document.getElementById('bestAngle').textContent = (bestAngleEp.avgAngle || 0).toFixed(1) + '°';

            const log = document.getElementById('episodeLog');
            const entry = document.createElement('div');
            entry.className = `episode-entry ${ep.outcome}`;
            entry.innerHTML = `<span>${ep.episode}</span><span>${ep.steps}</span><span>${ep.return.toFixed(0)}</span><span>${(ep.avgAngle || 0).toFixed(1)}°</span>`;
            log.insertBefore(entry, log.children[1]);

            while (log.children.length > 51) {
                log.removeChild(log.lastChild);
            }

            updateLearningIndicator();
        }

        socket.on('connect', () => {
            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Connected';
        });

        socket.on('disconnect', () => {
            document.getElementById('statusDot').classList.remove('connected');
            document.getElementById('statusText').textContent = 'Disconnected';
        });

        socket.on('reward_update', (data) => {
            const now = Date.now();

            document.getElementById('statusDot').classList.add('connected');
            document.getElementById('statusText').textContent = 'Live';

            const isNewEpisode = (data.step < currentEp.lastStep) ||
                                 (now - currentEp.lastDataTime > 3000 && currentEp.steps > 0);

            if (isNewEpisode) {
                monitorEpisodeCount++;
                let outcome = 'timeout';
                if ((data.total_successes || 0) > 0) outcome = 'success';
                else if ((data.total_collisions || 0) > 0) outcome = 'collision';

                const avgAngle = currentEp.angles.length > 0
                    ? currentEp.angles.reduce((a,b) => a+b, 0) / currentEp.angles.length
                    : 90;

                addEpisode({
                    episode: monitorEpisodeCount,
                    steps: currentEp.steps,
                    return: currentEp.totalReward,
                    avgAngle: avgAngle,
                    outcome: outcome
                });

                currentEp = { steps: 0, totalReward: 0, lastStep: -1, angles: [], lastDataTime: now };
            }

            currentEp.steps = (data.step || 0) + 1;
            currentEp.totalReward = typeof data.episode_return === 'number'
                ? data.episode_return
                : (data.rewards && typeof data.rewards.total === 'number' ? data.rewards.total : 0);
            currentEp.lastStep = data.step || 0;
            currentEp.lastDataTime = now;

            const state = data.state || {};
            const angleRaw = typeof state.goal_angle !== 'undefined'
                ? Number(state.goal_angle)
                : 0;
            const angle = Number.isFinite(angleRaw) ? angleRaw : 0;
            const angleDisplay = Math.abs(angle);
            const angleEl = document.getElementById('goalAngle');

            angleEl.textContent = angleDisplay.toFixed(1);
            if (angleDisplay < 5) {
                angleEl.style.color = '#10b981';
            } else if (angleDisplay < 30) {
                angleEl.style.color = '#f59e0b';
            } else {
                angleEl.style.color = '#ef4444';
            }

            if (Number.isFinite(angle)) {
                currentEp.angles.push(Math.abs(angle));
            }

            if (typeof data.episode === 'number') {
                monitorEpisodeCount = data.episode;
            }
            document.getElementById('currentEpisode').textContent = monitorEpisodeCount;
            document.getElementById('currentStep').textContent = data.step || 0;

            const r = data.rewards || {};
            updateReward('rProgress', r.progress);
            updateReward('rMovement', r.movement);
            updateReward('rAlign', r.alignment);
            updateReward('rExplore', r.exploration);
            updateReward('rSlow', r.slowness);
            updateReward('rObstacle', r.obstacle);
            updateReward('rGoal', r.goal);
            updateReward('rFail', r.fail);
            updateReward('rTotal', r.total);
        });

        function updateReward(id, value) {
            const el = document.getElementById(id);
            if (!el || value === undefined || value === null) return;
            const num = Number(value);
            if (!Number.isFinite(num)) return;
            el.textContent = num.toFixed(2);
            el.className = `reward-value ${num >= 0 ? 'positive' : 'negative'}`;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

class RewardMonitorNode(Node):
    def __init__(self):
        super().__init__("reward_monitor")
        self.reward_sub = self.create_subscription(
            String,
            "reward_breakdown",
            self.reward_callback,
            10,
        )
        self.step_count = 0
        self.get_logger().info("Analytics Monitor → http://localhost:5000")

    def reward_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
            self.step_count += 1
            if self.step_count % 50 == 0:
                r = data.get("rewards", {})
                s = data.get("state", {})
                self.get_logger().info(
                    f"Step {data.get('step', 0):03d} | "
                    f"Angle={s.get('goal_angle', 0):06.1f}° | "
                    f"R:align={r.get('alignment', 0):06.2f} "
                    f"R:total={r.get('total', 0):06.2f}"
                )
            socketio.emit("reward_update", data)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def run_ros_node():
    rclpy.init()
    node = RewardMonitorNode()
    rclpy.spin(node)
    rclpy.shutdown()

def main():
    ros_thread = threading.Thread(target=run_ros_node, daemon=True)
    ros_thread.start()

    print("=" * 50)
    print("RL Training Analytics")
    print("→ http://localhost:5000")
    print("=" * 50)

    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    main()
