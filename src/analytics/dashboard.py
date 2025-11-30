#!/usr/bin/env python3
"""
Real-time Analytics Dashboard
==============================

Web-based dashboard for monitoring portfolio performance.

Features:
- Real-time equity curve
- Position monitoring
- Performance metrics display
- Trade history table
- Alert notifications

Author: Stoic Citadel Team
License: MIT
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# HTML template for dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stoic Citadel - Portfolio Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, sans-serif; 
            background: #0f1419; 
            color: #e7e9ea;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1f25 0%, #2d3748 100%);
            padding: 1rem 2rem;
            border-bottom: 1px solid #2d3748;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 1.5rem; color: #38bdf8; }
        .status { display: flex; align-items: center; gap: 0.5rem; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; }
        .status-dot.connected { background: #22c55e; }
        .status-dot.disconnected { background: #ef4444; }
        
        .container { padding: 1rem 2rem; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: #1a1f25;
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid #2d3748;
        }
        .metric-label { font-size: 0.875rem; color: #9ca3af; margin-bottom: 0.5rem; }
        .metric-value { font-size: 1.75rem; font-weight: 700; }
        .metric-value.positive { color: #22c55e; }
        .metric-value.negative { color: #ef4444; }
        .metric-value.neutral { color: #38bdf8; }
        .metric-change { font-size: 0.875rem; margin-top: 0.25rem; }
        
        .charts-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        @media (max-width: 1024px) {
            .charts-grid { grid-template-columns: 1fr; }
        }
        
        .chart-container {
            background: #1a1f25;
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid #2d3748;
        }
        .chart-title { font-size: 1rem; margin-bottom: 1rem; color: #9ca3af; }
        
        .positions-table {
            background: #1a1f25;
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid #2d3748;
            margin-bottom: 1.5rem;
        }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #2d3748; }
        th { color: #9ca3af; font-weight: 500; }
        tr:hover { background: rgba(56, 189, 248, 0.05); }
        
        .alerts {
            position: fixed;
            top: 1rem;
            right: 1rem;
            z-index: 1000;
        }
        .alert {
            background: #1a1f25;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.5rem;
            border-left: 4px solid #38bdf8;
            animation: slideIn 0.3s ease;
        }
        .alert.warning { border-color: #f59e0b; }
        .alert.error { border-color: #ef4444; }
        .alert.success { border-color: #22c55e; }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    </style>
</head>
<body>
    <header class="header">
        <h1>⚔️ Stoic Citadel - Live Dashboard</h1>
        <div class="status">
            <span class="status-dot" id="statusDot"></span>
            <span id="statusText">Connecting...</span>
            <span id="lastUpdate" style="margin-left: 1rem; color: #9ca3af;"></span>
        </div>
    </header>
    
    <div class="container">
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Equity</div>
                <div class="metric-value neutral" id="equity">$0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value" id="pnl">$0.00</div>
                <div class="metric-change" id="pnlPct">0.00%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Unrealized P&L</div>
                <div class="metric-value" id="unrealizedPnl">$0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Open Positions</div>
                <div class="metric-value neutral" id="positions">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Win Rate</div>
                <div class="metric-value" id="winRate">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value negative" id="maxDD">0%</div>
            </div>
        </div>
        
        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Equity Curve</div>
                <canvas id="equityChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">P&L Distribution</div>
                <canvas id="pnlChart"></canvas>
            </div>
        </div>
        
        <div class="positions-table">
            <div class="chart-title">Open Positions</div>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Quantity</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Duration</th>
                    </tr>
                </thead>
                <tbody id="positionsBody"></tbody>
            </table>
        </div>
    </div>
    
    <div class="alerts" id="alerts"></div>
    
    <script>
        // Chart initialization
        const equityCtx = document.getElementById('equityChart').getContext('2d');
        const equityChart = new Chart(equityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity',
                    data: [],
                    borderColor: '#38bdf8',
                    backgroundColor: 'rgba(56, 189, 248, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: '#2d3748' }, ticks: { color: '#9ca3af' } },
                    y: { grid: { color: '#2d3748' }, ticks: { color: '#9ca3af' } }
                }
            }
        });
        
        const pnlCtx = document.getElementById('pnlChart').getContext('2d');
        const pnlChart = new Chart(pnlCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'P&L',
                    data: [],
                    backgroundColor: [],
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: '#2d3748' }, ticks: { color: '#9ca3af' } },
                    y: { grid: { color: '#2d3748' }, ticks: { color: '#9ca3af' } }
                }
            }
        });
        
        // WebSocket connection
        let ws;
        const wsUrl = 'ws://' + window.location.host + '/ws';
        
        function connect() {
            ws = new WebSocket(wsUrl);
            
            ws.onopen = () => {
                document.getElementById('statusDot').className = 'status-dot connected';
                document.getElementById('statusText').textContent = 'Connected';
            };
            
            ws.onclose = () => {
                document.getElementById('statusDot').className = 'status-dot disconnected';
                document.getElementById('statusText').textContent = 'Disconnected';
                setTimeout(connect, 3000);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };
        }
        
        function updateDashboard(data) {
            // Update metrics
            const snapshot = data.snapshot || {};
            const metrics = data.metrics || {};
            
            document.getElementById('equity').textContent = formatCurrency(snapshot.total_equity);
            
            const pnl = snapshot.total_pnl || 0;
            document.getElementById('pnl').textContent = formatCurrency(pnl);
            document.getElementById('pnl').className = 'metric-value ' + (pnl >= 0 ? 'positive' : 'negative');
            
            const pnlPct = metrics.total_return || 0;
            document.getElementById('pnlPct').textContent = (pnlPct >= 0 ? '+' : '') + pnlPct.toFixed(2) + '%';
            document.getElementById('pnlPct').className = 'metric-change ' + (pnlPct >= 0 ? 'positive' : 'negative');
            
            const unrealized = snapshot.unrealized_pnl || 0;
            document.getElementById('unrealizedPnl').textContent = formatCurrency(unrealized);
            document.getElementById('unrealizedPnl').className = 'metric-value ' + (unrealized >= 0 ? 'positive' : 'negative');
            
            document.getElementById('positions').textContent = snapshot.open_positions || 0;
            document.getElementById('winRate').textContent = (metrics.win_rate || 0).toFixed(1) + '%';
            document.getElementById('maxDD').textContent = '-' + (metrics.max_drawdown || 0).toFixed(2) + '%';
            
            // Update equity chart
            if (data.equity_history) {
                equityChart.data.labels = data.equity_history.map(s => 
                    new Date(s.timestamp * 1000).toLocaleTimeString()
                );
                equityChart.data.datasets[0].data = data.equity_history.map(s => s.total_equity);
                equityChart.update('none');
            }
            
            // Update positions table
            const tbody = document.getElementById('positionsBody');
            tbody.innerHTML = '';
            (data.positions || []).forEach(pos => {
                const pnlClass = pos.unrealized_pnl >= 0 ? 'positive' : 'negative';
                tbody.innerHTML += `
                    <tr>
                        <td>${pos.symbol}</td>
                        <td>${pos.side}</td>
                        <td>${pos.quantity.toFixed(4)}</td>
                        <td>${formatCurrency(pos.entry_price)}</td>
                        <td>${formatCurrency(pos.current_price)}</td>
                        <td class="${pnlClass}">${formatCurrency(pos.unrealized_pnl)}</td>
                        <td class="${pnlClass}">${pos.unrealized_pnl_pct.toFixed(2)}%</td>
                        <td>${formatDuration(pos.hold_duration_seconds)}</td>
                    </tr>
                `;
            });
            
            document.getElementById('lastUpdate').textContent = 
                'Updated: ' + new Date().toLocaleTimeString();
        }
        
        function formatCurrency(value) {
            return '$' + (value || 0).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            });
        }
        
        function formatDuration(seconds) {
            if (seconds < 60) return Math.floor(seconds) + 's';
            if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
            return Math.floor(seconds / 3600) + 'h';
        }
        
        function showAlert(message, type = 'info') {
            const alerts = document.getElementById('alerts');
            const alert = document.createElement('div');
            alert.className = 'alert ' + type;
            alert.textContent = message;
            alerts.appendChild(alert);
            setTimeout(() => alert.remove(), 5000);
        }
        
        connect();
    </script>
</body>
</html>
"""


class AnalyticsDashboard:
    """
    Web-based real-time analytics dashboard.
    
    Usage:
        dashboard = AnalyticsDashboard(tracker, port=8080)
        await dashboard.start()
    """
    
    def __init__(
        self,
        portfolio_tracker,
        host: str = "0.0.0.0",
        port: int = 8080,
        update_interval: float = 1.0
    ):
        self.tracker = portfolio_tracker
        self.host = host
        self.port = port
        self.update_interval = update_interval
        self._running = False
        self._clients = set()
        self._app = None
    
    async def start(self):
        """Start the dashboard server."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp not installed. Install with: pip install aiohttp")
            return
        
        self._running = True
        
        # Create web app
        self._app = web.Application()
        self._app.router.add_get('/', self._handle_index)
        self._app.router.add_get('/ws', self._handle_websocket)
        self._app.router.add_get('/api/data', self._handle_api_data)
        
        # Start broadcast task
        asyncio.create_task(self._broadcast_loop())
        
        # Start server
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"Dashboard running at http://{self.host}:{self.port}")
    
    async def stop(self):
        """Stop the dashboard server."""
        self._running = False
        for client in self._clients:
            await client.close()
        self._clients.clear()
    
    async def _handle_index(self, request):
        """Serve dashboard HTML."""
        from aiohttp import web
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')
    
    async def _handle_websocket(self, request):
        """Handle WebSocket connections."""
        from aiohttp import web
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self._clients.add(ws)
        logger.info(f"Dashboard client connected. Total: {len(self._clients)}")
        
        try:
            async for msg in ws:
                pass  # Just keep connection alive
        finally:
            self._clients.discard(ws)
            logger.info(f"Dashboard client disconnected. Total: {len(self._clients)}")
        
        return ws
    
    async def _handle_api_data(self, request):
        """REST API endpoint for data."""
        from aiohttp import web
        data = self._get_dashboard_data()
        return web.json_response(data)
    
    async def _broadcast_loop(self):
        """Broadcast updates to all connected clients."""
        while self._running:
            if self._clients:
                data = self._get_dashboard_data()
                message = json.dumps(data)
                
                # Send to all clients
                closed = []
                for client in self._clients:
                    try:
                        await client.send_str(message)
                    except:
                        closed.append(client)
                
                for client in closed:
                    self._clients.discard(client)
            
            await asyncio.sleep(self.update_interval)
    
    def _get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data."""
        tracker_data = self.tracker.to_dict()
        
        # Add equity history for chart
        equity_history = self.tracker.get_equity_history(limit=100)
        tracker_data["equity_history"] = [s.to_dict() for s in equity_history]
        
        return tracker_data
