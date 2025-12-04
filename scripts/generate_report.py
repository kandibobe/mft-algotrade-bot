#!/usr/bin/env python3
"""
Stoic Citadel - Backtest Report Generator
==========================================

Generates HTML reports from Freqtrade backtest results.

Usage:
    python scripts/generate_report.py --input results.json --output report.html
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import html


def load_backtest_results(file_path: str) -> Dict[str, Any]:
    """Load backtest results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def generate_html_report(
    results: Dict[str, Any],
    output_path: str,
    title: str = "Stoic Citadel Backtest Report"
) -> None:
    """
    Generate HTML report from backtest results.
    
    Args:
        results: Backtest results dictionary
        output_path: Path for output HTML file
        title: Report title
    """
    # Extract key metrics
    strategy_name = list(results.get('strategy', {}).keys())[0] if results.get('strategy') else 'Unknown'
    strategy_data = results.get('strategy', {}).get(strategy_name, {})
    
    total_trades = strategy_data.get('total_trades', 0)
    profit_total = strategy_data.get('profit_total', 0)
    profit_total_pct = strategy_data.get('profit_total_pct', 0)
    winrate = strategy_data.get('winrate', 0)
    max_drawdown = strategy_data.get('max_drawdown', 0)
    sharpe = strategy_data.get('sharpe', 0)
    sortino = strategy_data.get('sortino', 0)
    
    # Get trades list
    trades = strategy_data.get('trades', [])
    
    # Generate HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --text-primary: #eaeaea;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --success: #00d26a;
            --warning: #ffc107;
            --danger: #ff4757;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            padding: 30px 0;
            border-bottom: 2px solid var(--accent);
            margin-bottom: 30px;
        }}
        
        h1 {{
            color: var(--accent);
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1em;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .metric-card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 25px;
            text-align: center;
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        
        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.9em;
            text-transform: uppercase;
        }}
        
        .positive {{ color: var(--success); }}
        .negative {{ color: var(--danger); }}
        .neutral {{ color: var(--warning); }}
        
        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            color: var(--accent);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(233, 69, 96, 0.3);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        th {{
            background: rgba(233, 69, 96, 0.2);
            color: var(--accent);
            font-weight: 600;
        }}
        
        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}
        
        footer {{
            text-align: center;
            padding: 30px 0;
            color: var(--text-secondary);
            font-size: 0.9em;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .badge-success {{ background: rgba(0, 210, 106, 0.2); color: var(--success); }}
        .badge-danger {{ background: rgba(255, 71, 87, 0.2); color: var(--danger); }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏛️ {html.escape(title)}</h1>
            <p class="subtitle">Strategy: {html.escape(strategy_name)} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if profit_total >= 0 else 'negative'}">
                    {profit_total:.4f}
                </div>
                <div class="metric-label">Total Profit</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if profit_total_pct >= 0 else 'negative'}">
                    {profit_total_pct:.2f}%
                </div>
                <div class="metric-label">Profit %</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if winrate >= 50 else 'negative'}">
                    {winrate:.1f}%
                </div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value negative">{max_drawdown:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if sharpe >= 1 else 'neutral' if sharpe >= 0 else 'negative'}">
                    {sharpe:.2f}
                </div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Trade History (Last 20)</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Pair</th>
                        <th>Profit %</th>
                        <th>Open Date</th>
                        <th>Close Date</th>
                        <th>Duration</th>
                        <th>Exit Reason</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    # Add trade rows (last 20)
    for i, trade in enumerate(trades[-20:], 1):
        profit_pct = trade.get('profit_percent', 0) * 100
        profit_class = 'positive' if profit_pct >= 0 else 'negative'
        badge_class = 'badge-success' if profit_pct >= 0 else 'badge-danger'
        
        html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{html.escape(str(trade.get('pair', 'N/A')))}</td>
                        <td class="{profit_class}">{profit_pct:.2f}%</td>
                        <td>{trade.get('open_date', 'N/A')}</td>
                        <td>{trade.get('close_date', 'N/A')}</td>
                        <td>{trade.get('trade_duration', 'N/A')} min</td>
                        <td><span class="badge {badge_class}">{html.escape(str(trade.get('exit_reason', 'N/A')))}</span></td>
                    </tr>
"""
    
    html_content += """
                </tbody>
            </table>
        </div>
        
        <footer>
            <p>🏛️ Stoic Citadel Trading Bot | "The wise man accepts losses with equanimity."</p>
            <p>Report generated automatically by Stoic Citadel CI/CD Pipeline</p>
        </footer>
    </div>
</body>
</html>
"""
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate HTML report from backtest results')
    parser.add_argument('--input', '-i', required=True, help='Input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='Output HTML file path')
    parser.add_argument('--title', '-t', default='Stoic Citadel Backtest Report', help='Report title')
    
    args = parser.parse_args()
    
    try:
        results = load_backtest_results(args.input)
        generate_html_report(results, args.output, args.title)
    except FileNotFoundError:
        print(f"❌ Input file not found: {args.input}")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in: {args.input}")
        exit(1)
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        exit(1)


if __name__ == '__main__':
    main()
