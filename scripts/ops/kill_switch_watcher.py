#!/usr/bin/env python3
"""
Kill Switch Watcher
===================

External process to monitor risk state and trigger emergency stop
if thresholds are breached. Acts as a second layer of defense.
"""

import time
import sqlite3
import pickle
import logging
from pathlib import Path
from src.utils.logger import log
from src.config.unified_config import load_config
from src.notification.telegram import TelegramBot

def monitor_kill_switch():
    u_cfg = load_config()
    db_path = u_cfg.paths.user_data_dir / "risk_state_v2.db"
    telegram = TelegramBot()
    
    max_drawdown = u_cfg.max_daily_drawdown_pct
    max_losses = u_cfg.max_consecutive_losses
    
    log.info(f"Kill Switch Watcher started. Monitoring {db_path}")
    log.info(f"Thresholds: Drawdown={max_drawdown:.2%}, Consecutive Losses={max_losses}")
    
    while True:
        try:
            if db_path.exists():
                with sqlite3.connect(db_path) as conn:
                    row = conn.execute("SELECT data FROM risk_v2 WHERE id = 1").fetchone()
                    if row:
                        state = pickle.loads(row[0])
                        metrics = state.get("metrics")
                        
                        if metrics:
                            # Access attributes if it's a dataclass, or keys if it's a dict
                            if hasattr(metrics, 'current_drawdown_pct'):
                                dd = float(metrics.current_drawdown_pct)
                            else:
                                dd = float(metrics.get('current_drawdown_pct', 0))

                            # Handle consecutive losses from circuit_breaker status or metrics
                            cb_status = state.get("circuit_breaker") or {}
                            losses = int(cb_status.get("consecutive_losses", 0))
                            
                            if dd >= max_drawdown or losses >= max_losses:
>>>>+++ REPLACE

                                reason = "Max Drawdown" if dd >= max_drawdown else "Consecutive Losses"
                                log.critical(f"EXTERNAL KILL SWITCH TRIGGERED: {reason} (DD={dd:.2%}, Losses={losses})")
                                
                                # In a real scenario, this would send an emergency signal to the main process
                                # or directly call exchange APIs to cancel all orders.
                                # For this implementation, we log and notify.
                                telegram.send_message(
                                    f"🚨 <b>EXTERNAL KILL SWITCH TRIGGERED</b> 🚨\n"
                                    f"Reason: {reason}\n"
                                    f"Current Drawdown: {dd:.2%}\n"
                                    f"Consecutive Losses: {losses}"
                                )
                                # Stop monitoring after trigger to avoid spam
                                break
            
            time.sleep(5) # Check every 5 seconds
        except Exception as e:
            log.error(f"Kill Switch Watcher Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    monitor_kill_switch()
