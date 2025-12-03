#!/usr/bin/env python3
"""
Stoic Citadel - Health Monitoring & Auto-Restart Service

Features:
- Automatic health checks every 30 seconds
- Auto-restart unhealthy containers
- Telegram alerts on issues
- Metrics export for Prometheus
- Detailed logging
"""

import docker
import time
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import requests
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

CHECK_INTERVAL = 30  # seconds
MAX_RESTART_ATTEMPTS = 3
RESTART_COOLDOWN = 300  # 5 minutes

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

CONTAINERS_TO_MONITOR = [
    'stoic_freqtrade',
    'stoic_frequi',
    'stoic_postgres',
    'stoic_jupyter'
]

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('user_data/logs/health_monitor.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# TELEGRAM NOTIFICATIONS
# ============================================================================

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.enabled = bool(token and chat_id and token != '<YOUR_TELEGRAM_BOT_TOKEN>')
        
        if self.enabled:
            logger.info("✅ Telegram notifications enabled")
        else:
            logger.info("⚠️  Telegram notifications disabled (configure in .env)")
    
    def send(self, message: str, level: str = "INFO"):
        if not self.enabled:
            return
        
        emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅"
        }.get(level, "📢")
        
        formatted_message = f"{emoji} *Stoic Citadel*\n\n{message}"
        
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": formatted_message,
                "parse_mode": "Markdown"
            }
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")

# ============================================================================
# CONTAINER HEALTH CHECKER
# ============================================================================

class HealthMonitor:
    def __init__(self):
        try:
            self.client = docker.from_env()
            logger.info("✅ Connected to Docker")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Docker: {e}")
            sys.exit(1)
        
        self.notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
        self.restart_counts: Dict[str, int] = {}
        self.last_restart_time: Dict[str, float] = {}
        
        # Send startup notification
        self.notifier.send(
            f"🚀 Health Monitor Started\n"
            f"Monitoring: {', '.join(CONTAINERS_TO_MONITOR)}\n"
            f"Check Interval: {CHECK_INTERVAL}s",
            "SUCCESS"
        )
    
    def get_container(self, name: str):
        try:
            return self.client.containers.get(name)
        except docker.errors.NotFound:
            return None
        except Exception as e:
            logger.error(f"Error getting container {name}: {e}")
            return None
    
    def check_health(self, container) -> tuple[bool, str]:
        """Check if container is healthy. Returns (is_healthy, status)"""
        try:
            # Check if running
            if container.status != 'running':
                return False, f"Not running (status: {container.status})"
            
            # Check health status if available
            health = container.attrs.get('State', {}).get('Health', {})
            if health:
                health_status = health.get('Status', 'none')
                if health_status == 'healthy':
                    return True, "Healthy"
                elif health_status == 'starting':
                    return True, "Starting (OK)"
                else:
                    return False, f"Unhealthy (status: {health_status})"
            
            # If no health check defined, consider running = healthy
            return True, "Running (no healthcheck)"
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return False, f"Error: {str(e)}"
    
    def should_restart(self, container_name: str) -> bool:
        """Check if we should attempt restart based on cooldown and max attempts"""
        # Check restart count
        if self.restart_counts.get(container_name, 0) >= MAX_RESTART_ATTEMPTS:
            logger.error(f"❌ {container_name} exceeded max restart attempts")
            self.notifier.send(
                f"Container: {container_name}\n"
                f"Status: FAILED\n"
                f"Reason: Exceeded {MAX_RESTART_ATTEMPTS} restart attempts\n\n"
                f"⚠️  MANUAL INTERVENTION REQUIRED",
                "ERROR"
            )
            return False
        
        # Check cooldown
        last_restart = self.last_restart_time.get(container_name, 0)
        if time.time() - last_restart < RESTART_COOLDOWN:
            logger.warning(f"⏳ {container_name} in restart cooldown")
            return False
        
        return True
    
    def restart_container(self, container_name: str) -> bool:
        """Attempt to restart a container"""
        try:
            container = self.get_container(container_name)
            if not container:
                logger.error(f"❌ Container {container_name} not found")
                return False
            
            logger.info(f"🔄 Restarting {container_name}...")
            container.restart(timeout=30)
            
            # Update restart tracking
            self.restart_counts[container_name] = self.restart_counts.get(container_name, 0) + 1
            self.last_restart_time[container_name] = time.time()
            
            # Wait for container to start
            time.sleep(10)
            
            # Check if restart was successful
            container.reload()
            is_healthy, status = self.check_health(container)
            
            if is_healthy:
                logger.info(f"✅ {container_name} restarted successfully")
                self.notifier.send(
                    f"Container: {container_name}\n"
                    f"Action: Auto-restart\n"
                    f"Result: SUCCESS\n"
                    f"Status: {status}",
                    "SUCCESS"
                )
                return True
            else:
                logger.error(f"❌ {container_name} still unhealthy after restart")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to restart {container_name}: {e}")
            self.notifier.send(
                f"Container: {container_name}\n"
                f"Action: Auto-restart\n"
                f"Result: FAILED\n"
                f"Error: {str(e)}",
                "ERROR"
            )
            return False
    
    def check_all_containers(self):
        """Check health of all monitored containers"""
        results = []
        
        for container_name in CONTAINERS_TO_MONITOR:
            container = self.get_container(container_name)
            
            if not container:
                logger.warning(f"⚠️  {container_name} not found")
                results.append((container_name, False, "Not found"))
                continue
            
            is_healthy, status = self.check_health(container)
            results.append((container_name, is_healthy, status))
            
            if not is_healthy:
                logger.warning(f"⚠️  {container_name}: {status}")
                
                # Attempt auto-restart
                if self.should_restart(container_name):
                    self.restart_container(container_name)
            else:
                # Reset restart count on successful health check
                if container_name in self.restart_counts:
                    self.restart_counts[container_name] = 0
        
        return results
    
    def get_metrics(self) -> Dict:
        """Get metrics for Prometheus export"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "containers": {}
        }
        
        for container_name in CONTAINERS_TO_MONITOR:
            container = self.get_container(container_name)
            if not container:
                continue
            
            try:
                stats = container.stats(stream=False)
                cpu_percent = self._calculate_cpu_percent(stats)
                mem_usage = stats['memory_stats'].get('usage', 0) / (1024 * 1024)  # MB
                
                metrics['containers'][container_name] = {
                    "status": container.status,
                    "cpu_percent": round(cpu_percent, 2),
                    "memory_mb": round(mem_usage, 2),
                    "restart_count": self.restart_counts.get(container_name, 0)
                }
            except Exception as e:
                logger.error(f"Error getting metrics for {container_name}: {e}")
        
        return metrics
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from Docker stats"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * 100.0
                return cpu_percent
        except:
            pass
        return 0.0
    
    def run(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting health monitor...")
        logger.info(f"📊 Monitoring containers: {', '.join(CONTAINERS_TO_MONITOR)}")
        logger.info(f"⏱️  Check interval: {CHECK_INTERVAL}s")
        
        try:
            while True:
                results = self.check_all_containers()
                
                # Log summary
                healthy_count = sum(1 for _, is_healthy, _ in results if is_healthy)
                logger.info(
                    f"📊 Health check complete: "
                    f"{healthy_count}/{len(results)} containers healthy"
                )
                
                # Export metrics
                metrics = self.get_metrics()
                with open('user_data/logs/health_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                time.sleep(CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            logger.info("\n🛑 Stopping health monitor...")
            self.notifier.send(
                "🛑 Health Monitor Stopped\n\n"
                "Manual shutdown detected",
                "WARNING"
            )
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}")
            self.notifier.send(
                f"🚨 Health Monitor CRASHED\n\n"
                f"Error: {str(e)}\n\n"
                f"⚠️  IMMEDIATE ATTENTION REQUIRED",
                "ERROR"
            )
            raise

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    monitor = HealthMonitor()
    monitor.run()
