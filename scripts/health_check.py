#!/usr/bin/env python3
"""
Stoic Citadel Health Check Script
Performs comprehensive health checks on all components.
"""

import subprocess
import sys
import json
import socket
from datetime import datetime
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
NC = "\033[0m"


def print_header():
    """Print script header."""
    print(f"""
{CYAN}╔════════════════════════════════════════════════════════════════════╗
║                 STOIC CITADEL HEALTH CHECK                          ║
╚════════════════════════════════════════════════════════════════════╝{NC}
""")


def check_docker():
    """Check Docker status."""
    print(f"{CYAN}🐳 Checking Docker...{NC}")
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  {GREEN}✅ Docker: {result.stdout.strip()}{NC}")
            return True
        else:
            print(f"  {RED}❌ Docker not running{NC}")
            return False
    except Exception as e:
        print(f"  {RED}❌ Docker error: {e}{NC}")
        return False



def check_containers():
    """Check running containers."""
    print(f"\n{CYAN}📦 Checking Containers...{NC}")
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=30
        )
        if result.stdout:
            containers = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    containers.append(json.loads(line))
            
            for c in containers:
                name = c.get('Name', 'unknown')
                state = c.get('State', 'unknown')
                status = c.get('Status', '')
                
                if state == 'running':
                    print(f"  {GREEN}✅ {name}: {status}{NC}")
                else:
                    print(f"  {RED}❌ {name}: {state}{NC}")
            return len([c for c in containers if c.get('State') == 'running'])
        else:
            print(f"  {YELLOW}⚠️ No containers running{NC}")
            return 0
    except Exception as e:
        print(f"  {RED}❌ Error checking containers: {e}{NC}")
        return 0


def check_ports():
    """Check if required ports are available/in-use."""
    print(f"\n{CYAN}🔌 Checking Ports...{NC}")
    
    ports = {
        3000: "FreqUI",
        8080: "Freqtrade API",
        8888: "Jupyter Lab",
        5432: "PostgreSQL",
        9000: "Portainer"
    }
    
    for port, service in ports.items():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', port))
        sock.close()
        
        if result == 0:
            print(f"  {GREEN}✅ Port {port} ({service}): In use{NC}")
        else:
            print(f"  {YELLOW}⚠️ Port {port} ({service}): Available{NC}")


def check_config():
    """Check configuration files."""
    print(f"\n{CYAN}⚙️ Checking Configuration...{NC}")
    
    config_files = [
        ".env",
        "docker-compose.yml",
        "user_data/config/config_dryrun.json",
        "user_data/config/config_production.json"
    ]
    
    for config in config_files:
        path = Path(config)
        if path.exists():
            print(f"  {GREEN}✅ {config}: Found{NC}")
        else:
            print(f"  {RED}❌ {config}: Missing{NC}")


def check_data():
    """Check market data."""
    print(f"\n{CYAN}📊 Checking Market Data...{NC}")
    
    data_dir = Path("user_data/data")
    if data_dir.exists():
        pairs = list(data_dir.glob("**/*.json"))
        if pairs:
            print(f"  {GREEN}✅ Found {len(pairs)} data files{NC}")
        else:
            print(f"  {YELLOW}⚠️ No data files found. Run: make download{NC}")
    else:
        print(f"  {RED}❌ Data directory not found{NC}")


def main():
    """Run all health checks."""
    print_header()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Timestamp: {timestamp}\n")
    
    # Run checks
    docker_ok = check_docker()
    containers = check_containers() if docker_ok else 0
    check_ports()
    check_config()
    check_data()
    
    # Summary
    print(f"\n{CYAN}{'='*60}{NC}")
    print(f"{CYAN}SUMMARY{NC}")
    print(f"{CYAN}{'='*60}{NC}")
    
    if docker_ok and containers >= 2:
        print(f"{GREEN}✅ System is HEALTHY{NC}")
        return 0
    elif docker_ok:
        print(f"{YELLOW}⚠️ System is PARTIAL (some services down){NC}")
        return 1
    else:
        print(f"{RED}❌ System is DOWN{NC}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
