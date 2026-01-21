#!/bin/bash
# Stoic Citadel - Docker Maintenance Script

echo "Cleaning up Docker logs..."
# Find all docker log files and truncate them
find /var/lib/docker/containers/ -name "*.log" -exec truncate -s 0 {} \;

echo "Cleaning up unused Docker images and volumes..."
docker image prune -f
docker volume prune -f

echo "Cleaning up stopped containers..."
docker container prune -f

echo "Maintenance complete."