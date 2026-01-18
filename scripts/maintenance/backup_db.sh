#!/bin/bash
# scripts/maintenance/backup_db.sh
# Postgres Backup Script for Stoic Citadel
# Goal: Daily transaction history safety.

# Set project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKUP_DIR="${PROJECT_ROOT}/user_data/backups/postgres"
mkdir -p "${BACKUP_DIR}"

# Load environment variables if .env exists
if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

# Configuration
DB_NAME=${POSTGRES_DB:-trading_analytics}
DB_USER=${POSTGRES_USER:-stoic_trader}
DB_HOST=${POSTGRES_HOST:-localhost}
DB_PORT=${POSTGRES_PORT:-5434}
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FILENAME="${BACKUP_DIR}/${DB_NAME}_${TIMESTAMP}.sql.gz"

echo "Starting backup of ${DB_NAME} to ${FILENAME}..."

# Execute pg_dump inside the docker container if it's running, 
# otherwise try local pg_dump
if docker ps | grep -q "stoic_postgres"; then
    echo "Detected running stoic_postgres container, using docker exec..."
    docker exec stoic_postgres pg_dump -U "${DB_USER}" "${DB_NAME}" | gzip > "${FILENAME}"
else
    echo "Container not found, attempting local pg_dump..."
    PGPASSWORD="${POSTGRES_PASSWORD}" pg_dump -h "${DB_HOST}" -p "${DB_PORT}" -U "${DB_USER}" "${DB_NAME}" | gzip > "${FILENAME}"
fi

# Check success
if [ $? -eq 0 ]; then
    echo "Backup completed successfully: ${FILENAME}"
    # Keep only last 7 days of backups
    find "${BACKUP_DIR}" -name "*.sql.gz" -mtime +7 -delete
    echo "Cleaned up backups older than 7 days."
else
    echo "Backup FAILED!"
    exit 1
fi
