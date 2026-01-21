param (
    [string]$command = "up"
)

$COMPOSE_FILE = "deploy/docker-compose.yml"

switch ($command) {
    "up" { docker-compose -f $COMPOSE_FILE up -d --build }
    "down" { docker-compose -f $COMPOSE_FILE down }
    "logs" { docker-compose -f $COMPOSE_FILE logs -f freqtrade }
    "restart" { 
        docker-compose -f $COMPOSE_FILE down
        docker-compose -f $COMPOSE_FILE up -d
    }
    "chaos" { python scripts/risk/chaos_test.py }
    "verify" { python scripts/verify_deployment.py }
    default { Write-Host "Usage: .\manage.ps1 [up|down|logs|restart|chaos|verify]" }
}