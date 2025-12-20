# Test Setup Script - Validates entire Stoic Citadel installation
# Run this after setup to verify everything works

$ErrorActionPreference = "Continue"
$PROJECT_DIR = "C:\mft-algotrade-bot"

function Write-Test($message, $status) {
    $color = if ($status -eq "PASS") { "Green" } elseif ($status -eq "FAIL") { "Red" } else { "Yellow" }
    $symbol = if ($status -eq "PASS") { "[OK]" } elseif ($status -eq "FAIL") { "[FAIL]" } else { "[WARN]" }
    
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $color
    Write-Output "$symbol $message"
    $host.UI.RawUI.ForegroundColor = $fc
}

Write-Host ""
Write-Host "============================================================"
Write-Host "       STOIC CITADEL - INSTALLATION TEST SUITE"
Write-Host "============================================================"
Write-Host ""

Set-Location $PROJECT_DIR

# Test 1: Docker installed
Write-Host "[1/10] Testing Docker installation..."
try {
    $dockerVersion = docker --version
    Write-Test "Docker installed: $dockerVersion" "PASS"
} catch {
    Write-Test "Docker not found" "FAIL"
    exit 1
}

# Test 2: Docker Compose
Write-Host "[2/10] Testing Docker Compose..."
try {
    $composeVersion = docker-compose --version
    Write-Test "Docker Compose: $composeVersion" "PASS"
} catch {
    Write-Test "Docker Compose not found" "FAIL"
    exit 1
}

# Test 3: .env file exists
Write-Host "[3/10] Testing .env configuration..."
if (Test-Path ".env") {
    Write-Test ".env file exists" "PASS"
    
    # Check for default passwords
    $envContent = Get-Content ".env" -Raw
    if ($envContent -match "ChangeMe") {
        Write-Test "WARNING: Default passwords detected in .env" "WARN"
        Write-Host "  Run: .\stoic.ps1 generate-secrets"
    } else {
        Write-Test "Passwords configured" "PASS"
    }
} else {
    Write-Test ".env file missing" "FAIL"
    Write-Host "  Run: .\stoic.ps1 setup"
}

# Test 4: Required directories
Write-Host "[4/10] Testing directory structure..."
$requiredDirs = @("user_data", "user_data/logs", "user_data/data")
$dirTest = $true
foreach ($dir in $requiredDirs) {
    if (-not (Test-Path $dir)) {
        $dirTest = $false
        Write-Test "Missing directory: $dir" "FAIL"
    }
}
if ($dirTest) {
    Write-Test "Directory structure OK" "PASS"
}

# Test 5: Docker containers
Write-Host "[5/10] Testing Docker containers..."
$containers = @("stoic_freqtrade", "stoic_frequi", "stoic_postgres", "stoic_jupyter")
$runningCount = 0
foreach ($container in $containers) {
    $running = docker ps --filter "name=$container" --format "{{.Names}}" 2>$null
    if ($running) {
        $runningCount++
    }
}

if ($runningCount -eq $containers.Count) {
    Write-Test "All containers running ($runningCount/$($containers.Count))" "PASS"
} elseif ($runningCount -gt 0) {
    Write-Test "Some containers running ($runningCount/$($containers.Count))" "WARN"
} else {
    Write-Test "No containers running" "WARN"
    Write-Host "  Run: .\stoic.ps1 start"
}

# Test 6: FreqUI accessibility
Write-Host "[6/10] Testing FreqUI dashboard..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    Write-Test "FreqUI accessible on port 3000" "PASS"
} catch {
    Write-Test "FreqUI not accessible" "WARN"
    Write-Host "  Check: .\stoic.ps1 logs frequi"
}

# Test 7: Jupyter accessibility
Write-Host "[7/10] Testing Jupyter Lab..."
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8888" -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
    Write-Test "Jupyter Lab accessible on port 8888" "PASS"
} catch {
    Write-Test "Jupyter Lab not accessible" "WARN"
    Write-Host "  Check: .\stoic.ps1 logs jupyter"
}

# Test 8: PostgreSQL port
Write-Host "[8/10] Testing PostgreSQL..."
try {
    $tcpTest = Test-NetConnection -ComputerName localhost -Port 5433 -WarningAction SilentlyContinue
    if ($tcpTest.TcpTestSucceeded) {
        Write-Test "PostgreSQL listening on port 5433" "PASS"
    } else {
        Write-Test "PostgreSQL not responding on port 5433" "WARN"
    }
} catch {
    Write-Test "Cannot test PostgreSQL port" "WARN"
}

# Test 9: Historical data
Write-Host "[9/10] Testing historical data..."
if (Test-Path "user_data/data/binance") {
    $dataFiles = Get-ChildItem -Path "user_data/data/binance" -Filter "*.json" -Recurse
    if ($dataFiles.Count -gt 0) {
        Write-Test "Historical data present ($($dataFiles.Count) files)" "PASS"
    } else {
        Write-Test "No historical data found" "WARN"
        Write-Host "  Run: .\stoic.ps1 download-data"
    }
} else {
    Write-Test "Data directory missing" "WARN"
}

# Test 10: Strategies
Write-Host "[10/10] Testing strategies..."
if (Test-Path "user_data/strategies") {
    $strategies = Get-ChildItem -Path "user_data/strategies" -Filter "*.py"
    if ($strategies.Count -gt 0) {
        Write-Test "Strategies found ($($strategies.Count))" "PASS"
        foreach ($strat in $strategies | Select-Object -First 3) {
            Write-Host "  - $($strat.Name)"
        }
    } else {
        Write-Test "No strategies found" "FAIL"
    }
} else {
    Write-Test "Strategy directory missing" "FAIL"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "                    TEST SUMMARY"
Write-Host "============================================================"
Write-Host ""
Write-Host "If all tests passed, you're ready to trade!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Configure API keys in .env (if trading live)"
Write-Host "  2. Run: .\stoic.ps1 trade-dry"
Write-Host "  3. Open: .\stoic.ps1 dashboard"
Write-Host "  4. Monitor: .\stoic.ps1 health"
Write-Host ""
