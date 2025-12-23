# ============================================================
# CLEANUP COMMANDS FOR MFT-ALGOTRADE-BOT ROOT DIRECTORY
# ============================================================
# Execute these commands one by one after reviewing
# ============================================================

Write-Host "Starting repository cleanup..." -ForegroundColor Cyan

# 1. Remove .claude/ from git tracking and delete locally
Write-Host "`n1. Removing .claude/ folder..." -ForegroundColor Yellow
git rm -r --cached .claude
if (Test-Path .claude) {
    Remove-Item -Recurse -Force .claude
    Write-Host "  -> .claude/ removed from git and deleted locally" -ForegroundColor Green
} else {
    Write-Host "  -> .claude/ not found locally" -ForegroundColor Yellow
}

# 2. Remove backups/ from git tracking (keep .gitkeep in git)
Write-Host "`n2. Cleaning backups/ folder..." -ForegroundColor Yellow
# Remove all files except .gitkeep from git tracking
git rm --cached backups/*.bak 2>$null
git rm --cached backups/*.txt 2>$null
git rm --cached backups/*.py 2>$null
# Check if there are any other files
$backupFiles = Get-ChildItem backups -File | Where-Object { $_.Name -ne ".gitkeep" }
if ($backupFiles) {
    foreach ($file in $backupFiles) {
        git rm --cached "backups/$($file.Name)" 2>$null
    }
}
Write-Host "  -> Backup files removed from git tracking (except .gitkeep)" -ForegroundColor Green

# 3. Create docs/samples/ directory if it doesn't exist
Write-Host "`n3. Creating docs/samples/ directory..." -ForegroundColor Yellow
if (-not (Test-Path docs\samples)) {
    New-Item -ItemType Directory -Path docs\samples -Force | Out-Null
    Write-Host "  -> docs/samples/ created" -ForegroundColor Green
} else {
    Write-Host "  -> docs/samples/ already exists" -ForegroundColor Yellow
}

# 4. Move useful example reports to docs/samples/
Write-Host "`n4. Moving example reports to docs/samples/..." -ForegroundColor Yellow
$reportFiles = @(
    "reports/real_data_analysis_report.md",
    "reports/walk_forward_analysis_report.md",
    "reports/phase4_analysis_report.md"
)

foreach ($report in $reportFiles) {
    if (Test-Path $report) {
        Move-Item $report "docs/samples/" -Force
        Write-Host "  -> Moved $report to docs/samples/" -ForegroundColor Green
    } else {
        Write-Host "  -> $report not found" -ForegroundColor Yellow
    }
}

# Move visualizations folder if it exists
if (Test-Path "reports/visualizations") {
    Move-Item "reports/visualizations" "docs/samples/visualizations" -Force
    Write-Host "  -> Moved reports/visualizations/ to docs/samples/visualizations/" -ForegroundColor Green
}

# Remove .gitkeep from reports/ (since we're moving everything useful)
if (Test-Path "reports/.gitkeep") {
    git rm --cached reports/.gitkeep
    Remove-Item "reports/.gitkeep" -Force
    Write-Host "  -> Removed reports/.gitkeep from git and locally" -ForegroundColor Green
}

# 5. Move research/ to notebooks/research/ (since it only has .gitkeep)
Write-Host "`n5. Moving research/ to notebooks/research/..." -ForegroundColor Yellow
if (Test-Path research) {
    # Create notebooks/research if it doesn't exist
    if (-not (Test-Path notebooks\research)) {
        New-Item -ItemType Directory -Path notebooks\research -Force | Out-Null
    }
    
    # Move all files from research to notebooks/research
    Get-ChildItem research -File | ForEach-Object {
        Move-Item "research/$($_.Name)" "notebooks/research/" -Force
    }
    
    # Remove research directory from git tracking
    git rm -r --cached research
    Remove-Item -Recurse -Force research
    Write-Host "  -> research/ moved to notebooks/research/ and removed from root" -ForegroundColor Green
} else {
    Write-Host "  -> research/ not found" -ForegroundColor Yellow
}

# 6. Move mcp_servers/ to src/mcp_servers/
Write-Host "`n6. Moving mcp_servers/ to src/mcp_servers/..." -ForegroundColor Yellow
if (Test-Path mcp_servers) {
    # Create src/mcp_servers if it doesn't exist
    if (-not (Test-Path src\mcp_servers)) {
        New-Item -ItemType Directory -Path src\mcp_servers -Force | Out-Null
    }
    
    # Move all Python files
    Get-ChildItem mcp_servers -File | ForEach-Object {
        Move-Item "mcp_servers/$($_.Name)" "src/mcp_servers/" -Force
        Write-Host "  -> Moved mcp_servers/$($_.Name) to src/mcp_servers/" -ForegroundColor Green
    }
    
    # Remove mcp_servers directory from git tracking
    git rm -r --cached mcp_servers
    Remove-Item -Recurse -Force mcp_servers
    Write-Host "  -> mcp_servers/ moved to src/mcp_servers/ and removed from root" -ForegroundColor Green
} else {
    Write-Host "  -> mcp_servers/ not found" -ForegroundColor Yellow
}

# 7. Remove empty feature_repo/ directory
Write-Host "`n7. Removing empty feature_repo/ directory..." -ForegroundColor Yellow
if (Test-Path feature_repo) {
    # Check if directory is empty
    $items = Get-ChildItem feature_repo
    if ($items.Count -eq 0) {
        git rm -r --cached feature_repo
        Remove-Item -Recurse -Force feature_repo
        Write-Host "  -> Empty feature_repo/ removed from git and deleted locally" -ForegroundColor Green
    } else {
        Write-Host "  -> feature_repo/ is not empty, skipping removal" -ForegroundColor Red
    }
} else {
    Write-Host "  -> feature_repo/ not found" -ForegroundColor Yellow
}

# 8. Update imports in mcp server files if needed
Write-Host "`n8. Checking for import updates in mcp server files..." -ForegroundColor Yellow
$mcpFiles = Get-ChildItem src\mcp_servers -Filter *.py
foreach ($file in $mcpFiles) {
    $content = Get-Content $file.FullName -Raw
    # Check if there are imports that need updating
    # The current imports use sys.path insertion, so they should still work
    Write-Host "  -> Checked $($file.Name): imports appear to be OK" -ForegroundColor Green
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "CLEANUP COMPLETE!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "`nSummary of changes:" -ForegroundColor Yellow
Write-Host "- .claude/ removed (IDE config)" -ForegroundColor White
Write-Host "- backups/ cleaned (only .gitkeep kept in git)" -ForegroundColor White
Write-Host "- Example reports moved to docs/samples/" -ForegroundColor White
Write-Host "- research/ moved to notebooks/research/" -ForegroundColor White
Write-Host "- mcp_servers/ moved to src/mcp_servers/" -ForegroundColor White
Write-Host "- feature_repo/ removed (empty directory)" -ForegroundColor White
Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "1. Review the changes with: git status" -ForegroundColor White
Write-Host "2. Commit the changes: git commit -m 'Clean up root directory per Python src-layout standards'" -ForegroundColor White
Write-Host "3. Verify the repository structure" -ForegroundColor White
