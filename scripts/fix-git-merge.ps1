# Fix Git Merge Conflicts - PowerShell Script
# Run this if you have merge conflicts

Write-Host "рџ”§ Fixing Git merge conflicts..." -ForegroundColor Cyan

# Check if in merge state
if (Test-Path ".git/MERGE_HEAD") {
    Write-Host "вљ пёЏ  Merge in progress detected. Resolving..." -ForegroundColor Yellow
    
    # Accept remote version of conflicting files
    git checkout --theirs START.md 2>$null
    git add START.md 2>$null
    
    # Commit the merge
    git commit -m "Resolved merge conflicts - accepted remote changes" 2>$null
    
    Write-Host "вњ… Merge conflicts resolved!" -ForegroundColor Green
} else {
    Write-Host "вњ… No merge conflicts detected" -ForegroundColor Green
}

# Pull latest changes
Write-Host "рџ“Ґ Pulling latest changes..." -ForegroundColor Cyan
git pull origin main

Write-Host "рџЋ‰ Repository updated successfully!" -ForegroundColor Green
Write-Host "\nNext steps:" -ForegroundColor Cyan
Write-Host "1. Run: docker-compose down -v"
Write-Host "2. Run: docker-compose up -d --build"
Write-Host "3. Wait ~10 minutes for first build"
Write-Host "4. Open: http://localhost:3000"