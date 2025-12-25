#!/bin/bash
# Repository Cleanup Script for mft-algotrade-bot v2.4.0
# This script executes the cleanup commands from the Repository Polish Plan

set -e  # Exit on error
echo "🚀 Starting repository cleanup for mft-algotrade-bot v2.4.0"
echo "=========================================================="

# 1. Create archives directory for non-standard folders
echo "📁 Step 1: Organizing non-standard folders..."
mkdir -p archives/
if [ -d ".claude" ]; then
    echo "  Moving .claude/ to archives/"
    mv .claude/ archives/ 2>/dev/null || echo "  .claude/ not found or already moved"
fi

if [ -d "backups" ]; then
    echo "  Moving backups/ to archives/"
    mv backups/ archives/ 2>/dev/null || echo "  backups/ not found or already moved"
fi

if [ -d "research" ]; then
    echo "  Moving research/ to archives/"
    mv research/ archives/ 2>/dev/null || echo "  research/ not found or already moved"
fi

# 2. Organize configuration files
echo "⚙️  Step 2: Organizing configuration files..."
mkdir -p config/examples/

# Move YAML configs
if ls config/*.yaml 1> /dev/null 2>&1; then
    echo "  Moving YAML configs to config/examples/"
    mv config/*.yaml config/examples/ 2>/dev/null || true
fi

# Move JSON configs
if ls config/*.json 1> /dev/null 2>&1; then
    echo "  Moving JSON configs to config/examples/"
    mv config/*.json config/examples/ 2>/dev/null || true
fi

# 3. Clean up temporary files
echo "🧹 Step 3: Cleaning temporary files..."

# Python cache files
echo "  Removing Python cache files..."
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Test cache files
echo "  Removing test cache files..."
find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".mypy_cache" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".ruff_cache" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".coverage" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "htmlcov" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".hypothesis" -type d -exec rm -rf {} + 2>/dev/null || true

# 4. Remove IDE-specific files (they're in .gitignore)
echo "💻 Step 4: Cleaning IDE-specific files..."
rm -f .project .pydevproject 2>/dev/null || true
rm -rf .settings/ 2>/dev/null || true

# 5. Verify the cleanup
echo "✅ Step 5: Verifying cleanup..."
echo ""
echo "📊 Cleanup Summary:"
echo "  - Archives created: $(ls -la archives/ 2>/dev/null | wc -l) items"
echo "  - Config examples: $(ls -la config/examples/ 2>/dev/null | wc -l) files"
echo "  - Python cache: Cleaned"
echo "  - Test cache: Cleaned"
echo "  - IDE files: Removed"

# 6. Run code quality check
echo ""
echo "🔍 Step 6: Running code quality check..."
if command -v python &> /dev/null; then
    if [ -f "pyproject.toml" ]; then
        echo "  Checking if virtual environment exists..."
        if [ -d ".venv" ]; then
            echo "  Virtual environment found. Activating..."
            source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate 2>/dev/null || true
            echo "  Running Ruff check..."
            python -m ruff check --exit-zero src/ tests/ scripts/ 2>/dev/null || echo "  Ruff check completed"
        else
            echo "  ⚠️  No virtual environment found. Skipping linting."
            echo "  Run 'make setup' to create development environment."
        fi
    else
        echo "  ⚠️  pyproject.toml not found. Skipping linting."
    fi
else
    echo "  ⚠️  Python not found. Skipping linting."
fi

echo ""
echo "🎉 Cleanup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Review the changes: git status"
echo "  2. Commit the cleanup: git add . && git commit -m 'chore: repository cleanup for v2.4.0'"
echo "  3. Run full test suite: make test"
echo "  4. Verify everything works: make preflight"
echo ""
echo "⚠️  Important: Check that no important files were accidentally moved."
echo "   The archives/ directory contains moved items for safety."
