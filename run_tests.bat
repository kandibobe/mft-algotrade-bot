@echo off
REM Stoic Citadel - Test Runner
REM Run from worktree directory

echo Running Order Management tests...
pytest tests/test_order_manager/ -v --tb=short

echo.
echo Running with coverage...
pytest tests/test_order_manager/ --cov=src.order_manager --cov-report=term-missing

pause
