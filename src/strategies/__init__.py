"""
Stoic Citadel - Strategies Module
==================================

Modular, testable trading strategies.

Each strategy follows a standard interface:
- populate_indicators(): Add technical indicators
- populate_entry_trend(): Define entry signals
- populate_exit_trend(): Define exit signals

Strategies use the utilities from src/utils for:
- Technical indicators
- Risk management
- Regime detection
"""

from .base_strategy import BaseStoicStrategy

__all__ = ["BaseStoicStrategy"]
