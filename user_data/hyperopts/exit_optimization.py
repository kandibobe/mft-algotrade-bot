"""
Hyperopt for exit optimization only.
This file inherits from StoicEnsembleStrategyV4 but overrides parameters
to fix the DecimalParameter issue.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from user_data.strategies.StoicEnsembleStrategyV4 import StoicEnsembleStrategyV4
from freqtrade.strategy import DecimalParameter, CategoricalParameter


class ExitOptimizationHyperopt(StoicEnsembleStrategyV4):
    """
    Hyperopt class for exit optimization only.
    Inherits from StoicEnsembleStrategyV4 but overrides parameters
    to work with Freqtrade's hyperopt.
    """
    
    # Override stoploss to be a float for hyperopt compatibility
    # We'll keep it as DecimalParameter but with a default value that can be converted to float
    stoploss = -0.05  # Default value as float
    
    # These parameters will be optimized
    hyperopt_stoploss = DecimalParameter(-0.10, -0.01, default=-0.05, space="stoploss")
    hyperopt_trailing_stop_positive = DecimalParameter(0.005, 0.03, default=0.01, space="trailing")
    hyperopt_trailing_stop_positive_offset = DecimalParameter(0.01, 0.05, default=0.02, space="trailing")
    hyperopt_exit_profit_only = CategoricalParameter([True, False], default=False, space="roi")
    hyperopt_ignore_roi_if_entry_signal = CategoricalParameter([True, False], default=False, space="roi")
    
    def __init__(self, config: dict) -> None:
        # Call parent init
        super().__init__(config)
        
        # Map hyperopt parameters to actual strategy parameters
        self.stoploss = self.hyperopt_stoploss
        self.trailing_stop_positive = self.hyperopt_trailing_stop_positive
        self.trailing_stop_positive_offset = self.hyperopt_trailing_stop_positive_offset
        self.exit_profit_only = self.hyperopt_exit_profit_only
        self.ignore_roi_if_entry_signal = self.hyperopt_ignore_roi_if_entry_signal
