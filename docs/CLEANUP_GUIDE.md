# Audit and Cleanup Guide

This document provides recommendations for cleaning up old strategies and unused features.

## Old Strategies

The following strategy files are likely outdated and can be removed:

- `user_data/strategies/StoicEnsembleStrategyV4.py`

Before removing, please ensure that there are no active backtests or production deployments that rely on this version.

## Unused Features

Identifying unused features is a more complex task. Here is a recommended process:

1.  **Run Feature Importance Analysis:** Use the `scripts/analyze_feature_importance.py` script to generate a feature importance report. This will give you a good idea of which features are not contributing to the model's predictions.

2.  **Analyze Strategy Code:** Manually review the strategy code (e.g., `user_data/strategies/StoicEnsembleStrategyV5.py` and `src/strategies/core_logic.py`) to identify which features are used in the `populate_indicators` and `populate_entry_exit_signals` methods.

3.  **Cross-reference with Model:** Compare the list of features used in the strategy with the list of features used by the trained models. The model files (e.g., in `user_data/models`) should contain information about the features they were trained on.

4.  **Create a List of Unused Features:** Based on the analysis above, create a list of features that are not used by either the strategy or the models.

5.  **Remove Unused Features:** Remove the unused features from the feature engineering pipeline (`src/ml/training/feature_engineering.py`) and the strategy code.

By following this process, you can safely remove unused features and simplify the codebase.
