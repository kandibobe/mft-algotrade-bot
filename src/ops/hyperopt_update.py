"""
Hyperopt Automation Utility
===========================

Automates the update of configuration files with hyperopt results.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def update_config_from_hyperopt(hyperopt_results_path: str, target_config_path: str):
    """
    Read hyperopt results and update the target configuration file.

    Args:
        hyperopt_results_path: Path to freqtrade hyperopt results JSON or MD
        target_config_path: Path to the configuration file to update
    """
    results_path = Path(hyperopt_results_path)
    config_path = Path(target_config_path)

    if not results_path.exists():
        logger.error(f"Hyperopt results not found at {hyperopt_results_path}")
        return

    if not config_path.exists():
        logger.error(f"Target config not found at {target_config_path}")
        return

    try:
        # Freqtrade hyperopt results are usually in a specific format
        # For simplicity, we assume we're reading a JSON export or a simplified format
        with open(results_path) as f:
            if results_path.suffix == ".json":
                results = json.load(f)
            else:
                # Placeholder for parsing MD or other formats
                logger.warning(
                    "Only JSON hyperopt results are supported for automatic updates currently."
                )
                return

        # Load existing config
        with open(config_path) as f:
            config_data = json.load(f)

        # Update strategy parameters
        if "params" in results:
            strategy_params = results["params"]
            if "strategy" not in config_data:
                config_data["strategy"] = {}

            for key, value in strategy_params.items():
                config_data["strategy"][key] = value
                logger.info(f"Updated {key} to {value}")

        # Save updated config
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Successfully updated config at {target_config_path}")

    except Exception as e:
        logger.error(f"Failed to update config from hyperopt: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.ops.hyperopt_update <results_json> <target_config_json>")
    else:
        update_config_from_hyperopt(sys.argv[1], sys.argv[2])
