"""
Integration Tests for Configuration Manager
"""
import json
import os

import pytest

from src.config.manager import ConfigurationManager
from src.config.unified_config import TradingConfig


class TestConfigIntegration:

    @pytest.fixture
    def valid_config_file(self, tmp_path):
        """Create a temporary valid config file."""
        config_data = {
            "max_open_trades": 5,
            "stake_currency": "USDT",
            "stake_amount": 100.0,
            "dry_run": True,
            "pairs": ["BTC/USDT", "ETH/USDT"],
            "exchange": {
                "name": "binance",
                "api_key": "test_key",
                "api_secret": "test_secret",
                "sandbox": True
            },
            "timeframe": "5m"
        }

        file_path = tmp_path / "valid_config.json"
        with open(file_path, "w") as f:
            json.dump(config_data, f)
        return str(file_path)

    def test_load_valid_config(self, valid_config_file):
        """Test loading a valid configuration file."""
        # Reset singleton
        ConfigurationManager._instance = None

        config = ConfigurationManager.initialize(valid_config_file)

        assert isinstance(config, TradingConfig)
        assert config.max_open_trades == 5
        assert config.stake_currency == "USDT"
        assert config.exchange.name == "binance"
        assert len(config.pairs) == 2

    def test_export_freqtrade_config(self, valid_config_file, tmp_path):
        """Test exporting config to Freqtrade format."""
        ConfigurationManager._instance = None
        ConfigurationManager.initialize(valid_config_file)

        output_path = tmp_path / "freqtrade_config.json"
        ConfigurationManager.export_freqtrade_config(str(output_path))

        assert os.path.exists(output_path)

        with open(output_path) as f:
            exported_data = json.load(f)

        assert exported_data["max_open_trades"] == 5
        assert exported_data["exchange"]["name"] == "binance"
        # Check standard freqtrade structure
        assert "entry_pricing" in exported_data
        assert "pairlists" in exported_data
