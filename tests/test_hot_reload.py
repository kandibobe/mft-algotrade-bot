import json
import time
import unittest
from pathlib import Path

from src.config.unified_config import ConfigWatcher, load_config


class TestHotReload(unittest.TestCase):
    def setUp(self):
        self.test_config_path = Path("tests/test_hot_reload_config.json")
        self.initial_config = {
            "dry_run": True,
            "pairs": ["BTC/USDT"],
            "timeframe": "1h",
            "stake_amount": 100.0,
            "max_open_trades": 3
        }
        with open(self.test_config_path, "w") as f:
            json.dump(self.initial_config, f)

        self.config = load_config(str(self.test_config_path))
        self.watcher = ConfigWatcher(self.config, str(self.test_config_path))
        self.reload_called = False

    def tearDown(self):
        if hasattr(self, 'watcher'):
            self.watcher.stop()
        if self.test_config_path.exists():
            self.test_config_path.unlink()

    def test_manual_reload(self):
        # Change file
        updated_config = self.initial_config.copy()
        updated_config["stake_amount"] = 200.0
        with open(self.test_config_path, "w") as f:
            json.dump(updated_config, f)

        # Manual reload
        self.config.reload(str(self.test_config_path))
        self.assertEqual(self.config.stake_amount, 200.0)

    def test_watcher_reload(self):
        def on_reload(config):
            self.reload_called = True

        self.watcher.add_callback(on_reload)
        self.watcher.start()

        # Give observer time to start
        time.sleep(0.5)

        # Change file
        updated_config = self.initial_config.copy()
        updated_config["max_open_trades"] = 5
        with open(self.test_config_path, "w") as f:
            json.dump(updated_config, f)
            f.flush()

        # Give observer time to detect change and reload
        # Watchdog can be slow on some systems, wait up to 2 seconds
        for _ in range(20):
            if self.reload_called:
                break
            time.sleep(0.1)

        self.assertTrue(self.reload_called, "Reload callback was not called")
        self.assertEqual(self.config.max_open_trades, 5)

if __name__ == "__main__":
    unittest.main()
