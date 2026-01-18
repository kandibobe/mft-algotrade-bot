import json
import threading
import time
import unittest
from pathlib import Path

from src.config.unified_config import ConfigWatcher, load_config


class TestConfigWatcherStress(unittest.TestCase):
    def setUp(self):
        self.test_config_path = Path("tests/stress_test_config.json")
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
        self.reload_count = 0
        self.lock = threading.Lock()

    def tearDown(self):
        if hasattr(self, 'watcher'):
            self.watcher.stop()
        if self.test_config_path.exists():
            self.test_config_path.unlink()

    def test_rapid_modifications(self):
        """Test how the watcher handles rapid fire file changes."""
        def on_reload(config):
            with self.lock:
                self.reload_count += 1

        self.watcher.add_callback(on_reload)
        self.watcher.start()

        time.sleep(0.5)

        # Bombard the file with 10 changes in quick succession
        iterations = 10
        for i in range(iterations):
            updated_config = self.initial_config.copy()
            updated_config["stake_amount"] = 100.0 + i
            with open(self.test_config_path, "w") as f:
                json.dump(updated_config, f)
                f.flush()
            # Very short sleep to trigger multiple events
            time.sleep(0.05)

        # Wait for potential reloads to settle (considering debounce)
        time.sleep(2.0)

        with self.lock:
            print(f"\nTotal reloads detected: {self.reload_count}")
            # We expect at least 1 reload, but significantly less than 'iterations' due to debounce (1s)
            self.assertTrue(self.reload_count > 0, "No reloads detected")
            self.assertTrue(self.reload_count < iterations, "Debounce mechanism failed to group rapid changes")

if __name__ == "__main__":
    unittest.main()
