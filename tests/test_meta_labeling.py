import unittest
import pandas as pd
import numpy as np
from src.ml.meta_labeling import MetaLabeler

class TestMetaLabeling(unittest.TestCase):
    def test_create_meta_labels(self):
        # Create dummy trade results
        trades_data = {
            'return_pct': [0.02, -0.01, 0.05, -0.02, 0.001],
            'barrier_hit': ['take_profit', 'stop_loss', 'take_profit', 'stop_loss', 'time_barrier']
        }
        df = pd.DataFrame(trades_data)
        
        meta_labels = MetaLabeler.create_meta_labels(df)
        
        # Check expected labels (1 for TP, 0 otherwise)
        expected = [1, 0, 1, 0, 0]
        self.assertEqual(meta_labels.tolist(), expected)

    def test_filter_signals(self):
        signals = pd.Series([1, 1, 1, 1, 1])
        # Probabilities from meta-model
        meta_probs = pd.Series([0.9, 0.2, 0.8, 0.4, 0.6])
        
        # Filter with threshold 0.5
        filtered = MetaLabeler.filter_signals(signals, meta_probs, threshold=0.5)
        
        expected = [1, 0, 1, 0, 1]
        self.assertEqual(filtered.tolist(), expected)

if __name__ == "__main__":
    unittest.main()
