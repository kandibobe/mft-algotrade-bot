
"""
Robustness tests for Feature Engineering Pipeline
Focusing on edge cases like duplicate indices, duplicate columns, and outliers.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.ml.training.feature_engineering import FeatureEngineer

# Configure logging to see output during tests
logging.basicConfig(level=logging.INFO)

class TestFeatureEngineerRobustness:
    """Test FeatureEngineer robustness against dirty data."""

    def test_duplicate_indices_outlier_check(self):
        """Test validation with duplicate indices (regression test for crash)."""
        # Create data with duplicate indices
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        # Duplicate the middle part
        dates_duplicated = dates.append(dates[50:60])

        # Ensure we have enough rows for rolling windows if needed,
        # but here we test validation logic mostly.

        df = pd.DataFrame({
            'open': np.random.rand(110),
            'high': np.random.rand(110),
            'low': np.random.rand(110),
            'close': np.random.rand(110),
            'volume': np.random.rand(110),
            'feature_1': np.random.randn(110) # Normal dist
        }, index=dates_duplicated)

        # Add outliers (needs > 1% to be flagged)
        # 110 rows -> 1.1 outlier is 1%. So need 2 outliers.
        df.iloc[0, df.columns.get_loc('feature_1')] = 100.0
        df.iloc[1, df.columns.get_loc('feature_1')] = -100.0

        fe = FeatureEngineer()

        # This should NOT crash
        is_valid, issues = fe.validate_features(df, fix_issues=True, raise_on_error=False)

        # Should detect the outlier
        outlier_cols = [item['column'] for item in issues['outlier_columns']]
        assert 'feature_1' in outlier_cols

        # Should have clipped the value if fix_issues=True
        # Check if the max/min value is now reasonable (less than 100 magnitude)
        assert df['feature_1'].max() < 90.0
        assert df['feature_1'].min() > -90.0

    def test_duplicate_columns_outlier_check(self):
        """Test validation with duplicate columns."""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'open': np.random.rand(100),
            'high': np.random.rand(100),
            'low': np.random.rand(100),
            'close': np.random.rand(100),
            'volume': np.random.rand(100),
            'feature_1': np.random.randn(100)
        }, index=dates)

        # Force duplicate column
        df.insert(5, 'feature_1', np.random.randn(100), allow_duplicates=True)

        # Add outliers to 'feature_1' column (needs > 1% to trigger)
        # Get integer index of first 'feature_1' column
        feat_locs = np.where(df.columns == 'feature_1')[0]
        first_feat_idx = feat_locs[0]

        df.iloc[0, first_feat_idx] = 100.0
        df.iloc[1, first_feat_idx] = 100.0

        fe = FeatureEngineer()

        # This should NOT crash
        is_valid, issues = fe.validate_features(df, fix_issues=True, raise_on_error=False)

        # Should detect outlier in 'feature_1'
        outlier_cols = [item['column'] for item in issues['outlier_columns']]
        assert 'feature_1' in outlier_cols

    def test_outlier_detection_logic(self):
        """Verify outlier logic counts correctly."""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='h')
        df = pd.DataFrame({
            'open': np.random.rand(200),
            'high': np.random.rand(200),
            'low': np.random.rand(200),
            'close': np.random.rand(200),
            'volume': np.random.rand(200),
            'test_feat': np.random.randn(200) * 1.0 # std=1
        }, index=dates)

        # Add 3 outliers (> 5 std)
        # Mean is approx 0, std approx 1. So > 5 is outlier.
        df.iloc[0, df.columns.get_loc('test_feat')] = 10.0
        df.iloc[1, df.columns.get_loc('test_feat')] = -10.0
        df.iloc[2, df.columns.get_loc('test_feat')] = 10.0

        fe = FeatureEngineer()
        is_valid, issues = fe.validate_features(df, fix_issues=False, raise_on_error=False)

        # Find the issue for test_feat
        issue = next((i for i in issues['outlier_columns'] if i['column'] == 'test_feat'), None)
        assert issue is not None
        assert issue['count'] == 3

    def test_fit_transform_crash_check(self):
        """End-to-end crash check with duplicate indices."""
        dates = pd.date_range(start='2023-01-01', periods=300, freq='h')
        # Duplicate some indices
        dates_duplicated = dates.append(dates[100:150])

        df = pd.DataFrame({
            'open': np.random.uniform(100, 200, 350),
            'high': np.random.uniform(100, 200, 350),
            'low': np.random.uniform(100, 200, 350),
            'close': np.random.uniform(100, 200, 350),
            'volume': np.random.uniform(1000, 5000, 350),
        }, index=dates_duplicated)

        fe = FeatureEngineer()

        # Should run without crashing
        try:
            result = fe.fit_transform(df)
            assert not result.empty
        except Exception as e:
            pytest.fail(f"fit_transform crashed with duplicate indices: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
