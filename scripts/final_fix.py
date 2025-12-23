import sys
from pathlib import Path

# Read original content
with open('temp_original.txt', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add sys.path insertion after the docstring
# Find the line after the docstring (the line with "import pickle")
lines = content.splitlines(keepends=True)
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    if line.strip() == '"""' and i > 0 and 'Version:' in lines[i-1]:
        # This is the end of the docstring
        # Insert sys.path after the next line (which should be empty line)
        # Actually we need to insert after the docstring closing line
        # Look ahead for the import line
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith('import'):
            new_lines.append(lines[j])
            j += 1
        # Now at the import line
        # Insert sys.path before the imports
        new_lines.append('import sys\n')
        new_lines.append('from pathlib import Path\n')
        new_lines.append('# Add project root to sys.path to allow imports of src modules\n')
        new_lines.append('sys.path.insert(0, str(Path(__file__).parent.parent.parent))\n')
        new_lines.append('\n')
        # Add the rest of the lines
        new_lines.extend(lines[j:])
        break
    i += 1

if len(new_lines) == len(lines):
    # Didn't find the pattern, fallback: replace the whole header
    content = content.replace(
        '"""\n\nimport pickle',
        '"""\n\nimport sys\nfrom pathlib import Path\n# Add project root to sys.path to allow imports of src modules\nsys.path.insert(0, str(Path(__file__).parent.parent.parent))\n\nimport pickle'
    )
else:
    content = ''.join(new_lines)

# 2. Replace populate_entry_trend method
old_method = '''    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions based on ML model predictions with dynamic threshold.
        
        Entry Logic:
        - Use dynamic probability threshold based on prediction distribution
        - Adaptive to current market conditions and signal density
        - Additional filters to improve signal quality
        
        Dynamic Threshold Calculation:
        - Use percentile-based threshold (e.g., 75th percentile of recent predictions)
        - Adjust based on market regime
        - Minimum threshold of 0.55 to ensure quality
        """
        
        # Calculate dynamic threshold based on recent predictions
        # Use adaptive threshold based on signal density
        # With 4.8% signal density, we need threshold around 0.05-0.10
        if len(dataframe) >= 100:
            recent_predictions = dataframe['ml_prediction'].tail(100)
            # Calculate signal density in recent predictions
            signal_density = (recent_predictions > 0.5).mean()
            # Dynamic threshold: 50th percentile for normal density, adjust for low density
            if signal_density > 0.1:  # Normal signal density (>10%)
                dynamic_threshold = np.percentile(recent_predictions, 75)
            elif signal_density > 0.01:  # Low signal density (1-10%)
                # Use lower percentile for low density
                dynamic_threshold = np.percentile(recent_predictions, 50)
            else:  # Very low signal density (<1%)
                # Use even lower threshold to generate some signals
                dynamic_threshold = np.percentile(recent_predictions, 25)
            
            # Apply reasonable bounds: 0.05 to 0.75
            dynamic_threshold = max(0.05, min(dynamic_threshold, 0.75))
        else:
            # Default threshold for initial data
            dynamic_threshold = 0.1
        
        # Adjust threshold based on regime
        if self._regime_mode == 'defensive':
            # Higher threshold in defensive mode (more conservative)
            dynamic_threshold = max(dynamic_threshold, 0.6)
        elif self._regime_mode == 'aggressive':
            # Lower threshold in aggressive mode (more opportunities)
            dynamic_threshold = max(0.5, dynamic_threshold * 0.9)
        
        # Base conditions - ML prediction probability > dynamic threshold
        base_conditions = [
            # ML model prediction probability with dynamic threshold
            (dataframe['ml_prediction'] > dynamic_threshold),
            
            # ML confidence threshold (if available)
            (dataframe['ml_confidence'] > 0.55),
            
            # Trend filter - only trade in uptrend
            (dataframe['close'] > dataframe['ema_200']),
            (dataframe['ema_50'] > dataframe['ema_100']),
            
            # Volume confirmation - avoid low liquidity
            (dataframe['volume_ratio'] > 0.7),
            
            # Not already overbought
            (dataframe['rsi'] < 70),
            
            # Volatility filter - avoid extreme volatility
            (dataframe['bb_width'] > self.buy_bb_width_min.value),
            (dataframe['bb_width'] < 0.2),
        ]
        
        # Regime-adjusted conditions
        if self._regime_mode == 'defensive':
            # In defensive mode, require stronger signals
            base_conditions.append(dataframe['adx'] > 25)
            base_conditions.append(dataframe['rsi'] < 50)
            base_conditions.append(dataframe['ml_confidence'] > 0.65)
        elif self._regime_mode == 'aggressive':
            # In aggressive mode, relax some conditions
            base_conditions.append(dataframe['volume_ratio'] > 0.4)
        
        # Combine conditions
        if base_conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, base_conditions),
                'enter_long'
            ] = 1
        
        # Log entry signals with dynamic threshold
        entry_count = dataframe['enter_long'].sum()
        if entry_count > 0 or len(dataframe) % 100 == 0:
            last_row = dataframe.iloc[-1]
            logger.info(
                f"📊 {metadata['pair']}: Dynamic threshold={dynamic_threshold:.3f}, "
                f"Signals={entry_count}, "
                f"ML prob: {last_row['ml_prediction']:.3f}, "
                f"ML conf: {last_row['ml_confidence']:.3f}, "
                f"Regime: {self._regime_mode}"
            )
        
        return dataframe'''

new_method = '''    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Define entry conditions based solely on ML model probability.
        
        Entry Logic:
        - Buy when ML prediction probability > 0.55
        - Simple threshold to generate signals
        """
        
        # Debug: print max probability
        max_prob = dataframe['ml_prediction'].max() if 'ml_prediction' in dataframe.columns else 0
        print(f"Max ML Probability for {metadata['pair']}: {max_prob:.3f}")
        
        # Simple condition: ML prediction > 0.55
        dataframe.loc[
            dataframe['ml_prediction'] > 0.55,
            'enter_long'
        ] = 1
        
        # Log entry signals
        entry_count = dataframe['enter_long'].sum()
        if entry_count > 0:
            logger.info(
                f"📊 {metadata['pair']}: {entry_count} entry signals "
                f"(max prob: {max_prob:.3f})"
            )
        
        return dataframe'''

# Replace the old method with new method
content = content.replace(old_method, new_method)

# Write to the actual file
with open('user_data/strategies/StoicEnsembleStrategyV4.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Strategy file updated successfully.")
