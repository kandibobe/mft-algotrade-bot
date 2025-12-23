import sys
from pathlib import Path

# Read original file
file_path = Path("user_data/strategies/StoicEnsembleStrategyV4.py")
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add sys.path insertion after the docstring
# Find the line after the docstring (the line with "import pickle")
lines = content.splitlines(keepends=True)
new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    new_lines.append(line)
    if line.strip() == '"""' and i > 0 and lines[i-1].strip().startswith('Version:'):
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
# Find the method definition
start_marker = '    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:'
end_marker = '    # ==========================================================================\n    # EXIT LOGIC'

# New simplified method
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
        
        return dataframe
'''

# Split content into lines again
lines = content.splitlines(keepends=True)
new_content_lines = []
i = 0
while i < len(lines):
    if lines[i].strip() == start_marker.strip():
        # Found the method start
        # Skip until we find the next method at same indentation level
        new_content_lines.append(new_method)
        # Skip the old method lines
        i += 1
        while i < len(lines) and not (lines[i].strip().startswith('    def ') and lines[i].strip().endswith(':')):
            i += 1
        # Now i points to the next method definition (or end of file)
        # Continue adding lines
        continue
    new_content_lines.append(lines[i])
    i += 1

content = ''.join(new_content_lines)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Strategy file updated successfully.")
