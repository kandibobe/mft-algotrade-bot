import sys

# Read original file
with open('temp_original2.txt', 'r', encoding='utf-8') as f:
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
# Find the method definition
old_method_start = '    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:'
old_method_end = '        return dataframe'

# Split content into lines
lines = content.splitlines(keepends=True)
new_content_lines = []
i = 0
while i < len(lines):
    if lines[i].strip() == old_method_start.strip():
        # Found the method start
        # Skip until we find a line that is just '        return dataframe' at same indentation
        # Actually we need to skip the entire method
        # We'll replace with new method
        new_content_lines.append('    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:\n')
        new_content_lines.append('        """\n')
        new_content_lines.append('        Define entry conditions based solely on ML model probability.\n')
        new_content_lines.append('        \n')
        new_content_lines.append('        Entry Logic:\n')
        new_content_lines.append('        - Buy when ML prediction probability > 0.55\n')
        new_content_lines.append('        - Simple threshold to generate signals\n')
        new_content_lines.append('        """\n')
        new_content_lines.append('        \n')
        new_content_lines.append('        # Debug: print max probability\n')
        new_content_lines.append('        max_prob = dataframe[\'ml_prediction\'].max() if \'ml_prediction\' in dataframe.columns else 0\n')
        new_content_lines.append('        print(f"Max ML Probability for {metadata[\'pair\']}: {max_prob:.3f}")\n')
        new_content_lines.append('        \n')
        new_content_lines.append('        # Simple condition: ML prediction > 0.55\n')
        new_content_lines.append('        dataframe.loc[\n')
        new_content_lines.append('            dataframe[\'ml_prediction\'] > 0.55,\n')
        new_content_lines.append('            \'enter_long\'\n')
        new_content_lines.append('        ] = 1\n')
        new_content_lines.append('        \n')
        new_content_lines.append('        # Log entry signals\n')
        new_content_lines.append('        entry_count = dataframe[\'enter_long\'].sum()\n')
        new_content_lines.append('        if entry_count > 0:\n')
        new_content_lines.append('            logger.info(\n')
        new_content_lines.append('                f"📊 {metadata[\'pair\']}: {entry_count} entry signals "\n')
        new_content_lines.append('                f"(max prob: {max_prob:.3f})"\n')
        new_content_lines.append('            )\n')
        new_content_lines.append('        \n')
        new_content_lines.append('        return dataframe\n')
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

# Write to the actual file
with open('user_data/strategies/StoicEnsembleStrategyV4.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Strategy file updated successfully.")
