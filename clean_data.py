import re

for fname in ['train.csv', 'val.csv']:
    path = f'/workspace/data/processed/{fname}'
    lines = open(path).readlines()
    cleaned = []
    removed = 0
    for line in lines:
        parts = line.split('|')
        if len(parts) > 1:
            text = parts[1]
            if '_letter' in text or '\u200c' in text or '\u200d' in text:
                removed += 1
                continue
        cleaned.append(line)
    open(path, 'w').writelines(cleaned)
    print(f'{fname}: removed {removed}, kept {len(cleaned)}')