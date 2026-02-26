for fname in ['train.csv', 'val.csv']:
    path = f'/workspace/data/processed/{fname}'
    lines = open(path).readlines()
    fixed = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            fixed.append(parts[0] + '|' + parts[1] + '|' + parts[1] + '\n')
    open(path, 'w').writelines(fixed)
    print(f'{fname}: fixed {len(fixed)} lines')