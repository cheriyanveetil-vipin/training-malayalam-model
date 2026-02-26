import json

# Extract all unique chars from your dataset
chars = set()
with open('/workspace/data/processed/train.csv') as f:
    for line in f:
        parts = line.strip().split('|')
        if len(parts) > 1:
            for ch in parts[1]:
                chars.add(ch)

chars = ''.join(sorted(chars))
print('Found chars:', chars)

c = json.load(open('/workspace/config.json'))
c['characters'] = {
    'pad': '<PAD>',
    'eos': '<EOS>',
    'bos': '<BOS>',
    'blank': '<BLNK>',
    'characters': chars,
    'punctuations': '!,.?- ',
    'phonemes': None
}
json.dump(c, open('/workspace/config.json','w'), indent=2, ensure_ascii=False)
print('Done')