import os
import pandas as pd
import librosa
import soundfile as sf
RAW_DIR = "/workspace/data/raw"
OUT_WAVS = "/workspace/data/processed/wavs"
OUT_META = "/workspace/data/processed/metadata.csv"
os.makedirs(OUT_WAVS, exist_ok=True)
rows = []
for tsv_file in ["line_index_female.tsv", "line_index_male.tsv"]:
    tsv_path = os.path.join(RAW_DIR, tsv_file)
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["filename", "text"])
    print("Processing " + tsv_file + ": " + str(len(df)) + " samples")
    for _, row in df.iterrows():
        src = os.path.join(RAW_DIR, row["filename"] + ".wav")
        if not os.path.exists(src):
            continue
        y, _ = librosa.load(src, sr=22050, mono=True)
        duration = len(y) / 22050
        if duration < 1.0 or duration > 15.0:
            continue
        dst = os.path.join(OUT_WAVS, row["filename"] + ".wav")
        sf.write(dst, y, 22050)
        text = row["text"].strip()
        rows.append(row["filename"] + "|" + text + "|" + text)
with open(OUT_META, "w", encoding="utf-8") as f:
    f.write("\n".join(rows))
print("Done! " + str(len(rows)) + " samples saved.")
