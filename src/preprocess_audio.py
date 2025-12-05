import os
import csv
import numpy as np
import librosa
from tqdm import tqdm

BASE = "Full_Music_Emotion_Project/data_raw"
MANIFEST_DIR = "data_processed/manifests"
OUT_BASE = "data_processed/audio"   # mel outputs saved here
SR = 22050
DURATION = 4.0   # seconds (pad/truncate)
N_MELS = 128
HOP_LENGTH = 512

os.makedirs(OUT_BASE, exist_ok=True)

def load_wav(path, sr=SR, duration=DURATION):
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        if len(y) < target_len:
            y = np.pad(y, (0, max(0, target_len - len(y))), 'constant')
    return y

def wav_to_mel(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)

def process_manifest(manifest_csv, out_subdir):
    out_dir = os.path.join(OUT_BASE, out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_rows = []
    with open(manifest_csv, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        entries = list(reader)

    for r in tqdm(entries, desc=f"proc {os.path.basename(manifest_csv)}"):
        wav_path = r.get('filepath') or r.get('path') or r.get('file')
        label = r.get('label', '')
        if not wav_path or not os.path.exists(wav_path):
            continue

        try:
            y = load_wav(wav_path)
            mel = wav_to_mel(y)
            rel = os.path.relpath(wav_path, BASE)
            # safe filename for npy
            safe = rel.replace(os.sep, '__').replace('/', '__')
            base_name = os.path.splitext(safe)[0]
            out_path = os.path.join(out_dir, base_name + ".npy")
            np.save(out_path, mel)
            out_rows.append((out_path, label))
        except Exception as e:
            print("ERR", wav_path, e)

    # write a small manifest for processed files
    out_csv = os.path.join(out_dir, "manifest_processed.csv")
    with open(out_csv, "w", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["npy_path", "label"])
        writer.writerows(out_rows)

    return len(out_rows)

if __name__ == "__main__":
    manifests = [
        (os.path.join(MANIFEST_DIR, "audio_manifest.csv"), "ravdess_audio"),
        (os.path.join(MANIFEST_DIR, "audio_manifest_cremad.csv"), "cremad_audio"),
    ]

    total = 0
    for m, name in manifests:
        if os.path.exists(m):
            print("Processing manifest:", m)
            n = process_manifest(m, name)
            print(" -> saved:", n, "items")
            total += n
        else:
            print("Manifest missing:", m)

    print("All done. Total processed:", total)
