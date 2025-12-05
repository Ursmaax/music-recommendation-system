# src/make_mel_images.py
import os, glob, numpy as np, librosa, matplotlib.pyplot as plt
from PIL import Image
import librosa.feature

BASE = "data_processed/audio"
OUT_BASE = "data_processed/mel_images"
os.makedirs(OUT_BASE, exist_ok=True)

# helper to compute delta and chroma and stack into 3 channels
def mel_to_rgb_arrays(mel):
    # mel expected shape (n_mels, time)
    n_mels, time_frames = mel.shape
    
    # 1) normalize mel to 0-255
    m = mel - mel.min()
    if mel.max() - mel.min() > 0:
        m = m / (mel.max() - mel.min())
    m_img = (m * 255).astype('uint8')  # channel 0

    # 2) delta (first derivative) -> channel 1
    # librosa.feature.delta might change shape, so we pad it back
    delta = librosa.feature.delta(m)
    # Ensure delta has the same shape as m
    if delta.shape != m.shape:
        # Pad or trim to match
        if delta.shape[1] < m.shape[1]:
            delta = np.pad(delta, ((0,0), (0, m.shape[1] - delta.shape[1])), mode='edge')
        else:
            delta = delta[:, :m.shape[1]]
    
    d = delta - delta.min()
    if delta.max() - delta.min() > 0:
        d = d / (delta.max() - delta.min())
    d_img = (d * 255).astype('uint8')

    # 3) chroma (compute from original audio? we approximate using mel->stft)
    # If we cannot compute chroma from mel alone, compute a pseudo chroma by summing mel into 12 bands
    bins_per_chroma = max(1, n_mels // 12)
    chroma = np.zeros((12, time_frames))
    for i in range(12):
        start = i * bins_per_chroma
        end = min((i + 1) * bins_per_chroma, n_mels)
        if start < n_mels:
            chroma[i, :] = mel[start:end, :].sum(axis=0)
    
    c = chroma - chroma.min()
    if chroma.max() - chroma.min() > 0:
        c = c / (chroma.max() - chroma.min())
    # resize chroma to mel height by repeat and ensure exact dimensions
    c_img_temp = (c * 255).astype('uint8')
    # Repeat each chroma bin to match n_mels height
    c_img = np.repeat(c_img_temp, repeats=max(1, n_mels//12), axis=0)
    # Trim or pad to exactly n_mels
    if c_img.shape[0] < n_mels:
        c_img = np.pad(c_img, ((0, n_mels - c_img.shape[0]), (0,0)), mode='edge')
    else:
        c_img = c_img[:n_mels, :]
    
    # Final safety check: ensure all three channels have exactly the same shape
    assert m_img.shape == d_img.shape == c_img.shape, f"Shape mismatch: m={m_img.shape}, d={d_img.shape}, c={c_img.shape}"
    
    return m_img, d_img, c_img

def save_rgb_image(ch0, ch1, ch2, outpath, size=(224,224)):
    # stack -> H x W x 3
    H,W = ch0.shape
    rgb = np.stack([ch0, ch1, ch2], axis=-1)
    im = Image.fromarray(rgb)
    im = im.resize(size, resample=Image.BILINEAR)
    im.save(outpath)

# process each processed folder (ravdess_audio, cremad_audio)
for sub in os.listdir(BASE):
    proc_csv = os.path.join(BASE, sub, "manifest_processed.csv")
    if not os.path.exists(proc_csv):
        continue
    out_sub = os.path.join(OUT_BASE, sub)
    os.makedirs(out_sub, exist_ok=True)
    count = 0
    with open(proc_csv, newline='', encoding='utf-8') as fh:
        import csv
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            npy = row[0]
            label = row[1] if len(row)>1 else "unknown"
            if not os.path.exists(npy):
                continue
            try:
                mel = np.load(npy)
                if mel.ndim==3: mel = mel[...,0]
                ch0, ch1, ch2 = mel_to_rgb_arrays(mel)
                # create outpath using label folders
                labdir = os.path.join(out_sub, label if label else "unknown")
                os.makedirs(labdir, exist_ok=True)
                fname = os.path.splitext(os.path.basename(npy))[0] + ".png"
                outpath = os.path.join(labdir, fname)
                save_rgb_image(ch0, ch1, ch2, outpath, size=(224,224))
                count += 1
            except Exception as e:
                print("ERR", npy, e)
    print(f"Processed {count} images for {sub} -> {out_sub}")
print("DONE")
