import os

base = "Full_Music_Emotion_Project/data_raw"

folders = {
    "ravdess_audio": "ravdess_audio",
    "ravdess_video": "ravdess_video",
    "cremad_AudioWAV": "cremad/AudioWAV",
    "fer2013_train": "fer2013/train",
    "fer2013_test": "fer2013/test"
}

print("\n=== DATASET CHECK ===\n")

for name, path in folders.items():
    full_path = os.path.join(base, path)
    exists = os.path.exists(full_path)
    count = 0

    if exists:
        for root, dirs, files in os.walk(full_path):
            for file in files:
                if file.lower().endswith((".wav", ".mp4", ".flv", ".jpg", ".png")):
                    count += 1

    print(f"{name:20} exists={exists}   files={count}")

print("\nDone.\n")
