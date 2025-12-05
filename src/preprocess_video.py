import os, csv, cv2
from pathlib import Path

BASE = "Full_Music_Emotion_Project/data_raw"
MANIFEST = "data_processed/manifests/video_manifest.csv"
OUT_BASE = "data_processed/video"
os.makedirs(OUT_BASE, exist_ok=True)

# Haar cascade for face detection (uses OpenCV bundled path)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise SystemExit("Face cascade not found at: " + cascade_path)

def process_video(video_path, label, out_dir, sample_rate=1):
    # sample_rate: frames per second to sample (1 = one frame each second)
    saved = 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("WARN: cannot open", video_path)
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps else 0
    # step frames = fps / sample_rate
    step = max(1, int(round(fps / sample_rate)))
    frame_idx = 0
    out_label_dir = os.path.join(out_dir, label)
    os.makedirs(out_label_dir, exist_ok=True)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # convert to gray for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
            if len(faces) > 0:
                # choose largest face
                x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
                # expand box a little
                pad = int(0.2 * max(w,h))
                x0 = max(0, x-pad); y0 = max(0, y-pad)
                x1 = min(frame.shape[1], x + w + pad); y1 = min(frame.shape[0], y + h + pad)
                face_img = frame[y0:y1, x0:x1]
                # resize to 224x224
                face_img = cv2.resize(face_img, (224,224), interpolation=cv2.INTER_LINEAR)
                # output file name
                base = Path(video_path).stem
                out_name = f"{base}_f{frame_idx}.jpg"
                out_path = os.path.join(out_label_dir, out_name)
                cv2.imwrite(out_path, face_img)
                saved += 1
        frame_idx += 1
    cap.release()
    return saved

def main():
    if not os.path.exists(MANIFEST):
        raise SystemExit("Manifest not found: " + MANIFEST)
    entries = []
    with open(MANIFEST, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            # manifest columns: filepath, dataset, relative_path, label
            path = r.get("filepath") or r.get("path") or r.get("file")
            label = r.get("label","").strip()
            if path and os.path.exists(path) and label:
                entries.append((path, label))
    print("Videos to process:", len(entries))
    total_faces = 0
    processed_videos = 0
    out_manifest_rows = []
    for video_path, label in entries:
        saved = process_video(video_path, label, OUT_BASE, sample_rate=1)
        if saved >= 0:
            processed_videos += 1
            total_faces += saved
            # record saved images
            from glob import glob
            pattern = os.path.join(OUT_BASE, label, f"{Path(video_path).stem}_f*.jpg")
            for img in glob(pattern):
                out_manifest_rows.append((img, label))
        if processed_videos % 50 == 0:
            print(f"Processed videos: {processed_videos}  faces saved so far: {total_faces}")
    # write manifest of face images
    out_csv = os.path.join(OUT_BASE, "manifest_faces.csv")
    with open(out_csv, "w", newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(["img_path", "label"])
        writer.writerows(out_manifest_rows)
    print("Done. Processed videos:", processed_videos, "Total faces saved:", total_faces)
    print("Face manifest saved to:", out_csv)

if __name__ == '__main__':
    main()
