# src/gradio_audio_demo.py
import os
import numpy as np
import librosa
import tensorflow as tf
import json
import gradio as gr

# SETTINGS - match training preprocessing
SR = 22050
DURATION = 4.0     # seconds used in training
N_MELS = 128
HOP_LENGTH = 512

MODEL_PATH = "models/audio_cnn_best.h5"   # change if different
LABELS_JSON = "models/labels.json"        # created earlier by eval step, fallback below

# Fallback: build label list from processed manifests if labels.json not found
def load_labels():
    if os.path.exists(LABELS_JSON):
        with open(LABELS_JSON, "r") as fh:
            labels = json.load(fh)
            return labels
    # fallback: scan processed manifests
    labels_set = set()
    base = "data_processed/audio"
    for sub in os.listdir(base):
        m = os.path.join(base, sub, "manifest_processed.csv")
        if os.path.exists(m):
            import csv
            with open(m, newline='', encoding='utf-8') as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        labels_set.add(row[1])
    labels = sorted(list(labels_set))
    # save for future use
    os.makedirs("models", exist_ok=True)
    with open(LABELS_JSON, "w") as fh:
        json.dump(labels, fh)
    return labels

labels = load_labels()
print("Loaded labels:", labels)

# load model
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded from", MODEL_PATH)

# Preprocessing: load audio, pad/truncate to DURATION, create mel, to same dtype as training
def audio_to_model_input(audio_path):
    y, sr = librosa.load(audio_path, sr=SR, mono=True)
    target_len = int(SR * DURATION)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        if len(y) < target_len:
            y = np.pad(y, (0, max(0, target_len - len(y))), 'constant')
    S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH, fmax=8000)
    S_db = librosa.power_to_db(S, ref=np.max)
    # model expects shape (None, None, 1) and model has a Resizing layer -> we'll keep as 2D + channel
    arr = S_db.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    # add batch dim
    arr = np.expand_dims(arr, axis=0)
    return arr

# Map prediction to songs (example small mapping; you can expand with your library)
RECOMMEND = {
    "happy": ["Song: Uptempo 1", "Song: Energetic 2"],
    "sad":   ["Song: Calm 1", "Song: Soft 2"],
    "angry": ["Song: Relaxing 1", "Song: Calm 2"],
    "neutral":["Song: Chill 1","Song: Background 2"],
    "fear":  ["Song: Comfort 1","Song: Warm 2"],
    "surprise":["Song: Bright 1","Song: Upbeat 2"],
    "disgust":["Song: Soft 3","Song: Easy 4"],
    "calm":  ["Song: Smooth 1", "Song: Gentle 2"]
}

def predict_audio(file_obj):
    # file_obj is a path-like object provided by Gradio
    if file_obj is None:
        return {"emotion":"no_audio", "score":0.0, "recommendations":[]}
    arr = audio_to_model_input(file_obj)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    emotion = labels[idx] if idx < len(labels) else "unknown"
    score = float(preds[idx])
    recs = RECOMMEND.get(emotion, ["No songs configured"])
    return {"emotion": emotion, "score": round(score,3), "recommendations": recs}

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## Real-time Demo — Audio Emotion Recognition (Prototype)")
    with gr.Row():
        audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Upload or record a short audio (4s recommended)")
        btn = gr.Button("Predict")
    output_text = gr.JSON(label="Prediction")
    def run_and_format(filepath):
        res = predict_audio(filepath)
        # format for display
        return {
            "predicted_emotion": res["emotion"],
            "confidence": res["score"],
            "top_recommendations": res["recommendations"]
        }
    btn.click(run_and_format, inputs=[audio_in], outputs=[output_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
