# src/preprocess_helpers.py
"""
Unified preprocessing utilities for audio and video inputs.
Ensures consistent shape and normalization for the fusion model.
"""
import numpy as np
import librosa
import cv2
import tensorflow as tf

# Audio constants
SR = 22050
DURATION = 4.0
N_MELS = 128
HOP_LENGTH = 512
TARGET_MEL_WIDTH = 128

# Video constants
FACE_SIZE = (224, 224)

def preprocess_audio_bytes(audio_bytes_or_path, sr=SR, duration=DURATION):
    """
    Convert audio bytes or file path to mel-spectrogram (128, 128, 1).
    
    Args:
        audio_bytes_or_path: bytes or str path to audio file
        sr: sample rate
        duration: target duration in seconds
    
    Returns:
        np.array of shape (128, 128, 1), normalized 0-1
    """
    # Load audio
    if isinstance(audio_bytes_or_path, (str, bytes)):
        y, _ = librosa.load(audio_bytes_or_path, sr=sr, mono=True)
    else:
        y = audio_bytes_or_path
    
    # Pad or trim to target length
    target_len = int(sr * duration)
    if len(y) > target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, max(0, target_len - len(y))), 'constant')
    
    # Compute mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Ensure width is 128
    if mel_db.shape[1] > TARGET_MEL_WIDTH:
        start = (mel_db.shape[1] - TARGET_MEL_WIDTH) // 2
        mel_db = mel_db[:, start:start+TARGET_MEL_WIDTH]
    else:
        pad_width = TARGET_MEL_WIDTH - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0,0), (0, pad_width)), mode='constant')
    
    # Normalize to 0-1
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    
    # Add channel dimension
    mel_norm = mel_norm[..., np.newaxis]  # (128, 128, 1)
    
    return mel_norm.astype(np.float32)


def extract_face_from_frame(frame, face_cascade=None):
    """
    Extract and resize face from a video frame.
    
    Args:
        frame: numpy array (H, W, 3) BGR
        face_cascade: cv2.CascadeClassifier (optional, will create if None)
    
    Returns:
        np.array of shape (224, 224, 3) RGB, preprocessed for EfficientNet, or None if no face
    """
    if face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(48,48))
    
    if len(faces) == 0:
        # No face detected, return center crop
        h, w = frame.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        face_img = frame[start_h:start_h+size, start_w:start_w+size]
    else:
        # Use largest face
        x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        # Expand box slightly
        pad = int(0.2 * max(w, h))
        x0 = max(0, x-pad)
        y0 = max(0, y-pad)
        x1 = min(frame.shape[1], x + w + pad)
        y1 = min(frame.shape[0], y + h + pad)
        face_img = frame[y0:y1, x0:x1]
    
    # Resize to 224x224
    face_resized = cv2.resize(face_img, FACE_SIZE, interpolation=cv2.INTER_LINEAR)
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # Preprocess for EfficientNet (will be done in model, but normalize to 0-255 range)
    face_rgb = face_rgb.astype(np.float32)
    
    return face_rgb


def preprocess_video_frames(video_path_or_frames, max_frames=10):
    """
    Extract faces from video and return average face representation.
    
    Args:
        video_path_or_frames: str path to video or list of frames
        max_frames: maximum number of frames to sample
    
    Returns:
        np.array of shape (224, 224, 3), averaged face, or None if no faces
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    if isinstance(video_path_or_frames, str):
        # Load from video file
        cap = cv2.VideoCapture(video_path_or_frames)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // max_frames)
        
        faces = []
        frame_idx = 0
        while len(faces) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step == 0:
                face = extract_face_from_frame(frame, face_cascade)
                if face is not None:
                    faces.append(face)
            frame_idx += 1
        cap.release()
    else:
        # Process provided frames
        faces = []
        for frame in video_path_or_frames[:max_frames]:
            face = extract_face_from_frame(frame, face_cascade)
            if face is not None:
                faces.append(face)
    
    if len(faces) == 0:
        return None
    
    # Average faces
    avg_face = np.mean(faces, axis=0).astype(np.float32)
    
    return avg_face


def preprocess_for_fusion(audio_input, video_input):
    """
    Preprocess both audio and video for fusion model.
    
    Args:
        audio_input: audio file path or bytes
        video_input: video file path or list of frames
    
    Returns:
        tuple: (audio_tensor, video_tensor) ready for model input
    """
    audio_mel = preprocess_audio_bytes(audio_input)
    video_face = preprocess_video_frames(video_input)
    
    if video_face is None:
        # Create blank face if no detection
        video_face = np.zeros((224, 224, 3), dtype=np.float32)
    
    return audio_mel, video_face
