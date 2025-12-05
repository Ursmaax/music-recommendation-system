# src/fusion_inference.py
"""
Fusion model inference wrapper with preprocessing.
"""
import os
import numpy as np
import tensorflow as tf
from preprocess_helpers import preprocess_for_fusion
import json

# Load labels
LABELS_PATH = "models/labels.json"
MODEL_PATH = "models/fusion_model_best.keras"

# Global model (loaded once)
_model = None
_labels = None

def load_model_and_labels():
    """Load fusion model and labels once."""
    global _model, _labels
    
    if _model is None:
        print("Loading fusion model...")
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Model loaded successfully")
    
    if _labels is None:
        if os.path.exists(LABELS_PATH):
            with open(LABELS_PATH, 'r') as f:
                _labels = json.load(f)
        else:
            # Fallback labels
            _labels = ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        print(f"Labels: {_labels}")
    
    return _model, _labels


def predict_from_live(audio_input, video_input):
    """
    Predict emotion from audio and video inputs.
    
    Args:
        audio_input: audio file path, bytes, or numpy array
        video_input: video file path or list of frames
    
    Returns:
        dict: {
            'emotion': str,
            'confidence': float,
            'scores': dict {emotion: probability},
            'all_probs': list of floats
        }
    """
    try:
        # Load model
        model, labels = load_model_and_labels()
        
        # Preprocess
        audio_tensor, video_tensor = preprocess_for_fusion(audio_input, video_input)
        
        # Add batch dimension
        audio_batch = np.expand_dims(audio_tensor, axis=0)  # (1, 128, 128, 1)
        video_batch = np.expand_dims(video_tensor, axis=0)  # (1, 224, 224, 3)
        
        # Predict
        predictions = model.predict([audio_batch, video_batch], verbose=0)
        probs = predictions[0]  # (8,)
        
        # Get top emotion
        top_idx = np.argmax(probs)
        emotion = labels[top_idx]
        confidence = float(probs[top_idx])
        
        # Create scores dict
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'scores': scores,
            'all_probs': probs.tolist()
        }
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return {
            'emotion': 'neutral',
            'confidence': 0.0,
            'scores': {},
            'all_probs': [],
            'error': str(e)
        }


def get_emotion_color(emotion):
    """Get color theme for emotion."""
    colors = {
        'happy': '#FFD700',      # Gold
        'calm': '#87CEEB',       # Sky Blue
        'sad': '#4682B4',        # Steel Blue
        'angry': '#DC143C',      # Crimson
        'fear': '#8B008B',       # Dark Magenta
        'surprise': '#FF69B4',   # Hot Pink
        'disgust': '#556B2F',    # Dark Olive Green
        'neutral': '#808080'     # Gray
    }
    return colors.get(emotion, '#808080')


def get_emotion_description(emotion):
    """Get friendly description for emotion."""
    descriptions = {
        'happy': 'Feeling joyful and upbeat! 🎉',
        'calm': 'Peaceful and relaxed 🌊',
        'sad': 'Feeling down or melancholic 😔',
        'angry': 'Frustrated or upset 😠',
        'fear': 'Anxious or worried 😰',
        'surprise': 'Amazed or shocked! 😲',
        'disgust': 'Feeling aversion 😖',
        'neutral': 'Balanced and composed 😐'
    }
    return descriptions.get(emotion, 'Emotion detected')
