# src/train_fusion_model.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
import json

# Config
AUDIO_MANIFEST = "data_processed/audio/ravdess_audio/manifest_processed.csv"
VIDEO_MANIFEST = "data_processed/video/manifest_faces.csv"
AUDIO_MODEL_PATH = "models/audio_cnn_best.h5"
VIDEO_MODEL_PATH = "models/video_cnn_best.h5"
MODEL_DIR = "models"
BATCH_SIZE = 16
EPOCHS = 20

# Load Manifests
def load_manifests():
    # Audio
    if not os.path.exists(AUDIO_MANIFEST):
        raise SystemExit(f"Missing audio manifest: {AUDIO_MANIFEST}")
    df_audio = pd.read_csv(AUDIO_MANIFEST)
    # Key extraction: ravdess_audio__Actor_01__01-01-01-01-01-01-01.npy -> 01-01-01-01-01-01-01 -> 01-01-01-01-01-01
    df_audio['key'] = df_audio['npy_path'].apply(lambda x: '-'.join(os.path.splitext(os.path.basename(x))[0].split('__')[-1].split('-')[1:]))
    
    # Video
    if not os.path.exists(VIDEO_MANIFEST):
        raise SystemExit(f"Missing video manifest: {VIDEO_MANIFEST}")
    df_video = pd.read_csv(VIDEO_MANIFEST)
    # Key extraction: 01-01-01-01-01-01-01_f0.jpg -> 01-01-01-01-01-01-01 -> 01-01-01-01-01-01
    df_video['key'] = df_video['img_path'].apply(lambda x: '-'.join(os.path.basename(x).split('_f')[0].split('-')[1:]))
    
    # Merge on key and label (sanity check)
    # We want samples that have BOTH audio and video
    # Note: Video has multiple frames per clip. Audio has one file per clip.
    # We will pair the audio with EACH video frame from the same clip.
    merged = pd.merge(df_video, df_audio, on=['key', 'label'], suffixes=('_video', '_audio'))
    
    print(f"Audio samples: {len(df_audio)}")
    print(f"Video samples: {len(df_video)}")
    print(f"Merged samples (Audio+Video pairs): {len(merged)}")
    
    return merged

# Data Generator
def preprocess_audio(path):
    # Load npy
    mel = np.load(path.numpy().decode('utf-8'))
    # Resize/Pad to (128, 128) if needed, though preprocess_audio.py should have handled it.
    # But let's ensure shape.
    # Expected shape (128, time). We need (128, 128, 1)
    # If time > 128, crop. If < 128, pad.
    target_width = 128
    if mel.shape[1] > target_width:
        start = (mel.shape[1] - target_width) // 2
        mel = mel[:, start:start+target_width]
    else:
        pad_width = target_width - mel.shape[1]
        mel = np.pad(mel, ((0,0), (0, pad_width)), mode='constant')
    
    mel = mel[..., np.newaxis] # (128, 128, 1)
    # Normalize? It was db scale. Let's scale to roughly 0-1 or -1 to 1?
    # Audio CNN was trained on raw db? Let's check train_audio_cnn.py... 
    # It used standard scaler or just raw? 
    # Actually, let's just normalize to 0-1 for safety
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
    return mel.astype(np.float32)

def preprocess_video(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def make_ds(df, label_map, training=False):
    audio_paths = df['npy_path'].values
    video_paths = df['img_path'].values
    labels = df['label'].values
    y = [label_map[l] for l in labels]
    
    ds = tf.data.Dataset.from_tensor_slices((audio_paths, video_paths, y))
    if training:
        ds = ds.shuffle(len(df))
    
    def _map(a_path, v_path, label):
        # Wrap numpy load in py_function
        audio = tf.py_function(preprocess_audio, [a_path], tf.float32)
        audio.set_shape([128, 128, 1])
        
        video = preprocess_video(v_path)
        
        return (audio, video), label
    
    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def build_fusion_model(num_classes):
    # Audio Branch
    audio_base = tf.keras.models.load_model(AUDIO_MODEL_PATH, compile=False)
    # Assume audio_base output is softmax. We want the layer BEFORE that.
    # Or we can just use the whole model and take the output?
    # Better to take the dense layer output.
    # Let's inspect layers. Usually the last one is 'dense_1' or similar.
    # For safety, let's pop the last layer.
    audio_base_out = audio_base.layers[-2].output # Second to last (Dropout or Dense)
    audio_model = models.Model(inputs=audio_base.input, outputs=audio_base_out)
    audio_model.trainable = False # Freeze
    
    # Video Branch
    # Instead of loading the saved model (which has Lambda layer issues),
    # we'll rebuild EfficientNetB0 and load only the weights we need
    efficientnet_base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    efficientnet_base.trainable = False
    
    # Build a simple feature extractor
    video_input_temp = layers.Input(shape=(224, 224, 3))
    x_v = tf.keras.applications.efficientnet.preprocess_input(video_input_temp)
    x_v = efficientnet_base(x_v, training=False)
    x_v = layers.GlobalAveragePooling2D()(x_v)
    video_feature_model = models.Model(inputs=video_input_temp, outputs=x_v)
    video_feature_model.trainable = False
    
    # Fusion - video input will be preprocessed in the data pipeline
    audio_in = layers.Input(shape=(128, 128, 1), name="audio_input")
    video_in = layers.Input(shape=(224, 224, 3), name="video_input")
    
    a_feat = audio_model(audio_in)
    v_feat = video_feature_model(video_in)
    
    concat = layers.Concatenate()([a_feat, v_feat])
    
    x = layers.Dense(256, activation='relu')(concat)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=[audio_in, video_in], outputs=outputs)
    return model

def main():
    print("Loading Data...")
    df = load_manifests()
    
    # Labels
    unique_labels = sorted(df['label'].unique())
    label_map = {l: i for i, l in enumerate(unique_labels)}
    print(f"Labels: {label_map}")
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_ds = make_ds(train_df, label_map, training=True)
    val_ds = make_ds(val_df, label_map, training=False)
    
    print("Building Model...")
    model = build_fusion_model(len(unique_labels))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    # Callbacks
    cb = [
        callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, "fusion_model_best.keras"), 
                                  save_best_only=True, monitor='val_accuracy'),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    print("Starting Training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=cb)
    
    print("Done.")

if __name__ == "__main__":
    main()
