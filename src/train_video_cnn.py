# src/train_video_cnn.py
"""
Train a video-face emotion classifier using transfer learning (EfficientNetB0).
Usage:
    python src/train_video_cnn.py
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

# ---- CONFIG ----
DATA_DIR = "data_processed/video"            # folders: angry, calm, disgust, ...
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS_WARMUP = 6
EPOCHS_FINETUNE = 20
FINE_TUNE_AT = 100   # layer index to unfreeze from base model (tune later)
BASE_MODEL = "EfficientNetB0"  # or "MobileNetV2" (fallback)

# ---- build data generators ----
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    brightness_range=(0.8, 1.2),
    zoom_range=0.08,
    fill_mode='nearest'
)

# use the same preprocessing for validation except augmentation
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=SEED
)

val_gen = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    seed=SEED,
    shuffle=False
)

labels = list(train_gen.class_indices.keys())
num_classes = len(labels)
print("Labels:", labels, "Num classes:", num_classes)

# compute class weights (helps with imbalance)
y_train = train_gen.classes
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
print("Class weights:", class_weights_dict)

# save label mapping and class weights
with open(os.path.join(MODEL_DIR, "video_labels.json"), "w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)
with open(os.path.join(MODEL_DIR, "video_class_weights.json"), "w", encoding="utf-8") as f:
    json.dump(class_weights_dict, f, indent=2)

# ---- build model (transfer learning) ----
def build_model(num_classes):
    if BASE_MODEL == "EfficientNetB0":
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    else:
        base = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base.trainable = False  # freeze for warmup

    inp = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = layers.Lambda(preprocess)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_model(num_classes)
model.summary()

# compile for warmup
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# callbacks
checkpoint_path = os.path.join(MODEL_DIR, "video_cnn_best.h5")
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
]

# ---- warmup training (train head only) ----
print("Starting warmup training (head only)...")
history_warm = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_WARMUP,
    class_weight=class_weights_dict,
    callbacks=cb
)

# ---- fine-tune: unfreeze top layers of base ----
print("Unfreezing top layers and fine-tuning...")
# unfreeze base from FINE_TUNE_AT
base_model = None
for layer in model.layers:
    # find the base model instance
    if isinstance(layer, tf.keras.Model) or 'efficientnet' in layer.name or 'mobilenetv2' in layer.name:
        base_model = layer
        break

if base_model is None:
    # fallback: find the biggest submodel
    for layer in model.layers:
        if hasattr(layer, 'trainable') and layer.count_params() > 10000:
            base_model = layer
            break

if base_model is None:
    print("Warning: could not detect base model to fine-tune. Skipping fine-tune stage.")
else:
    base_model.trainable = True
    # set low learning rate and freeze lower layers
    total_layers = len(base_model.layers)
    fine_at = min(FINE_TUNE_AT, total_layers-10)
    print(f"Total base layers: {total_layers}, unfreezing from: {fine_at}")
    for i, layer in enumerate(base_model.layers):
        layer.trainable = (i >= fine_at)

    # recompile with lower LR
    model.compile(
        optimizer=optimizers.Adam(learning_rate=2.5e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history_finetune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINETUNE,
        initial_epoch=history_warm.epoch[-1] + 1 if len(history_warm.epoch) else 0,
        class_weight=class_weights_dict,
        callbacks=cb
    )

print("Training finished. Best model saved to:", checkpoint_path)
