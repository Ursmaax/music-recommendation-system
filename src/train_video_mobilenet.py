# src/train_video_mobilenet.py
import os, csv, json, random
from glob import glob
from collections import Counter
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

ROOT = os.getcwd()  # project root, expected C:\Music_Rec_System
MANIFEST = "data_processed/video/manifest_faces.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (224,224)
BATCH = 32   # reduce to 16 if GPU memory low
SEED = 42
VAL_SPLIT = 0.12
HEAD_EPOCHS = 6
FINETUNE_EPOCHS = 6
AUTOTUNE = tf.data.AUTOTUNE

if not os.path.exists(MANIFEST):
    raise SystemExit("Manifest not found: " + MANIFEST)

# load manifest (img_path,label)
items = []
with open(MANIFEST, newline='', encoding='utf-8') as fh:
    reader = csv.reader(fh)
    hdr = next(reader, None)
    for r in reader:
        if not r: continue
        p = r[0]; lab = r[1] if len(r)>1 else ""
        if p and os.path.exists(p) and lab:
            items.append((p, lab))
print("Total face images found:", len(items))
if len(items) == 0:
    raise SystemExit("No face images found in manifest")

random.seed(SEED)
random.shuffle(items)
paths, labels = zip(*items)
paths = list(paths)
labels = list(labels)

# label -> index
unique_labels = sorted(list(set(labels)))
label_to_idx = {l:i for i,l in enumerate(unique_labels)}
print("Labels:", label_to_idx)

y = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

# compute class weights and save
cw_path = os.path.join(MODEL_DIR, "video_class_weights.json")
if os.path.exists(cw_path):
    class_weight_idx = json.load(open(cw_path))
    print("Loaded existing class weights.")
else:
    cnt = Counter(y)
    total = sum(cnt.values())
    n_classes = len(cnt)
    class_weight_idx = {int(k): float(total / (n_classes * v)) for k,v in cnt.items()}
    with open(cw_path, "w") as fh:
        json.dump(class_weight_idx, fh, indent=2)
    print("Computed and saved class weights to", cw_path)

# stratified split
train_p, val_p, train_y, val_y = train_test_split(paths, y, test_size=VAL_SPLIT, random_state=SEED, stratify=y)
print("Train/Val sizes:", len(train_p), len(val_p))

# dataset helpers
def preprocess(path, label, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    img = tf.image.resize(img, IMG_SIZE)
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.08)
        img = tf.image.random_contrast(img, 0.9, 1.1)
    return img, label

def make_ds(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=max(2048, len(paths)))
    ds = ds.map(lambda p,l: tf.py_function(lambda a,b: preprocess(a,b,training), [p,l], [tf.float32, tf.int32]),
                num_parallel_calls=AUTOTUNE)
    # set shapes
    def set_shapes(img, lab):
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        lab.set_shape([])
        return img, lab
    ds = ds.map(set_shapes, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

train_ds = make_ds(train_p, train_y, training=True)
val_ds = make_ds(val_p, val_y, training=False)

# build model
base = tf.keras.applications.MobileNetV2(input_shape=(IMG_SIZE[0],IMG_SIZE[1],3),
                                         include_top=False, weights='imagenet')
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0],IMG_SIZE[1],3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(unique_labels), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR, 'video_mobilenet_best.h5'),
                                       save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

# Phase 1: train head
print("Training head for %d epochs..." % HEAD_EPOCHS)
history1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS,
                     class_weight=class_weight_idx, callbacks=callbacks)

# Phase 2: fine-tune
base.trainable = True
fine_tune_at = len(base.layers) - 30
for layer in base.layers[:fine_tune_at]:
    layer.trainable = False
for layer in base.layers[fine_tune_at:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Fine-tuning for %d epochs..." % FINETUNE_EPOCHS)
history2 = model.fit(train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS,
                     class_weight=class_weight_idx, callbacks=callbacks)

# save final model and history
model.save(os.path.join(MODEL_DIR, "video_mobilenet_last.h5"))
hist = {}
for k,v in (getattr(history1, 'history', {}).items()):
    hist[k] = v
for k,v in (getattr(history2, 'history', {}).items()):
    hist.setdefault(k, []).extend(v)
with open(os.path.join(MODEL_DIR, "train_history_video_mobilenet.json"), "w") as fh:
    json.dump(hist, fh)
print("Training finished. Best model at models/video_mobilenet_best.h5")
