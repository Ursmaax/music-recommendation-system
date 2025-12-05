# src/train_audio_model_v2.py
import os, csv, json, numpy as np, tensorflow as tf
from sklearn.model_selection import train_test_split

BASE_AUDIO_PROC = "data_processed/audio"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# load class weights (label -> weight)
cw_path = os.path.join(MODEL_DIR, "class_weights.json")
if not os.path.exists(cw_path):
    raise SystemExit(f"Missing class weights: {cw_path}")
label_weights = json.load(open(cw_path, "r"))

# build manifests list (npy_path, label)
def load_manifest_csv(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for r in reader:
            if not r: continue
            p = r[0]; lab = r[1] if len(r)>1 else ""
            rows.append((p, lab))
    return rows

manifests = []
for sub in os.listdir(BASE_AUDIO_PROC):
    m = os.path.join(BASE_AUDIO_PROC, sub, "manifest_processed.csv")
    if os.path.exists(m):
        manifests.extend(load_manifest_csv(m))

if not manifests:
    raise SystemExit("No processed manifests found")

# build label->index map
labels = sorted(set(l for _,l in manifests if l))
label_to_idx = {l:i for i,l in enumerate(labels)}
print("Labels:", label_to_idx)

# convert class weights label->index
class_weight_idx = {}
for lab, w in label_weights.items():
    if lab in label_to_idx:
        class_weight_idx[label_to_idx[lab]] = float(w)
print("Class weights (index):", class_weight_idx)

# keep only valid labeled entries
valid = [(p,label_to_idx[l]) for p,l in manifests if l in label_to_idx and os.path.exists(p)]
print("Valid samples:", len(valid))
paths, labs = zip(*valid)
paths = list(paths)
labs = np.array(labs, dtype=np.int32)

# train/val split
train_p, val_p, train_y, val_y = train_test_split(paths, labs, test_size=0.12, random_state=42, stratify=labs)
print("Train / Val sizes:", len(train_p), len(val_p))

# dataset factory with augmentation
AUTOTUNE = tf.data.AUTOTUNE
BATCH = 32

def load_npy_tf(path):
    def _load(p):
        import numpy as np
        a = np.load(p.decode('utf-8'))
        if a.ndim == 2:
            a = a[..., np.newaxis]
        return a.astype(np.float32)
    arr = tf.numpy_function(_load, [path], tf.float32)
    arr.set_shape([None, None, 1])
    return arr

def augment(arr):
    # arr: (n_mels, time, 1)
    # random time shift
    if tf.random.uniform([]) < 0.5:
        t = tf.shape(arr)[1]
        limit = tf.cast(tf.cast(t, tf.float32) * 0.1, tf.int32)
        shift = tf.random.uniform([], minval=-limit, maxval=limit, dtype=tf.int32)
        arr = tf.roll(arr, shift=shift, axis=1)
    # add small gaussian noise
    if tf.random.uniform([]) < 0.5:
        noise = tf.random.normal(tf.shape(arr), mean=0.0, stddev=0.005)
        arr = arr + noise
    return arr

def make_dataset(paths, y, shuffle=True, augment_prob=True):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=max(2048, len(paths)))
    def loader(p, label):
        arr = load_npy_tf(p)
        if augment_prob:
            arr = augment(arr)
        # normalize
        mean = tf.math.reduce_mean(arr)
        std = tf.math.reduce_std(arr)
        arr = (arr - mean) / (std + 1e-6)
        return arr, label
    ds = ds.map(loader, num_parallel_calls=AUTOTUNE)
    ds = ds.padded_batch(BATCH, padded_shapes=([None, None, 1], []))
    ds = ds.prefetch(AUTOTUNE)
    return ds

train_ds = make_dataset(train_p, train_y, shuffle=True, augment_prob=True)
val_ds = make_dataset(val_p, val_y, shuffle=False, augment_prob=False)

# model
def build_model(num_classes):
    inp = tf.keras.Input(shape=(None, None, 1))
    x = tf.keras.layers.Resizing(128,128)(inp)
    x = tf.keras.layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    out = tf.keras.layers.Dense(num_classes,activation='softmax')(x)
    return tf.keras.Model(inp,out)

num_classes = len(label_to_idx)
model = build_model(num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# callbacks
cb_list = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(MODEL_DIR,'audio_cnn_best.h5'), save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
]

EPOCHS = 20
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, class_weight=class_weight_idx, callbacks=cb_list)

# save final model and history
model.save(os.path.join(MODEL_DIR,'audio_cnn_last.h5'))
import json
with open(os.path.join(MODEL_DIR,'train_history_v2.json'),'w') as fh:
    json.dump({k:[float(x) for x in v] for k,v in history.history.items()}, fh)
print("Training finished. Best model saved to models/audio_cnn_best.h5")
