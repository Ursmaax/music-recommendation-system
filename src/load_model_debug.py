# src/load_model_debug.py
import os, traceback, tensorflow as tf
model_path = 'models/video_mobilenet_best.h5'
print('Model path exists:', os.path.exists(model_path))
try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print('Model loaded successfully')
    model.summary()
except Exception as e:
    print('Error loading model:')
    traceback.print_exc()
