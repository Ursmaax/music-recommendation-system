"""
Convert fusion_model_best.keras to .h5 format for better compatibility
"""
import tensorflow as tf

print("Loading .keras model...")
model = tf.keras.models.load_model('models/fusion_model_best.keras', compile=False)

print("Saving as .h5 format...")
model.save('models/fusion_model_best_converted.h5')

print("✅ Conversion complete!")
print("Now upload 'fusion_model_best_converted.h5' to Hugging Face")
