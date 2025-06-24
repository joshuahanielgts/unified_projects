import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model('asl_model.h5')

# Set test image path
TEST_IMAGE_PATH = 'test_images/B_test.jpg'

# Set image size
IMG_SIZE = (64, 64)

# Get class names (from training folder)
train_dir = 'dataset/asl_alphabet_train/'
class_names = sorted(os.listdir(train_dir))

# Preprocess image
img = image.load_img(TEST_IMAGE_PATH, target_size=IMG_SIZE)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = class_names[np.argmax(prediction)]

# Output
print(f"✅ Prediction: The ASL sign is '{predicted_class}'")
