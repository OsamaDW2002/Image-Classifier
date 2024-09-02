import tensorflow as tf
from PIL import Image
import numpy as np

image_size = 224

def process_image(image):
    image = tf.convert_to_tensor(image)
    
    image = tf.image.resize(image, (image_size, image_size))
    image = image / 255.0
    
    return image.numpy()


def predict(image_path, model, top_k=5):
    image = Image.open(image_path)
    image = np.asarray(image)
    
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = model.predict(processed_image)
    top_k_probs, top_k_indices = tf.nn.top_k(predictions, k=top_k)
    
    top_k_probs = top_k_probs.numpy().flatten()
    top_k_indices = top_k_indices.numpy().flatten()
    
    return top_k_probs, top_k_indices