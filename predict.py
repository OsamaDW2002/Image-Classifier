import argparse
import json
import tensorflow as tf
import tensorflow_hub as hub
from utils import predict

TOP_K_LIMIT = 103

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Predict flower class from an image.')

    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the saved model file.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top classes to return.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')

    return parser.parse_args()

def load_category_names(category_names_path):
    """Load category names from a JSON file."""
    with open(category_names_path, 'r') as f:
        return json.load(f)

def map_classes_to_names(classes, category_names):
    """Map class labels to human-readable names."""
    return [category_names.get(str(cls), str(cls)) for cls in classes]

def main():
    args = parse_arguments()
    
    print(f"Image Path: {args.image_path}")
    print(f"Model Path: {args.model_path}")
    print(f"Top K: {args.top_k}")
    print(f"Category Names Path: {args.category_names}")

    if 0 < args.top_k < TOP_K_LIMIT:
        # Load the model
        model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
        
        # Predict
        probs, classes = predict(args.image_path, model, args.top_k)
        
        print("\nTop Predictions:")
        
        if args.category_names:
            category_names = load_category_names(args.category_names)
            class_names = map_classes_to_names(classes, category_names)
            for name, prob in zip(class_names, probs):
                print(f"Class: {name}, Probability: {prob:.4f}")
        else:
            for cls, prob in zip(classes, probs):
                print(f"Class: {cls}, Probability: {prob:.4f}")
    else:
        print('Top K must be between 1 and 102.')

if __name__ == "__main__":
    main()
