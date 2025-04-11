import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# Preprocess image for TFLite
def load_and_preprocess_image(image_path, img_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(img_size)
    image = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(image, axis=0)

# Load TFLite models
def load_tflite_models(model_paths):
    interpreters = {}
    for name, path in model_paths.items():
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        interpreters[name] = interpreter
    return interpreters

# Predict using TFLite models
def predict_image_tflite(interpreters, image_path, class_names, img_size=(224, 224), plot=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    image = load_and_preprocess_image(image_path, img_size)
    predictions = []

    for name, interpreter in interpreters.items():
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        predictions.append(output)

    avg_pred = np.mean(np.stack(predictions), axis=0)
    predicted_class = np.argmax(avg_pred[0])
    probabilities = avg_pred[0]

    print(f"\n Predicted Label: {class_names[predicted_class]}")
    print(" Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {class_names[i]}: {prob * 100:.2f}%")

    if plot:
        plt.figure(figsize=(6, 4))
        plt.bar(class_names, probabilities, color='skyblue')
        plt.ylabel("Probability")
        plt.title("Prediction Confidence")
        plt.ylim([0, 1])
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    return class_names[predicted_class], probabilities

# -------- Main --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict uniform class from an image using TFLite ensemble models")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    # Paths to your TFLite models
    model_paths = {
        'resnet50': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\ResNet50.tflite',
        'resnet101': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\resnet101.tflite',
        'inceptionv3': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\InceptionV3.tflite',
    }

    class_names = ['BSF', 'CRPF', 'J&K Police']

    interpreters = load_tflite_models(model_paths)
    predict_image_tflite(interpreters, args.image_path, class_names)
