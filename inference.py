import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the weights for cross-entropy loss
def weighted_cce(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true_idx = tf.argmax(y_true, axis=1)
        weights_per_sample = tf.gather(weights, y_true_idx)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return ce * weights_per_sample
    return loss

# Load Models
def load_models(model_paths, custom_objects=None):
    models = {}
    for name, path in model_paths.items():
        models[name] = tf.keras.models.load_model(path, custom_objects=custom_objects)
    return models

# Preprocess Image
def load_and_preprocess_image(image_path, img_size=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return tf.expand_dims(image, axis=0)

# Predict
def predict_image(models, image_path, class_names, img_size=(224, 224), plot=True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")
    
    image = load_and_preprocess_image(image_path, img_size=img_size)
    preds = [model.predict(image, verbose=0) for model in models.values()]
    avg_pred = tf.reduce_mean(tf.stack(preds), axis=0)

    predicted_class = tf.argmax(avg_pred, axis=1).numpy()[0]
    probabilities = avg_pred.numpy()[0]

    print(f"\n Predicted Label: {class_names[predicted_class]}")
    print(" Class Probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {class_names[i]}: {prob*100:.2f}%")

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
    parser = argparse.ArgumentParser(description="Predict uniform class from an image using ensemble models")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    model_paths = {
        'resnet50': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\uniform_classification\models\ResNet50_best_model.h5',
        'resnet101': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\uniform_classification\models\resnet101_best_model.h5',
        'inceptionv3': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\uniform_classification\models\InceptionV3_best_model.h5',
    }

    class_names = ['BSF', 'CRPF', 'J&K Police']
    weights = [1.0, 1.2, 0.9]  # Adjust if needed
    custom_objects = {'loss': weighted_cce(weights)}

    # Load models and predict
    models = load_models(model_paths, custom_objects=custom_objects)
    predict_image(models, args.image_path, class_names)
