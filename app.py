import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import tensorflow as tf

# ----------- Model and Prediction Logic -----------

def weighted_cce(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true_idx = tf.argmax(y_true, axis=1)
        weights_per_sample = tf.gather(weights, y_true_idx)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return ce * weights_per_sample
    return loss

def load_models(model_paths, custom_objects=None):
    models = {}
    for name, path in model_paths.items():
        models[name] = tf.keras.models.load_model(path, custom_objects=custom_objects)
    return models

def load_and_preprocess_image(image_path, img_size=(224, 224)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return tf.expand_dims(image, axis=0)

def predict_image(models, image_path, class_names):
    image = load_and_preprocess_image(image_path)
    preds = [model.predict(image, verbose=0) for model in models.values()]
    avg_pred = tf.reduce_mean(tf.stack(preds), axis=0)

    predicted_class = tf.argmax(avg_pred, axis=1).numpy()[0]
    probabilities = avg_pred.numpy()[0]
    return class_names[predicted_class], probabilities

# ----------- GUI -----------

class UniformClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Uniform Classifier")
        self.root.geometry("500x600")

        # Class and model setup
        self.class_names = ['BSF', 'CRPF', 'J&K Police']
        weights = [1.0, 1.2, 0.9]
        custom_objects = {'loss': weighted_cce(weights)}

        self.model_paths = {
            'resnet50': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\uniform_classification\models\ResNet50_best_model.h5',
            'resnet101': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\uniform_classification\models\resnet101_best_model.h5',
            'inceptionv3': r'C:\Users\Ajeet\Downloads\Masters\Coding\Python\uniform_classification\models\InceptionV3_best_model.h5',
        }

        self.models = load_models(self.model_paths, custom_objects=custom_objects)
        self.image_path = None

        # GUI Elements
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, font=('Arial', 12))
        self.upload_btn.pack(pady=10)

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.result_label = tk.Label(root, text="", font=("Helvetica", 16), fg="blue")
        self.result_label.pack(pady=20)

        self.predict_btn = tk.Button(root, text="Predict", command=self.run_prediction, font=('Arial', 12), state='disabled')
        self.predict_btn.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        self.image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        self.canvas.configure(image=img_tk)
        self.canvas.image = img_tk
        self.result_label.config(text="")
        self.predict_btn.config(state='normal')

    def run_prediction(self):
        if not self.image_path:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        label, _ = predict_image(self.models, self.image_path, self.class_names)
        self.result_label.config(text=f"Predicted: {label}")

# ----------- Driver code -----------

if __name__ == "__main__":
    root = tk.Tk()
    app = UniformClassifierApp(root)
    root.mainloop()
