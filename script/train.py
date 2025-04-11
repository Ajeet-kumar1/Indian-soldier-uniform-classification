
import tensorflow as tf
import numpy as np
import os
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


# -------- Parameters --------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 3
DATA_DIR = r"C:\Users\Ajeet\Downloads\Masters\Coding\Python\uniform_classification\dataset"

# -------- Load image paths and labels --------
class_names = sorted(os.listdir(DATA_DIR))
image_paths = []
labels = []

for idx, cls in enumerate(class_names):
    cls_folder = os.path.join(DATA_DIR, cls)
    for fname in os.listdir(cls_folder):
        image_paths.append(os.path.join(cls_folder, fname))
        labels.append(idx)

image_paths = np.array(image_paths)
labels = np.array(labels)

# -------- Train/Val split --------
val_indices = []
train_indices = []

for cls in range(len(class_names)):
    cls_indices = np.where(labels == cls)[0]
    cls_val_indices = np.random.choice(cls_indices, size=30, replace=False)
    cls_train_indices = np.setdiff1d(cls_indices, cls_val_indices)
    
    val_indices.extend(cls_val_indices)
    train_indices.extend(cls_train_indices)

x_train, y_train = image_paths[train_indices], labels[train_indices]
x_val, y_val = image_paths[val_indices], labels[val_indices]

# -------- Class Weights --------
class_counts = Counter(y_train)
total = sum(class_counts.values())
class_weights = {i: total / (len(class_counts) * count) for i, count in class_counts.items()}
print("Class weights:", class_weights)

# -------- Image Preprocessing --------
def process_img(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return image, tf.one_hot(label, depth=NUM_CLASSES)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(process_img).shuffle(500).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(process_img).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# -------- Custom Loss with Class Weights --------
def weighted_cce(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_true_idx = tf.argmax(y_true, axis=1)
        weights_per_sample = tf.gather(weights, y_true_idx)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        return ce * weights_per_sample
    return loss

loss_fn = weighted_cce([class_weights[i] for i in range(NUM_CLASSES)])

# -------- Base Model Builder --------
def build_model(base, name):
    base_model = base(include_top=False, weights='imagenet', input_shape=(*IMG_SIZE, 3))
    base_model.trainable = True

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ], name=name)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss_fn, metrics=['accuracy'])
    return model

# -------- Train Each Base Model --------
models = {
    "ResNet50": build_model(tf.keras.applications.ResNet50, "resnet50"),
    "ResNet101": build_model(tf.keras.applications.ResNet101, "resnet101"),
    "InceptionV3": build_model(tf.keras.applications.InceptionV3, "inceptionv3"),
}

histories = {}

for name, model in models.items():
    print(f"\nTraining {name}...\n")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{name}_best_model.h5", save_best_only=True, monitor='val_accuracy', mode='max')
    earlystop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy', mode='max')

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint, earlystop])
    histories[name] = history

# -------- Ensemble Evaluation (Soft Voting) --------
def evaluate_ensemble(models, dataset):
    preds = []
    for model in models.values():
        preds.append(model.predict(dataset))

    avg_preds = tf.reduce_mean(tf.stack(preds), axis=0)
    y_true = tf.concat([y for _, y in dataset], axis=0)
    y_true_labels = tf.argmax(y_true, axis=1).numpy()
    y_pred_labels = tf.argmax(avg_preds, axis=1).numpy()

    # -------- Accuracy --------
    acc = tf.reduce_mean(tf.cast(tf.equal(y_pred_labels, y_true_labels), tf.float32))
    print(f"\n Ensemble Accuracy: {acc.numpy()*100:.2f}%")

    # -------- Confusion Matrix --------
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # -------- Precision, Recall, F1 --------
    print("\n Classification Report:")
    print(classification_report(y_true_labels, y_pred_labels, target_names=class_names))

    # -------- ROC Curve --------
    fpr = {}
    tpr = {}
    roc_auc = {}
    y_true_np = y_true.numpy()
    avg_preds_np = avg_preds.numpy()

    plt.figure(figsize=(8, 6))
    for i in range(NUM_CLASSES):
        fpr[i], tpr[i], _ = roc_curve(y_true_np[:, i], avg_preds_np[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (One-vs-Rest)")
    plt.legend()
    plt.grid(True)
    plt.show()

evaluate_ensemble(models, val_ds)