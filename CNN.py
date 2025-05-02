import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, Model

# Set random seeds (Ensures the same output of graphs each time)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Load data
data = pd.read_csv("data.csv")
data = data.drop(["id", "Unnamed: 32"], axis=1)
data["diagnosis"] = LabelEncoder().fit_transform(data["diagnosis"])
X = data.drop("diagnosis", axis=1).values
y = data["diagnosis"].values
X = StandardScaler().fit_transform(X)
X = X.reshape(-1, 5, 6, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# Build and train model
model = Sequential(
    [
        Conv2D(32, (2, 2), activation="relu", input_shape=(5, 6, 1), name="conv_1"),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),
        Conv2D(64, (2, 2), activation="relu", name="conv_2"),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, shuffle=False
)

model.evaluate(X_test, y_test)

# Ensure model is built
_ = model(X_test[:1])
last_conv_layer = model.get_layer("conv_2")

# Feature names
feature_names = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

feature_grid = np.array(feature_names).reshape(5, 6)


# Compute Grad-CAM and return heatmap
def compute_gradcam(sample):
    grad_model = Model(
        inputs=model.inputs[0], outputs=[last_conv_layer.output, model.outputs[0]]
    )
    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(sample)
        loss = prediction[0]

    grads = tape.gradient(loss, conv_output)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1)
    cam = np.maximum(cam, 0)
    pred_score = prediction.numpy().item()  # ✅ fixed: convert to scalar safely
    if np.max(cam) == 0:
        return None, pred_score
    return tf.image.resize(cam[..., tf.newaxis], (5, 6)).numpy().squeeze(), pred_score


# Find one benign and one malignant sample
benign_cam, malignant_cam = None, None
benign_pred, malignant_pred = None, None
benign_idx, malignant_idx = None, None

for i in range(len(X_test)):
    sample = X_test[i : i + 1]
    label = y_test[i]
    cam, pred = compute_gradcam(sample)
    if cam is not None:
        if label == 0 and benign_cam is None:
            benign_cam, benign_pred, benign_idx = cam, pred, i
        elif label == 1 and malignant_cam is None:
            malignant_cam, malignant_pred, malignant_idx = cam, pred, i
    if benign_cam is not None and malignant_cam is not None:
        break

print(f"Benign sample index: {benign_idx}, prediction score: {benign_pred:.4f}")

print(
    f"Malignant sample index: {malignant_idx}, prediction score: {malignant_pred:.4f}"
)

# Plot both heatmaps side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

for ax, cam, title, pred in zip(
    axs,
    [benign_cam, malignant_cam],
    ["Benign", "Malignant"],
    [benign_pred, malignant_pred],
):
    heatmap = ax.imshow(cam, cmap="jet", alpha=0.7)
    ax.set_title(f"{title} (Pred: {pred:.2f})", fontsize=14)
    for i in range(5):
        for j in range(6):
            ax.text(
                j,
                i,
                feature_grid[i, j],
                ha="center",
                va="center",
                fontsize=5,
                color="white",
                fontweight="bold",
            )
    plt.colorbar(heatmap, ax=ax)

# Plot Grad Cam heatmap
plt.tight_layout()
plt.show()

# Predict probabilities
y_probs = model.predict(X_test).ravel()
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"Mean ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="black", lw=2, linestyle="--", label="Random Guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("CNN - ROC Curve", fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True)
plt.tight_layout()
plt.show()

val_accuracies = history.history["val_accuracy"]

plt.figure(figsize=(7, 5))
box = plt.boxplot(val_accuracies, patch_artist=False, boxprops=dict(linewidth=2))

for median in box["medians"]:
    median.set(color="black", linewidth=3)

# Plot boxplot
plt.title("CNN Validation Accuracy Distribution Across 20 Epochs", fontsize=13)
plt.ylabel("Accuracy", fontsize=11)
plt.xticks([1], ["CNN (20 Epochs)"])
plt.grid(True, axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
