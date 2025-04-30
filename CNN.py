import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Load the datasetaqa!
data = pd.read_csv("data.csv")
data = data.drop(["id", "Unnamed: 32"], axis=1)

# Encode target: M=1, B=0
data["diagnosis"] = LabelEncoder().fit_transform(data["diagnosis"])

# Get features and labels
X = data.drop("diagnosis", axis=1).values
y = data["diagnosis"].values

# Normalize features
X = StandardScaler().fit_transform(X)

# Reshape for CNN: (n_samples, 5, 6, 1)
X = X.reshape(-1, 5, 6, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build CNN model
model = Sequential(
    [
        Conv2D(32, (2, 2), activation="relu", input_shape=(5, 6, 1)),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),
        Conv2D(64, (2, 2), activation="relu"),
        MaxPooling2D(pool_size=(1, 1)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="sigmoid"),
    ]
)

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Re-load data
data = pd.read_csv("/Users/elvinnguyen/Desktop/CS4210/CS-4210-Project/data.csv").drop(
    ["id", "Unnamed: 32"], axis=1
)
data["diagnosis"] = LabelEncoder().fit_transform(data["diagnosis"])

X = StandardScaler().fit_transform(data.drop("diagnosis", axis=1).values)
X = X.reshape(-1, 5, 6, 1)
y = data["diagnosis"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Functional model for Grad-CAM
inputs = Input(shape=(5, 6, 1))
x = Conv2D(16, (2, 2), activation="relu", name="conv")(inputs)
x = MaxPooling2D((1, 1))(x)
x = Flatten()(x)
x = Dense(64, activation="relu")(x)
outputs = Dense(1, activation="sigmoid")(x)
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Pick input for Grad-CAM
sample = X_test[0:1]
pred_class = int(model.predict(sample)[0] > 0.5)

# Grad-CAM
grad_model = Model(
    inputs=model.inputs, outputs=[model.get_layer("conv").output, model.output]
)
with tf.GradientTape() as tape:
    conv_output, prediction = grad_model(sample)
    loss = prediction[0]

grads = tape.gradient(loss, conv_output)[0]
weights = tf.reduce_mean(grads, axis=(0, 1))
cam = tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1)
cam = np.maximum(cam, 0)
cam = cam / np.max(cam)

# Resize CAM to match input shape (5x6)
cam_resized = tf.image.resize(cam[..., tf.newaxis], (5, 6)).numpy().squeeze()

# Define 5x6 feature grid (30 features)
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

# Plot only labeled Grad-CAM heatmap
fig, ax = plt.subplots(figsize=(10, 7))
heatmap = ax.imshow(cam_resized, cmap="jet", alpha=0.7)
plt.colorbar(heatmap, ax=ax)
ax.set_title(
    f"Grad-CAM with Feature Labels (Class: {'Malignant' if pred_class else 'Benign'})"
)

# Overlay text labels
for i in range(5):
    for j in range(6):
        ax.text(
            j,
            i,
            feature_grid[i, j],
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            wrap=True,
        )

plt.tight_layout()
plt.show()
