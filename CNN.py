import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
# from keras.models import Sequential


# Load the dataset
data = pd.read_csv("data.csv")
df = pd.read_csv("/Users/elvinnguyen/Desktop/CS4210/CS-4210-Project/data.csv")
df = df.drop(["id", "Unnamed: 32"], axis=1)

# Encode target
df["diagnosis"] = LabelEncoder().fit_transform(df["diagnosis"])  # M=1, B=0

# Get features and labels 
X = df.drop("diagnosis", axis=1).values
y = df["diagnosis"].values

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

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
