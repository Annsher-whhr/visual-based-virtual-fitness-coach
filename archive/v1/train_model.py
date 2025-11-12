import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
layers = tf.keras.layers

# Load dataset files relative to this script's directory so users can run from project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_PATH = os.path.join(BASE_DIR, "X.npy")
Y_PATH = os.path.join(BASE_DIR, "y.npy")

if not os.path.exists(X_PATH) or not os.path.exists(Y_PATH):
    raise FileNotFoundError(
        f"Required dataset files not found. Please run 'python generate_data.py' in '{BASE_DIR}' or copy X.npy/y.npy there.\n"
        f"Expected files:\n  {X_PATH}\n  {Y_PATH}\n"
    )

X = np.load(X_PATH)
y = np.load(Y_PATH)

model = keras.Sequential([
    layers.Input(shape=(X.shape[1],)),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(64, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X, y, batch_size=32, epochs=40, validation_split=0.15, verbose=2)

model.save("taichi_mlp.h5")
print("✅ Model saved as taichi_mlp.h5")
