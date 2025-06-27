import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Paths
PROC_DIR = os.path.join("data", "processed", "dynamic")
MODEL_DIR = os.path.join("models")

# 1. Load all sequences and labels
X, y = [], []
for label_idx, label in enumerate(("J", "Z")):
    seq_files = glob.glob(os.path.join(PROC_DIR, label, "*.npy"))
    for fp in seq_files:
        seq = np.load(fp)        # shape: (30,2)
        X.append(seq.flatten())  # shape: (60,)
        y.append(label_idx)

X = np.stack(X)
y = np.array(y)

print(f"Loaded {len(X)} samples: {np.bincount(y)} (J, Z)")

# 2. Train/validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build a small MLP
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(2, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16
)

# 5. Save the model
os.makedirs(MODEL_DIR, exist_ok=True)
model.save(os.path.join(MODEL_DIR, "asl_dynamic_ml.keras"))

print("Dynamic J/Z model saved to models/asl_dynamic_ml.keras")
