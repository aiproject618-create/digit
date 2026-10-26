"""
Train a compact CNN on the MNIST handwritten digit dataset.
Run: python train_model.py
Output: digit_model.keras
"""
import tensorflow as tf
from tensorflow.keras import layers, models

SEED = 42
tf.keras.utils.set_random_seed(SEED)

print("Downloading/loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# MNIST images are 28x28 grayscale; pixel values are scaled to [0, 1].
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., None]
x_test = x_test[..., None]

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.25),
    layers.Dense(10, activation="softmax"),
], name="digit_recognition_cnn")

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    restore_best_weights=True,
)

print("Training model...")
model.fit(
    x_train,
    y_train,
    validation_split=0.1,
    epochs=8,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1,
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"MNIST test accuracy: {accuracy * 100:.2f}%")
model.save("digit_model.keras")
print("Saved trained model to digit_model.keras")
