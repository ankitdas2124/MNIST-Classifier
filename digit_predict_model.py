import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers

# data preprocessing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# input verification and visualization
print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Testing data shape: {x_test.shape}")
print(f"Pixel value range: min={x_train.min()}, max={x_train.max()}")
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    img = x_train[i].squeeze()
    plt.imshow(img, cmap="gray")
    plt.title(f"Label: {y_train[i]}")
plt.tight_layout()
plt.show()

# 1. CNN Architecture
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)
model.summary()

# 2. Compilation
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# 3. Training
print("\n--- Starting Training ---")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# 4. Final Evaluation
print("\n--- Final Evaluation ---")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

# 5.saving the model
model.save("mnist_model.keras")
