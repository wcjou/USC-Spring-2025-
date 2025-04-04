# William Jou
# ITP 259, Spring 2025
# Homework 6
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf



fashion = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)  
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)


class_names = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

plt.figure(figsize=(8, 5))
plt.hist(train_labels, edgecolor='black', alpha=0.7)
plt.xticks(range(10), class_names)
plt.xlabel("Apparel Type")
plt.ylabel("Count")
plt.title("Distribution of Apparel in Training Data")
plt.show()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    idx = np.random.randint(0, train_images.shape[0])
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(class_names[train_labels[idx]])
    plt.axis('off')
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_images, train_labels, epochs=100, batch_size=32, validation_split=0.2)

plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

plt.imshow(test_images[0], cmap='gray')
prediction = model.predict(test_images[0].reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
plt.title(f"Actual: {class_names[test_labels[0]]} | Predicted: {class_names[predicted_label]}")
plt.show()

predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
misclassified_idx = np.where(predicted_labels != test_labels)[0][0]
plt.imshow(test_images[misclassified_idx], cmap='gray')
plt.title(f"Actual: {class_names[test_labels[misclassified_idx]]} | Predicted: {class_names[predicted_labels[misclassified_idx]]}")
plt.show()