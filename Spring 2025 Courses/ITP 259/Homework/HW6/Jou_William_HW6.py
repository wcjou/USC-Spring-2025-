# William Jou
# ITP 259, Spring 2025
# Homework 6

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# 1. Loading the data
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()

#2. Printing the shape of the data
print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)  
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)

# 3 & 4. Creating a dictionary to map the label numbers to the class names
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

# 5. Plotting the distribution of the labels in the training data
class_counts = np.bincount(train_labels)
colors = plt.cm.get_cmap('tab10', len(class_counts))
plt.figure(figsize=(8, 5))
bars = plt.bar(range(10), class_counts, edgecolor='black', alpha=0.7, color=colors(range(10)))
plt.xticks(range(10), class_names.values(), rotation=45)
plt.xlabel("Apparel Type")
plt.ylabel("Count")
plt.title("Distribution of Apparel in Training Data")
plt.show()

# 6. Plotting 25 random images from the training data
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    idx = np.random.randint(0, train_images.shape[0])
    plt.imshow(train_images[idx], cmap='gray')
    plt.title(class_names[train_labels[idx]])
    plt.axis('off')
plt.show()

# 7. Scaling the train and test features
train_images = train_images / 255.0
test_images = test_images / 255.0

# 8 & 9. Creating a keras model with 1 flatten layer, 2 dense layers, and 1 output layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 10. Getting a summary of the model
model.summary()

# 11. Compiling the model with SGD optimizer and sparse categorical crossentropy loss function and accuracy metric
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 12. Fitting themodel to the training data with 100 epochs and a batch size of 32
history = model.fit(train_images, train_labels, epochs=100, batch_size=32, validation_split=0.2)

# 13. Plotting the loss curves for both training and validation data
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 14. Printing the accuracy of the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc:.4f}")

# 15. Displaying the first image from the test set and its predicted label
plt.imshow(test_images[0], cmap='gray')
prediction = model.predict(test_images[0].reshape(1, 28, 28))
predicted_label = np.argmax(prediction)
plt.title(f"Actual: {class_names[test_labels[0]]} | Predicted: {class_names[predicted_label]}")
plt.show()

# 16. Displaying a misclassified image from the test set
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
misclassified_idx = np.where(predicted_labels != test_labels)[0][0]
plt.imshow(test_images[misclassified_idx], cmap='gray')
plt.title(f"Actual: {class_names[test_labels[misclassified_idx]]} | Predicted: {class_names[predicted_labels[misclassified_idx]]}")
plt.show()