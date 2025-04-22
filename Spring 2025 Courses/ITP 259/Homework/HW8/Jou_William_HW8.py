from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 1. Loading the CIFAR-100 dataset and storing them as train and test sets

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')


# 2. Printing the shape of the train and test sets
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle'
               , 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle'
               , 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur'
               , 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard'
               , 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain'
               , 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree'
               , 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket'
               , 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider'
               , 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor'
               , 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']


# 3. Visualizing the first 30 images from the training dataset

plt.figure(figsize=[10,10])
for i in range (30):
  plt.subplot(5, 6, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(X_train[i])
  plt.xlabel(class_names[y_train[i,0]])
plt.show()


# 4. Scale the pixel values

X_train = X_train/255
X_test = X_test/255


# 5. One-hot encode the classes to use the categorical cross-entropy loss function
n_classes = len(class_names)
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


# 6. Build a sequential model

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPool2D(pool_size=(2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])


# 7. Use the loss function categorical cross-entropy when compiling the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 8. Train the model with atleast 10 epochs

history = model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_data=(X_test, Y_test))


# 9. Plot the loss and accuracy curves for both train and validation sets

# Loss curve
plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)

plt.show()

# Accuracy curve
plt.figure(figsize=[6,4])
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)

plt.show()


#  10. Visualize the predicted and actual image labels for the first 30 images in teh dataset

predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(10, 10))
for i in range (30):
    plt.subplot(5, 6, i+1).imshow(X_test[i])
    plt.subplot(5, 6, i+1).set_title("True: %s \nPredict: %s" %
                      (class_names[y_test[i, 0]],
                       class_names[predicted_labels[i]]))
    plt.subplot(5, 6, i+1).axis('off')

plt.subplots_adjust(hspace=1)
plt.show()


# 11. Visualize 30 random misclassified images
misclassified_indices = np.where(predicted_labels != y_test.flatten())[0]
random_indices = np.random.choice(misclassified_indices, size=30, replace=False)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(random_indices):
    plt.subplot(5, 6, i+1).imshow(X_test[idx])
    plt.subplot(5, 6, i+1).set_title("True: %s \nPredict: %s" %
                      (class_names[y_test[idx, 0]],
                       class_names[predicted_labels[idx]]))
    plt.subplot(5, 6, i+1).axis('off')
plt.subplots_adjust(hspace=1)
plt.show()
