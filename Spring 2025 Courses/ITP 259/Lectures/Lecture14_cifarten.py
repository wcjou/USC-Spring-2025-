from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
# x_train: uint8 NumPy array of grayscale image data with shapes (50000, 32, 32, 3),
# containing the training data. Pixel values range from 0 to 255.
#
# y_train: uint8 NumPy array of labels (integers in range 0-9)
# with shape (50000, 1) for the training data.
#
# x_test: uint8 NumPy array of grayscale image data with shapes (10000, 32, 32, 3),
# containing the test data. Pixel values range from 0 to 255.
#
# y_test: uint8 NumPy array of labels (integers in range 0-9)
# with shape (10000, 1) for the test data.

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualize 25 images from train dataset
plt.figure(figsize=[10,10])
for i in range (25):
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(X_train[i])
  plt.xlabel(class_names[y_train[i,0]])
plt.show()

# Scale the data for faster convergence during draining
X_train = X_train/255
X_test = X_test/255

# one-hot encoding (create dummy variables for the target variable). This is required
# for the loss function categorical cross-entropy.
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# Build a stack of layers with the sequential model from keras
model = Sequential()

# convolutional layer
# padding = "valid" means no padding.
# "same" results in padding with zeros evenly to the left/right
# or up/down of the input such that output has the same height/width dimension as the input.
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1),
                 padding='same', activation='relu', input_shape=(32, 32, 3)))

# convolutional layer
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

# The Dropout layer randomly sets input units to 0 with a frequency of rate
# at each step during training time, which helps prevent overfitting.
# In Keras, the dropout rate is the fraction of nodes to be dropped in each epoch
model.add(Dropout(0.25))

model.add(Conv2D(100, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.3))

# output layer
model.add(Dense(10, activation='softmax'))

model.summary()

# compile the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# train the model for 50 epochs
history = model.fit(X_train, Y_train, batch_size=64, epochs=30, validation_data=(X_test, Y_test))

# Loss curve
plt.figure(figsize=[6,4])
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)

# Accuracy curve
plt.figure(figsize=[6,4])
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)

# Predict
pred = model.predict(X_test)

# Convert the predictions into label index
pred_classes = np.argmax(pred, axis=1)

# Plot the Actual vs. Predicted results
plt.figure(figsize=[10,10])
for i in range (25):
    plt.subplot(5, 5, i+1).imshow(X_test[i])
    plt.subplot(5, 5, i+1).set_title("True: %s \nPredict: %s" %
                      (class_names[y_test[i, 0]],
                       class_names[pred_classes[i]]))
    plt.subplot(5, 5, i+1).axis('off')

plt.subplots_adjust(hspace=1)
plt.show()