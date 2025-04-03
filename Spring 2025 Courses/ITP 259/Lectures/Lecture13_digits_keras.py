# keras digits mnist

import random
from tensorflow import keras
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Import datasets
train_df = pd.read_csv("mnist_train.csv")
test_df = pd.read_csv("mnist_test.csv")

# print(train_df.head())
# print(test_df.head())

# Explore the distribution of digits in the train dataset
plt.figure(1)
sb.countplot(x="label", data=train_df)
plt.show()

# Visualize the first digit in the training dataset
print("The digit is", train_df.iloc[0,0])
pixel = train_df.iloc[0,1:]
print(pixel)

# Shape pixels as a 28x28 array
pixel = np.array(pixel)
pixel = pixel.reshape(28,28)
# print(pixel)

# Plot pixel array
plt.imshow(pixel, cmap="gray")
plt.title("The digit is " + str(train_df.iloc[0,0]))
plt.show()

# Train test datasets
X_train = train_df.iloc[:,1:]
Y_train = train_df.iloc[:,0]
X_test = test_df.iloc[:,1:]
Y_test = test_df.iloc[:,0]
print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

# Scale the feature set
X_train = X_train/255
X_test = X_test/255

# Instantiate keras model
# Three hidden layers each with 100 neurons
model = keras.models.Sequential()
model.add(keras.Input(shape=(784,)))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

# Display model summary
model.summary()

# set the loss function and the optimizer
model.compile(loss = "sparse_categorical_crossentropy",
              optimizer = "sgd", metrics = ["accuracy"])
# Training
h = model.fit(X_train, Y_train, epochs = 25, verbose=1)

# Plot loss curve
pd.DataFrame(h.history).plot()
plt.show()

# Accuracy
model.evaluate(X_test,Y_test)

# Confusion matrix
y_pred = model.predict(X_test)
# print(y_pred)

# What do you observe about y_pred above? Why do we use argmax below?
y_pred = np.argmax(y_pred, axis=1)
# print(y_pred)

cm = confusion_matrix(y_pred, Y_test)
print("The confusion matrix", cm)

# Predict
image_number = random.randint(0, 10000)
test_sample = np.array(X_test.iloc[image_number]).reshape(28, 28)
plt.imshow(test_sample, cmap="gray")
plt.title("The predicted digit is "+ str(y_pred[image_number]))
plt.show()

# Display a failed prediction
# Filter the test dataframe to those cases where the prediction failed
failed_df = X_test[y_pred != Y_test]
# failed_df is a dataframe with 255 rows and 784 columns. The index is filtered from X_Test
# where the predicted and actual y did not match.
print("Dataframe of incorrect predictions ", failed_df)

# Pick a random row index from the failed dataframe
failed_index = failed_df.sample(n=1).index
print("The index of the row of a random incorrect prediction ", failed_index)

# Now unflatten the row at the failed index.
failed_sample = np.array(X_test.iloc[failed_index]).reshape(28, 28)
print("The failed sample is ", failed_sample)

# plot the incorrectly predicted digit.
# Show its actual and its predicted values
plt.imshow(failed_sample, cmap="gray")

# y_pred is a 1D numpy array. (10000, ). We can access an array element
# with an index number which is the same index as the failed_index.
# Y_test is a pandas series. We can access its value using the same
# index as the failed_index.

plt.title("The failed predicted digit is "+ str(y_pred[failed_index]) +
          " whereas the actual digit is" + str(Y_test[failed_index].values))
plt.show()