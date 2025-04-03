##### DISCLAIMER ######
# This file is intended as an overview how to solve image classification problems using MLP and Keras
# This file will NOT run in Python as is
# Throughout the file implementation details have been abstracted and need to be
# adapted to the correct variable names and parameters
# Also this file presents an overview of various code paths. In a true implementation
# only one particular path will be chosen.

# IMPORTS

import random
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sb
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Import dataset
dataframe = pd.read_csv("...")
# or load from a web source (example from HW 8):
(X_train, y_train), (X_test, y_test) = cifar100.load_data("fine")

# Get a feel for the data using Python
print(dataframe.head())
print(dataframe.shape)

# Extract features and target variable
# make sure to exclude a header row if present
# make sure to correctly identify the target colum (first one vs. last one for example)
X = dataframe.iloc[0:,1:]
y = dataframe.iloc[0:,0]
# print(X.shape)
# print(y.shape)

# Create a mapping dictionary to convert categorical variables to integers
# You could also just use an array and make sure to define the right order
# Mapping numbers to letters
dict = {0:class 1 ,
        1: class 2,
        2: ,
        ...
        }

# Explore the distribution of classes in the train dataset
plt.figure(1)
ax = sb.countplot(x="label", data=dataframe)
ax.set_xticklabels(dict.values())
plt.show()

# Display zzz random samples
plt.figure(figsize=( , ))
for i in range(zzz):
    plt.subplot(yyy,yyy,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    r = random.randint(1, dataframe.shape[0]) # Visualize a random row in the dataset
    pixel = X.iloc[r]
    pixel = np.array(pixel)
    # in some problem sets you won't need to reshape, for example in HW 8
    pixel = pixel.reshape(xxx,xxx) # Shape pixels as a bbb x bbb array
    plt.imshow(pixel, cmap=plt.cm.binary)
    plt.title(dict[y[r]])
plt.show()

#################
# Data Preprocessing
#################

# Train test datasets; obviously not needed in case the data already comes
# in a split fashion, like in HW 8
X_train, X_test , Y_train, Y_test = \
    train_test_split(X, y,test_size=0.30, random_state=2023, stratify=y)

# Always keep track of shapes, it's the biggest source of errors
print(X_train.shape, Y_train.shape, X_test.shape,Y_test.shape)

# Normalize
# Normalize when there is a min and max to the data
X_train = X_train/255
X_test = X_test/255

# Normalization using Normalizer()
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(X)
X = pd.DataFrame(normalizer.transform(X), columns=X.columns)
#################

#################
# Building the Model
# Below is an MLP example (1)
# A Keras FF NN example (2)
# A Keras CNN example (3)
#################

# 1  MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(a, b, ...), activation=" ",
                    max_iter= , alpha=1e-3, solver="adam",
                    random_state=2023, learning_rate_init=   , verbose=True)

# MLP Training
mlp.fit(X_train, Y_train)

#################

# 2  this is how a FF NN looks like in Keras:
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(bbb, bbb)))
model.add(keras.layers.Dense( , activation="relu"))
model.add(keras.layers.Dense( , activation="relu"))
model.add(keras.layers.Dense(NUMBER_OUTPUT_CLASSES, activation="softmax"))

#################

# 3  CNN layer setup will look like this:
model.add(Conv2D(Number Filters in Layer 1, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# The Dropout layer randomly sets input units to 0 with a frequency of rate
# at each step during training time, which helps prevent overfitting.
model.add(Dropout(0.25))
# [... MORE LAYERS ...]

# flatten output of conv
model.add(Flatten())

# hidden layer
model.add(Dense( , activation='relu'))
model.add(Dropout(0.4))
# [... MORE LAYERS ...]

# output layer
model.add(Dense(NUMBER_OUTPUT_CLASSES, activation='softmax'))

#################

model.summary()
# compile the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# train the model
h = model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=(X_test, Y_test))

# we had to specify the loss function above.
# categorical_crossentropy requires that your data is one-hot encoded and hence converted into categorical format.
# Often, this is not what your dataset looks like when you'll start creating your models. Rather, you
# likely have feature vectors with integer targets - such as 0 to 9 for the numbers 0 to 9.

# This means that you'll have to convert these targets first. In Keras, this can be done with to_categorical,
# which essentially applies one-hot encoding to your training set's targets. When applied, you can start using
# categorical crossentropy.

# This code does the conversion:
from keras.utils import to_categorical
n_classes = 100
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

# When you have integer targets instead of categorical vectors as targets, you can use
# sparse categorical crossentropy. It's an integer-based version of the categorical crossentropy
# loss function, which means that we don't have to convert the targets into categorical format anymore.

# Set the loss function and the optimizer for SCCE:
model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd", metrics=["accuracy"])

#################
# Model Evaluation & Plotting the Loss
#################

# Plot loss curve for MLP
plt.plot(mlp.loss_curve_)
plt.show()

# Model Evaluation for MLP
print("The accuracy is", mlp.score(X_test,Y_test))

# Evaluation for Keras
# Plot the loss curve
pd.DataFrame(h.history).plot()
plt.show()

# Evaluate the Keras model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy:', test_acc)

# Loss curve for Keras
plt.figure(figsize=[ , ])
plt.plot(h.history['loss'], 'black', linewidth=2.0)
plt.plot(h.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)

# 9 Accuracy curve
plt.figure(figsize=[ , ])
plt.plot(h.history['accuracy'], 'black', linewidth=2.0)
plt.plot(h.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)

#################
# Prediction
#################

# Predict is identical for both models:
# MLP.predict returns the predicted classes. To get the probabilities, we need to use predict_proba()
y_pred = mlp.predict(X_test)
# Keras returns the probabilities for each class, that is why we need to run argmax later one to identify the
# 'winning' class
y_pred = model.predict(test_images)
# Convert the predictions into label index
y_pred = np.argmax(y_pred, axis=1)

# Confusion matrix for MLP and Keras are identical, of course use the right labels
# we need the reverse dictionary to correctly label the ConfusionMatrix or displays
cm = confusion_matrix(respective labels, y_pred, labels=respective classes)

ConfusionMatrixDisplay(confusion_matrix=cm,
                       display_labels=[...]).plot()
plt.show()

#################
# Showing results; identical between Keras and MLP
#################

# Predicted vs actual
# We use the dictionary to convert the integers back to the categories (e.g. letters)
# you only need to reshape to get back to a bbb x bbb format. In other problems we already have that format (e.g. HW 6)
sample = np.array(X_test.iloc[0,:]).reshape(bbb, bbb)
plt.imshow(sample, cmap="gray")
plt.title("The predicted target is " + str(dict[y_pred[0]])
          + " and the actual label is " + str(dict[Y_test.iloc[0]]))
# ^ mind the .iloc for a dataframe!
plt.show()

# Display a failed prediction
# Filter the test dataframe to those cases where the prediction failed
#### CAREFUL: HERE WAS AN ERROR, y_test needs to be lower case
failed_df = X_test[y_pred != y_test]

# Pick a random row from the failed dataframe
# .sample() only works for a dataframe
failed_index = failed_df.sample(n=1).index
# print(failed_df)
# However, we need to retrieve the index of the failed sample in our pandas df, we just pick the first
# one in the failed_index list:
#### CAREFUL: HERE WAS AN ERROR, y_test needs to be lower case
req_id = y_test.index.to_list().index(failed_index)
# or
# req_id = Y_test.index.get_loc(failed_index[0])

#################
# If you only have an array (not a dataframe), we need to find the failed indices manually:
# get all failed prediction indexes
failed_indices = []
idx = 0

# clunky way of identifying failed labels
for i in test_labels:
    if i != y_pred[idx]:
        failed_indices.append(idx)
    idx = idx + 1

# pick a random failed predict
random_select = np.random.randint(0, len(failed_indices))
failed_index = failed_indices[random_select]
#################

failed_sample = np.array(X_test.loc[failed_index]).reshape(bbb,bbb)
#### CAREFUL: HERE WAS AN ERROR, y_test needs to be lower case
req_id = y_test.index.to_list().index(failed_index)
# or
# req_id = y_test.index.get_loc(failed_index[0])

# #plot the target and its predicted value
plt.imshow(failed_sample, cmap="gray")
#### CAREFUL: HERE WAS AN ERROR, y_test needs to be lower case
plt.title("The failed predicted target is "+ str(dict[int(y_pred[req_id])])
         + " whereas the actual label is " + str(dict[int(y_test.iloc[req_id])]))
plt.show()

#################
# Showing misclassification results in a subplot=array
#################

# Plot code for random 30 misclassified images
for i in range (30):
    random_select = np.random.randint(0, len(failed_indices))
    failed_index = failed_indices[random_select]
    plt.subplot(, , i+1).imshow(X_test[failed_index])
#### CAREFUL: HERE WAS AN ERROR, y_test needs to be lower case
    plt.subplot(, , i+1).set_title("True: %s \nPredict: %s" %
                      (dict[y_test[ failed_index,0]],
                       dict[y_pred[failed_index]]))
    plt.subplot(, , i+1).axis('off')
